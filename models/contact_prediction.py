import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl

from pathlib import Path
from typing import Dict, List, Tuple

from torch.optim.lr_scheduler import LambdaLR

from data.protein_entity import ProteinExt
from utils.query_local_db import get_top_seqs

from esm.modules import TransformerLayer, gelu, apc, symmetrize


class TransformerLayerCrossAttention(TransformerLayer):

    def forward(
        self, x, key, value, self_attn_mask=None, self_attn_padding_mask=None, need_head_weights=False
    ):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, attn = self.self_attn(
            query=x,
            key=key,
            value=value,
            key_padding_mask=self_attn_padding_mask,
            need_weights=True,
            need_head_weights=need_head_weights,
            attn_mask=self_attn_mask,
        )
        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x, attn
    

class ContactPredictor(nn.Module):
    def __init__(self, d_esm, d_se, num_layers, num_heads, use_bias, top_n_similar_seqs):
        super(ContactPredictor, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerLayerCrossAttention(
                    d_esm,
                    4 * d_esm,
                    num_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                    use_rotary_embeddings=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.query_proj_top1 = nn.Linear(d_esm, d_esm)
        self.key_proj_top1 = nn.Linear(d_se, d_esm)
        self.value_proj_top1 = nn.Linear(d_se, d_esm)

        self.query_proj_top2 = nn.Linear(d_esm, d_esm)
        self.key_proj_top2 = nn.Linear(d_se, d_esm)
        self.value_proj_top2 = nn.Linear(d_se, d_esm)

        self.query_proj_top3 = nn.Linear(d_esm, d_esm)
        self.key_proj_top3 = nn.Linear(d_se, d_esm)
        self.value_proj_top3 = nn.Linear(d_se, d_esm)

        self.query_proj_top = [self.query_proj_top1, self.query_proj_top2, self.query_proj_top3]
        self.key_proj_top = [self.key_proj_top1, self.key_proj_top2, self.key_proj_top3]
        self.value_proj_top = [self.value_proj_top1, self.value_proj_top2, self.value_proj_top3]

        for layer in self.key_proj_top:
            nn.init.xavier_uniform_(layer.weight)

        for layer in self.value_proj_top:
            nn.init.xavier_uniform_(layer.weight)

        for layer in self.query_proj_top:
            nn.init.xavier_uniform_(layer.weight)

        self.attn_weights = nn.Parameter(torch.Tensor([1, 1, 1]), requires_grad=True)
        self.regression = nn.Linear(num_layers * num_heads, 1, use_bias)
        nn.init.xavier_uniform_(self.regression.weight)
        self.final_activation = nn.Sigmoid()
    
    def forward(self, esm_emb, esm_emb_mask, top_seqs, top_seqs_mask):

        attention_weights = []
        
        for i, (emb, mask) in enumerate(zip(top_seqs, top_seqs_mask)):
            query = self.query_proj_top[i](esm_emb).transpose(0, 1)
            key  = self.key_proj_top[i](emb).transpose(0, 1)
            value = self.value_proj_top[i](emb).transpose(0, 1)
            
            emb_attention_weights = []

            for layer in self.layers:
                query, attn = layer(
                    x=query,
                    key=key,
                    value=value,
                    self_attn_mask=esm_emb_mask,
                    self_attn_padding_mask=mask,
                    need_head_weights=True,
                )
                emb_attention_weights.append(attn.transpose(1, 0))

            attention_weights.append(torch.stack(emb_attention_weights, 1))

        normalized_attn_weights = F.softmax(self.attn_weights, dim=0)
        normalized_attn_weights = normalized_attn_weights.view(1, -1, 1, 1, 1, 1)
        average_weights = torch.sum(torch.stack(attention_weights, dim=1) * normalized_attn_weights, dim=1)

        batch_size, layers, heads, seqlen, _ = average_weights.size()
        average_weights = average_weights.view(batch_size, layers * heads, seqlen, seqlen)

        average_weights = average_weights.to(
            self.regression.weight.device
        )
        average_weights = apc(symmetrize(average_weights))
        average_weights = average_weights.permute(0, 2, 3, 1)

        return self.final_activation(self.regression(average_weights).squeeze(3))


class ContactPredictionLightningModule(pl.LightningModule):
    TOP_N_SIMILAR_SEQS = 3
    def __init__(self, 
                 esm2_model, 
                 structure_encoder, 
                 collate_fn, 
                 pdb_files_dir, 
                 local_db_name, 
                 d_esm, 
                 d_se, 
                 num_heads, 
                 num_layers, 
                 use_bias,
                 warmup_steps,
                 learning_rate=1e-4):
        
        super(ContactPredictionLightningModule, self).__init__()
        self.save_hyperparameters(ignore=["esm2_model", "structure_encoder"])
        
        self.esm2_model = esm2_model
        self.structure_encoder = structure_encoder

        self.d_esm = d_esm
        self.d_se = d_se

        #this is for performing similar sequences search in our local db
        self.collate_fn = collate_fn
        self.pdb_files_dir = pdb_files_dir
        self.local_db_name = local_db_name
        
        # Freeze parameters in the pretrained models.
        for param in self.esm2_model.parameters():
            param.requires_grad = False
        for param in self.structure_encoder.parameters():
            param.requires_grad = False
        

        self.contact_predictor = ContactPredictor(d_esm=d_esm, 
                                                  d_se=d_se,
                                                  num_layers=num_layers,
                                                  num_heads=num_heads,
                                                  use_bias=use_bias, 
                                                  top_n_similar_seqs=self.TOP_N_SIMILAR_SEQS)
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        
        self.loss = nn.BCELoss()


    @staticmethod
    def pad_and_mask(embedding, matrices):
        B, T, E = embedding.shape
        N = max([matrice.shape[1] for matrice in matrices])
        
        max_len = max(T, N)
        
        if T < max_len:
            padding = torch.zeros(B, max_len - T, E, device=embedding.device)
            padded_embedding = torch.cat([embedding, padding], dim=1)
            emb_mask = torch.full((padded_embedding.shape[1], padded_embedding.shape[1]), 
                                False, 
                                device=padded_embedding.device)
            
            emb_mask[:, - (max_len - T):] = True
            emb_mask[- (max_len - T):, :] = True
        else:
            padded_embedding = embedding
            emb_mask = torch.full((T, T), False, device=embedding.device)
        
        padded_matrices = []
        matrix_masks = []
        for matrix in matrices:
            B, T, E = matrix.shape
            if T < max_len:
                padding = torch.zeros(B, max_len - T, E, device=matrix.device)
                padded_matrix = torch.cat([matrix, padding], dim=1)
                matrix_mask = torch.cat([torch.full((B, T), False, device=matrix.device), 
                                        torch.full((B, max_len - T), True, device=matrix.device)], dim=1)
            else:
                padded_matrix = matrix
                matrix_mask = torch.full((B, T), False, device=matrix.device)
            
            padded_matrices.append(padded_matrix)
            matrix_masks.append(matrix_mask)
        
        return padded_embedding, emb_mask, padded_matrices, matrix_masks

    
    @staticmethod
    def _to_esm2_input(batch_info: Dict[str, List]) -> List[Tuple]:
        return [
            (f"{id_}_{chain_id}", seq) for (id_, chain_id, seq) in zip(
                batch_info['id'], batch_info['chain_id'], batch_info['residue_codes']
                )
                ]
    
    def get_similar_seq_structure(self, esm2_input: List[Tuple]) -> List[Dict]:

        similar_seqs = [
            get_top_seqs(seq, id_, top_n=self.TOP_N_SIMILAR_SEQS, db_name=self.local_db_name) for (id_, seq) in esm2_input
            ]

        top1_sim = self.collate_fn([
        ProteinExt.from_pdb_file_path(
                Path(self.pdb_files_dir, seq[0].split('_')[0]).with_suffix('.pdb'), chain_id=seq[0].split('_')[1]
                ).to_torch().to_dict() for seq in similar_seqs if seq[0] is not None
    ])
        top1_mask = [seq[0] is None for seq in similar_seqs]

        top2_sim = self.collate_fn([
            ProteinExt.from_pdb_file_path(
                    Path(self.pdb_files_dir, seq[1].split('_')[0]).with_suffix('.pdb'), chain_id=seq[1].split('_')[1]
                    ).to_torch().to_dict() for seq in similar_seqs if seq[1] is not None
        ])

        top2_mask = [seq[1] is None for seq in similar_seqs]

        top3_sim = self.collate_fn([
            ProteinExt.from_pdb_file_path(
                    Path(self.pdb_files_dir, seq[2].split('_')[0]).with_suffix('.pdb'), chain_id=seq[2].split('_')[1]
                    ).to_torch().to_dict() for seq in similar_seqs if seq[2] is not None
        ])

        top3_mask = [seq[2] is None for seq in similar_seqs]

        return [(top1_sim, top1_mask), (top2_sim, top2_mask), (top3_sim, top3_mask)]

    
    def forward(self, esm2_input, structure_encoder_inputs):

        esm2_emb = self.esm2_model(esm2_input)[:, 1:-1, :]
        
        top_structure_embs = []

        with torch.no_grad():
            for input_, masks in structure_encoder_inputs:

                dummy_size = (1, 10, self.d_se)
                embs = []
                if input_:
                    embs = self.structure_encoder(**input_)[0]
                    dummy_size = (1, embs.shape[1], embs.shape[2])

                non_masked_ind = [i for (i, m) in enumerate(masks) if not m]

                all_embs = [torch.zeros(dummy_size, dtype=torch.float32).to(esm2_emb.device)] * esm2_emb.shape[0]
                for (ind, emb) in zip(non_masked_ind, embs):
                    all_embs[ind] = emb.unsqueeze(0)
                
                top_structure_embs.append(torch.concat(all_embs, dim=0))

        input = self.pad_and_mask(esm2_emb, top_structure_embs)
        contact_probs = self.contact_predictor(*input)
        return contact_probs, input[1]
    

    def training_step(self, batch, batch_idx):
        esm2_input = self._to_esm2_input(batch)
        structure_encoder_inputs = self.get_similar_seq_structure(esm2_input)

        contact_probs, residue_masks = self(esm2_input, structure_encoder_inputs)
        contact_probs = torch.masked_select(
            contact_probs, ~residue_masks
            ).view(batch['residue_contacts'].shape)
        
        loss = self.loss(contact_probs, batch['residue_contacts'])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False, logger=True)
        self.log('weight1', self.contact_predictor.attn_weights[0], on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('weight2', self.contact_predictor.attn_weights[1], on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('weight3', self.contact_predictor.attn_weights[2], on_step=False, on_epoch=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        esm2_input = self._to_esm2_input(batch)
        structure_encoder_inputs = self.get_similar_seq_structure(esm2_input)
        
        contact_probs, residue_masks = self(esm2_input, structure_encoder_inputs)

        contact_probs = torch.masked_select(
            contact_probs, ~residue_masks
            ).view(batch['residue_contacts'].shape)
        
        loss = self.loss(contact_probs, batch['residue_contacts'])
        
        preds = (contact_probs > 0.5).float()
        accuracy = (preds == batch['residue_contacts']).float().mean()
        
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", accuracy, prog_bar=True, logger=True)
        return {"val_loss": loss, "val_accuracy": accuracy}
    
    def predict_step(self, batch, batch_idx):
        esm2_input = self._to_esm2_input(batch)
        structure_encoder_inputs = self.get_similar_seq_structure(esm2_input)
        
        contact_probs, residue_masks = self(esm2_input, structure_encoder_inputs)

        contact_probs = torch.masked_select(
            contact_probs, ~residue_masks
            )
        return contact_probs
    
    def configure_optimizers(self):
        optimizer = AdamW(self.contact_predictor.parameters(), lr=self.learning_rate)

        if self.warmup_steps:
            scheduler = LambdaLR(optimizer, lr_lambda=self.noam_lr_lambda)
            return [optimizer], [scheduler]
        else:
            return optimizer
    

    def noam_lr_lambda(self, step):
        if step == 0:
            step = 1
        return (self.d_esm ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
