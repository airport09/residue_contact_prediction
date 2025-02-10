import torch
import torch.nn as nn

from typing import List

from modelgenerator.structure_tokenizer.models.equiformer_encoder import EquiformerEncoder


class StructureEncoder(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str="genbio-ai/AIDO.StructureEncoder") -> None:
        super().__init__()
        self.encoder = EquiformerEncoder.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path
        )
        self.encoder.quantize.frozen = True

    def forward(
        self,
        atom_positions: torch.Tensor,
        attention_mask: torch.Tensor,
        residue_index: torch.Tensor,
        id: List,
        chain_id: List,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        
        output = self.encoder(
            atom_positions=atom_positions,
            attention_mask=attention_mask,
            residue_index=residue_index,
        )
    
        min_encoding_indices = output["idx"]
        struct_tokens = {
            f"{id_}_{chain_id_}": {
                "struct_tokens": min_indices[mask],
                "residue_index": res[mask],
                "emb": emb[mask],
            }
            for id_, chain_id_, min_indices, res, mask, emb in zip(
                id, chain_id, min_encoding_indices, residue_index, attention_mask, output['emb']
            )
        }

        return output['emb'], struct_tokens