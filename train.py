from functools import partial

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import hydra
from omegaconf import DictConfig

from models.esm2 import ESM2
from models.structure_encoder import StructureEncoder
from models.contact_prediction import ContactPredictionLightningModule
from data.protein_dataset import ProteinDataset, ProteinDataModule


@hydra.main(version_base=None, config_path="conf", config_name="train")
def main(cfg: DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm2_model = ESM2(device=device)
    structure_encoder = StructureEncoder()

    datamodule = ProteinDataModule(train_dir=cfg.train_ds.dir,
                                   batch_size=cfg.train_ds.batch_size)
    
    model = ContactPredictionLightningModule(
                    esm2_model=esm2_model,
                    structure_encoder=structure_encoder,
                    collate_fn=partial(ProteinDataset.collate_fn, device=device),
                    pdb_files_dir=cfg.model.pdb_files_dir,
                    local_db_name=cfg.model.local_db_name,
                    d_esm=1280,
                    d_se=384,
                    num_heads=cfg.model.num_heads,
                    num_layers=cfg.model.num_layers,
                    use_bias=cfg.model.use_bias,
                    learning_rate=cfg.model.learning_rate,
                    warmup_steps=cfg.model.warmup_steps,
                )
    
    EXP_NAME = f'train_lr_{cfg.model.learning_rate}_ws_{cfg.model.warmup_steps}_nl_{cfg.model.num_layers}_nh_{cfg.model.num_heads}'


    checkpoint_callback = ModelCheckpoint(
    dirpath=f'{cfg.trainer.checkpoints_dir}/{EXP_NAME}',
    filename='model-{epoch:02d}-{train_loss:.4f}',
    save_top_k=-1,
    every_n_epochs=1,
    monitor='train_loss'
    )
    
    wandb_logger = WandbLogger(name=EXP_NAME, project=cfg.wandb_logger.project, resume="must")

    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs,
                         accelerator=device,
                         logger=wandb_logger, 
                         check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch, 
                         log_every_n_steps=cfg.trainer.log_every_n_steps,
                         accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
                         limit_train_batches=cfg.trainer.limit_train_batches,
                         callbacks=[checkpoint_callback],
                         strategy='ddp_find_unused_parameters_true')

    trainer.fit(model, 
                datamodule=datamodule,
                ckpt_path=cfg.trainer.checkpoint_path)



if __name__ == "__main__":
    main()