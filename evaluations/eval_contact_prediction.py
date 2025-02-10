import json
from pathlib import Path

import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig

from evaluations.utils import get_metrics
from models.esm2 import ESM2
from models.structure_encoder import StructureEncoder
from models.contact_prediction import ContactPredictionLightningModule
from data.protein_dataset import ProteinDataModule


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):

    Path(cfg.results.dir).mkdir(exist_ok=True, parents=True)
    results_file = Path(cfg.results.dir, cfg.results.filename)
    results_file = open(results_file, 'w')

    try:

        device = 'cuda' if torch.cuda.is_available() else "cpu"

        esm2_model = ESM2(device=device)
        structure_encoder = StructureEncoder()

        datamodule = ProteinDataModule(train_dir=None,
                                       test_dir=cfg.test_ds.dir,
                                       batch_size=cfg.test_ds.batch_size)

        model = ContactPredictionLightningModule.load_from_checkpoint(cfg.checkpoint.path, 
                                                                      esm2_model=esm2_model, 
                                                                      structure_encoder=structure_encoder)

        trainer = pl.Trainer(devices=[0])
        predictions = trainer.predict(model, datamodule=datamodule)

        for pred, batch in zip(predictions, datamodule.predict_dataloader()):
            pred = pred.view(batch['residue_contacts'].shape)
            metrics = get_metrics(batch['residue_contacts'], pred)

            id = f"{batch['id'][0]}_{batch['chain_id'][0]}"
            seq = batch['residue_codes']
            metrics.update({"id": id, 'seq_len': len(seq[0])})
            json.dump(metrics, results_file)
            results_file.write("\n")
            results_file.flush()
    except Exception as e:
        raise e  
    finally:
        results_file.close()

if __name__ == "__main__":
    main()