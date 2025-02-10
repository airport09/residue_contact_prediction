import json
from pathlib import Path

import torch
import esm
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig

from evaluations.utils import get_metrics
from data.protein_dataset import ProteinDataset


@hydra.main(version_base=None, config_path="../conf", config_name="eval")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()

    dataset = ProteinDataset(dir=cfg.test_ds.dir)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=cfg.test_ds.batch_size,
                            shuffle=False,
                            collate_fn=dataset.collate_fn)
    
    
    Path(cfg.results.dir).mkdir(exist_ok=True, parents=True)
    results_file = Path(cfg.results.dir, cfg.results.filename)
    results_file = open(results_file, 'w')

    try:
        for i, batch in tqdm(enumerate(dataloader)):
            batch_data = [(f"{id}_{chain_id}", codes) for (id, chain_id, codes) in zip(batch['id'], batch['chain_id'], batch['residue_codes'])]
            gt_contacts = batch['residue_contacts']
            
            batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            
            batch_tokens = batch_tokens.to("cuda")

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)

            for id_seq, gt, contacts, batch_len in zip(batch_data, gt_contacts, results['contacts'], batch_lens):
                pred = contacts[:batch_len - 2, :batch_len - 2].cpu()

                metrics = get_metrics(gt, pred)
                metrics.update({"id": id_seq[0], 'seq_len': len(id_seq[1])})
                json.dump(metrics, results_file)
                results_file.write("\n")
                results_file.flush()
    except Exception as e:
        raise e  
    finally:
        results_file.close()
    

if __name__ == "__main__":
    main()

    

    

    
    
    

    
        








