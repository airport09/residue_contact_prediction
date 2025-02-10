import random
from typing import Union, Optional
from pathlib import Path
from functools import partial

from tqdm import tqdm
from biotite.structure.io import pdb

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pytorch_lightning as pl

from data.protein_entity import ProteinExt
from modelgenerator.structure_tokenizer.datasets.protein import ( 
    ProteinChainEmpty
)


class ProteinDataset(Dataset):
    def __init__(
        self,
        dir: Union[str, Path],
    ) -> None:
        super().__init__()

        if isinstance(dir, str):
            dir = Path(dir)

        self.pdb_files = list(dir.glob('*.pdb'))
        self.separate_by_chain_ids()

    def separate_by_chain_ids(self):
        self.proteins_chains = []
        for pdb_file in tqdm(self.pdb_files, desc='Separating Proteins by their Chain IDs'):
            atom_array = pdb.get_structure(pdb.PDBFile.read(pdb_file), model=1)
            chains = set(atom_array.chain_id)
            self.proteins_chains.extend([(pdb_file, chain) for chain in chains])
            
    def __getitem__(self, idx):
        protein = None
        while not protein:
            pdb_file, chain = self.proteins_chains[idx]
            try:
                protein = ProteinExt.from_pdb_file_path(pdb_file, chain_id=chain)
            except (ProteinChainEmpty, ProteinChainEmpty):
                idx = random.randint(0, len(self.proteins_chains))

        return protein.to_torch().to_dict()

    def __len__(self):
        return len(self.proteins_chains)

    @staticmethod
    def collate_fn(
        batch: list[dict[str, torch.Tensor | str | None]],
        device: torch.device = 'cpu',
    ) -> dict[str, torch.Tensor | list[str | None]]:
        
        batch_dict = {}

        if not batch:
            return batch_dict

        for key, value in batch[0].items():
            values_for_batch = []

            for protein in batch:
                values_for_batch.append(protein[key])
                
            if key == "attention_mask":
                padding_value = False
            elif key == "residue_contacts":
                padding_value = None 
                max_shape = max([val.shape[0] for val in values_for_batch])
                values_for_batch = [
                    F.pad(val, (0, max_shape - val.shape[0], 0, max_shape - val.shape[0])) for val in values_for_batch
                    ]
                values_for_batch = torch.stack(values_for_batch).to(device)
            elif type(value) == torch.Tensor:
                padding_value = 0
            else:
                padding_value = None 

            if padding_value is not None:
                values_for_batch = pad_sequence(values_for_batch, batch_first=True, padding_value=padding_value).to(device)
            
            batch_dict[key] = values_for_batch

        return batch_dict
    

class ProteinDataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_dir: Union[str, Path],
        test_dir: Optional[Union[str, Path]] = None,
        device: torch.device = 'cpu',
        random_seed: int = 42,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.device = device

        self.random_seed = random_seed
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_dataset = ProteinDataset(dir=self.train_dir)
            self.train_dataset, self.val_dataset = random_split(
                    self.train_dataset, [0.95, 0.05], generator=torch.Generator().manual_seed(42)
                )
        else:
            self.test_dataset = ProteinDataset(dir=self.test_dir)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=partial(self.train_dataset.dataset.collate_fn, device=self.device),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(self.val_dataset.dataset.collate_fn, device=self.device),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(self.test_dataset.collate_fn, device=self.device),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(self.test_dataset.collate_fn, device=self.device),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
