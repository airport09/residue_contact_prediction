import torch
import numpy as np
from typing import Dict
from dataclasses import dataclass, asdict, fields
import biotite.structure as bs
from biotite.structure.io import pdb
from modelgenerator.structure_tokenizer.datasets.protein import (
    Protein, 
    ProteinChainEmpty, 
    _process_atom_array, 
    MIN_NB_RES)


CODES_MAPPING = {
    "ALA": "A",
    "ARG": "R",
    "ASP": "D",
    "CYS": "C",
    "CYX": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "HIE": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "ASN": "N",
    "PHE": "F",
    "PRO": "P",
    "SEC": "U",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V"
}

# two residues will be in contact 
# if the distance between their Ca atoms < MAX_DIST
MAX_DIST = 8  

@dataclass
class ProteinExt(Protein):
    """Data class for storing necessary information on Protein."""
    residue_codes: str
    residue_contacts: np.ndarray
    attention_mask: np.ndarray

    @classmethod
    def from_pdb_file(cls, pdb_file: pdb.PDBFile, id: str, chain_id: str = "nan") -> "Protein":
        atom_array = pdb.get_structure(pdb_file, model=1, extra_fields=["b_factor"])
        if chain_id == "nan":
            atom_array = atom_array[
                bs.filter_amino_acids(atom_array) & ~atom_array.hetero
            ]
        else:
            atom_array = atom_array[
                bs.filter_amino_acids(atom_array)
                & ~atom_array.hetero
                & (atom_array.chain_id == chain_id)
            ]

        if atom_array.array_length() == 0:
            raise ProteinChainEmpty(f"id {id} chain_id {chain_id} has no valid atoms.")

        aatype, atom_positions, atom_mask, residue_index, confidence = (
            _process_atom_array(atom_array=atom_array)
        )

        if len(atom_positions) == 0:
            raise ProteinChainEmpty(f"id {id} chain_id {chain_id} has no valid atoms.")
        if len(atom_positions) < MIN_NB_RES:
            raise ProteinChainEmpty(
                f"id {id} chain_id {chain_id} has {len(atom_positions)} residues which is below the minimum number {MIN_NB_RES}."
            )
        
        residue_codes = cls.get_residue_codes(atom_array)
        residue_contacts = cls.get_residue_contacts(atom_array)
        attention_mask = np.full(atom_positions.shape[0], True, dtype=bool)
        
        return cls(
            id=id,
            entity_id=None,
            resolution=0.0,
            chain_id=chain_id,
            residue_index=residue_index,
            atom_positions=atom_positions,
            atom_mask=atom_mask,
            aatype=aatype,
            b_factors=confidence,
            plddt=None,
            residue_codes=residue_codes,
            residue_contacts=residue_contacts,
            attention_mask=attention_mask
        )
    
    @staticmethod
    def get_residue_codes(atom_array: bs.AtomArray) -> str:
        codes_sequence = "".join(
        [
            CODES_MAPPING.get(monomer[0].res_name, "X")
            for monomer in bs.residue_iter(atom_array)
        ]
    )
        return codes_sequence
    

    @staticmethod
    def get_residue_contacts(atom_array: bs.AtomArray) -> np.ndarray:
        ca_coords = []
        for res in bs.residue_iter(atom_array):
            for atom in res:
                if atom.atom_name == "CA":
                    ca_coords.append(atom.coord)

        ca_coords = np.array(ca_coords)
        sum_sq = np.sum(ca_coords**2, axis=1)
        sq_dists = sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :] - 2 * ca_coords.dot(ca_coords.T)
        np.fill_diagonal(sq_dists, 0)
        dists = np.sqrt(sq_dists)

        return (dists < MAX_DIST).astype(np.float32)
    

    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_torch(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                setattr(self, f.name, torch.from_numpy(value))
        return self
    
    @classmethod
    def get_dummy_instance(cls):
        return cls(
            id="dummy",
            entity_id=None,
            resolution=0.0,
            chain_id="dummy",
            residue_index=np.zeros((10, ), dtype=np.int64),
            atom_positions=np.zeros((10, 37, 3), dtype=np.float32),
            atom_mask=np.zeros((10, 37)),
            aatype=np.zeros((10, )),
            b_factors=np.zeros((10, )),
            plddt=None,
            residue_codes="dummy_code",
            residue_contacts=np.zeros((10, 10)),
            attention_mask=np.full((10, ), True)
        )




        
