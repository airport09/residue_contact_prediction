import torch
import torch.nn as nn

import esm

from typing import List, Tuple

class ESM2(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.device = device

    def forward(
        self,
        data: List[Tuple[str, str]],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        results = self.model(batch_tokens, repr_layers=[33])

        return results["representations"][33]