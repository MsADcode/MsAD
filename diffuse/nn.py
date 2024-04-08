from typing import List, Optional, Type
import torch
from torch import nn
from torch.nn import Dropout, LayerNorm, Linear, Module, Sequential, BatchNorm1d
from torch.nn import TransformerEncoder, TransformerEncoderLayer



import random
import numpy as np
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class DiffAtt(Module):
    def __init__(self, n_nodes: int,dim: int) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.dim = dim
        small_layer = max(self.dim, 2 * self.n_nodes)

        self.main_block = nn.Sequential(
            nn.Linear(self.n_nodes + 1, small_layer, bias=False),
            nn.LeakyReLU(),
            nn.LayerNorm([small_layer]),
            nn.Dropout(0.2)
        )
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=small_layer, nhead=4),
            num_layers=2
        )
        self.output_layer = nn.Linear(small_layer, self.n_nodes)

    def forward(self, X, t):
        X_t = torch.cat([X, t.unsqueeze(1)], axis=1)

        output = self.main_block(X_t)

        res = output
        output = self.transformer_encoder(output) + res

        output = self.output_layer(output)

        return output

