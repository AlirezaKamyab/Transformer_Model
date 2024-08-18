import torch
from typing import Optional

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model:int, max_length:int):
        """
        Implements absolute positional encoding. It gives a unique vector with d_model dimension to each
        position.

        args:
            d_model: integer specifying the dimension for each position
            max_length: integer specifying maximum length for each seqeuence


        call:
            x: with the shape [Batch, Sequence length, units]
        """
        super(PositionalEncoding, self).__init__()

        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_length, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]