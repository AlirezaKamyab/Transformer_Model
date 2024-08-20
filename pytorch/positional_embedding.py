import torch
from .positional_encoding import PositionalEncoding


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, 
                 d_model:int, 
                 max_length: int = 4000, 
                 padding_idx: int = 0):
        """
        Implements embedding with absolute positional encoding. Given a vocab and its position, first
        it takes the vector for that vocab and adds its position vector using absolute positional encoding

        args:
            vocab_size: integer specifying the size of the vocabulary
            d_model: integer specifying the embedding dimension for each vocab
            max_length: integer specifying maximum sequence length
            padding_idx: integer specifying the value for [PAD]

        call:
            x: takes input with shape [Batch Size, Sequence length] with type integer
        
        returns: the positional embedding with the shape [Batch, Sequence length, d_model]
        """
        super(PositionalEmbedding, self).__init__()

        self.d_model = d_model
        self.embedding = torch.nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_length)

    def forward(self, x):
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x = x * torch.math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        return x