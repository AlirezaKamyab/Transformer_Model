import torch
from .attention import SelfAttention
from .feedforward import FeedForward
from .positional_embedding import PositionalEmbedding


class Encoder(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, dff:int, dropout:float = 0.0):
        """
        Implements a single encoder

        args:
            d_model: integer specifying embedding dimension of the inputs
            num_heads: integer specifying the number of heads for multiheaded attention
            dff: units for relu transformation
            dropout: float specifying dropout_rate
        """
        super(Encoder, self).__init__()

        self.selfattention = SelfAttention(d_model=d_model, 
                                            num_heads=num_heads, 
                                            dropout=dropout)

        self.feedforward = FeedForward(d_model=d_model, 
                                       dff=dff, 
                                       dropout=dropout)

    def forward(self, x):
        x = self.selfattention(x)
        x = self.feedforward(x)
        return x
    

class EncoderLayer(torch.nn.Module):
    """
    Implements a stack of encoders.
    from the paper [Attention is all you need]

    args:
            d_model: integer specifying embedding dimension of the inputs
            num_heads: integer specifying the number of heads for multiheaded attention
            N: number of encoders to stack
            dff: units for relu transformation
            dropout: float specifying dropout_rate
            vocab_size: integer specifying the size of the vocabulary
            max_length: integer specifying maximum sequence length
            padding_idx: integer specifying the value for [PAD]
    """
    def __init__(self, *, 
                 d_model:int, 
                 num_heads:int, 
                 N:int, 
                 dff:int,
                 dropout:float=0.0, 
                 vocab_size:int, 
                 max_length:int=4000,
                 padding_idx:int=0):
        
        super(EncoderLayer, self).__init__()

        self.dropout_rate = dropout

        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, 
                                                        d_model=d_model, 
                                                        max_length=max_length, 
                                                        padding_idx=padding_idx)
        self.encoder_layers = torch.nn.ModuleList([Encoder(d_model=d_model, 
                                                           num_heads=num_heads,
                                                           dff=dff,
                                                           dropout=dropout) for _ in range(N)])
        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=dropout)


    def forward(self, x):
        x = self.positional_embedding.cuda()(x)
        
        if self.dropout_rate > 0.0:
            x = self.dropout(x)
        
        for n in range(len(self.encoder_layers)):
            x = self.encoder_layers[n](x)

        return x