import torch
from .attention import CausalAttention, CrossAttention
from .feedforward import FeedForward
from .positional_embedding import PositionalEmbedding


class Decoder(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, dff:int, dropout:float=0.0):
        """
        Implements a single decoder

        args:
            d_model: integer specifying embedding dimension of the inputs
            num_heads: integer specifying the number of heads for multiheaded attention
            dff: units for relu transformation
            dropout: float specifying dropout_rate
        """
        super(Decoder, self).__init__()

        self.causalattention = CausalAttention(d_model=d_model, 
                                               num_heads=num_heads, 
                                               dropout=dropout)
        self.crossattention = CrossAttention(d_model=d_model, 
                                             num_heads=num_heads, 
                                             dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, 
                                       dff=dff, 
                                       dropout=dropout)

    def forward(self, x, context):
        x = self.causalattention(x)
        x = self.crossattention(x, context)
        self.attention_weights = self.crossattention.attention_weights
        x = self.feedforward(x)
        return x
    

class DecoderLayer(torch.nn.Module):
    def __init__(self, *, 
                 d_model:int,
                 num_heads:int,
                 N:int, 
                 dff:int,
                 vocab_size:int, 
                 dropout:float = 0.0, 
                 max_length:int = 4000,
                 padding_idx:int = 0):
        """
        Implements a stack of decoders.
        from the paper [Attention is all you need]

        args:
                d_model: integer specifying embedding dimension of the inputs
                num_heads: integer specifying the number of heads for multiheaded attention
                N: number of decoders to stack
                dff: units for relu transformation
                dropout: float specifying dropout_rate
                vocab_size: integer specifying the size of the vocabulary
                max_length: integer specifying maximum sequence length
                padding_idx: integer specifying the value for [PAD]
        """
        
        super(DecoderLayer, self).__init__()

        self.dropout_rate = dropout
        self.positional_embedding = PositionalEmbedding(d_model=d_model, 
                                                        vocab_size=vocab_size, 
                                                        max_length=max_length, 
                                                        padding_idx=padding_idx)
        
        self.decoder_layers = torch.nn.ModuleList([Decoder(d_model=d_model,
                                                           num_heads=num_heads,
                                                           dff=dff,
                                                           dropout=dropout) for _ in range(N)])

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=dropout)


    def forward(self, x, context):
        x = self.positional_embedding(x)
        if self.dropout_rate > 0.0:
            x = self.dropout(x)

        for n in range(len(self.decoder_layers)):
            x = self.decoder_layers[n](x, context=context)
        self.attention_weights = self.decoder_layers[-1].attention_weights
        
        return x