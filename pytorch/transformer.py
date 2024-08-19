import torch
from encoder import EncoderLayer
from decoder import DecoderLayer

class Transformer(torch.nn.Module):
    def __init__(self, *, 
                 d_model:int, 
                 num_heads:int,
                 N:int, 
                 dff:int,
                 source_vocab_size:int,
                 target_vocab_size:int, 
                 dropout:float = 0.0, 
                 max_length:int = 4000, 
                 source_padding_idx:int = 0, 
                 target_padding_idx:int = 0):
        """
        Implements trasformer model [Attention is all you need]

        args:
            d_model: integer specifying embedding dimension of the inputs
            num_heads: integer specifying the number of heads for multiheaded attention
            N: number of encoder and decoder stacks
            dff: units for feedforward modules
            dropout: float specifying dropout_rate for modules
            source_vocab_size: integer specifying the size of the vocabulary for source
            target_vocab_size: integer specifying the size of the vocabulary for target
            max_length: integer specifying maximum sequence length
            source_padding_idx: integer specifying the value for [PAD] in source sequence
            target_padding_idx: integer specifying the value for [PAD] in target sequence
        """
        super(Transformer, self).__init__()

        self.encoder_layer = EncoderLayer(d_model=d_model, 
                                          num_heads=num_heads, 
                                          N=N, 
                                          dff=dff,
                                          vocab_size=source_vocab_size,
                                          dropout=dropout, 
                                          max_length=max_length,
                                          padding_idx=source_padding_idx)

        self.decoder_layer = DecoderLayer(d_model=d_model, 
                                          num_heads=num_heads, 
                                          N=N,
                                          dff=dff,
                                          vocab_size=target_vocab_size, 
                                          dropout=dropout, 
                                          max_length=max_length,
                                          padding_idx=target_padding_idx)

        self.classifier = torch.nn.Linear(in_features=d_model, out_features=target_vocab_size)


    def forward(self, source, target):
        context = self.encoder_layer(source)
        output = self.decoder_layer(target, context=context)
        output = self.classifier(output)
        return output
