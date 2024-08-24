import torch
from .attention import SelfAttention
from .feedforward import FeedForward
from .positional_embedding import PositionalEmbedding


class Encoder(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, dff:int, dropout:float = 0.0):
        """
        Implements a single encoder layer as described in the Transformer architecture.

        Args:
            d_model (int): The embedding dimension of the inputs.
            num_heads (int): The number of attention heads for the multi-headed self-attention.
            dff (int): The number of units in the feedforward network (usually larger than d_model).
            dropout (float): The dropout rate applied to the attention and feedforward layers. Default is 0.0.
        """
        super(Encoder, self).__init__()

        self.selfattention = SelfAttention(d_model=d_model, 
                                            num_heads=num_heads, 
                                            dropout=dropout)

        self.feedforward = FeedForward(d_model=d_model, 
                                       dff=dff, 
                                       dropout=dropout)

    def forward(self, x, mask=None):
        """
        Forward pass for the encoder layer.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_length, d_model].
            mask (torch.Tensor, optional): The attention mask tensor to prevent attention to padding tokens. 
                                           Shape [batch_size, seq_length]. Default is None.

        Returns:
            torch.Tensor: The output tensor of shape [batch_size, seq_length, d_model] after 
                          applying self-attention and feedforward transformations.
        """
        x = self.selfattention(x, mask=mask)
        x = self.feedforward(x)
        return x
    

class EncoderStacks(torch.nn.Module):
    """
    Implements a stack of encoder layers as described in the Transformer architecture 
    from the paper "Attention is All You Need".

    Args:
        d_model (int): The embedding dimension of the inputs.
        num_heads (int): The number of attention heads for the multi-headed self-attention.
        N (int): The number of encoder layers to stack.
        dff (int): The number of units in the feedforward networks (usually larger than d_model).
        dropout (float): The dropout rate applied to the attention, feedforward, and embedding layers. Default is 0.0.
        vocab_size (int): The size of the vocabulary for the input sequences.
        max_length (int): The maximum length of the input sequences. Default is 4000.
        padding_idx (int): The index representing the padding token in the input sequences.
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
        
        super(EncoderStacks, self).__init__()

        self.positional_embedding = PositionalEmbedding(vocab_size=vocab_size, 
                                                        d_model=d_model, 
                                                        max_length=max_length, 
                                                        padding_idx=padding_idx)
        self.encoder_layers = torch.nn.ModuleList([Encoder(d_model=d_model, 
                                                           num_heads=num_heads,
                                                           dff=dff,
                                                           dropout=dropout) for _ in range(N)])
        self.dropout = torch.nn.Dropout(p=dropout)


    def forward(self, x, mask=None):
        """
        Forward pass through the stack of encoder layers.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_length].
            mask (torch.Tensor, optional): The attention mask tensor to prevent attention to padding tokens. 
                                           Shape [batch_size, seq_length]. Default is None.

        Returns:
            torch.Tensor: The output tensor of shape [batch_size, seq_length, d_model] after 
                          passing through the positional embedding, dropout, and stack of encoder layers.
        """
        x = self.positional_embedding(x)
        x = self.dropout(x)
        
        for n in range(len(self.encoder_layers)):
            x = self.encoder_layers[n](x, mask=mask)

        return x