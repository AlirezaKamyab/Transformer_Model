import torch
from .attention import CausalAttention, CrossAttention
from .feedforward import FeedForward


class Decoder(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, k:int, dff:int, dropout:float=0.0):
        """
        Implements a single decoder layer as described in the Transformer architecture.

        Args:
            d_model (int): The embedding dimension of the inputs.
            num_heads (int): The number of attention heads for the multi-headed attention.
            k (int): The clip value for relative position weights
            dff (int): The number of units in the feedforward network (usually larger than d_model).
            dropout (float): The dropout rate applied to the attention and feedforward layers. Default is 0.0.
        """
        super(Decoder, self).__init__()

        self.causalattention = CausalAttention(d_model=d_model, 
                                               num_heads=num_heads, 
                                               k=k,
                                               dropout=dropout)
        self.crossattention = CrossAttention(d_model=d_model, 
                                             num_heads=num_heads, 
                                             k=k,
                                             dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, 
                                       dff=dff, 
                                       dropout=dropout)

    def forward(self, x, context, mask=None):
        """
        Forward pass for the decoder layer.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_length, d_model].
            context (torch.Tensor): The context tensor (output from the encoder) of shape [batch_size, context_length, d_model].
            mask (torch.Tensor, optional): The attention mask tensor to prevent attention to padding tokens. 
                                           Shape [batch_size, seq_length]. Default is None.

        Returns:
            torch.Tensor: The output tensor of shape [batch_size, seq_length, d_model] after 
                          applying causal self-attention, cross-attention, and feedforward transformations.
        """
        x = self.causalattention(x)
        x = self.crossattention(x, context, mask=mask)
        self.attention_weights = self.crossattention.attention_weights
        x = self.feedforward(x)
        return x
    

class DecoderStacks(torch.nn.Module):
    def __init__(self, *, 
                 d_model:int,
                 num_heads:int,
                 k:int,
                 N:int, 
                 dff:int,
                 vocab_size:int, 
                 dropout:float = 0.0, 
                 padding_idx:int = 0):
        """
        Implements a stack of decoder layers as described in the Transformer architecture 
        from the paper "Attention is All You Need".

        Args:
            d_model (int): The embedding dimension of the inputs.
            num_heads (int): The number of attention heads for the multi-headed attention.
            k (int): The clip value for relative position weights
            N (int): The number of decoder layers to stack.
            dff (int): The number of units in the feedforward networks (usually larger than d_model).
            vocab_size (int): The size of the vocabulary for the input sequences.
            dropout (float): The dropout rate applied to the attention, feedforward, and embedding layers. Default is 0.0.
            padding_idx (int): The index representing the padding token in the input sequences.
        """
        
        super(DecoderStacks, self).__init__()

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, 
                                            embedding_dim=d_model,  
                                            padding_idx=padding_idx)
        
        self.decoder_layers = torch.nn.ModuleList([Decoder(d_model=d_model,
                                                           num_heads=num_heads,
                                                           k=k,
                                                           dff=dff,
                                                           dropout=dropout) for _ in range(N)])

        
        self.dropout = torch.nn.Dropout(p=dropout)


    def forward(self, x, context, mask=None):
        """
        Forward pass through the stack of decoder layers.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_length].
            context (torch.Tensor): The context tensor (output from the encoder) of shape [batch_size, context_length, d_model].
            mask (torch.Tensor, optional): The attention mask tensor to prevent attention to padding tokens. 
                                           Shape [batch_size, seq_length]. Default is None.

        Returns:
            torch.Tensor: The output tensor of shape [batch_size, seq_length, d_model] after 
                          passing through the positional embedding, dropout, and stack of decoder layers.
                          The final attention weights from the cross-attention layer of the last decoder layer are also stored.
        """
        x = self.embedding(x)
        x = self.dropout(x)

        for n in range(len(self.decoder_layers)):
            x = self.decoder_layers[n](x, context=context, mask=mask)
        self.attention_weights = self.decoder_layers[-1].attention_weights
        
        return x