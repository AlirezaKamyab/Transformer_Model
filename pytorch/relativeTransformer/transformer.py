import torch
from .encoder import EncoderStacks
from .decoder import DecoderStacks

class Transformer(torch.nn.Module):
    def __init__(self, *, 
                 d_model:int, 
                 num_heads:int,
                 k:int,
                 N:int, 
                 dff:int,
                 source_vocab_size:int,
                 target_vocab_size:int, 
                 dropout:float = 0.0, 
                 source_padding_idx:int = 0, 
                 target_padding_idx:int = 0):
        """
        Implements the Transformer model as described in the paper 
        "Attention is All You Need" by Vaswani et al. (2017).
        "Self-Attention with Relative Position Representation" by Shaw at al. (2018).

        Args:
            d_model (int): The dimensionality of the embeddings and hidden states.
            num_heads (int): The number of attention heads for multi-headed attention.
            k (int): The clip value for relative position weights
            N (int): The number of layers (or blocks) in both the encoder and decoder stacks.
            dff (int): The number of units in the feedforward layers.
            dropout (float): Dropout rate applied to various layers. Default is 0.0.
            source_vocab_size (int): The size of the vocabulary for the source language.
            target_vocab_size (int): The size of the vocabulary for the target language.
            source_padding_idx (int): The index representing the padding token in the source sequences.
            target_padding_idx (int): The index representing the padding token in the target sequences.
        """
        super(Transformer, self).__init__()

        self.source_padding_idx = source_padding_idx
        self.target_padding_idx = target_padding_idx

        self.encoder_layer = EncoderStacks(d_model=d_model, 
                                          num_heads=num_heads, 
                                          k=k,
                                          N=N, 
                                          dff=dff,
                                          vocab_size=source_vocab_size,
                                          dropout=dropout, 
                                          padding_idx=source_padding_idx)

        self.decoder_layer = DecoderStacks(d_model=d_model, 
                                          num_heads=num_heads, 
                                          k=k,
                                          N=N,
                                          dff=dff,
                                          vocab_size=target_vocab_size, 
                                          dropout=dropout, 
                                          padding_idx=target_padding_idx)

        self.classifier = torch.nn.Linear(in_features=d_model, out_features=target_vocab_size)


    def generate_mask(self, source):
        """
        Generates a padding mask for the source sequences.

        Args:
            source (torch.Tensor): The input source tensor of shape [batch_size, seq_length].

        Returns:
            torch.Tensor: A boolean mask tensor of shape [batch_size, seq_length], where `True`
                          indicates the positions of padding tokens.
        """
        src_mask = source == self.source_padding_idx
        return src_mask
        

    def forward(self, source, target):
        """
        Forward pass through the Transformer model.

        Args:
            source (torch.Tensor): The source input tensor of shape [batch_size, src_seq_length].
            target (torch.Tensor): The target input tensor of shape [batch_size, tgt_seq_length].

        Returns:
            torch.Tensor: The output logits tensor of shape [batch_size, tgt_seq_length, target_vocab_size],
                          representing the predicted token probabilities for each position in the target sequence.
        """
        # Generate the mask for the source sequences, if applicable
        mask = None
        if self.source_padding_idx is not None:
            mask = self.generate_mask(source=source)
            
        # Pass through the encoder
        context = self.encoder_layer(source, mask=mask)
        
        # Pass through the decoder with the encoded context
        output = self.decoder_layer(target, context=context, mask=mask)
        
        # Pass through the classifier to get the final predictions
        output = self.classifier(output)
        return output
