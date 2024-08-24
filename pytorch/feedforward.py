import torch

class FeedForward(torch.nn.Module):
    def __init__(self, d_model:int, dff:int, dropout:float=0.0):
        """
        Implements the feedforward module from the paper [Attention is All You Need].
        This module includes two linear layers with a ReLU activation in between,
        followed by a dropout, a residual connection, and layer normalization.

        Args:
            d_model (int): The input and output dimension of the model.
            dff (int): The dimension of the feedforward layer (hidden layer).
            dropout (float): Dropout rate to be applied after the linear layers. Default is 0.0.
        """
        super(FeedForward, self).__init__()

        self.W1 = torch.nn.Linear(in_features=d_model, out_features=dff)
        self.relu = torch.nn.ReLU()
        self.W2 = torch.nn.Linear(in_features=dff, out_features=d_model)
        self.layernorm = torch.nn.LayerNorm(d_model)

        self.dropout = torch.nn.Dropout(p=dropout)


    def forward(self, x):
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
        output = self.W1(x)
        output = self.relu(output)
        output = self.W2(output)
        
        output = self.dropout(output)
            
        x = x + output
        x = self.layernorm(x)
        return x