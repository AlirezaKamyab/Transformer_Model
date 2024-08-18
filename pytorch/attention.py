import torch

class BaseAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        """
        Implements base attention
        """
        super(BaseAttention, self).__init__()

        self.mha = torch.nn.MultiheadAttention(embed_dim=d_model, 
                                               num_heads=num_heads, 
                                               dropout=dropout, 
                                               batch_first=True)
        self.layernorm = torch.nn.LayerNorm(d_model)


class SelfAttention(BaseAttention):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        """
        Implements self-multiheaded-attention where query, key, value is the same input.

        args:
            d_model: integer specifying embedding dimension of the inputs
            num_heads: integer specifying the number of heads for multiheaded attention
            dropout: float specifying dropout_rate
        """
        super(SelfAttention, self).__init__(d_model=d_model, 
                                            num_heads=num_heads, 
                                            dropout=dropout)

    def forward(self, x):
        output, _ = self.mha(query=x, key=x, value=x)
        x = x + output
        x = self.layernorm(x)
        return x
    

class CausalAttention(BaseAttention):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        """
        Implements masked-attention or causal-attention, which gives an attention mask to the multihead attention
        to not to attend to the future.

        args:
            d_model: integer specifying embedding dimension of the inputs
            num_heads: integer specifying the number of heads for multiheaded attention
            dropout: float specifying dropout_rate
        
        """
        super(CausalAttention, self).__init__(d_model=d_model,
                                             num_heads=num_heads, 
                                             dropout=dropout)

    def forward(self, x):
        # x has the shape [Batch, Seq_length, d_model]
        seq_length = x.shape[1]
        # create a mask with shape [Seq_length, Seq_length]
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.float32), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)

        output, _ = self.mha(query=x, key=x, value=x, attn_mask=causal_mask)
        x = x + output
        x = self.layernorm(x)
        return output
    

class CrossAttention(BaseAttention):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        """
        Implements cross-attention to attend to the input and the output from the encoder.

        args:
            d_model: integer specifying embedding dimension of the inputs
            num_heads: integer specifying the number of heads for multiheaded attention
            dropout: float specifying dropout_rate
        """
        super(CrossAttention, self).__init__(d_model=d_model, 
                                             num_heads=num_heads, 
                                             dropout=dropout)

    def forward(self, x, context):
        output, attention_weights = self.mha(query=x, key=context, value=context)
        self.attention_weights = attention_weights
        x = x + output
        x = self.layernorm(x)
        return x