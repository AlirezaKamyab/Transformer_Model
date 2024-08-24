import torch

class BaseAttention(torch.nn.Module):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        """
        Implements base attention mechanism using multi-head attention.

        Args:
            d_model (int): The embedding dimension of the inputs.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate to apply after attention. Default is 0.0.
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
        Implements self-attention mechanism where query, key, and value are the same input.

        Args:
            d_model (int): The embedding dimension of the inputs.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate to apply after attention. Default is 0.0.
        """
        super(SelfAttention, self).__init__(d_model=d_model, 
                                            num_heads=num_heads, 
                                            dropout=dropout)
        self.num_heads = num_heads
        
    def compute_attention_mask(self, mask):
        """
        Computes the attention mask for self-attention. The mask is expanded 
        across the number of attention heads and reshaped to match the 
        dimensions required by the multi-head attention module.

        Args:
            mask (torch.Tensor): A binary mask tensor of shape [batch_size, source_seq_length], 
                                 where False indicates a valid token and True indicates a padding token.

        Returns:
            torch.Tensor: The computed attention mask of shape [batch_size * num_heads, source_seq_length, source_seq_length].
        """
        batch_size = mask.shape[0]
        source_seq_length = mask.shape[1]
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask.expand(batch_size, self.num_heads, source_seq_length, source_seq_length)
        mask = mask.reshape(batch_size * self.num_heads, source_seq_length, source_seq_length)
        mask = mask.bool()
        return mask
    

    def forward(self, x, mask=None):
        """
        Performs the forward pass of self-attention.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_length, d_model].
            mask (torch.Tensor, optional): The binary mask tensor to prevent attention to padding tokens.
                                           Shape [batch_size, seq_length]. Default is None.

        Returns:
            torch.Tensor: The output tensor after applying self-attention and layer normalization.
                          Shape [batch_size, seq_length, d_model].
        """
        if mask is not None:
            mask = self.compute_attention_mask(mask)

        output, _ = self.mha(query=x, key=x, value=x, attn_mask=mask)
        x = x + output
        x = self.layernorm(x)
        return x
    

class CausalAttention(BaseAttention):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        """
        Implements causal attention (masked attention) where the attention mask 
        prevents the model from attending to future tokens. This is typically 
        used in autoregressive models.

        Args:
            d_model (int): The embedding dimension of the inputs.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate to apply after attention. Default is 0.0.
        """
        super(CausalAttention, self).__init__(d_model=d_model,
                                             num_heads=num_heads, 
                                             dropout=dropout)

    def forward(self, x):
        """
        Performs the forward pass of causal attention.

        Args:
            x (torch.Tensor): The input tensor of shape [batch_size, seq_length, d_model].

        Returns:
            torch.Tensor: The output tensor after applying causal attention and layer normalization.
                          Shape [batch_size, seq_length, d_model].
        """
        seq_length = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_length, seq_length, dtype=torch.float32), diagonal=1).bool()
        causal_mask = causal_mask.to(x.device)

        output, _ = self.mha(query=x, key=x, value=x, attn_mask=causal_mask)
        x = x + output
        x = self.layernorm(x)
        return x
    

class CrossAttention(BaseAttention):
    def __init__(self, d_model:int, num_heads:int, dropout:float=0.0):
        """
        Implements cross-attention, where the model attends to a different 
        context (e.g., encoder output) from the input query.

        Args:
            d_model (int): The embedding dimension of the inputs.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate to apply after attention. Default is 0.0.
        """
        super(CrossAttention, self).__init__(d_model=d_model, 
                                             num_heads=num_heads, 
                                             dropout=dropout)
        self.num_heads = num_heads


    def compute_attention_mask(self, query, source_mask):
        """
        Computes the attention mask for cross-attention, ensuring that the 
        model does not attend to padding tokens in the source context.

        Args:
            query (torch.Tensor): The query tensor of shape [batch_size, query_length, d_model].
            source_mask (torch.Tensor): A binary mask tensor of shape [batch_size, source_seq_length], 
                                        where False indicates a valid token and True indicates a padding token.

        Returns:
            torch.Tensor: The computed attention mask of shape [batch_size * num_heads, 
                                                               query_length, source_seq_length].
        """
        batch_size = query.shape[0]
        query_length = query.shape[1]
        source_mask_length = source_mask.shape[1]
        mask = torch.ones(batch_size, query_length, source_mask_length, device=query.device) * source_mask.unsqueeze(1)
        mask = mask.unsqueeze(1)
        mask = mask.expand(batch_size, self.num_heads, query_length, source_mask_length)
        mask = mask.reshape(batch_size * self.num_heads, query_length, source_mask_length)
        mask = mask.bool()
        return mask
    

    def forward(self, x, context, mask=None):
        """
        Performs the forward pass of cross-attention.

        Args:
            x (torch.Tensor): The query input tensor of shape [batch_size, query_length, d_model].
            context (torch.Tensor): The context tensor to attend to (e.g., encoder output) of shape 
                                    [batch_size, context_length, d_model].
            mask (torch.Tensor, optional): The binary mask tensor to prevent attention to padding tokens 
                                           in the context. Shape [batch_size, context_length]. Default is None.

        Returns:
            torch.Tensor: The output tensor after applying cross-attention and layer normalization.
                          Shape [batch_size, query_length, d_model].
        """
        if mask is not None:
            mask = self.compute_attention_mask(query=x, source_mask=mask)
            
        output, attention_weights = self.mha(query=x, key=context, value=context, attn_mask=mask)
        self.attention_weights = attention_weights
        x = x + output
        x = self.layernorm(x)
        return x