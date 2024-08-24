import torch

class RelativeMultiheadAttention(torch.nn.Module):
    def __init__(self, *,
                 d_model:int, 
                 num_heads:int,
                 k:int, 
                 k_dim:int=None, 
                 v_dim:int=None, 
                 dropout:float = 0.0):
        
        super(RelativeMultiheadAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.k_dim = k_dim if k_dim is not None else d_model
        self.v_dim = v_dim if v_dim is not None else d_model
        self.k = k

        assert self.k_dim % num_heads == 0, "k_dim should be divisible by num_heads!"
        assert self.v_dim % num_heads == 0, "v_dim should be divisible by num_heads!"
        self.d_head = d_model // num_heads

        self.WQ = torch.nn.Linear(in_features=d_model, out_features=self.k_dim)
        self.WK = torch.nn.Linear(in_features=d_model, out_features=self.k_dim)
        self.WV = torch.nn.Linear(in_features=d_model, out_features=self.v_dim)
        self.WO = torch.nn.Linear(in_features=self.v_dim, out_features=d_model)
        self.embeddingK = torch.nn.Embedding(num_embeddings=2 * k + 1, embedding_dim=self.k_dim)
        self.embeddingV = torch.nn.Embedding(num_embeddings=2 * k + 1, embedding_dim=self.v_dim)
        self.dropout = torch.nn.Dropout(dropout)


    def split_heads(self, x:torch.Tensor) -> torch.Tensor:
        # x has the shape [B, seq, d_model]
        batch, seq, dim = x.shape
        x = x.reshape(batch, seq, self.num_heads, -1)
        x = x.permute(0, 2, 1, 3)
        return x


    def get_relative_matrix(self, 
                            xi:torch.Tensor,
                            xj:torch.Tensor) -> torch.Tensor:
        # xi has the shape [B, h, length_xi, d_head]
        # xj has the shape [B, h, length_xj, d_head]
        length_xi = xi.shape[2]
        length_xj = xj.shape[2]
        arange_xi = torch.arange(length_xi, device=xi.device)[:, None]
        arange_xj = torch.arange(length_xj, device=xj.device)[None, :]
        matrix = arange_xj - arange_xi
        matrix = torch.clip(matrix, -self.k, self.k)
        matrix = matrix + self.k
        return matrix


    def get_relative_embeddingK(self, 
                                xi:torch.Tensor, 
                                xj:torch.Tensor) -> torch.Tensor:
        length_xi = xi.shape[2]
        length_xj = xj.shape[2]
        matrix = self.get_relative_matrix(xi, xj)
        rel_embed = self.embeddingK(matrix)
        rel_embed = rel_embed.reshape(length_xi, length_xj, self.num_heads, -1)
        rel_embed = rel_embed.permute(2, 0, 1, 3)
        return rel_embed

    
    def get_relative_embeddingV(self, 
                                xi:torch.Tensor, 
                                xj:torch.Tensor) -> torch.Tensor:
        length_xi = xi.shape[2]
        length_xj = xj.shape[2]
        matrix = self.get_relative_matrix(xi, xj)
        rel_embed = self.embeddingV(matrix)
        rel_embed = rel_embed.reshape(length_xi, length_xj, self.num_heads, -1)
        rel_embed = rel_embed.permute(2, 0, 1, 3)
        return rel_embed


    def scaled_dot_product_attention(self, 
                                     Q:torch.Tensor, 
                                     K:torch.Tensor, 
                                     V:torch.Tensor, 
                                     mask:torch.Tensor=None) -> torch.Tensor:
        
        _, _, _, k_dim = Q.shape
        
        # Equation (5) from the paper
        qk = torch.einsum('bhqd,bhkd->bhqk', Q, K)
        rel_embeddingK = torch.einsum('bhqd,hqkd->bhqk', Q, self.get_relative_embeddingK(Q, K))
        attn_weights = (qk + rel_embeddingK) / torch.sqrt(torch.tensor(k_dim, dtype=torch.float32))
        
        # Mask the attention_weights
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask, -torch.inf)
            
        attn = torch.nn.functional.softmax(attn_weights, dim=-1)
        
        # Equation (1) from the paper
        outputs = torch.einsum('bhqk,bhkv->bhqv', attn, V)
        outputs = outputs + torch.einsum('bhqk,hqkv->bhqv', attn, self.get_relative_embeddingV(Q, K))
        return outputs, attn

    
    def forward(self, 
                query:torch.Tensor, 
                key:torch.Tensor,
                value:torch.Tensor,
                attn_mask:torch.Tensor=None) -> torch.Tensor:
        
        batch_size, sequence_length, _ = query.shape
        # query has the shape [B, q_seq, d_model] --> [B, q_seq, k_dim]
        # key has the shape [B, k_seq, d_model] --> [B, q_seq, k_dim]
        # value has the shape [B, k_seq, d_model] --> [B, q_seq, v_dim]
        query = self.WQ(query)
        key = self.WK(key)
        value = self.WV(value)

        # split heads to make query, key value = [B, h, seq_len, d_head]
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        outputs, attn = self.scaled_dot_product_attention(Q=query, K=key, V=value, mask=attn_mask)
        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.reshape(batch_size, -1, self.v_dim)

        outputs = self.dropout(outputs)
        outputs = self.WO(outputs)
        return outputs, attn