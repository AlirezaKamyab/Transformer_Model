import torch

class FeedForward(torch.nn.Module):
    def __init__(self, d_model:int, dff:int, dropout:float=0.0):
        """
        Implements feedforward module from the paper [Attention is all you need]
        """
        super(FeedForward, self).__init__()

        self.dropout_rate = dropout
        self.W1 = torch.nn.Linear(in_features=d_model, out_features=dff)
        self.relu = torch.nn.ReLU()
        self.W2 = torch.nn.Linear(in_features=dff, out_features=d_model)
        self.layernorm = torch.nn.LayerNorm(d_model)

        if dropout > 0.0:
            self.dropout = torch.nn.Dropout(p=dropout)


    def forward(self, x):
        output = self.W1(x)
        output = self.relu(output)
        output = self.W2(output)
        if self.dropout_rate > 0.0:
            output = self.dropout(output)
            
        x = x + output
        x = self.layernorm(x)
        return x