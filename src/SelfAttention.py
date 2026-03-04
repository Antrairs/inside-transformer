import torch
from torch import nn
from torch.nn import functional as F

# 简单的注意力机制实现
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_out = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / (self.d_model)**0.5

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return self.w_out(output)