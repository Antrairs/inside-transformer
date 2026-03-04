import torch
from torch import nn
from .MaskMultiHeadAttention import MaskMultiHeadAttention

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.attention = MaskMultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 这里用的是主流的Pre-Norm 也可以用Post-Norm
        x_norm = self.norm1(x)
        x = x + self.attention(x_norm, x_norm, x_norm)
        x = x + self.feed_forward(self.norm2(x))
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, d_ff, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)