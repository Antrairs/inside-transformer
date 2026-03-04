import torch
from torch import nn
from torch.nn import functional as F
from .MaskMultiHeadAttention import MaskMultiHeadAttention

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, dropout):
        super().__init__()
        self.self_attn = MaskMultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MaskMultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask):
        x_norm = self.norm1(x)
        x = x + self.dropout(self.self_attn(x_norm, x_norm, x_norm, tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory))
        x = x + self.dropout(self.feed_forward(self.norm3(x)))
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.norm(x)