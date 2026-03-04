import torch
from torch import nn
from .TransformerEncoder import TransformerEncoder
from .TransformerDecoder import TransformerDecoder

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim,
                d_model, d_ff, num_heads, num_layers, max_len=512, 
                dropout=0.1):
        super().__init__()
        
        self.src_embedding = nn.Embedding(input_dim, d_model)
        self.src_pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        self.tgt_embedding = nn.Embedding(input_dim, d_model)
        self.tgt_pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        self.dropout = nn.Dropout(dropout)
        
        self.encoder = TransformerEncoder(d_model, d_ff, num_heads, num_layers)
        self.decoder = TransformerDecoder(d_model, d_ff, num_heads, num_layers)

        self.out = nn.Linear(d_model, output_dim)

    def generate_mask(self, size):
        return torch.triu(torch.ones(size, size), diagonal=1).bool()
    
    def forward(self, src, tgt):
        src = self.src_embedding(src)
        src_seq_len = src.size(1)
        src = src + self.src_pos_embedding[:, :src_seq_len, :]
        src = self.dropout(src)

        memory = self.encoder(src)

        tgt_seq_len = tgt.size(1)
        tgt_mask = self.generate_mask(tgt_seq_len).to(src.device)
        tgt = self.tgt_embedding(tgt)
        tgt = tgt + self.tgt_pos_embedding[:, :tgt_seq_len, :]
        tgt = self.dropout(tgt)

        decoder_output = self.decoder(tgt, memory, tgt_mask)
        output = self.out(decoder_output)
        return output