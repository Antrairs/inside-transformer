# %% [markdown]
# 现在将多头注意力, 层归一化, 前馈神经网络, 残差连接组合起来形成TransformerEncoderBlock
# 
# 由于现在离散的token输入入口变为了TransformerEncoderBlock, 需要把MultiHeadAttention里的embedding移到block

# %%
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads): # 由于embedding层被移到block里了，所以不需要vocab_size了, output_dim也不需要了，因为多头注意力的输出维度是d_model，而不是output_dim
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model) # 注意这里的输出维度改为了d_model，因为多头注意力的输出维度是d_model，而不是output_dim
        self.attn_w = None

    def _rotate_half(self, x):
        # x: (batch, num_heads, seq_len, d_k)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _rope(self, q, k, base=10000):
        # q, k: (batch, num_heads, seq_len, d_k)
        seq_len = q.size(-2)   # ← 注意是 -2，不是 1
        freqs = torch.arange(0, self.d_k, 2, device=q.device) / self.d_k
        freqs = base ** freqs                                           # (d_k/2,)
        angles = torch.arange(seq_len, device=q.device)[:, None] / freqs[None, :]  # (seq_len, d_k/2)
        sin = angles.sin().repeat_interleave(2, dim=-1)[None, None]   # (1, 1, seq_len, d_k)
        cos = angles.cos().repeat_interleave(2, dim=-1)[None, None]   # (1, 1, seq_len, d_k)
        return q * cos + self._rotate_half(q) * sin, \
            k * cos + self._rotate_half(k) * sin
    
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size = x.size(0)
        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        Q, K = self._rope(Q, K)
        score = Q @ K.transpose(-2, -1) / self.d_k ** 0.5
        attn_w = torch.softmax(score, dim=-1)
        self.attn_w = attn_w.detach()
        output = attn_w @ V # (batch, num_heads, seq_len, d_k)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        return self.out(output) # (batch, seq_len, d_model)
    
class TransformerEncoderBlock(nn.Module): # 新增block
    def __init__(self, vocab_size, output_dim, ffn_dim, d_model, num_heads):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) # 将embedding放在block里，保证每个block都有自己的embedding层
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, output_dim) # 将output_dim在这里应用
    
    def forward(self, x): # Pre-Norm
        x = self.embedding(x)  # (batch, seq_len, d_model)
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm)
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return self.out(x)



# %% [markdown]
# TransformerEncoderBlock里的forward有两种写法, 一种是上面的Pre-Norm写法
# 
# 一种是下面的Transformer论文的经典写法Post-Norm, 先做注意力/FFN, 并残差连接, 再norm
# 
# ```python
# def forward(self, x):
#         x = x + self.attn(x) # (batch, seq_len, output_dim) 先注意力并残差
#         x = self.norm1(x) # 再层归一化
#         x = x + self.ffn(x) # (batch, seq_len, d_model) 先FFN并残差
#         x = self.norm2(x) # 再层归一化
#         return x
# ```
# 一般来说, Pre-Norm写法对于深层网络训练更稳定, 且是现在大模型的主流写法, 所以我们用这种

# %% [markdown]
# ## 实验1：序列排序
# 

# %%
def get_sort_batch(batch_size=64, seq_len=6, vocab_size=10):
    """
    排序任务数据集
    输入: 乱序的整数序列      e.g. [3, 1, 4, 1, 5, 2]
    标签: 升序排列后的序列    e.g. [1, 1, 2, 3, 4, 5]
    """
    x = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    y = x.sort(dim=-1).values  # (batch_size, seq_len), long
    return x, y

# 验证一下
x, y = get_sort_batch(4, 6, 10)
for i in range(4):
    print(f"input: {x[i].tolist()}  →  sorted: {y[i].tolist()}")


epochs = 3000

model = TransformerEncoderBlock(vocab_size=10, ffn_dim=16, output_dim=10, d_model=64, num_heads=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

model.train()
for i in range(epochs):
    train_x, train_y = get_sort_batch(1024, 10, 10)
    optimizer.zero_grad()
    pred = model(train_x)                      # (batch, seq_len, vocab_size)
    loss = criterion(pred.transpose(1, 2), train_y)
    loss.backward()
    optimizer.step()
    if (i+1) % 100 == 0:
        print(f"Epoch {i+1}, Loss: {loss.item():.4f}")

# %%
test_x, test_y = get_sort_batch(4, 6, 10)
model.eval()
with torch.no_grad():
    pred = model(test_x)                   # 触发 forward，缓存 attn_w
    pred_tokens = pred[0].argmax(dim=-1)   # (seq_len,)

print("input :", test_x[0].tolist())
print("pred  :", pred_tokens.tolist())
print("target:", test_x[0].sort().values.tolist())



