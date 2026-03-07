# %% [markdown]

# 为了将TransformerBlock组装成更强大的TransformerEncoder, 我们需要在TransformerBlock的基础上新增一个embedding层和一个输出层, 以及一个TransformerEncoder类来管理多个TransformerBlock

# %%
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert (d_model // num_heads) % 2 == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_w = None

    def _rotate_half(self, x):
        # x: (batch, num_heads, seq_len, d_k)
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def _rope(self, q, k, base=10000):
        # q, k: (batch, num_heads, seq_len, d_k)
        seq_len = q.size(-2)
        freqs = torch.arange(0, self.d_k, 2, device=q.device) / self.d_k
        freqs = base ** freqs                                           # (d_k/2,)
        angles = torch.arange(seq_len, device=q.device)[:, None] / freqs[None, :]  # (seq_len, d_k/2)
        sin = angles.sin().repeat_interleave(2, dim=-1)[None, None]   # (1, 1, seq_len, d_k)
        cos = angles.cos().repeat_interleave(2, dim=-1)[None, None]   # (1, 1, seq_len, d_k)
        return q * cos + self._rotate_half(q) * sin, \
            k * cos + self._rotate_half(k) * sin
    
    def forward(self, x, attn_mask=None): # 新增attn_mask参数
        """
        x: (batch, seq_len, d_model)
        """
        batch_size = x.size(0)
        Q = self.w_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        Q, K = self._rope(Q, K)
        score = Q @ K.transpose(-2, -1) / self.d_k ** 0.5
        
        if attn_mask is not None:
            key_mask = attn_mask[:, None, None, :]
            score = score.masked_fill(key_mask == 0, -1e9)
        
        attn_w = torch.softmax(score, dim=-1)
        self.attn_w = attn_w.detach()
        output = attn_w @ V # (batch, num_heads, seq_len, d_k)
        output = output.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        return self.out(output) # (batch, seq_len, d_model)
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, ffn_dim, d_model, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, attn_mask=None): # 新增attn_mask参数
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, attn_mask=attn_mask)
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x

class TransformerEncoder(nn.Module): # 新增encoder
    def __init__(self, vocab_size, output_dim, ffn_dim, d_model, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.token_type_embedding = nn.Embedding(2, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(ffn_dim, d_model, num_heads)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, output_dim)
    
    def forward(self, x, token_type_ids=None, attn_mask=None): # 新增attn_mask参数
        x = self.embedding(x)  # (batch, seq_len, d_model)
        if token_type_ids is not None:
            x = x + self.token_type_embedding(token_type_ids)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.out(x)
# %% [markdown]
# 准备数据集 由于transformer输入要求整数id, 所以需要将中文句子token化

# %%
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
ds = load_dataset("clue", "afqmc")
print(ds)
for i in range(3):
    print(ds["train"][i])

from collections import Counter

class AFQMCDataset(Dataset):
    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    def __init__(self, hf_dataset, vocab=None, max_len=64):
        self.data = hf_dataset
        self.max_len = max_len

        if vocab is None:
            self.vocab = self.build_vocab(hf_dataset)
        else:
            self.vocab = vocab

        self.PAD_IDX = self.vocab["[PAD]"]
        self.UNK_IDX = self.vocab["[UNK]"]
        self.CLS_IDX = self.vocab["[CLS]"]
        self.SEP_IDX = self.vocab["[SEP]"]

    def build_vocab(self, hf_dataset):
        vocab = {tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)}
        counter = Counter()

        for item in hf_dataset:
            counter.update(self.tokenize(item["sentence1"]))
            counter.update(self.tokenize(item["sentence2"]))

        for ch in counter:
            if ch not in vocab:
                vocab[ch] = len(vocab)

        return vocab
    
    def tokenize(self, text):
        return list(text)

    def encode_text(self, text):
        return [self.vocab.get(ch, self.UNK_IDX) for ch in self.tokenize(text)]

    def encode_pair(self, sentence1, sentence2):
        s1_ids = self.encode_text(sentence1)
        s2_ids = self.encode_text(sentence2)

        input_ids = [self.CLS_IDX] + s1_ids + [self.SEP_IDX] + s2_ids + [self.SEP_IDX]
        token_type_ids = [0] * (len(s1_ids) + 2) + [1] * (len(s2_ids) + 1)

        input_ids = input_ids[:self.max_len]
        token_type_ids = token_type_ids[:self.max_len]

        attn_mask = [1] * len(input_ids)
        pad_len = self.max_len - len(input_ids)

        if pad_len > 0:
            input_ids += [self.PAD_IDX] * pad_len
            token_type_ids += [0] * pad_len
            attn_mask += [0] * pad_len

        return input_ids, token_type_ids, attn_mask


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids, token_type_ids, attn_mask = self.encode_pair(
            item["sentence1"], item["sentence2"]
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attn_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(item["label"], dtype=torch.long),
    }
    
MAX_LEN = 128
train_dataset = AFQMCDataset(ds["train"], max_len=MAX_LEN)
vocab = train_dataset.vocab
val_dataset = AFQMCDataset(ds["validation"], vocab=vocab, max_len=MAX_LEN)
test_dataset = AFQMCDataset(ds["test"], vocab=vocab, max_len=MAX_LEN)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
train_dataset[0]


# %%
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = TransformerEncoder(
    vocab_size=len(vocab),
    d_model=64,
    ffn_dim=128,
    num_heads=4,
    num_layers=2,
    output_dim=2
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
criterion = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attn_mask = batch["attn_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        pred = model(input_ids, token_type_ids=token_type_ids, attn_mask=attn_mask)  # (batch, seq_len, output_dim)
        pred_cls = pred[:, 0, :]  # (batch, output_dim)
        loss = criterion(pred_cls, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            attn_mask = batch["attn_mask"].to(device)
            labels = batch["labels"].to(device)

            pred = model(input_ids, token_type_ids=token_type_ids, attn_mask=attn_mask)   # (batch, seq_len, 2)
            pred_cls = pred[:, 0, :]                       # (batch, 2)

            loss = criterion(pred_cls, labels)
            val_loss += loss.item()

            pred_labels = pred_cls.argmax(dim=-1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total

    print(
        f"Epoch {epoch+1:02d} | "
        f"train_loss={avg_train_loss:.4f} | "
        f"val_loss={avg_val_loss:.4f} | "
        f"val_acc={val_acc:.4f}"
    )
# %%
