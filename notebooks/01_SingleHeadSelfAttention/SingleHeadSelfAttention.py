# %% [markdown]
#  # 单头自注意力模块
# 
# SelfAttention做的事本质上是:
# 
# 序列中的每个token都去看序列里其他token, 在训练中判断自己应该关注谁, 然后把这些信息加权汇总得到新的表示.
# 
#  下面是实现最小可用版本的自注意力的具体步骤:
# 
#  1. 首先输入要求为一个序列x, 这里为了方便处理默认输入一个离散的数字序列, 序列中每个数字就是一个token.
# 
#  2. 经过`nn.Embedding`将每个token映射到维度为`d_model`的空间中, 得到`x: (batch, seq_len, d_model)`.
# 
#  3. 然后同一个输入x分别线性映射成`Q/K/V`, 虽然输入都是同一个x, 但是这里用了三个不同的线性层, 所以会得到三个不同的结果, 可以把它理解为每个token同时扮演了三种身份:
#     - `Q (query)`: 倾向于表示当前token想找什么, 即查询信息, 当前token根据什么依据搜索别的token
#     - `K (key)`: 倾向于表示当前token能被如何匹配, 即该token对外展示我是什么类型的信息
#     - `V (value)`: 倾向于表示当前token真正提供给别人的内容, 注意力根据Q和K匹配出该看什么后, 在V里取对应的数据
#  4. `Q @ K.transpose(-2, -1)`这一段代码是计算每个位置的查询Q和所有位置的键K之间的相关性分数, 为避免计算出的结果太大, 所以后面要缩放, 除以`self.d_model ** 0.5`, 然后对最后一维进行`softmax`得到注意力权重`attn_w: (batch, seq_len, seq_len)`
#     >  经典线性代数教材是列向量优先时$S=KQ$, 但这里是行向量优先, 行向量优先时$S=QK^T$, 两者在数学上完全等价
# 
#  1. 用注意力权重对 `V` 加权求和得到`output: (batch, seq_len, d_model)`, 再经过输出层得到最终 logits.

# %%
import torch
from torch import nn

class SelfAttention(nn.Module):
    def __init__(self, vocab_size, output_dim, d_model):
        super().__init__()
        # d_model: token 的隐藏维度
        # output_dim: 每个位置输出的类别数（例如实验1是1路位置分类，实验2是2路位置分类）
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, output_dim)
        self.attn_w = None  # 缓存最近一次的注意力权重用于可视化
    
    def forward(self, x):
        """
        x: (batch, seq_len)
        """
        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.embedding(x)  # (batch, seq_len, d_model)
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        # score: (batch, query, key) --- 序列内各位置之间的注意力, query为token位置, key为看的那个位置
        score = Q @ K.transpose(-2, -1) / self.d_model ** 0.5 # @等价于torch.matmul函数 为方便使用@
        attn_w = torch.softmax(score, dim=-1)
        self.attn_w = attn_w.detach()  # 缓存
        output = attn_w @ V          # (batch, seq_len, d_model)
        return self.out(output)      # (batch, seq_len, output_dim)


# %% [markdown]
#  ## 注意力热力图可视化函数
# 
#  该函数会绘制 batch 前 4 个样本的注意力矩阵, 帮助初学者观察:
# 
#  - 每个 query 位置更关注哪些 key 位置
# 
#  - 注意力是否集中到任务相关位置

# %%
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

def plot_attention(attn_w, tokens=None, title="Batch", cmap="Blues"):
    # attn_w: (batch, T, T) 或 (T, T)
    if attn_w.dim() == 2:
        attn_w = attn_w.unsqueeze(0)
    attn_w = attn_w[:4].detach().cpu().float()   # 取前4个样本

    seq_len = attn_w.shape[1]
    if tokens is None:
        tokens = [str(i) for i in range(seq_len)]

    cell = seq_len * 0.5 + 0.5
    fig, axes = plt.subplots(2, 2, figsize=(cell * 2, cell * 2), constrained_layout=True)

    for n in range(4):
        ax = axes[n // 2][n % 2]
        im = ax.imshow(attn_w[n], cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(seq_len)); ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(seq_len)); ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"{title} [{n}]", fontsize=9)
        for i in range(seq_len):
            for j in range(seq_len):
                val = attn_w[n, i, j].item()
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if val > 0.6 else "black")

    plt.colorbar(im, ax=axes, fraction=0.02, pad=0.04)
    plt.show()


# %% [markdown]
#  # 实验1 找最大值下标
# 
# ## 1. 数据集构建
#  我们选一个任务来把这个最简单的自注意力用起来, 从一堆乱序数字中找出最大值的下标
# 
#  数据集构建: 
#  - 数据: 一堆乱序的数字 [[3, 2, 1, 5, 7],
#                     [1, 6, 2, 6, 1]]
#  - 标签: [4, 1]
# 
#  实验目标: 输出最大的数字下标
# 
#  为了测试这个模块的效果, 我们来构造一个数据集

# %%
def get_batch(batch, sql_len, low, high):
    # x: 随机整数序列，y: 最大值所在下标
    x = torch.randint(low, high, (batch, sql_len), dtype=torch.long)  # (batch_size, n)
    y = torch.argmax(x, dim=-1)  # (batch_size,)
    return x, y
print(get_batch(2, 5, 0, 10))


# %% [markdown]
# ## 2. 模型训练
#  输出层设置为 `output_dim=1`, 表示每个位置只输出1个logit.
# 
#  模型输出形状是 `(batch, seq_len, 1)`, 压缩后变成 `(batch, seq_len)`.
# 
#  用 `CrossEntropyLoss` 把 `seq_len` 视为类别数, 即预测哪个位置是最大值.

# %%

epochs = 1000

model = SelfAttention(vocab_size=100, output_dim=1, d_model=4)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

model.train()
for i in range(epochs):
    train_x, train_y = get_batch(1024, 10, 0, 100)
    optimizer.zero_grad()
    pred = model(train_x).squeeze(-1)  # (batch, seq_len, 1) → (batch, seq_len)
    loss = criterion(pred, train_y)
    loss.backward()
    optimizer.step()
    if (i+1) % 100 == 0:
        print(f"Epoch {i+1}, Loss: {loss.item():.4f}")


# %% [markdown]
#  ## 3. 评估结果
# 
#  查看预测下标与真实下标, 并结合注意力图理解模型行为.

# %%
test_x, test_y = get_batch(4, 10, 1, 100)
sample_tokens = [str(v) for v in test_x[0].tolist()]

model.eval()
with torch.no_grad():
    logits = model(test_x) # (batch, seq_len, 1) → (batch, seq_len)
    pred = logits.argmax(dim=-1)  # (batch,) 取每行最大值的索引作为预测结果

    for i in range(test_x.size(0)):
        print(f"input : {test_x[i].tolist()}")
        print(f"pred  : {pred[i].tolist()}")
        print(f"target: {test_y[i].tolist()}")
        print("-" * 30)

    plot_attention(model.attn_w, tokens=sample_tokens)


# %% [markdown]
#  # 实验2 同时找最大值和最小值下标
# 
# ## 1. 数据集构建
# 
#  与实验1相比, 这里同一输入需要同时完成两个目标:
# 
#  - 预测最大值位置
# 
#  - 预测最小值位置
# 

# %%
def get_batch_max_min(batch, sql_len, low, high):
    # y[:, 0] 是 max 下标，y[:, 1] 是 min 下标
    train_x = torch.randint(low, high, (batch, sql_len), dtype=torch.long)  # (batch_size, n)
    max_y = torch.argmax(train_x, dim=-1)  # (batch_size,)
    min_y = torch.argmin(train_x, dim=-1)  # (batch_size,)
    return train_x, torch.stack([max_y, min_y], dim=-1)  # (batch_size, 2)
get_batch_max_min(2, 5, 0, 10)

# %% [markdown]
#  ## 2. 训练实验
# 
#  因此输出层设为 `output_dim=2`, 分别对应 max/min 两路预测.
# 
#  `pred` 形状是 `(batch, seq_len, 2)`：
# 
#  - `pred[:, :, 0]` 用来做最大值位置分类
# 
#  - `pred[:, :, 1]` 用来做最小值位置分类
# 
#  总损失为两路交叉熵之和.

# %%
epochs = 300
model = SelfAttention(vocab_size=100, output_dim=2, d_model=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

model.train()
for i in range(epochs):
    train_x, train_y = get_batch_max_min(1024, 10, 0, 100)
    optimizer.zero_grad()
    pred = model(train_x)  # (batch, seq_len, 2)

    pred_max = pred[:, :, 0]  # (batch, seq_len)
    pred_min = pred[:, :, 1]  # (batch, seq_len)
    loss_max = criterion(pred_max, train_y[:, 0])
    loss_min = criterion(pred_min, train_y[:, 1])
    loss = loss_max + loss_min

    loss.backward()
    optimizer.step()
    if (i+1) % 10 == 0:
        print(
            f"Epoch {i+1}, Loss: {loss.item():.4f}, "
            f"MaxLoss: {loss_max.item():.4f}, MinLoss: {loss_min.item():.4f}"
        )

# %% [markdown]
#  ## 3. 评估实验
# 
#  同时打印 max/min 两路预测结果, 并观察注意力图是否出现对极值位置的关注模式

# %%
test_x, test_y = get_batch_max_min(4, 10, 1, 100)
sample_tokens = [str(v) for v in test_x[0].tolist()]

model.eval()
with torch.no_grad():
    pred = model(test_x)
    pred_max = pred[:, :, 0].argmax(dim=-1)
    pred_min = pred[:, :, 1].argmax(dim=-1)

    for i in range(test_x.size(0)):
        print(f"input : {test_x[i].tolist()}")
        print(f"pred  : [max={pred_max[i].item()}, min={pred_min[i].item()}]")
        print(f"target: [max={test_y[i,0].item()}, min={test_y[i,1].item()}]")
        print("-" * 30)

    plot_attention(model.attn_w, tokens=sample_tokens, title="Exp2 Max-Min Attention")




