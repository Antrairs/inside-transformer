# Transformer 研究笔记与从零开始的复现

Transformer 是当今 AI 大模型的核心理论基石. 可以说, 没有 Transformer 这篇开创性论文, 就很难有今天大模型的快速发展. 其重要性不言而喻, 而系统学习 Transformer 架构, 也是我们深入理解大模型原理最根本, 最有效的路径之一.

这个仓库记录了我学习并用 Pytorch 复现 Transformer 的全过程. 我希望把抽象的公式变成可以运行, 可以观察, 可以复现实验结果的代码.

我并非严格按照 Attention Is All You Need 原论文的架构做，而是做了简化并使用了现代更常用的实现.

项目适合初学者按顺序阅读和跟着跑实验, 每个章节都有对应的代码讲解和实验.

## 你会在这里看到什么

### 从零拆解 Transformer 的核心组件

这个仓库不是直接给出一个完整模型, 而是把 Transformer 拆成多个小模块, 按照学习顺序逐步复现. 你会先看到单头注意力, 再到多头注意力, RoPE 位置编码, Transformer Block, 最后组装成可训练的 Transformer. 每一章都尽量围绕一个明确的概念展开, 方便从公式是什么过渡到代码怎么写.

### 可以直接运行和观察的 Notebook

内容按章节推进, 从单头注意力开始, 逐步加入多头机制, RoPE, Transformer Block, 最后到一个可训练的 Encoder. 我的写法会尽量保持朴素, 不依赖复杂数学推理, 全部代码实验均可复现, 方便直接对照张量形状和计算过程.

### 兼顾经典论文和现代实践

项目参考了经典的 Transformer 思路, 但并不拘泥于原论文的所有细节, 而是结合了更常见, 更适合学习和实践的实现方式. 比如位置编码部分会使用现代更主流的 RoPE, 而不是只停留在最早的正余弦编码方案上.

### 适合按顺序学习和做实验

如果你是第一次系统学习注意力机制, 这个仓库可以作为一条完整的练手路线图. 章节之间是连续推进的, 内容从基础到完整模型逐步加深, 实验设计得尽可能简单直观, 适合一边阅读一边跟着实验, 最后形成对 Transformer 的整体认知.

## 章节进度

可以直接点击章节链接阅读:

| 章节 | 当前状态 | 说明 |
| --- | --- | --- |
| [前言](notebooks/00_Preface/Preparation.ipynb) | 完成 | PyTorch 基本组件与训练流程预热 |
| [单头自注意力](notebooks/01_SingleHeadSelfAttention/SingleHeadSelfAttention.ipynb) | 完成 | 单头注意力的代码解释和实验验证效果 |
| [多头注意力](notebooks/02_MultiHeadAttention/MultiHeadAttention.ipynb) | 完成 | 在单头注意力的基础上实现多头注意力的代码解释和实验 |
| [RoPE位置编码](notebooks/03_AttentionWithRoPE/AttentionWithRoPE.ipynb) | 完成 | 用RoPE位置编码注入位置信息, 包含代码解释和实验 |
| [Transformer编码器](notebooks/04_TransformerEncoder/TransformerEncoder.ipynb) | 进行中 | 将组件组装并堆叠N层为 TransformerEncoder |
| [Transformer解码器](notebooks/05_TransformerDecoder/TransformerDecoder.ipynb) | 待做 | TransformerDecoder 用于生成序列 |

## 本地运行

### 第一步: 准备 Python 环境

建议使用 Python 3.13 或更高版本. 先克隆并进入项目根目录:

```bash
git clone https://github.com/Antrairs/inside-transformer.git
cd inside-transformer
```

### 第二步: 选择工具

- 如果你使用 uv:

```bash
# 1. 下载所需要的包
uv sync

# 2. 进入虚拟环境
.venv/Scripts/activate
```

- 如果你使用 pip:

```bash
# 创建虚拟环境
python -m venv .venv

# 进入虚拟环境
.venv/Scripts/activate

# 安装依赖
pip install -e .
```

### 第三步: 运行 Notebook

在项目根目录启动:

```bash
jupyter notebook
```

各个章节的 `ipynb` 文件在 `notebooks/` 目录下, 点击即可阅读和运行代码

## 主要参考

首先就是神级经典论文**Attention Is All You Need**, 没有这篇论文也就没有现在的大语言模型. <https://arxiv.org/abs/1706.03762>

然后是李沐老师的**动手学深度学习**一书, 本项目主要参考了该书的预备知识和第十章的注意力机制, 该书在网上能免费看. <https://zh.d2l.ai>
