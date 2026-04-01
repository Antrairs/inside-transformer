# Transformer 研究笔记与从零开始的复现

这个仓库记录了我学习并用 Pytorch 复现 Transformer 的全过程。我希望把抽象的公式变成可以运行、可以观察、可以复现实验结果的代码。

我并非严格按照 Attention Is All You Need 原论文的架构做，而是做了简化并使用了现代更常用的实现

项目适合初学者按顺序阅读和跟着跑实验，每个章节都有对应的代码讲解和实验。

## 你会在这里看到什么

内容按章节推进，从单头注意力开始，逐步加入多头机制、RoPE、Transformer Block，最后到一个可训练的 Encoder。我的写法会尽量保持朴素，不依赖复杂封装，方便直接对照张量形状和计算过程。

如果你是第一次系统学注意力机制，这个仓库可以当作练手路线图。

## 章节进度

| 章节 | 当前状态 | 说明 |
| --- | --- | --- |
| [Preface](notebooks/00_Preface/Preparation.ipynb) | 完成 | PyTorch 基本组件与训练流程预热 |
| [SingleHeadSelfAttention](notebooks/01_SingleHeadSelfAttention//SingleHeadSelfAttention.ipynb) | 完成 | 单头注意力的代码解释和实验验证效果 |
| [MulitHandAttention](notebooks/02_MulitHandAttention/MulitHandAttention.ipynb) | 完成 | 在单头注意力的基础上实现多头注意力的代码解释和实验 |
| [AttentionWithRoPE](notebooks/03_AttentionWithRoPE/AttentionWithRoPE.ipynb) | 进行中 | 用RoPE位置编码注入位置信息, 包含代码解释和实验 |
| [TransformerBlock](notebooks/04_TransformerBlock/TransformerBlock.ipynb) | 待做 | 将组件组装为 TransformerBlock |
| [TransformerEncoder](notebooks/05_TransformerEncoder/TransformerEncoder.ipynb) | 待做 | 用多层 Encoder 进行真实的语义分类实验 |

## 如何使用

### 第一步：准备 Python 环境

建议使用 Python 3.13 或更高版本。先克隆并进入项目根目录：

```bash
git clone https://github.com/Antrairs/inside-transformer.git
cd inside-transformer
```

如果你使用 uv 执行：

```bash
uv sync
```

如果你使用 pip：

创建虚拟环境

```bash
python -m venv .venv
```

进入虚拟环境

```bash
.venv/Scripts/activate
```

安装依赖

```bash
pip install -e .
```

### 第二步：运行 Notebook

在项目根目录启动：

```bash
jupyter notebook
```

## 主要参考

首先就是神级经典论文**Attention Is All You Need**, 没有这篇论文也就没有现在的大语言模型 <https://arxiv.org/abs/1706.03762>

然后是李沐老师的**动手学深度学习**一书, 本项目参考了该书的预备知识和第十章的注意力机制, 该书在网上能免费看 <https://zh.d2l.ai>
