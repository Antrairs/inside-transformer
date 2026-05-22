<div align="center">
  <h1>Inside Transformer</h1>
  <p><strong>一个以 Notebook 为主的 Transformer 架构学习指南，通过最小可运行的 PyTorch 实现拆解核心组件。</strong></p>
  <p><a href="README.md">English</a> | <strong>中文</strong></p>
  <p>
    <img alt="Python 3.13+" src="https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white">
    <img alt="PyTorch 2.10+" src="https://img.shields.io/badge/PyTorch-2.10%2B-EE4C2C?logo=pytorch&logoColor=white">
    <img alt="Jupyter Notebook" src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white">
    <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-blue">
  </p>
</div>

这个项目把 *Attention Is All You Need* 背后的核心思想拆成小而可运行的实验。它遵循原论文的主要精神，同时在更有助于学习的地方采用现代实践，比如 RoPE。

## 亮点

- 从论文到代码的学习路线：按顺序重建 self-attention、multi-head attention、RoPE、Transformer Encoder 和 GPT 风格的 Decoder。
- 可运行的章节式 Notebook：每个主题都结合理论解释、PyTorch 代码、张量形状、训练循环和实验结果。
- 优先保持最小实现：让核心逻辑保持可见，而不是隐藏在大型框架或生产级库之后。
- 可以观察的实验：包含玩具任务、真实数据集训练、评估步骤，以及在合适位置加入的注意力可视化。
- 在有帮助的地方使用现代实践：例如使用 RoPE，同时解释它和原始 Transformer 论文之间的关系。

## 章节

| 章节 | 状态 | 说明 |
| --- | --- | --- |
| [前言](notebooks/00_Preface/Preparation.ipynb) | 完成 | PyTorch 基础组件与训练流程 |
| [单头自注意力](notebooks/01_SingleHeadSelfAttention/SingleHeadSelfAttention.ipynb) | 完成 | 最小 self-attention 实现与实验 |
| [多头注意力](notebooks/02_MultiHeadAttention/MultiHeadAttention.ipynb) | 完成 | 基于单头注意力构建 multi-head attention |
| [RoPE 位置编码](notebooks/03_AttentionWithRoPE/AttentionWithRoPE.ipynb) | 完成 | 使用 RoPE 注入位置信息 |
| [Transformer 编码器](notebooks/04_TransformerEncoder/TransformerEncoder.ipynb) | 完成 | TransformerEncoder 在真实数据集上的实验 |
| [Transformer 解码器](notebooks/05_TransformerDecoder/TransformerDecoder.ipynb) | 进行中 | 用于序列生成的 TransformerDecoder |

## 快速开始

建议使用 Python 3.13 或更高版本。

```bash
git clone https://github.com/Antrairs/inside-transformer.git
cd inside-transformer
```

使用 uv：

```bash
uv sync
.venv/Scripts/activate
```

或使用 pip：

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
```

在项目根目录启动 Jupyter Notebook：

```bash
jupyter notebook
```

然后打开 `notebooks/` 目录下的 Notebook，并按章节顺序阅读和运行。

## 参考资料

- *Attention Is All You Need*: <https://arxiv.org/abs/1706.03762>
- *动手学深度学习*: <https://zh.d2l.ai>
