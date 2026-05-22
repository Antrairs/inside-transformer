<div align="center">
  <h1>Inside Transformer</h1>
  <p><strong>A notebook-first guide to the Transformer architecture, built from minimal, runnable PyTorch implementations.</strong></p>
  <p><strong>English</strong> | <a href="README_CN.md">中文</a></p>
  <p>
    <img alt="Python 3.13+" src="https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white">
    <img alt="PyTorch 2.10+" src="https://img.shields.io/badge/PyTorch-2.10%2B-EE4C2C?logo=pytorch&logoColor=white">
    <img alt="Jupyter Notebook" src="https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white">
    <img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-blue">
  </p>
</div>

This project turns the ideas behind *Attention Is All You Need* into small, runnable experiments. It follows the spirit of the original paper while using modern implementation choices, such as RoPE, when they make the learning path clearer.

## Highlights

- Paper-to-code learning path: rebuilds self-attention, multi-head attention, RoPE, Transformer Encoder, and a GPT-style Decoder in order.
- Runnable, chapter-sized notebooks: each topic pairs explanation with PyTorch code, tensor shapes, training loops, and experiment results.
- Minimal implementations first: keeps the core logic visible instead of hiding it behind a large framework or production library.
- Experiments you can inspect: includes toy tasks, real-dataset training, evaluation steps, and attention visualizations where useful.
- Modern where it helps: uses practical choices such as RoPE while still explaining how they relate to the original Transformer paper.

## Chapters

| Chapter | Status | Description |
| --- | --- | --- |
| [Preface](notebooks/00_Preface/Preparation.ipynb) | Complete | Basic PyTorch components and the training workflow |
| [Single-Head Self-Attention](notebooks/01_SingleHeadSelfAttention/SingleHeadSelfAttention.ipynb) | Complete | A minimal self-attention implementation and experiment |
| [Multi-Head Attention](notebooks/02_MultiHeadAttention/MultiHeadAttention.ipynb) | Complete | Multi-head attention built on top of single-head attention |
| [RoPE Positional Encoding](notebooks/03_AttentionWithRoPE/AttentionWithRoPE.ipynb) | Complete | Positional information injected with RoPE |
| [Transformer Encoder](notebooks/04_TransformerEncoder/TransformerEncoder.ipynb) | Complete | TransformerEncoder experiments on a real dataset |
| [Transformer Decoder](notebooks/05_TransformerDecoder/TransformerDecoder.ipynb) | In progress | TransformerDecoder for sequence generation |

## Quick Start

Python 3.13 or later is recommended.

```bash
git clone https://github.com/Antrairs/inside-transformer.git
cd inside-transformer
```

With uv:

```bash
uv sync
.venv/Scripts/activate
```

Or with pip:

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e .
```

Start Jupyter Notebook from the project root:

```bash
jupyter notebook
```

Then open the notebooks under `notebooks/` and follow the chapters in order.

## References

- *Attention Is All You Need*: <https://arxiv.org/abs/1706.03762>
- *Dive into Deep Learning*: <https://zh.d2l.ai>
