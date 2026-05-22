# Transformer Research Notes and a From-Scratch Implementation

**English** | [中文](README_CN.md)

Transformers are a core theoretical foundation of today's large AI models. It is no exaggeration to say that the field would look very different without the landmark paper *Attention Is All You Need*. Studying the Transformer architecture systematically is one of the most direct and effective ways to understand how modern large models work.

This repository documents my process of learning Transformer and re-implementing it with PyTorch. The goal is to turn abstract formulas into runnable, inspectable code and reproducible experiments.

The implementation does not strictly mirror the original architecture from *Attention Is All You Need*. I simplify some parts and adopt implementation choices that are more common in modern practice.

The project is designed for beginners to read sequentially and run experiments along the way. Each chapter combines theory, code walkthroughs, and hands-on experiments.

## What You Will Find Here

### My Own Research-Oriented Understanding

This repository is not a simple copy of papers, tutorials, or open-source code. It is a set of Transformer learning notes written with a research-oriented mindset.

I try to stay faithful to the classic papers and mainstream implementations, while also adding my own understanding, derivations, and experimental judgments. For example, when explaining Attention, I do not stop at the metaphor that Q is the query, K is the key, and V is the value. I care more about how `QK^T` constructs a relationship matrix between tokens. When explaining multi-head attention, RoPE, and Transformer Blocks, I also focus on what capabilities these modules actually give the model.

As a result, you will find explanations, experimental tradeoffs, and implementation choices shaped by my own thinking.

### Breaking Down the Core Components of Transformer

This repository breaks Transformer into small modules and rebuilds them step by step in learning order. You will start with single-head attention, then move on to multi-head attention, RoPE positional encoding, Transformer Blocks, and finally a trainable Transformer. Each chapter focuses on one clear concept, making it easier to move from the formula to the code.

### Notebooks You Can Run and Inspect Directly

The content progresses chapter by chapter: starting with single-head attention, then gradually adding multi-head attention, RoPE, Transformer Blocks, and finally a trainable Encoder/Decoder. I keep the explanations as plain as possible and avoid unnecessary mathematical complexity. All code experiments are reproducible, so you can directly inspect tensor shapes and computation steps.

### A Balance Between Classic Papers and Modern Practice

The project follows the core ideas of the original Transformer, but it does not stick to every detail of the paper. Instead, it uses implementation patterns that are more common and more suitable for learning and practice today. For example, the positional encoding section uses the now widely used RoPE instead of focusing only on the original sinusoidal positional encoding.

### Designed for Sequential Learning and Experimentation

If this is your first systematic study of attention mechanisms, this repository can serve as a complete practice roadmap. The chapters build on each other, gradually moving from the basics to a complete model. The experiments are intentionally simple and intuitive, so you can read, run, and develop an overall understanding of Transformer along the way.

## Chapter Progress

You can open each chapter directly from the links below:

| Chapter | Status | Description |
| --- | --- | --- |
| [Preface](notebooks/00_Preface/Preparation.ipynb) | Complete | Warm-up with basic PyTorch components and the training workflow |
| [Single-Head Self-Attention](notebooks/01_SingleHeadSelfAttention/SingleHeadSelfAttention.ipynb) | Complete | Code explanation and experimental verification for single-head attention |
| [Multi-Head Attention](notebooks/02_MultiHeadAttention/MultiHeadAttention.ipynb) | Complete | Code explanation and experiments for implementing multi-head attention on top of single-head attention |
| [RoPE Positional Encoding](notebooks/03_AttentionWithRoPE/AttentionWithRoPE.ipynb) | Complete | Injecting positional information with RoPE, including code explanations and experiments |
| [Transformer Encoder](notebooks/04_TransformerEncoder/TransformerEncoder.ipynb) | Complete | TransformerEncoder experiments on a real dataset |
| [Transformer Decoder](notebooks/05_TransformerDecoder/TransformerDecoder.ipynb) | In progress | TransformerDecoder for sequence generation |

## Running Locally

### Step 1: Prepare the Python Environment

Python 3.13 or later is recommended. First clone the repository and enter the project root directory:

```bash
git clone https://github.com/Antrairs/inside-transformer.git
cd inside-transformer
```

### Step 2: Choose a Tool

#### uv

```bash
# 1. Download the required packages
uv sync

# 2. Activate the virtual environment
.venv/Scripts/activate
```

#### pip

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv/Scripts/activate

# Install dependencies
pip install -e .
```

### Step 3: Run the Notebooks

Start Jupyter Notebook from the project root directory:

```bash
jupyter notebook
```

Each chapter's `ipynb` file is in the `notebooks/` directory. Open one to read the notes and run the code.

## Main References

The first and most important reference is the classic paper *Attention Is All You Need*. Without it, modern large language models as we know them would not exist. <https://arxiv.org/abs/1706.03762>

I also referenced Li Mu's book *Dive into Deep Learning*, especially its prerequisite material and Chapter 10 on attention mechanisms. The book is available online for free. <https://zh.d2l.ai>
