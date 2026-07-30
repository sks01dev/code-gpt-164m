# CodeGPT: 164M Parameter Causal Language Model

A custom decoder-only Causal Transformer engineered completely from scratch in PyTorch following GPT-2 Small architectural specifications.

## 📌 Technical Overview
This repository contains the full implementation of a 162.5M parameter autoregressive language model trained on a Python instruction-following dataset (`Alpaca` schema). The core objective of this project is to demonstrate low-level Transformer implementation mechanics, multi-head attention algebra, CUDA memory management, and custom inference optimizations without relying on high-level model abstractions.

## 🛠️ Architecture Specifications
- **Architecture:** Decoder-Only Causal Transformer (GPT-2 Small topology)
- **Parameters:** 162.5M total (~124M Transformer Backbone + 38.5M Embedding Layer)
- **Hyperparameters:** 12 Layers, 12 Attention Heads, 768 Embedding Dimension, GELU non-linearities
- **Context Window:** 1024 tokens
- **Tokenizer:** GPT-2 Byte-Pair Encoding (50,257 vocabulary size) with ~2.67x token compression over character mapping
- **Optimization:** Automatic Mixed Precision (FP16 `autocast` + `GradScaler`)

## ⚡ Inference Engineering: KV-Caching Layer
To optimize autoregressive sequence generation, a custom Key-Value (KV) Caching layer was implemented within the attention heads:
- **Standard Attention:** Recomputes Key and Value projections for all past tokens at every step ($O(N^2)$ per generated token).
- **KV-Cached Attention:** Preserves historical Key/Value tensors across time steps, projecting only the incoming token ($T=1$), successfully shifting sequence generation complexity to $O(1)$ per step.

## 📊 Training Dynamics
- **Hardware:** NVIDIA T4 GPU
- **Optimizer:** AdamW ($\text{lr} = 3 \times 10^{-4}$)
- **Convergence:** Cross-entropy loss converged from **11.02** down to **2.38** across 2,500 steps.

## 📁 Repository Structure
- `src/model.py`: Modular PyTorch implementations of `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, and `CodeGPT`.
- `app.py`: Interactive Streamlit Web UI executing the KV-cached decoding loop.
- `requirements.txt`: Python package dependencies.
