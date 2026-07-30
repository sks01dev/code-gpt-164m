# CodeGPT: 164M Parameter Causal Language Model

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)

A lightweight, decoder-only causal Transformer model engineered completely from scratch in PyTorch following the **GPT-2 Small** architectural specification.

This repository demonstrates low-level Transformer implementation mechanics, attention algebra, GPU memory management, Automatic Mixed Precision (AMP), and custom **$O(1)$ Key-Value (KV) Cache** inference acceleration—built without relying on high-level model abstractions or trainer wrappers inspired from **Andrej Karpathy**.

---

## Decoder Architecture Diagram
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/6f68a7c1-d9e3-4698-85b6-ca04e3582a54" />

---

The system is organized into a clean, modular Python package alongside an automated cloud deployment architecture:

* **`src/model.py`**: Pure PyTorch implementation of the Transformer primitives (`Head`, `MultiHeadAttention`, `FeedForward`, `Block`, and `CodeGPT`).
* **`app.py`**: Streamlit web interface executing the $O(1)$ KV-cached autoregressive generation loop.
* **Hugging Face Model Hub**: Decoupled cloud storage host for the model checkpoint (`code_gpt.pt`), keeping the source repository lightweight.

---

## 🛠️ Model & Hardware Specifications

| Parameter | Specification | Description |
| --- | --- | --- |
| **Model Type** | Causal Decoder-Only | Autoregressive language model |
| **Total Parameters** | **~162.5M** | ~124M Transformer Backbone + 38.5M Tokenizer Embedding |
| **Layers / Depth** | **12 Blocks** | Stacked Transformer blocks with pre-LayerNorm |
| **Attention Heads** | **12 Heads** | Parallel causal multi-head attention (head dim = 64) |
| **Embedding Dim ($d_{\text{model}}$)** | **768** | Hidden state size across all layers |
| **Context Window** | **1024 Tokens** | Sequence block size limit |
| **Vocabulary** | **50,257 Tokens** | GPT-2 Byte-Pair Encoding (BPE) tokenizer |
| **Non-Linearity** | **GELU** | Gaussian Error Linear Unit inside FeedForward expansion |
| **Training Precision** | **FP16 AMP** | `torch.amp.autocast` + `GradScaler` for memory efficiency |

---

## ⚡ Inference Acceleration: $O(1)$ KV-Caching

Standard autoregressive sequence generation recomputes Key ($K$) and Value ($V$) projections for all preceding tokens at every single time step, leading to **$O(N^2)$ computational complexity**.

To accelerate inference, a custom **Key-Value (KV) Cache mechanism** was built directly into the `Head` and `MultiHeadAttention` modules:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

* **Mechanics:** During decoding, historical $K$ and $V$ tensors from previous iterations are retained in memory.
* **Impact:** The network projects $Q, K, V$ for **only the incoming single token** ($T=1$), appending $K$ and $V$ to the cached states. This shifts per-token matrix computation from **$O(N^2)$ to $O(1)$**, resulting in significantly lower generation latency.

---

## 📊 Training Dynamics

* **Dataset:** Python instruction-following dataset (`Alpaca` prompt structure).
* **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999, \text{lr} = 3 \times 10^{-4}$).
* **Hardware Execution:** NVIDIA T4 GPU (Google Colab).
* **VRAM Optimization:** Adjusted training batch size ($B=6$) to operate comfortably within peak VRAM allocations.
* **Loss Convergence:** Cross-entropy loss dropped steadily from **11.02** down to **2.38** over 2,500 training steps.

```text
Step    0 | Train Loss: 11.0293 | Val Loss: 11.0273
Step  600 | Train Loss:  3.1314 | Val Loss:  3.2314
Step 1200 | Train Loss:  2.6218 | Val Loss:  2.8181
Step 1800 | Train Loss:  2.4541 | Val Loss:  2.6794
Step 2499 | Train Loss:  2.1053 | Val Loss:  2.3813

```

---

## 📂 Repository Structure

```text
code-gpt-autocompleter/
├── app.py              # Streamlit Web Application with HF Hub integration
├── requirements.txt    # Production Python dependencies
├── README.md           # Comprehensive project documentation
├── .gitignore          # Ignores local checkpoints and cache files
└── src/
    ├── __init__.py     # Module initialization
    └── model.py        # PyTorch CodeGPT architecture + KV-Cache decoder

```

---

## 🚀 Local Setup & Installation

### 1. Clone Repository & Install Dependencies

```bash
git clone [https://github.com/YOUR_USERNAME/code-gpt-autocompleter.git](https://github.com/YOUR_USERNAME/code-gpt-autocompleter.git)
cd code-gpt-autocompleter
pip install -r requirements.txt

```

### 2. Run Streamlit Application Locally

```bash
streamlit run app.py

```

---

## 🌐 Cloud Deployment Architecture

The application is deployed on **Streamlit Community Cloud** using a decoupled cloud storage workflow:

1. **Model Weights (`code_gpt.pt`):** Hosted on **Hugging Face Model Hub** to bypass git size restrictions (~650MB).
2. **Dynamic Weight Retrieval:** On application startup, `app.py` leverages `huggingface_hub.hf_hub_download` to fetch and load weight tensors into memory automatically.
3. **Execution Engine:** Runs the KV-cached decoding pipeline on Streamlit's cloud backend.

```

```
