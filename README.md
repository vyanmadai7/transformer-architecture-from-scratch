# ⚡ Transformer from Scratch
### A clean PyTorch implementation of *Attention Is All You Need*

----------------------------

```
         src tokens                    trg tokens
              │                             │
     ┌────────▼────────┐         ┌──────────▼──────────┐
     │  Word Embedding  │         │   Word Embedding     │
     │  + Positional    │         │   + Positional       │
     └────────┬────────┘         └──────────┬──────────┘
              │                             │
     ┌────────▼────────┐         ┌──────────▼──────────┐
     │  Encoder Block   │ ──────► │   Decoder Block      │
     │  × N layers      │  enc_out│   × N layers         │
     └─────────────────┘         └──────────┬──────────┘
                                             │
                                   ┌─────────▼─────────┐
                                   │  Linear + Softmax  │
                                   └───────────────────┘
```

---------------------------

## 📌 Overview

This repo implements the full **Transformer architecture** in pure PyTorch — no HuggingFace, no shortcuts. Every component is built from the ground up, making it ideal for learning, research, or as a starting point for custom seq2seq models.

> Based on the seminal paper: [*Attention Is All You Need* — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

------------------------

## 🧱 Architecture

| Component | Description |
|---|---|
| `SelfAttention` | Scaled dot-product multi-head attention |
| `TransformerBlock` | Attention → Add & Norm → FFN → Add & Norm |
| `Encoder` | Word + positional embeddings + N transformer blocks |
| `DecoderBlock` | Masked self-attention + cross-attention + FFN |
| `Decoder` | Embedding + N decoder blocks + linear projection |
| `Transformer` | Full encoder-decoder with src/trg masking |

### Attention Formula

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

-----------------------

## 🚀 Quick Start

**Clone & install dependencies:**

```bash
git clone https://github.com/your-username/transformer-from-scratch.git
cd transformer-from-scratch
pip install torch
```

**Run the test forward pass:**

```bash
python transformer.py
# → torch.Size([2, 7, 10])
```

------------------------

## 🔧 Usage

```python
import torch
from transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    src_vocab_size=10,
    trg_vocab_size=10,
    src_pad_idx=0,
    trg_pad_idx=0,
    embed_size=512,    # d_model
    num_layers=6,      # encoder/decoder depth
    heads=8,           # attention heads
    forward_expansion=4,
    dropout=0.1,
    max_length=100,
    device=device
).to(device)

src = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0]]).to(device)
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2]]).to(device)

out = model(src, trg)
print(out.shape)  # → (batch, trg_len, trg_vocab_size)
```

-------------

## ⚙️ Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `embed_size` | `512` | Model dimensionality (dₘₒdₑₗ) |
| `num_layers` | `6` | Number of encoder & decoder layers |
| `heads` | `8` | Number of attention heads |
| `forward_expansion` | `4` | FFN hidden dim multiplier |
| `dropout` | `0.1` | Dropout rate |
| `max_length` | `100` | Max sequence length |

-------------

## 🔍 Key Implementation Details

- **Masking** — Padding mask (`src_mask`) hides pad tokens; causal mask (`trg_mask`) prevents attending to future tokens via `torch.tril`
- **Multi-Head Attention** — Uses `torch.einsum` for clean, readable tensor contractions
- **Positional Encoding** — Learned embeddings (rather than fixed sinusoidal), enabling flexibility
- **Layer Norm** — Applied post-residual connection (original paper style)

-------------

## 📁 File Structure

```
transformer-from-scratch/
└── transformer.py   # Full implementation + test
```

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP

-------------

## 📄 License

MIT
