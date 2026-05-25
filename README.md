# 🌳 Neural Forest

> **A sparse Mixture-of-Experts language model designed for local deployment with persistent memory.**

[![Status](https://img.shields.io/badge/status-PoC-yellow.svg)](https://github.com/your-username/neural-forest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5+-orange.svg)](https://pytorch.org/)

---

## ✨ What Makes Forest Different

Most LLMs share two limitations:
1. **They forget you** — every conversation starts from zero
2. **They need the cloud** — large models can't run locally

**Forest solves both:**

🧠 **Sparse architecture** — Only 37% of parameters active per token, enabling 1B+ models on 16GB VRAM

💾 **Persistent local memory** — Your conversations, projects, and preferences live on your device, not in the cloud

🎯 **Domain-specialized zones** — 10 expert modules dynamically loaded based on the query

🔒 **Zero forgetting** — Train new domains without losing existing capabilities

---

## 🏗️ Architecture

```
                    Input Tokens
                         │
                         ▼
            ┌────────────────────────┐
            │   Shared Spine (6L)    │  🔥 Always in VRAM
            │   - Attention layers   │
            │   - KV Cache           │
            └────────────┬───────────┘
                         │
                ┌────────▼────────┐
                │     Router      │
                │  (Top-1 / Top-2)│
                └────────┬────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼        ▼        ▼        ▼      ▼
     Zone 0   Zone 1   Zone 2   ...   Zone 10
     (Skip)   Math    Code     ...   Translation
                                          
              🟡 Loaded from RAM (~3ms)
                         │
                         ▼
                     LM Head
                         │
                         ▼
                    Next Token
```

### Key innovations:
- **Shared Spine**: 6-layer attention backbone keeps KV cache coherent across zone switches
- **Specialty Zones**: FFN-only experts that swap dynamically from RAM to VRAM
- **Skip Zone**: Function words bypass expert computation entirely (saves ~20% FLOPs)
- **Dynamic Top-K**: Top-1 for simple tokens, Top-2 for complex multi-domain queries

---

## 📊 Goals & Targets

| Metric | Minimum | Target | Status |
|--------|--------:|-------:|:------:|
| VRAM savings vs Dense | 50% | 65% | 🚧 |
| FLOPs reduction | 50% | 65% | 🚧 |
| Zero-forgetting accuracy | 90% | 95% | 🚧 |
| Local memory recall | 100% | 100% | 🚧 |

---

## 🚀 Roadmap

- [x] **Phase 4.0**: Architecture design & validation
- [ ] **Phase 4.1**: 50M smoke test (proof the architecture works)
- [ ] **Phase 4.2**: 125M proof-of-concept (measure VRAM savings)
- [ ] **Phase 4.3**: 500M scaling (prove scaling law)
- [ ] **Phase 5.0**: 1B production model (with funding)
- [ ] **Phase 6.0**: 30B target (local deployment on 16GB VRAM)

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/neural-forest.git
cd neural-forest

# Install dependencies
pip install -e .
```

**Requirements:**
- Python 3.11+
- PyTorch 2.5+
- CUDA 12+ (for training)
- 16GB+ VRAM (for largest models)

---

## 📁 Project Structure

```
neural-forest/
├── forest/                 # Core library
│   ├── core/              # Model components
│   ├── memory/            # SQLite memory + temporal
│   ├── training/          # Training infrastructure
│   └── utils/             # Benchmarks & monitoring
├── tests/                  # Unit tests
├── scripts/                # Training scripts
└── notebooks/              # Colab notebooks
```

See [DECISIONS.md](DECISIONS.md) for architectural decisions and [ARCHITECTURE.md](ARCHITECTURE.md) for technical deep-dive.

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{neural-forest-2026,
  author = {Naoufel},
  title = {Neural Forest: A Sparse MoE Architecture for Local LLMs},
  year = {2026},
  url = {https://github.com/your-username/neural-forest}
}
```

---

## 📜 License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

## 🤝 Contact

**Author:** Naoufel
**Location:** Laval, Quebec, Canada
**Status:** Proof of Concept (seeking funding for 30B scale)
