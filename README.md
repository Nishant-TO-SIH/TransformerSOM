# TransformerSOM

> **Bio-Plausible Autoregressive Self-Organizing Map with Multi-Head QKV Self-Attention & Hebbian Learning**

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://isocpp.org/)
[![OpenMP](https://img.shields.io/badge/Parallelism-OpenMP%20SIMD-green.svg)](https://www.openmp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TransformerSOM** is a novel neuromorphic language model that combines the spatial self-organizing properties of **Kohonen Self-Organizing Maps (SOM)** with the temporal context-routing of **Transformer Q, K, V Self-Attention**, operating completely free of backpropagation through time (BPTT).

---

## Key Architectural Highlights

- **Zero-Backprop Local Learning**: Uses **Oja's Information-Theoretic Plasticity** for online $W_q, W_k, W_v$ attention matrix updates and **Kohonen Topological Vector Quantization** for 2D semantic manifold formation.
- **Continuous Dense Manifold**: Powered by 384-dimensional dense semantic vectors (from `all-MiniLM-L6-v2`) forming 768-dimensional node states (`384d word + 384d context attention`).
- **Autoregressive Flow with Bayesian Fusion**: Combines learned Markov transition flow with real-time attentional energy and Nucleus Top-P ($p=0.9$) stochastic sampling.
- **Zero-Heap Hardware-Accelerated Engine**: Pure stack-allocated intermediate buffers, contiguous thread transition logging, and AVX2/FMA SIMD vectorization saturating 16 CPU cores at **17,000+ tokens/second**.

---

## Directory Structure

```
├── transformer_som.cpp   # Core C++ engine (SOM grid, QKV attention, Oja learning, inference)
├── eval_runner.py        # Automated multi-prompt benchmark suite & metric tracker
├── prepare_oasst.py      # Dataset pipeline (OpenAssistant Conversations, 384d MiniLM extraction)
├── oasst_corpus.txt      # Cleaned multi-turn conversational corpus
├── eval_summary.md       # Per-epoch evaluation metrics & diversity scores
└── .gitignore            # Git exclusion for large binaries and checkpoints
```

---

## Building & Running

### Requirements
- GCC 11+ with OpenMP support (`g++`)
- Python 3.8+ (for dataset preparation & automated evaluation)

### Compilation
```bash
g++ -O3 -mavx2 -mfma -fopenmp transformer_som.cpp -o transformer_som.exe
```

### Training
```bash
# Train for 15 epochs without blocking on interactive prompt
./transformer_som.exe --train 15 --no-interactive
```

### Interactive Chat Mode
```bash
./transformer_som.exe
```

### Automated Checkpoint Evaluation
```bash
python eval_runner.py
```
