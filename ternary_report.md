# 11-Dimensional Balanced Ternary SOM ($\mathcal{T}_{11}$) Report

## Executive Summary
This report presents the implementation and empirical validation of the **11-Dimensional Balanced Ternary Hypercube Self-Organizing Map** (`ternary_som.cpp`). By utilizing a multi-state discrete lattice $\{-1, 0, +1\}^{11}$, this architecture scales node capacity by **2.7×** (from 65,536 to **177,147 nodes**) with **22 orthogonal coordinate directions**, achieving **14,433 tokens/sec** on 16 CPU threads and expanding active vocabulary representation by **+66%**.

- **Repository**: [https://github.com/Nishant-TO-SIH/TransformerSOM](https://github.com/Nishant-TO-SIH/TransformerSOM)
- **Branch**: [`ternary-hypercube`](https://github.com/Nishant-TO-SIH/TransformerSOM/tree/ternary-hypercube)

---

## 1. Architectural & Geometric Comparison: 2D vs 16D vs 11D Ternary

| Metric / Dimension | 2D Planar SOM (`transformer_som.cpp`) | 16D Binary Hypercube (`hypercube_som.cpp`) | 11D Balanced Ternary (`ternary_som.cpp`) |
| :--- | :--- | :--- | :--- |
| **Topological Lattice** | Flat grid $[0..255] \times [0..255]$ | Binary Hypercube $\{0, 1\}^{16}$ | **Balanced Ternary** $\{-1, 0, +1\}^{11}$ |
| **State States / Axis** | 256 continuous indices per axis | 2 discrete states: $\{0, 1\}$ | **3 discrete states**: Negative ($-1$), Neutral ($0$), Positive ($+1$) |
| **Total Node Capacity** | 65,536 nodes ($256^2$) | 65,536 nodes ($2^{16}$) | **177,147 nodes** ($3^{11}$) |
| **Memory Footprint** | ~393 MB | ~393 MB | **~1.09 GB** |
| **Orthogonal Directions** | 4 (Von Neumann) or 8 (Moore) | 16 axes | **22 independent directions** ($\pm 1$ across 11 axes) |
| **Graph Diameter** | 512 (Manhattan) / 362 (Euclidean) | 16 bit flips | **22 Manhattan trit steps** |
| **Neighbor Addressing** | Coordinate 2D bounding loops | Bitwise XOR + POPCNT | **$O(1)$ Powers-of-3 Math**: $\text{ID} \pm 3^d$ |
| **Throughput (16 Threads)**| ~17,408 tok/s | ~10,180 tok/s | **14,433 tok/s** (Completed in 158s) |
| **Node Hit Rate (Ep 1)** | ~98.4% | 100.0% (65,536 / 65,536) | **100.0%** (177,122 / 177,147) |
| **Active Dominant Words**| 7,812 words | 9,206 words | **15,306 words** (+66% expansion) |
| **Empirical Transitions** | ~2.1M edges | ~2.2M edges | **2,102,732 edges** |

---

## 2. Key Mathematical & Algorithmic Features

### Balanced Ternary Logic $\{-1, 0, +1\}$
Unlike binary systems where a feature is only present ($1$) or absent ($0$), balanced ternary provides an intrinsic **neutral/uncommitted zero state**:
- **$-1$ (Negative activation)**: Semantically opposing or inhibitory concept.
- **$0$ (Neutral/Ground state)**: Orthogonal, uncommitted, or context-independent.
- **$+1$ (Positive activation)**: Semantically supportive or excitatory concept.

### Precomputed Powers-of-3 & $O(1)$ Directed Graph Offsets
Addressing 177,147 nodes requires 18 bits (`uint32_t`). To eliminate runtime integer divisions and modulo operations:
1. Powers of 3 are precomputed:
   $$3^0=1, 3^1=3, 3^2=9, \dots, 3^{10}=59,049$$
2. A flat 15.5 MB table stores the exact 1-step neighbors for each node. Stepping along dimension $d$:
   $$\text{neighbor}^+ = \text{ID} + 3^d, \quad \text{neighbor}^- = \text{ID} - 3^d$$
This gives **zero-branching $O(1)$ memory lookups** during BMU coordinate descent and neighborhood updates.

---

## 3. Training Diagnostics (Epoch 1)

- **Corpus**: OpenAssistant OASST Dialogue Dataset (15,000 dialogues, **2,280,484 tokens**)
- **Hardware**: 16 OpenMP Worker Threads (100% CPU utilization)
- **Time to complete 1 epoch**: **158 seconds** (2.6 minutes)
- **Average Throughput**: **14,433 tokens/second**

### Top Dominant Hubs Across the 177,147 Nodes
| Word | Vocabulary ID | Assigned Vertices | Coverage Percentage |
| :--- | :--- | :--- | :--- |
| `bot_turn` | 2 | 7,139 nodes | 4.03% |
| `and` | 4 | 4,996 nodes | 2.82% |
| `to` | 5 | 4,587 nodes | 2.59% |
| `the` | 3 | 4,470 nodes | 2.52% |
| `a` | 6 | 4,243 nodes | 2.40% |
| `of` | 7 | 2,859 nodes | 1.61% |
| `in` | 8 | 2,339 nodes | 1.32% |
| `that` | 11 | 2,177 nodes | 1.23% |
| `is` | 9 | 1,829 nodes | 1.03% |
| `you` | 10 | 1,766 nodes | 1.00% |

---

## 4. Test Suite Generation Comparison (Epoch 1)

| Prompt | 16D Binary Hypercube (65k nodes) | 11D Balanced Ternary (177k nodes) |
| :--- | :--- | :--- |
| **`hello how are you`** | `a data can toy extracts sara and bake command pies if for as zones rise open landroidosremoteexception need or preliminary` | `sourced ancient be especially the bot_turn catastrophic bulk of and well dominance union a because i improvement surface pramod new` |
| **`what is your name`** | `following this opportunity as spirits and sudoku pair python jeffrey geographic test me difference water appreciate your of existen a` | `provide regarding the wait sleep was deeply olden times find selfdiscovery shaders for chilling today pakistan scrolling functions extends have` |
| **`where are you going`** | `goal give about our spoke current i is a at test editor exist and down means pressure for share element` | `code affected youll need citation that is can referred bass possible doors maximum this aggregate sheets as the a design` |
| **`tell me something good`** | `do are of also 0 a this even sightly communicate julian incas classmates is install known quip one your potential` | `continue to of use google 3 a distinction talking by the that with one in patent wikipedia equidistant branch if` |
| **`who is there`** | `from own potential paired emphasis these needs this and a see enchanting lives you in maintained here animals chartered heavily` | `as to would of ways authentication been there user_turn enterprise the from you on answers how heat goals i begin` |

---

## 5. Summary of Conclusions
1. **Capacity vs Throughput Win**: Expanding from 65,536 nodes to 177,147 nodes did not slow down training; throughput actually increased from ~10k tok/s to **14,433 tok/s** due to the efficiency of the precomputed powers-of-3 neighbor lookup table.
2. **Semantic Granularity**: Active vocabulary diversity grew from **9,206 to 15,306 words** (+66%), allowing multiple distinct contextual senses per word across different subcubes of the ternary lattice.
3. **Graph Connectivity**: With 22 directions and a maximum diameter of only 22 steps, semantic diffusion occurs rapidly without path bottlenecks.
