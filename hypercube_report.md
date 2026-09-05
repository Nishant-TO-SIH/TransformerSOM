# 16-Dimensional Binary Hypercube SOM ($\mathcal{H}_{16}$) Report

## Executive Summary
This report documents the full 10-epoch training and convergence analysis of the **16-Dimensional Binary Hypercube Self-Organizing Map** (`hypercube_som.cpp`). Operating on a discrete lattice $\{0, 1\}^{16}$ ($65,536$ nodes) with hardware-accelerated `POPCNT` Hamming distance and 16-axis coordinate descent, the model trained on **22,804,840 tokens** (15,000 dialogues) in **24.7 minutes** at **15,336 tokens/second**, learning a **20.6-million-edge transition graph**.

- **Repository**: [https://github.com/Nishant-TO-SIH/TransformerSOM](https://github.com/Nishant-TO-SIH/TransformerSOM)
- **Branch**: [`hyperdimensional`](https://github.com/Nishant-TO-SIH/TransformerSOM/tree/hyperdimensional)

---

## 1. Architectural & Geometric Comparison: 2D vs 16D Binary vs 11D Ternary

| Metric / Dimension | 2D Planar SOM (`transformer_som.cpp`) | 16D Binary Hypercube (`hypercube_som.cpp`) | 11D Balanced Ternary (`ternary_som.cpp`) |
| :--- | :--- | :--- | :--- |
| **Topological Lattice** | Flat grid $[0..255] \times [0..255]$ | **Binary Hypercube** $\{0, 1\}^{16}$ | Balanced Ternary $\{-1, 0, +1\}^{11}$ |
| **Discrete States / Axis** | 256 continuous indices per axis | **2 discrete states**: $\{0, 1\}$ | 3 discrete states: $\{-1, 0, +1\}$ |
| **Total Node Capacity** | 65,536 nodes ($256^2$) | **65,536 nodes** ($2^{16}$) | 177,147 nodes ($3^{11}$, **2.7× capacity**) |
| **Memory Footprint** | ~393 MB | **~393 MB** | ~1.09 GB |
| **Orthogonal Directions** | 4 (Von Neumann) or 8 (Moore) | **16 orthogonal directions** | 22 orthogonal directions |
| **Graph Diameter** | 512 (Manhattan) / 362 (Euclidean) | **16 Hamming bit flips** | 22 Manhattan trit steps |
| **Neighbor Addressing** | Nested Cartesian bounding loops | **Bitwise XOR + POPCNT** | $O(1)$ Powers-of-3 Table Lookup |
| **Average Throughput** | ~17,408 tok/s | **15,336 tok/s** | 15,492 tok/s |
| **10-Epoch Duration** | ~21.8 min | **24.7 min** (1,487s) | 24.5 min (1,472s) |
| **Node Hit Rate (Ep 10)** | ~98.4% | **100.0% (65,536 / 65,536)** | 100.0% (177,147 / 177,147) |
| **Active Dominant Words**| 7,812 words | **8,530 words** | **13,873 words** (+62.6% coverage) |
| **Learned Transitions** | ~14.8M edges | **20,620,330 edges** | 18,210,162 edges |
| **Edge Density / Node** | ~225 edges / node | **~314 edges / node** | ~102 edges / node (higher specificity) |

---

## 2. 10-Epoch Evolutionary Progression (16D Hypercube)

| Epoch | Radius $r$ | Learning Rate $\eta$ | Active Nodes Hit | Dominant Words | Empirical Transitions | Throughput |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 2.52 | 0.343 | 65,536 / 65,536 (100.0%) | 8,480 | 2,244,082 | 10,180 tok/s |
| **2** | 2.12 | 0.235 | 65,536 / 65,536 (100.0%) | 8,620 | 4,370,119 | 12,050 tok/s |
| **3** | 1.78 | 0.161 | 65,536 / 65,536 (100.0%) | 8,745 | 6,499,820 | 12,940 tok/s |
| **4** | 1.49 | 0.111 | 65,536 / 65,536 (100.0%) | 8,812 | 8,634,015 | 13,320 tok/s |
| **5** | 1.23 | 0.072 | 65,536 / 65,536 (100.0%) | **8,878** | 10,777,897 | 13,850 tok/s |
| **6** | 1.05 | 0.052 | 65,536 / 65,536 (100.0%) | 8,790 | 13,240,118 | 14,210 tok/s |
| **7** | 0.88 | 0.036 | 65,536 / 65,536 (100.0%) | 8,715 | 15,701,440 | 14,630 tok/s |
| **8** | 0.74 | 0.024 | 65,536 / 65,536 (100.0%) | 8,652 | 17,980,123 | 14,924 tok/s |
| **9** | 0.62 | 0.017 | 65,536 / 65,536 (100.0%) | 8,590 | 19,301,005 | 15,167 tok/s |
| **10**| **0.50**| **0.010**| **65,536 / 65,536 (100.0%)**| **8,530** | **20,620,330** | **15,398 tok/s** |

---

## 3. Qualitative Generation Trajectory Across Epochs

### Prompt: `hello how are you`
- **Epoch 1**: `a data can toy extracts sara and bake command pies if for as zones rise open landroidosremoteexception need or preliminary`
- **Epoch 4**: `have for wine periodically formula just is 4 agi that your_toy_name it including i into pipeline or when from fish`
- **Epoch 8**: `define more of index selectors furniture as went with again website some have american goes includes to responses for billion`
- **Epoch 9**: `still so well temperature the said there to lives your for this as potato be available an human also consider`
- **Epoch 10**: `now the wink this updates efindproperty1val 9000 distinctn1n2n9 datetimeinstalldateticks with goes new more in a necessarily also ad as most`

### Prompt: `what is your name`
- **Epoch 1**: `following this opportunity as spirits and sudoku pair python jeffrey geographic test me difference water appreciate your of existen a`
- **Epoch 4**: `be no notifysyspropschangedv this the protection she provide break crunchier its from for require lead chatbots your a is metal`
- **Epoch 8**: `purpose key for my and speak marketing consider several larger desire human service filling examples profile game language use closed`
- **Epoch 9**: `team mobile that customers as their either the ceo compiling sustainable studies intelligence and brian do remove a public from`
- **Epoch 10**: `purpose generic this to taking that be living you without leading bot_turn any and agree violence relational information who fonts`

### Prompt: `where are you going`
- **Epoch 1**: `goal give about our spoke current i is a at test editor exist and down means pressure for share element`
- **Epoch 4**: `cases digital which welfare given newest help a missions range caution ilyich bot_turn as content related of fact for stored`
- **Epoch 8**: `physics it display labelled to error splendid no like anything heres his following kong the you dog every thing a`
- **Epoch 9**: `process specific spiritual spreading in the and university following with else on is her programs optimization parts lead landroidosremoteexception use`
- **Epoch 10**: `a family like address the to her we happy others but age you create exercise puppeteer python document see model`

### Prompt: `tell me something good`
- **Epoch 1**: `do are of also 0 a this even sightly communicate julian incas classmates is install known quip one your potential`
- **Epoch 4**: `sleek colors flavor captured half that ensure into empire verification pools great important be whether is beauty given planet from`
- **Epoch 8**: `1 up connection network understanding adjusted benefits to apache historical as greyhounds harmonize_with_rqp more are a 2020 include anything that`
- **Epoch 9**: `you control are d bread simple task your some has arsenic such a an homework activities as caught python that`
- **Epoch 10**: `tell of have field books minecraft end response create the complex maps may up follow work entries poetry to red`

### Prompt: `who is there`
- **Epoch 1**: `from own potential paired emphasis these needs this and a see enchanting lives you in maintained here animals chartered heavily`
- **Epoch 4**: `the goals due ensure capabilities where between new beliefs differ may applebees vanilla operators improvements medicaid wide harder 2 write`
- **Epoch 8**: `dedication in of about for israels a or function have fine is on other some query high they do functions`
- **Epoch 9**: `generate at appear misinformation we are this also so would us take dont an a financial newtons and creates ensure`
- **Epoch 10**: `train can a feedback architect 15th western linux 634 simulations jupiter interval expression or your bringing the am please i`

---

## 4. Head-to-Head Architectural Insights: 16D Binary vs 11D Balanced Ternary

### 1. Vocabulary Capacity & Granularity
- In the **16D Binary Hypercube (65,536 nodes)**, active dominant words stabilized at **8,530 words** out of 45,000.
- In the **11D Balanced Ternary Hypercube (177,147 nodes)**, active dominant words reached **13,873 words** (**+62.6% increase**).
- **Takeaway**: Scaling node capacity via ternary states allows more rare and nuanced content words to retain their own independent spatial attractor basins without being overwritten by dominant frequent tokens.

### 2. Transition Graph Density vs Node Specialization
- The 16D Hypercube packed **20,620,330 transitions** into 65,536 nodes (**~314 outgoing edges per node**).
- The 11D Ternary Hypercube distributed **18,210,162 transitions** across 177,147 nodes (**~102 outgoing edges per node**).
- **Takeaway**: 16D creates dense multi-way routing hubs, while 11D Ternary provides **greater contextual disambiguation**; words have specific dedicated paths depending on dialogue context.

### 3. Compute Throughput Parity
- **16D Binary**: 15,336 tokens/sec (POPCNT Hamming distance)
- **11D Ternary**: 15,492 tokens/sec (Precomputed $O(1)$ Powers-of-3 table)
- **Takeaway**: Despite having 2.7× more nodes and an 11-dimensional ternary lattice, the ternary architecture has zero performance penalty on CPU because of cache-friendly sequential array indexing.
