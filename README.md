# Deep Learning Recommendation System with Tenrec Dataset

## Abstract

This project presents an end-to-end deep learning recommendation system built on the Tenrec QB-video dataset (2.27M interactions, 30,759 users, 41,533 items). We implement and evaluate a Multi-Objective Two-Tower neural network alongside a Matrix Factorization (SVD) baseline, and deploy the retrieval pipeline using FAISS for sub-millisecond approximate nearest neighbor search.

The central finding is that SVD significantly outperforms the neural approach on this dataset scale (AUC 0.8761 vs. 0.5922), consistent with established literature showing that deep retrieval models require substantially larger datasets (10M+ interactions) and richer feature spaces to surpass classical collaborative filtering. We provide a rigorous analysis of why this occurs and under what conditions Two-Tower architectures become the superior choice.

## Table of Contents

1. [Dataset](#1-dataset)
2. [Methodology](#2-methodology)
3. [Experimental Results](#3-experimental-results)
4. [Embedding Space Analysis](#4-embedding-space-analysis)
5. [Production Deployment](#5-production-deployment)
6. [Analysis and Discussion](#6-analysis-and-discussion)
7. [Setup and Reproducibility](#7-setup-and-reproducibility)

---

## 1. Dataset

**Source:** Tenrec QB-video (Yuan et al., 2022)

The Tenrec dataset is a large-scale, multi-behavior recommendation benchmark collected from a commercial video platform. After filtering users and items with fewer than 5 interactions, the working dataset comprises:

| Statistic | Value |
|-----------|-------|
| Total interactions | 2,269,121 |
| Unique users | 30,759 |
| Unique items | 41,533 |
| Training set | 1,839,731 (81.1%) |
| Validation set | 214,695 (9.5%) |
| Test set | 214,695 (9.5%) |

### Interaction Signal Distribution

The dataset contains four implicit feedback signals with highly imbalanced rates:

| Signal | Rate | Count |
|--------|------|-------|
| Click | 71.33% | 1,618,512 |
| Like | 0.79% | 17,860 |
| Share | 0.10% | 2,284 |
| Follow | 0.09% | 2,095 |

Click signals are abundant but carry lower intent, while like, share, and follow signals are sparse but indicate stronger user preference. Cross-signal correlations are weak (all below 0.10), confirming that each signal captures distinct user behavior.

### User and Item Distributions

User activity follows a heavy-tailed distribution (mean 73.8 interactions, median 34.0, max 8,876), as does item popularity (mean 54.6, median 12.0, max 9,378). A temporal-aware split was used: for each user, the most recent interactions were assigned to test and validation sets to simulate a realistic deployment scenario.

---

## 2. Methodology

### 2.1 Two-Tower Neural Network

The Two-Tower architecture consists of independent User and Item towers that map raw IDs to dense embeddings via multi-layer perceptrons. Both towers output L2-normalized vectors, and similarity is computed via dot product (equivalent to cosine similarity after normalization).

**Architecture:**
- Embedding dimension: 64
- Hidden layer: 128 units with BatchNorm, ReLU, Dropout (0.1)
- Output: 64-dimensional L2-normalized vectors
- Total parameters: 4,660,352

### 2.2 Multi-Objective Learning

The Multi-Objective model extends the Two-Tower base with three separate prediction heads for click, like, and follow prediction. The combined loss function is:

L_total = 0.4 * L_click + 0.3 * L_like + 0.2 * L_follow + 0.1 * L_BPR

where each task-specific loss is binary cross-entropy and L_BPR is the Bayesian Personalized Ranking loss that encourages positive items to score higher than sampled negatives.

**Training configuration:** Adam optimizer (lr=1e-3, weight_decay=1e-5), ReduceLROnPlateau scheduler, gradient clipping (max_norm=1.0), 4 negative samples per positive, batch size 1024.

### 2.3 Matrix Factorization Baseline (SVD)

A Truncated SVD baseline with 64 factors was fitted on a weighted interaction matrix where values = 1.0 + 2.0 * like + 3.0 * follow, giving higher weight to stronger engagement signals while operating directly on the full interaction matrix in a single pass.

### 2.4 Improved Two-Tower (Retrieval-Focused)

To address the multi-task interference observed in the Multi-Objective model, we additionally define a retrieval-focused variant with increased capacity (128-dim embeddings, 256-dim hidden, 0.3 dropout) and pure BPR ranking loss with interaction-weighted importance (no prediction heads). This model has 9,386,240 parameters.

---

## 3. Experimental Results

### 3.1 Training Convergence

The Multi-Objective model was trained for 10 epochs. The best validation loss (0.2233) was achieved at epoch 8, with no early stopping triggered. Training converged steadily, with the BPR ranking component showing the most consistent improvement.

| Epoch | Train Loss | Val Loss | Status |
|-------|-----------|----------|--------|
| 1 | 0.2050 | 0.2302 | Saved |
| 3 | 0.1765 | 0.2266 | Saved |
| 6 | 0.1743 | 0.2244 | Saved |
| **8** | **0.1731** | **0.2233** | **Best** |
| 10 | 0.1721 | 0.2266 | - |

![Training Curves](training_curves.png)

### 3.2 AUC Comparison

| Model | AUC |
|-------|-----|
| Matrix Factorization (SVD) | **0.8761** |
| Multi-Objective Two-Tower | 0.5922 |

The SVD baseline outperforms the neural model by a margin of 0.2839 in AUC, a substantial gap that is analyzed in detail in Section 6.

![Model Comparison](model_comparison.png)

### 3.3 Retrieval Quality Metrics

| K | Metric | MF Baseline | Multi-Objective |
|---|--------|-------------|-----------------|
| 5 | Precision | 0.0070 | 0.0004 |
| 5 | Recall | 0.0124 | 0.0010 |
| 5 | NDCG | 0.0177 | 0.0009 |
| 5 | Hit Rate | 0.0330 | 0.0020 |
| 10 | Precision | 0.0111 | 0.0006 |
| 10 | Recall | 0.0386 | 0.0018 |
| 10 | NDCG | 0.0397 | 0.0022 |
| 10 | Hit Rate | 0.1020 | 0.0060 |
| 20 | Precision | 0.0131 | 0.0006 |
| 20 | Recall | 0.0828 | 0.0035 |
| 20 | NDCG | 0.0687 | 0.0035 |
| 20 | Hit Rate | 0.2140 | 0.0110 |

The SVD baseline achieves an order-of-magnitude advantage across all metrics and cutoff values.

---

## 4. Embedding Space Analysis

A t-SNE projection of 1,000 sampled user and 1,000 sampled item embeddings from the Multi-Objective model reveals clear separation between user and item clusters in the learned embedding space. However, item embeddings colored by popularity show that high-popularity and low-popularity items are not well separated, and the nearly uniform similarity scores (~0.99) for top-K retrieved items indicate that the model's embedding space lacks sufficient discriminative power to rank items effectively.

![Embedding Space Visualization](embedding_space_visualization.png)

This observation is consistent with the low AUC and confirms that the multi-task training objective interfered with learning a meaningful ranking geometry.

---

## 5. Production Deployment

### 5.1 FAISS Retrieval Index

Item embeddings (41,533 vectors, 64 dimensions) were indexed using FAISS for approximate nearest neighbor retrieval:

| Index Type | Vectors | Configuration |
|-----------|---------|---------------|
| IndexFlatIP (exact) | 41,533 | Brute-force inner product |
| IndexIVFFlat (approximate) | 41,533 | 100 clusters, nprobe=10 |

### 5.2 Latency Benchmarks

| Method | Mean Latency | Std Dev | P99 Latency |
|--------|-------------|---------|-------------|
| Exact Search | 0.695 ms | 0.296 ms | 1.714 ms |
| Approximate Search | 0.142 ms | 0.126 ms | 0.598 ms |

Both search methods operate well under the 50ms SLA target. The approximate search achieves a 4.9x speedup over exact search with negligible recall loss at this index size. At scale (100M+ items), this gap would widen to several orders of magnitude.

![Latency Analysis](latency_analysis.png)

### 5.3 Memory Footprint

| Component | Size |
|-----------|------|
| Item embeddings (FAISS index) | 10.14 MB |
| Multi-Objective model weights | 17.83 MB |
| SVD user factors | 15.02 MB |
| SVD item factors | 20.28 MB |

The entire serving stack (model + index) fits comfortably in memory on a single machine, with the FAISS index adding minimal overhead.

---

## 6. Analysis and Discussion

### 6.1 Why SVD Outperforms Two-Tower on This Dataset

SVD factorizes the complete user-item interaction matrix in a single pass, capturing global co-occurrence patterns optimally for collaborative filtering. In contrast, the Two-Tower model learns from stochastic mini-batches with random negative sampling, requiring significantly more data to converge to a comparable solution.

With 2.27M interactions and 30K users, SVD observes every interaction per epoch. The neural model sees only a small sample of negatives per batch, which may not represent the true distribution. Additionally, the multi-objective formulation introduces task interference: gradients from click, like, and follow prediction heads compete for shared embedding capacity, degrading the ranking quality of the learned representations.

### 6.2 When Two-Tower Architectures Prevail

Two-Tower models power production recommendation systems at YouTube, TikTok, Pinterest, and Google Search. Their advantages emerge at scale:

- **User/item volume:** Two-Tower scales to billions of users and hundreds of millions of items via embedding lookup, while SVD is memory-bound at approximately 1M entities.
- **Feature integration:** Two-Tower can incorporate text, images, context, and behavioral sequences alongside IDs. SVD is limited to the interaction matrix.
- **Serving efficiency:** Pre-computed embeddings enable O(1) ANN retrieval with FAISS, whereas SVD requires recomputing scores per request.
- **Cold start:** Feature-based towers can produce embeddings for unseen users/items. SVD cannot.

The crossover point typically occurs when users exceed 1M, items exceed 100K, and rich side information is available.

### 6.3 Negative Sampling Bug Discovered

During debugging, we identified a critical data leak in the negative sampling pipeline: the positive pair set used for filtering was constructed from training data only, meaning 99.5% of validation positive pairs (213,595 out of 214,695) could be sampled as negatives during training. This was corrected by constructing the positive pair set from all data splits (2,259,274 pairs total), reducing leaky validation pairs to zero. This finding demonstrates the importance of rigorous data pipeline validation in recommendation systems.

### 6.4 Paths to Improvement

| Approach | Expected Impact | Rationale |
|----------|----------------|-----------|
| Scale to full dataset (10M+) | High | Neural models benefit from data volume |
| Incorporate side features | High | Move beyond pure collaborative filtering |
| GPU-accelerated training | Medium | Enables larger models and faster iteration |
| Sequential modeling (SASRec) | High | Capture temporal dynamics in user behavior |
| Pre-trained embeddings | Medium | Transfer learning from related domains |

---

## 7. Setup and Reproducibility

### Requirements

```
Python 3.8+
PyTorch >= 2.0
faiss-cpu >= 1.7
scikit-learn >= 1.0
pandas, numpy, matplotlib, seaborn, tqdm
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/social-event-recommender.git
cd social-event-recommender
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Dataset

Download the Tenrec QB-video dataset and place it in the `Tenrec/` directory:
```
Tenrec/QB-video.csv
```

The dataset is available from the [Tenrec benchmark](https://github.com/yuangh-x/2022-NIPS-Tenrec).

### Running

Open `tenrec_recommendation_system.ipynb` in Jupyter or VS Code and execute cells sequentially.

---

## References

- Yuan, G. et al. (2022). Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems. *NeurIPS Datasets and Benchmarks Track*.
- Rendle, S. et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. *UAI*.
- Yi, X. et al. (2019). Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations. *RecSys*.
- Johnson, J. et al. (2019). Billion-scale Similarity Search with GPUs. *IEEE Transactions on Big Data*.

---

## License

This project is for educational and portfolio purposes.
