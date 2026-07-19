# Task 2: Wikipedia Next-Click Prediction

## Overview

Given a navigator's current article and target article on Wikipedia, predict which link on the current page they click next.

**Metric**: Accuracy (exact match of `next_article_id`).

## Dataset

| File | Rows | Description |
|---|---|---|
| `articles.csv` | 4604 | `article_id, title` |
| `categories.csv` | 5205 | `article_id, category` (multi-valued) |
| `states_train.csv` | 9000 | `state_id, current_article_id, target_article_id, next_article_id` |
| `states_test.csv` | 6000 | `state_id, current_article_id, target_article_id` |
| `sample_submission.csv` | 6000 | `state_id, predicted_next_article_id` |
| `screenshots/<id>.png` | 4604 | Article screenshots with links drawn in blue |

### Training columns

- `current_article_id`: page currently open
- `target_article_id`: destination the navigator is heading for
- `next_article_id` (train only): the next link clicked — the answer

## State of the Art

### Directly on Wikispeedia / This Task

| # | Title | Venue/Yr | Key Idea | DOI/Link |
|---|-------|----------|----------|----------|
| 1 | **GRETEL: Extrapolating Paths with Graph Neural Networks** (Cordonnier & Loukas) | IJCAI 2019 | First GNN for path inference on Wikipedia (Wikispeedia). Conditioned on a path prefix, predicts suffix via message passing on the link graph. Tested on both GPS road networks and Wikispeedia. | `10.48550/arXiv.1903.07518` |
| 2 | **Human Wayfinding in Information Networks** (West & Leskovec) | WWW 2012 | Introduced Wikispeedia dataset. Analyzed human navigation on 4604-article Wikipedia. Found navigation is goal-directed but not shortest-path. | [cs.stanford.edu/~jure/pubs/wayfinding-www12.pdf](https://cs.stanford.edu/~jure/pubs/wayfinding-www12.pdf) |
| 3 | **WikiNet — Recurrent GNN for Wikispeedia** (Hurtado & Iswara) | Stanford CS224W 2021 | GCN + LSTM to predict target article from path prefix on Wikispeedia. Attention-based aggregation of path nodes then predict target. | [medium.com/stanford-cs224w/wikinet](https://medium.com/stanford-cs224w/wikinet-an-experiment-in-recurrent-graph-neural-networks-3f149676fbf3) |
| 4 | **Learning to Navigate Wikipedia by Taking Random Walks** (Zaheer et al.) | NeurIPS 2022 | Behavioral cloning of random walks to learn link selection policy. 96%/92% success on 5/20-step nav on full Wikipedia (38M nodes). | `10.48550/arXiv.2211.00177` |
| 5 | **Dual Hypergraph Features for Path Inference in Wikipedia Links** (Toufa et al.) | IJCNN 2023 | Extends GRETEL with dual hypergraph transformation; captures higher-order edge features on Wikispeedia graph. | (IEEE) |
| 6 | **Constructing & Analyzing Different Density Graphs for Path Extrapolation on Wikipedia** (Sotiroudi et al.) | DBKDA 2024 | WCM dataset built from Wikipedia; dual hypergraph GRETEL performs better on dense vs sparse graphs. | `10.48550/arXiv.2406.19039` |

### Wikipedia Link Prediction & Embeddings

| # | Title | Venue/Yr | Key Idea | DOI/Link |
|---|-------|----------|----------|----------|
| 7 | **Link Prediction for Wikipedia Articles as a Natural Language Inference Task** (Phan et al.) | DSAA 2023 | Sentence-pair classification (NLI) formulation for link prediction. Achieved 0.99996 Macro F1. | `10.48550/arXiv.2308.16469` |
| 8 | **Wikipedia2Vec** (Yamada et al.) | EMNLP 2020 | Skip-gram embeddings from Wikipedia link graph + anchor context. State-of-the-art on KORE entity relatedness. | `10.48550/arXiv.1812.06280` |
| 9 | **Wikipedia Reader Navigation: When Synthetic Data Is Enough** (Arora et al.) | WSDM 2022 | Markov chain models for next-article prediction from clickstream. Synthetic clickstream approximates real navigation within 10%. | `10.48550/arXiv.2201.00812` |
| 10 | **Wiki-CS: A Wikipedia-Based Benchmark for GNNs** (Mernyei & Cangea) | ICML 2020 | Wikipedia CS article graph benchmark for node classification & link prediction. | `10.48550/arXiv.2007.02901` |

### Multimodal Knowledge Graph Link Prediction

| # | Title | Venue/Yr | Key Idea | DOI/Link |
|---|-------|----------|----------|----------|
| 11 | **IMKGA-SM: Interpretable Multimodal KG Answer Prediction via Sequence Modeling** (Wen et al.) | arXiv 2023 | Uses OCR + Vgg16 for multimodal KG link prediction; models as RL Markov decision process → sequence framework. | `10.48550/arXiv.2301.02445` |
| 12 | **MM-Transformer: Transformer-based KG Link Prediction Fusing Multimodal Features** | MDPI Symmetry 2024 | Transformer fusing text + image features for KG link prediction. Addresses modality interaction. | `10.3390/sym16080961` |
| 13 | **AFME: Adaptive Fusion and Modality Information Enhancement for MMKG Link Prediction** | Neural Networks 2025 | GAN-based adaptive fusion with relation-driven denoising; multi-layer self-attention for intra/inter-modal features. | `10.1016/j.neunet.2025.107771` |

### Foundational Graph Methods

| # | Title | Venue/Yr | Key Idea | DOI/Link |
|---|-------|----------|----------|----------|
| 14 | **Wikispeedia: An Online Game for Inferring Semantic Distances** (West et al.) | IJCAI 2009 | Introduced the Wikispeedia game; semantic distance inference from human navigation. | (IJCAI) |
| 15 | **GCN (Kipf & Welling)** | ICLR 2017 | Foundational graph convolutional network. | — |
| 16 | **GAT (Veličković et al.)** | ICLR 2018 | Graph attention networks. | — |
| 17 | **Node2Vec (Grover & Leskovec)** | KDD 2016 | Biased random walk node embeddings. | `10.1145/2939672.2939754` |

**Consensus approach**: Use the Wikispeedia link graph (119,880 edges, 100% overlap). Encode nodes with sentence-transformers (384d) + CLIP visual (512d) + Wikipedia2Vec (100d) + category multi-hot (129d) = 1125d combined. Use a **GNN (GCN/GAT)** over the link graph to classify which neighbor is the next click, conditioned on current + target nodes via labeling trick. Alternatively, a **Transformer** that scores candidate links given the current+target context. For maximum accuracy, fuse visual (screenshots via CLIP) + textual (Wikipedia2Vec/titles) + categorical modalities.

## Plan

### Phase 1: Literature Review + Setup
- ✔ `dataset-task2/task2.md` — this document
- ✔ `task2/src/__init__.py`, `ocr.py`, `graph.py`, `model.py`, `train.py` — scaffolded

### Phase 2: EDA & OCR Pipeline
- ✔ `01_eda.ipynb` — data loading, statistics, exploration, all features cached
- ✔ Link graph from Wikispeedia (auto-downloads from SNAP if missing)
- ✔ Article text embeddings (sentence-transformers all-MiniLM-L6-v2, 384d)
- ✔ CLIP visual embeddings (openai/clip-vit-base-patch32, 512d from screenshots)
- ✔ Wikipedia2Vec embeddings (enwiki_20180420_100d, 100d graph-aware)
- ✔ Category feature encoding (129 cats, multi-hot)

### Phase 3: Baselines
- ✔ Title similarity (cosine between target & candidate embeddings, 16.5%)
- ✔ Category overlap heuristic (12.2%)
- ✔ Most popular link per current article (62.5% train)
- ✔ XGBoost with engineered features (~50% train acc)
- ✔ Ensemble majority vote (20.8%)
- ✔ Each saved as `submissions/<timestamp>_<model>/` (5 submissions)

### Phase 4: Deep Learning
- ✔ `04_deep_learning.ipynb` — four architectures, self-contained:
  - **MLP Scorer** (428K params) — multimodal features → 3-layer MLP → candidate score
  - **GCN+GRETEL** (296K params) — 2-layer GCN with labeling trick (conditions on current/target)
  - **GAT Scorer** (363K params) — 2-layer graph attention over link neighborhood
  - **Transformer Scorer** (857K params) — self-attention over candidates + cross-attention with target
- All models share a unified training loop and prediction function
- Features: text (384d) + CLIP (512d) + Wiki2Vec (100d) + categories (129d) = up to 1125d

### Phase 5: Experiments & Submissions
- Ablations: text-only vs visual-only vs fused
- Hyperparameter tuning
- Ensemble
- Final submission
