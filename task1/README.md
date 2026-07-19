# Task 1: Multimodal Traffic Speed Prediction

## Overview

Predict traffic speed (km/h) for 1,260 Beijing road segments at 3 horizons (+20, +40, +60 min), given:
- 15-step speed history (1 hour)
- Event text describing road incidents
- Road network adjacency matrix

**Metric**: MSE averaged across all samples × horizons × roads.

## Dataset

### Shapes

| File | Shape | Description |
|---|---|---|
| `train/train_speed_m1_1_11160.npy` | (11160, 1260) | Block 1 speed series |
| `train/train_speed_m2_1_5039.npy` | (5039, 1260) | Block 2 speed series |
| `train/train_text_m1_1_11160.json` | dict[11160] | Block 1 event text per timestep |
| `train/train_text_m2_1_5039.json` | dict[5039] | Block 2 event text per timestep |
| `test/test_X_hist.npy` | (540, 15, 1260) | Test history windows |
| `test/test_texts.json` | dict[540] | Test event text per sample |
| `static/matrix.npy` | (1260, 1260) int8 | Adjacency matrix (5122 edges) |
| `static/Roads1260.json` | list[1260] | Road geometry + metadata |
| `static/active_mask.npy` | (1296,) bool | Spatial grid mask |

### Key properties
- Speed range: 0–151 km/h
- Road segments: each is a list of sub-segments with `coordList`, `formway`, `length`, `linkId`, `roadName`, `roadclass`
- Train text is keyed by `m1_<timestep>`, `m2_<timestep>`
- Test text is keyed by `test_<sample_id>`
- Submission: 540 × 3 × 1260 = 2,041,200 rows

### Train → Supervised Samples

For each block, generate sliding windows:
- Input: `speed[t:t+15]` + `text[t]`
- Target: `speed[t+15+h5]`, `speed[t+15+h10]`, `speed[t+15+h15]`
- Step size: 15 (non-overlapping windows) to match test format

## State of the Art

### Directly on BjTT / This Task

| # | Title | Venue/yr | Key Idea | DOI |
|---|---|---|---|---|
| 1 | BjTT: A Large-scale Multimodal Dataset for Traffic Prediction (Zhang et al.) | IEEE TITS 2024 | Introduced the dataset + MJ-Model (Graph WaveNet + LSTM text) | `10.1109/TITS.2024.3456798` |
| 2 | CrossTrafficLLM (Du et al.) | arXiv 2026 | Text-guided adaptive GCN + LLM for joint prediction & text generation | `10.48550/arXiv.2601.06042` |
| 3 | ChatTraffic: Text-to-Traffic Generation via Diffusion Model (Zhang et al.) | arXiv 2023 | Diffusion model + GCN for text-conditioned traffic generation | `10.48550/arXiv.2311.16203` |

### Multimodal Traffic Prediction with Event Text

| # | Title | Venue/yr | Key Idea | DOI |
|---|---|---|---|---|
| 4 | Emergency Events Traffic Flow Forecasting Using Text-Prompt-Guided Multimodal LLMs (Lu et al.) | IEEE TITS 2026 | Frozen LLM as text encoder → cross-attention into ST-GNN | `10.1109/TITS.2026.3524699` |
| 5 | Event Traffic Forecasting with Sparse Multimodal Data — T³ (Han et al.) | ACM MM 2024 | Cross-modal attention between text & time-series, encoder-decoder | `10.1145/3664647.3680706` |
| 6 | Incorporating Multimodal Context Information into Traffic Speed Forecasting through Graph Deep Learning (Zhang et al.) | IJGIS 2023 | Weather + event context fused into STGCN via knowledge graph | `10.1080/13658816.2023.2234959` |
| 7 | Context-aware Knowledge Graph Framework for Traffic Speed Forecasting using GNN (Zhang et al.) | IEEE TITS 2024 | Knowledge graph embedding for meteorological + event context | `10.1109/TITS.2024.3519854` |
| 8 | CadST: Correlation-aware Alignment and Disentanglement in Multi-modal ST Traffic Prediction (Liu et al.) | PAKDD 2026 | Disentangles shared vs modality-specific features with correlation alignment | `10.1007/978-981-92-1947-6_8` |
| 9 | Causal Spatio-temporal Prediction — CSTP (Huang et al.) | NeurIPS 2026 | Causal structure learning across modalities to filter spurious correlations | (in proceedings) |
| 10 | Mobile Traffic Prediction in Consumer Applications: A Multimodal Deep Learning Approach (Jiang et al.) | IEEE TCE 2024 | CNN-GNN hybrid + SMS/text event data | `10.1109/TCE.2024.3357116` |

### Foundational ST-GNN Backbones

| # | Title | Venue/yr | Key Idea | DOI |
|---|---|---|---|---|
| 11 | Graph WaveNet for Deep Spatial-Temporal Graph Modeling (Wu et al.) | IJCAI 2019 | Gated TCN + diffusion GCN, adaptive adjacency | — |
| 12 | Diffusion Convolutional Recurrent Neural Network — DCRNN (Li et al.) | ICLR 2018 | Diffusion convolution + GRU seq2seq | — |
| 13 | Spatio-Temporal Graph Convolutional Networks — STGCN (Yu et al.) | IJCAI 2018 | Chebyshev GCN + gated CNN | — |
| 14 | MTGNN: Multivariate Time Series Forecasting with GNNs (Wu et al.) | KDD 2020 | Adaptive graph learning + mix-hop propagation | `10.1145/3394486.3403118` |

**Consensus approach**: Pre-trained text encoder (MiniLM, LLM) + ST-GNN backbone (Graph WaveNet, MTGNN) + cross-modal attention fusion, trained end-to-end with multi-horizon MSE.

## Plan

### Phase 1: EDA & Data Pipeline
- `task1.md` — this document
- `src/__init__.py` — data loaders, feature builder, submission helpers, format validator
- `01_eda.ipynb` — data loading, visualization, statistics

### Phase 2: Baseline
- Persistence (last known speed): Val MSE 44.17
- Linear regression per road: Val MSE 34.83
- XGBoost with engineered features: Val MSE 30.50

### Phase 3: TrafficGNN + Text
- `src/model.py` — TrafficGNN (per-node Conv1D → text fusion → GCN stack → MLP decoder)
- `src/train.py` — text encoding (MiniLM), dataloaders, training loop, early stopping
- `03_gnn_text.ipynb` — train + evaluate + submit

### Phase 4: Ablations & Tuning
- Speed-only vs text (use_text flag)
- Fusion mode: add vs concat
- Hyperparameter sweep: d_model × num_layers grid
- `04_experiments.ipynb` — automated sweep + best config retrain

### Phase 5: Final Submission
- Train fresh or load saved model
- Generate test predictions → `submissions/` timestamped dir
- Format validation against `sample_submission.csv`
- `05_submission.ipynb` — end-to-end submission pipeline
