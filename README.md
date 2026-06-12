# BioPhysTCR: Multimodal TCR-pMHC Binding Prediction via Sequence, Structure, and Physicochemical Feature Integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Official repository for **BioPhysTCR**, a multimodal deep learning framework that integrates sequence information, structural features, and physicochemical properties to predict T-cell receptor (TCR) and peptide-MHC (pMHC) binding affinity.

---

## Overview

Accurate prediction of TCR-pMHC binding is vital for designing targeted immunotherapies, vaccine development, and cancer treatment. BioPhysTCR integrates three complementary modalities through a cross-attention fusion mechanism:

1. **Sequence Encoders** (ESM2, dim=1280): Captures evolutionary and contextual residue relationships using the ESM2 protein language model, processed through a 2-layer transformer encoder with 8 attention heads, projected to dim=256.
2. **Structural Encoders** (SaProt/GraphSAGE, dim=446): Utilizes SaProt foldseek-derived tokens processed by a 3-layer GraphSAGE network on a k-NN graph (k=10) built from Cα distances, followed by a bidirectional LSTM and a self-attention layer (8 heads), yielding a dim=256 embedding via global max-pooling.
3. **Physicochemical Encoders** (dim=8): Computes 8 residue-level biophysical descriptors — electrostatic potential, SASA, normalized SASA ratio, B-factor, hydrophobicity, net charge, and hydrogen bond donor/acceptor counts — processed through a 2-layer MLP with ReLU activation (dim=64).

These heterogeneous features are fused through a cross-attention mechanism: TCR and pMHC representations are each fused per-molecule via MLP (Stage 2), then cross-attention between the TCR and pMHC embeddings (d=256, 8 heads) produces context-aware representations that capture inter-molecule binding geometry.

**Key Contributions:**
- Cross-attention fusion mechanism for dynamic inter-molecule modality weighting
- Structure-aware feature extraction from TCR-pMHC 3D complexes
- Combined contrastive (InfoNCE, λ₁=0.5) and focal loss (λ₂=1.0) for training
- Strong zero-shot generalization to unseen epitopes (+6.6 pp over best baseline)

---

## Performance Summary

| Setting | AUROC | AUPR | MCC |
|---------|-------|------|-----|
| Standard Benchmark (IEDB) | **0.932** ± 0.010 | 0.918 ± 0.012 | 0.684 ± 0.018 |
| Zero-Shot (98 Unseen Epitopes) | **0.827** ± 0.015 | 0.808 ± 0.017 | — |
| COVID-19 TCR Repertoires (72 patients) | **0.887** | — | — |

**Ablation Study:**

| Configuration | AUROC | AUPR | MCC |
|---------------|-------|------|-----|
| Sequence Only | 0.861 | 0.841 | 0.551 |
| Sequence + Physicochemical | 0.872 | 0.853 | 0.569 |
| Sequence + Structure | 0.901 | 0.882 | 0.627 |
| Sequence + Structure + Phys (concat) | 0.917 | 0.898 | 0.651 |
| **Full (Seq+Str+Phys+CrossAttn)** | **0.932** | **0.918** | **0.684** |

---

## Repository Structure

```
BioPhysTCR/
├── configs/
│   └── config.yaml          # Model and training hyperparameters
├── data/                    # [Not Included] Raw and processed data
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_transfer_learning.ipynb
├── scripts/
│   ├── 01_extract_features.py
│   ├── 02_train.py
│   ├── 03_evaluate.py
│   └── README.md
├── src/
│   ├── features/            # ESM2, SaProt, physicochemical extraction
│   ├── models/              # BioPhysTCR architecture, fusion, encoders
│   ├── training/            # Trainer, losses (Focal + InfoNCE), metrics
│   └── utils/               # Dataset, data loading, graph batching
├── checkpoints/             # [Not Included] Model checkpoints
└── pretrained_weights/      # [Not Included] Pre-trained encoder weights
```

---

## Installation

We recommend using a virtual environment or conda to manage dependencies.

```bash
git clone https://github.com/Bharath-Erusalagandi/BioPhysTCR.git
cd BioPhysTCR

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:** `torch>=2.0.0`, `torch-geometric>=2.3.0`, `transformers>=4.30.0`, `fair-esm>=2.0.0`, `biopython>=1.81`, `scikit-learn>=1.3.0`

---

## Usage

### 1. Feature Extraction

Generate ESM2 embeddings, SaProt structural encodings, and physicochemical descriptors:

```bash
python scripts/01_extract_features.py
```

### 2. Training

```bash
# Standard training
python scripts/02_train.py --config configs/config.yaml

# Zero-shot evaluation (unseen epitopes)
python scripts/02_train.py --config configs/config.yaml --scenario 2

# Debug mode (quick verification)
python scripts/02_train.py --config configs/config.yaml --debug --subset 100 --epochs 5
```

### 3. Evaluation

```bash
python scripts/03_evaluate.py --checkpoint checkpoints/best_model.pt

# Zero-shot and per-epitope analysis
python scripts/03_evaluate.py --checkpoint checkpoints/best_model.pt --scenario 2 --per_epitope --tsne
```

### 4. Notebooks

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
jupyter notebook notebooks/02_transfer_learning.ipynb
```

---

## Model Architecture

The BioPhysTCR architecture consists of four processing stages:

1. **Stage 1 — Modality-Specific Encoding:**
   - Sequence: ESM2 (d=1280) → 2-layer transformer encoder (8 heads) → d=256
   - Structure: SaProt tokens (d=446) → 3-layer GraphSAGE (k-NN graph, k=10) → BiLSTM → self-attention (8 heads) → global max-pool → d=256
   - Physicochemical: 8 residue-level descriptors → 2-layer MLP (ReLU) → d=64

2. **Stage 2 — Cross-Attention Fusion:** TCR and pMHC modalities are each fused per-molecule via concatenation and a 2-layer MLP. Cross-attention between TCR and pMHC representations (d=256, 8 heads) produces context-aware embeddings.

3. **Stage 3 — Interaction Modeling:** Cross-attended TCR and pMHC embeddings are concatenated (d=512) and passed through a 2-layer MLP. A separate contrastive projection head maps each embedding to d=128 for InfoNCE loss.

4. **Stage 4 — Binding Prediction:** Final binding probability: p_bind = σ(MLP([h_TCR' ‖ h_pMHC']))

**Training:** Combined InfoNCE contrastive loss (τ=0.07, λ₁=0.5) and focal loss (α=0.25, γ=2.0, λ₂=1.0). AdamW optimizer (lr=1e-4, weight decay=0.01), cosine LR scheduling with warmup, gradient clipping (norm 1.0), early stopping on validation AUROC.

---

## Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| IEDB Benchmark | 48,752 TCR-pMHC pairs, 982 epitopes | [IEDB](https://www.iedb.org/) |
| PDB Structures | 217 TCR-pMHC complex structures | [PDB](https://www.rcsb.org/) |
| COVID-19 TCR | Single-cell TCR-seq from 72 patients (ImmuneCODE) | Wang et al., Genomics 2021 |

---

## Pre-Trained Models

Pre-trained weights are not publicly released due to data privacy constraints. Researchers can train models from scratch using the provided architecture and scripts on publicly available datasets (IEDB, VDJdb).

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{erusalagandi2026biophystcr,
  title={Multimodal TCR-pMHC Binding Prediction Integrating Sequence,
         Structure, and Physicochemical Features with Cross-Attention Fusion},
  author={Erusalagandi, Bharath},
  booktitle={Proceedings of the IEEE International Conference on Bioinformatics
             and Biomedicine (BIBM)},
  year={2026}
}
```

---

## Acknowledgments

We thank the creators of ESM2, SaProt, and the IEDB for making their data and models publicly available. We also acknowledge the structural biology community for maintaining the Protein Data Bank.
