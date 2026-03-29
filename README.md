# BioPhysTCR: Multimodal TCR-pMHC Binding Prediction via Sequence, Structure, and Physicochemical Feature Integration

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Official repository for **BioPhysTCR**, a multimodal deep learning framework that integrates sequence information, structural features, and physicochemical properties to predict T-cell receptor (TCR) and peptide-MHC (pMHC) binding affinity.

---

## Overview

Accurate prediction of TCR-pMHC binding is vital for designing targeted immunotherapies, vaccine development, and cancer treatment. BioPhysTCR integrates three complementary modalities through a cross-attention fusion mechanism:

1. **Sequence Encoders** (ESM2, dim=1280): Captures evolutionary and contextual residue relationships using the ESM2 protein language model, processed through a transformer encoder with self-attention and positional encoding.
2. **Structural Encoders** (SaProt/GNN, dim=446): Utilizes SaProt foldseek-derived tokens and a 3-layer graph attention network (GAT) with 8 attention heads to model 3D geometric complementarity.
3. **Physicochemical Encoders** (dim=8): Computes residue-level biophysical descriptors including electrostatic potential, solvent accessibility (SASA), hydrophobicity, net charge, and hydrogen bonding potential.

These heterogeneous features are fused through a cross-attention mechanism that dynamically weighs information across modalities, enabling the model to learn the interplay between sequence, structural geometry, and biophysical properties.

**Key Contributions:**
- Cross-attention fusion mechanism for multimodal integration
- Structure-aware feature extraction from TCR-pMHC 3D complexes
- Combined contrastive (InfoNCE) and focal loss for training
- Strong zero-shot generalization to unseen epitopes

---

## Performance Summary

| Setting | AUROC | AUPR | MCC |
|---------|-------|------|-----|
| Standard Benchmark (IEDB) | **0.932** +/- 0.010 | 0.918 +/- 0.012 | 0.684 +/- 0.018 |
| Zero-Shot (98 Unseen Epitopes) | **0.827** +/- 0.015 | 0.808 +/- 0.017 | -- |
| COVID-19 TCR Repertoires | **0.887** | -- | -- |

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

1. **Modality-Specific Encoding:**
   - Sequence: ESM2 embeddings -> Transformer encoder (2 layers, 8 heads) -> dim 256
   - Structure: SaProt tokens -> 3-layer GAT (8 heads) -> dim 256
   - Physicochemical: 8 residue-level properties -> 2-layer MLP with GELU -> dim 64

2. **Cross-Attention Fusion:** Six pairwise attention operations (seq-str, seq-phys, str-phys) with dimension d_k = 256

3. **Interaction Modeling:** Pairwise TCR-epitope attention with biological masking constraints

4. **Binding Prediction:** Mean-pooled fusion features + max-pooled interface features -> sigmoid

**Training:** Combined loss with InfoNCE contrastive loss (lambda=0.5) and focal loss (alpha=0.25, gamma=2.0)

---

## Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| IEDB Benchmark | 48,752 TCR-pMHC pairs, 982 epitopes | [IEDB](https://www.iedb.org/) |
| PDB Structures | 217 TCR-pMHC complex structures | [PDB](https://www.rcsb.org/) |
| COVID-19 TCR | Single-cell TCR-seq from 72 patients | [Reference r42] |

---

## Pre-Trained Models

Pre-trained weights are not publicly released due to data privacy constraints. Researchers can train models from scratch using the provided architecture and scripts on publicly available datasets (IEDB, VDJdb).

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{biophystcr2026,
  title={Multimodal TCR-pMHC Binding Prediction Integrating Sequence,
         Structure, and Physicochemical Features with Cross-Attention Fusion},
  author={Erusalagandi, Bharath},
  journal={...},
  year={2026}
}
```

---

## Acknowledgments

We thank the creators of ESM2, SaProt, and the IEDB for making their data and models publicly available. We also acknowledge the structural biology community for maintaining the Protein Data Bank.
