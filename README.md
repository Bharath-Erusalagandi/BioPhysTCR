# BioPhysTCR: Physics-Informed Multi-Modal Deep Learning for TCR-pMHC Binding Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

Official repository for **BioPhysTCR**, a multi-modal deep learning framework that integrates sequence information, structural features, and physicochemical properties to predict T-cell receptor (TCR) and peptide-MHC (pMHC) binding affinity. 

This repository contains the code for preparing datasets, extracting multi-modal features, and training our models.

---

## Overview

Accurate prediction of TCR-pMHC binding is vital for designing targeted immunotherapies. BioPhysTCR (BioPhysTCR architecture) integrates three distinct modalities:

1. **Sequence Encoders** (dim=200): Captures sequence motifs and binding patterns pre-trained on diverse TCR/peptide datasets.
2. **Structural Encoders** (dim=512): Utilizes graph neural networks to explicitly model 3D geometric complementarity between the TCR and the pMHC.
3. **Physicochemical Encoders** (dim=64): Computes novel biophysical parameters, including APBS-based electrostatic potentials and surface area binding propensities.

Our multi-modal fusion mechanism projects these integrated features via cross-attention into robust, generalized binding predictions.

**Performance Summary:**
| Benchmark Evaluation | AUC-ROC | AUPR |
|-------------------|---------|------|
| Standard Setting (VDJdb) | **0.9500** | 0.9378 |
| Zero-Shot (Unseen Epitopes) | **0.8420** | 0.8234 |

## Repository Structure

```text
BioPhysTCR/
├── configs/               # Hyperparameter configurations and YAML files
├── data/                  # [Not Included] Raw data and pre-computed features
├── notebooks/             # Tutorials for data preparation and transfer learning
├── scripts/               # Executable scripts for training and model evaluation
└── src/
    ├── features/          # Multi-modal feature extraction logic
    ├── models/            # Core neural network architectures
    ├── training/          # Training pipelines and utilities
    └── utils/             # Helper utilities and data loaders
```

## Installation

We highly recommend using a virtual environment or `conda` to manage dependencies.

```bash
git clone https://github.com/YourOrganization/BioPhysTCR.git
cd BioPhysTCR

# Setup the environment using the provided shell script
source scripts/setup_env.sh

# Alternatively, install standard dependencies via pip:
pip install -r requirements.txt
```

## Usage

### 1. Data Processing and Feature Extraction
First, generate the necessary representations. Note that this requires structural files (e.g., PDB formats).

```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```

### 2. Model Training
Train a new model natively from scratch or employ transfer learning utilizing pre-trained sequence and structure checkpoints:

```bash
# Fine-tuning via transfer learning
jupyter notebook notebooks/02_transfer_learning.ipynb

# Command-line training execution
python scripts/02_train.py --config configs/config.yaml
```

### 3. Evaluation
Reproduce the benchmark evaluations provided in the manuscript:

```bash
python scripts/03_evaluate.py --checkpoint checkpoints/best_model.pt
```

## Pre-Trained Models

Due to data privacy and patient confidentiality constraints associated with the clinical datasets, pre-trained weights for the fully trained BioPhysTCR models are not publicly released in this repository. Researchers can train models from scratch using the provided architecture and training scripts on publicly available datasets like VDJdb.

## Citation

If you incorporate this framework or our representations into your research, please cite:

```bibtex
@misc{biophystcr2026,
  title={BioPhysTCR: Physics-Informed Multi-Modal Deep Learning for TCR-pMHC Binding Prediction},
  author={Erusalagandi, Bharath},
  year={2026},
  howpublished={\url{https://github.com/YourOrganization/BioPhysTCR}}
}
```

## Acknowledgments

We are grateful to the creators of the pre-trained sequence and structural models utilized in this project for their contributions to the open-source structural biology community.

## License

This project is distributed under the MIT License - see the [LICENSE](LICENSE) file for complete details.
