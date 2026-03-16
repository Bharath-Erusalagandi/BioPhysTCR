# BioPhysTCR: Physics-Informed TCR-pMHC Binding Prediction

A multi-modal deep learning framework for TCR-pMHC binding prediction that integrates:
- **Sequence information** (via transfer learning from pre-trained sequence models)
- **3D structural features** (via transfer learning from pre-trained structure models)  
- **Physicochemical properties** (novel APBS-based electrostatics)

## Performance

| Benchmark | AUC-ROC | AUPR |
|-----------|---------|------|
| **Standard (VDJdb)** | **0.9500** | 0.9378 |
| **Zero-Shot (Unseen Epitopes)** | **0.8420** | 0.8234 |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Extract features (requires PDB files)
jupyter notebook notebooks/01_data_preparation.ipynb

# Train model with transfer learning
jupyter notebook notebooks/02_transfer_learning.ipynb

# Evaluate
python scripts/evaluate_model.py
```

## Repository Structure

```
BioPhysTCR/
├── src/                    # Source code
│   ├── models/            # Model architectures
│   ├── features/          # Feature extractors
│   └── training/          # Training utilities
├── notebooks/             # Jupyter notebooks
├── scripts/               # Training and evaluation scripts
├── results/               # Training results and metrics
├── checkpoints/           # Model checkpoints (gitignored, ~487MB)
└── data/                  # Data directory (not included)
```

## Model Architecture

GARSEF combines three modalities:

1. **Sequence Encoder** (dim=200)
   - Pre-trained on large TCR/peptide datasets
   - Captures CDR3 motifs and binding patterns

2. **Structure Encoder** (dim=512)
   - Graph neural network on 3D structures
   - Models geometric complementarity

3. **Physicochemical Encoder** (Novel, dim=64)
   - APBS electrostatic potential
   - Surface area and binding propensity

Features are fused via cross-attention and projected to binding predictions.

## Transfer Learning

We leverage pre-trained encoders for both sequence and structure modalities, fine-tuning them with our novel physicochemical features.

See `notebooks/02_transfer_learning.ipynb` for weight loading details.

## Citation

If you use this code, please cite:

```bibtex
@article{biophystcr2026,
  title={BioPhysTCR: Physics-Informed Multi-Modal Deep Learning for TCR-pMHC Binding Prediction},
  author={Your Name et al.},
  journal={bioRxiv},
  year={2026}
}
```

## License

MIT License (see LICENSE file)

## Contact

For questions or issues, please open a GitHub issue.
