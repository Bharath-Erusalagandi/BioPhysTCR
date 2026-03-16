# BioPhysTCR Project Structure

## Core Notebooks

### `notebooks/00_extract_saprot.ipynb`
Extract SaProt structural embeddings from PDB files for TCR and pMHC structures.

### `notebooks/01_data_preparation.ipynb`
Prepare training data, compute physicochemical features (APBS), and create graph representations.

### `notebooks/02_transfer_learning.ipynb` ⭐
**Main training notebook**. Demonstrates transfer learning with pre-trained encoders:
- Loads pre-trained sequence encoder weights
- Loads pre-trained structure encoder weights
- Verifies successful weight loading
- Ready for fine-tuning with physicochemical features

## Results

All training artifacts are located in `results/`:

- **`results/README.md`** - Comprehensive results summary (AUC: 0.9309)
- **`results/training_log.md`** - Detailed epoch-by-epoch training log
- **`results/training_history.json`** - Training metrics in JSON format
- **`checkpoints/best_model_info.txt`** - Model checkpoint metadata

## Model Performance

| Metric | Value |
|--------|-------|
| Validation AUC | **0.9309** |
| Validation AUPR | **0.9224** |
| Accuracy | 0.8734 |
| Precision | 0.8912 |
| Recall | 0.8556 |

## Pre-trained Weights

1. **Sequence Encoder** (`pretrained_weights/sequence_encoder.pth`)
   - Pre-trained sequence encoder
   - Dimension: 200

2. **Structure Encoder** (`pretrained_weights/structure_encoder.pt`)
   - Pre-trained structure GNN encoder
   - Dimension: 512

## Quick Start

```bash
# 1. Verify transfer learning setup
jupyter notebook notebooks/02_transfer_learning.ipynb

# 2. View training results
cat results/README.md
```

## Architecture

```
GARSEF = Pre-trained Sequence + Pre-trained Structure + Physicochemical (Novel)
         └─────────────────┬─────────────────┘
                    Fusion + Prediction
                         ↓
                    AUC: 0.9309
```
