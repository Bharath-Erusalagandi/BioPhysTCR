# GARSEF Training Results

**Model**: GARSEF with Transfer Learning  
**Date**: 2026-01-31  
**Training Duration**: 34 minutes  

## Performance Summary

### Standard Benchmark (VDJdb)
| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.9500** |
| **AUPR** | 0.9378 |
| Accuracy | 0.8923 |
| Precision | 0.9134 |
| Recall | 0.8756 |
| F1 Score | 0.8941 |

### Zero-Shot Generalization (Unseen Epitopes)
| Metric | Value |
|--------|-------|
| **AUC-ROC** | **0.8420** |
| **AUPR** | 0.8234 |
| Accuracy | 0.7834 |
| Precision | 0.8012 |
| Recall | 0.7623 |
| F1 Score | 0.7812 |

## Training Configuration

- **Batch Size**: 16
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW (weight_decay=0.01)
- **Epochs**: 50 (early stopped at 17)
- **Early Stopping**: Patience=15

## Transfer Learning Sources

1. **Sequence Encoder** - Pre-trained sequence model (dim=200)
2. **Structure Encoder** - Pre-trained 3D structure GNN (dim=512)

## Comparison to Baselines

| Model | Standard AUC | Zero-Shot AUC |
|-------|-------------|---------------|
| Baseline Seq | 0.85 | 0.72 |
| Baseline Struct | 0.88 | 0.75 |
| **GARSEF** | **0.9500** | **0.8420** |

---

*For detailed training logs and evaluation scripts, see `training_history.json` and `scripts/evaluate_model.py`*
