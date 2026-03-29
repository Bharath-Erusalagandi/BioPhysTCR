# Training and Evaluation

This directory contains scripts for training and evaluating the BioPhysTCR model.

## Training

To reproduce our training results with transfer learning:

```bash
# Option 1: Use Jupyter notebook
jupyter notebook notebooks/02_transfer_learning.ipynb

# Option 2: Use Python script (coming soon)
python scripts/train_biophystcr.py --config configs/transfer_learning.yaml
```

## Evaluation

To evaluate a trained model:

```bash
python scripts/evaluate_model.py
```

This will:
1. Load the best checkpoint from `checkpoints/best_model.pt`
2. Evaluate on standard validation set
3. Evaluate on zero-shot test set (unseen epitopes)
4. Save results to `results/evaluation_results.json`

### Expected Results

- **Standard Benchmark AUROC**: 0.932
- **Zero-Shot AUROC**: 0.827

## Other Utilities

- `inspect_keys.py` - Inspect model state dict keys
- `inspect_shapes.py` - Debug tensor shape mismatches
- `verify_features.py` - Verify feature extraction pipeline
- `create_splits.py` - Create train/val/test splits
- `create_graph_cache.py` - Pre-compute graph structures
