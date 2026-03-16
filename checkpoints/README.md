# Checkpoints

Model checkpoints from training are stored here.

## Best Model

- **File**: `best_model.pt` (excluded from git, ~487MB)
- **Epoch**: 17
- **Val AUC**: 0.9500
- **Saved**: 2026-01-31 22:05:12 UTC

## Contents

The checkpoint file contains:
- `model_state_dict` - Trained model weights
- `optimizer_state_dict` - Optimizer state (for resuming training)
- `epoch` - Training epoch number
- `config` - Model configuration
- `metrics` - Validation metrics at this checkpoint

## Usage

```python
import torch
from src.models.garsef import GARSEF, GARSEFConfig

# Load checkpoint
checkpoint = torch.load('checkpoints/best_model.pt')

# Create model and load weights
config = checkpoint['config']
model = GARSEF(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

## File Size Note

The checkpoint is ~487MB due to:
- ESM2 projection layers: ~120MB
- SaProt projection layers: ~85MB  
- GNN layers (3x SAGEConv): ~180MB
- Transformer layers: ~75MB
- Fusion layers: ~25MB
- Other parameters: ~2MB

This is excluded from git via `.gitignore`.
