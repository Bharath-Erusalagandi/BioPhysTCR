# Pre-trained Weights

This directory contains pre-trained encoder weights used for transfer learning.

## Files

- `sequence_encoder.pth` (~30MB) - Pre-trained sequence encoder weights
- `structure_encoder.pt` (~20MB) - Pre-trained structure GNN encoder weights

## Usage

These weights are loaded in `notebooks/02_transfer_learning.ipynb`:

```python
# Load sequence encoder
seq_data = torch.load('pretrained_weights/sequence_encoder.pth')
model.tcr_sequence_encoder.load_state_dict(seq_data, strict=False)

# Load structure encoder
struct_data = torch.load('pretrained_weights/structure_encoder.pt')
model.tcr_structure_encoder.load_state_dict(struct_data, strict=False)
```

## Architecture Compatibility

- **Sequence Encoder**: dim=200, 2 transformer layers
- **Structure Encoder**: dim=512, 3 GraphSAGE layers

These dimensions are configured in `GARSEFConfig` to match the pre-trained weights.

## Note

These files are excluded from git via `.gitignore` due to file size. They are loaded from this directory during training.
