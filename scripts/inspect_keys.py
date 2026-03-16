import torch
from pathlib import Path
from src.models.garsef import GARSEF, GARSEFConfig

def print_keys(state_dict, name):
    print(f"\n--- {name} Keys ---")
    keys = list(state_dict.keys())
    # Print first 5 and some distinctive ones
    for k in keys[:5]:
        print(k)
    print("...")
    # Search for layer specific keys
    for k in keys:
        if "layer" in k or "transformer" in k or "conv" in k or "lstm" in k:
            print(k)

# 1. Instantiate Our Model
print("Instantiating GARSEF...")
config = GARSEFConfig()
model = GARSEF(config)
our_state = model.state_dict()

# 2. Load TRAP
print("\nLoading TRAP...")
try:
    trap_data = torch.load('TRAP-main/2024-01-20_19_19_14_707076.pth', map_location='cpu')
    if 'model_state_dict' in trap_data: trap_data = trap_data['model_state_dict']
    print_keys(trap_data, "TRAP")
except Exception as e:
    print(f"Failed to load TRAP: {e}")

# 3. Load Structure Encoder
print("\nLoading structure encoder...")
try:
    struct_data = torch.load('pretrained_weights/structure_encoder.pt', map_location='cpu')
    print_keys(struct_data, "Structure Encoder")
except Exception as e:
    print(f"Failed to load structure encoder: {e}")

# 4. Print Our Relevant Keys
print("\n--- Our Relevant Keys (Sequence) ---")
for k in our_state:
    if "sequence_encoder" in k:
        print(k)

print("\n--- Our Relevant Keys (Structure) ---")
for k in our_state:
    if "structure_encoder" in k:
        print(k)
