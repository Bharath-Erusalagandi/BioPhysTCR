import torch
import sys
# sys.path.append('.')

def inspect_file(path, name):
    print(f"\n--- {name} Shapes ---")
    try:
        data = torch.load(path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in data: data = data['model_state_dict']
        
        # Print a few key shapes
        for k, v in list(data.items())[:20]:
            print(f"{k}: {v.shape}")
            
        # Look for embedding/linear layers to deduce hidden dim
        print("... Searching for dimensions ...")
        for k, v in data.items():
            if 'fc' in k or 'linear' in k or 'embedding' in k:
                if len(v.shape) == 2:
                    print(f"{k}: {v.shape}")
    except Exception as e:
        print(f"Error loading {name}: {e}")

inspect_file('pretrained_weights/structure_encoder.pt', 'Structure Encoder')
