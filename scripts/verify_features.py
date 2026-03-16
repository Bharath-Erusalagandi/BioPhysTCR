import os
import json
import pickle
import sys
from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parent.parent

def check_file(path, description):
    print(f"Checking {description}...")
    print(f"  Path: {path}")
    
    if not path.exists():
        print(f"  [MISSING] File not found: {path}")
        return False
    
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"  [FOUND] Size: {size_mb:.2f} MB")
    
    try:
        if path.suffix == '.json':
            with open(path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"  [VALID] JSON loaded. List length: {len(data)}")
                elif isinstance(data, dict):
                    print(f"  [VALID] JSON loaded. Keys: {len(data.keys())}")
                else:
                    print(f"  [VALID] JSON loaded. Type: {type(data)}")
        elif path.suffix == '.pkl':
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if hasattr(data, '__len__'):
                     print(f"  [VALID] Pickle loaded. Length: {len(data)}")
                else:
                     print(f"  [VALID] Pickle loaded. Type: {type(data)}")
    except Exception as e:
        print(f"  [ERROR] Failed to load file: {e}")
        return False
        
    return True

def main():
    root = get_project_root()
    data_dir = root / 'data' / 'processed'
    
    
    complex_dir = data_dir / 'complex_features'
    tcr_dir = data_dir / 'tcr_features'
    
    files_to_check = [
        (complex_dir / 'complex_interface_features.json', "Complex Interface Features"),
        (complex_dir / 'complex_bfactor_features.json', "Complex B-Factor Features"),
        (complex_dir / 'complex_apbs_features.json', "Complex APBS Energetics"),
        (complex_dir / 'complex_graphs.json', "Complex Graph Representations"),
        
        (tcr_dir / 'tcr_saprot_embeddings.pkl', "TCR SaProt Embeddings"),
        (tcr_dir / 'tcr_structure_features.json', "TCR Structure Features"),
        (tcr_dir / 'cdr3_esm2_embeddings.pkl', "TCR CDR3 ESM2 Embeddings"),
        (tcr_dir / 'cdr3_physicochemical.json', "TCR CDR3 Physicochemical Features")
    ]
    
    print(f"Verifying files in: {data_dir}\n")
    
    all_passed = True
    for path, desc in files_to_check:
        if not check_file(path, desc):
            all_passed = False
        print("-" * 60)
            
    if all_passed:
        print("\nAll files verified successfully!")
        sys.exit(0)
    else:
        print("\nSome files are missing or invalid.")
        sys.exit(1)

if __name__ == "__main__":
    main()
