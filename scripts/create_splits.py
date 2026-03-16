
import os
import json
import random
import pandas as pd
from pathlib import Path

def create_splits(seed=42):
    random.seed(seed)
    
    base_dir = Path("BioPhysTCR/data/raw")
    tcr_dir = base_dir / "TCR_complexes"
    tsv_path = Path("STCRDab Database Overview/20260101_0012741_summary.tsv")
    output_dir = Path("BioPhysTCR/data/splits")
    
    df = pd.read_csv(tsv_path, sep='\t')
    
    pdb_files = list(tcr_dir.glob("*.pdb"))
    pdb_ids = [f.name.split('.')[0] for f in pdb_files]
    
    print(f"Found {len(pdb_files)} PDB files in {tcr_dir}")
    
    
    valid_data = []
    
    for pdb_file in pdb_files:
        pdb_id = pdb_file.name.split('.')[0]
        
        
        
        if pdb_id in df['pdb'].values:
            valid_data.append({
                'pdb_id': pdb_id,
                'filename': pdb_file.name,
                'rel_path': str(pdb_file.relative_to(base_dir.parent))
            })
        else:
            print(f"Warning: {pdb_id} not found in TSV summary.")
            valid_data.append({
                'pdb_id': pdb_id,
                'filename': pdb_file.name,
                'rel_path': str(pdb_file.relative_to(base_dir.parent))
            })

    random.shuffle(valid_data)
    
    total = len(valid_data)
    n_train = int(0.9 * total)
    
    train_set = valid_data[:n_train]
    val_set = valid_data[n_train:]
    
    print(f"Total: {total}, Train: {len(train_set)}, Val: {len(val_set)}")
    
    splits = {
        'train': train_set,
        'val': val_set,
        'test': []
    }
    
    with open(output_dir / "splits_random.json", "w") as f:
        json.dump(splits, f, indent=2)
        
    print(f"Saved splits to {output_dir / 'splits_random.json'}")

if __name__ == "__main__":
    create_splits()
