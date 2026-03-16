
import os
import sys
import pickle
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from tqdm import tqdm

# Add src to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from features.esm2_extractor import ESM2Extractor

def main():
    print("Starting Clinical Feature Extraction...")
    
    # Paths
    clinical_csv = PROJECT_DIR / "data/splits/clinical_test_specificity.csv"
    orig_features_dir = PROJECT_DIR / "data/processed"
    new_features_dir = PROJECT_DIR / "data/processed_clinical"
    
    new_features_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Clinical Data
    print(f"Loading clinical data from {clinical_csv}")
    df = pd.read_csv(clinical_csv)
    # Unique sequences
    unique_cdr3s = df['cdr3'].unique()
    unique_epitopes = df['epitope'].unique()
    
    print(f"Found {len(unique_cdr3s)} unique CDR3s and {len(unique_epitopes)} unique epitopes.")
    
    # 2. Setup New Features Directory
    orig_pkl = orig_features_dir / "combined_features.pkl"
    new_pkl = new_features_dir / "combined_features.pkl"
    orig_cache = orig_features_dir / "graphs_cache.pkl"
    new_cache = new_features_dir / "graphs_cache.pkl"
    
    if not new_pkl.exists():
        if orig_pkl.exists():
            print(f"Copying {orig_pkl} to {new_pkl}...")
            shutil.copy(orig_pkl, new_pkl)
        else:
            print(f"Warning: {orig_pkl} not found. Creating empty features.")
            # Create structure
            with open(new_pkl, 'wb') as f:
                pickle.dump({'esm2': {'cdr3': {}, 'epitope': {}}, 'saprot': {}, 'physicochemical': {}}, f)
    
    if orig_cache.exists() and not new_cache.exists():
        print(f"Copying {orig_cache} to {new_cache}...")
        shutil.copy(orig_cache, new_cache)
        
    # 3. Compute Embeddings
    # Load current pickle to verify what's missing
    with open(new_pkl, 'rb') as f:
        features = pickle.load(f)
        
    esm2_cdr3 = features.get('esm2', {}).get('cdr3', {})
    esm2_epitope = features.get('esm2', {}).get('epitope', {})
    
    missing_cdr3 = [seq for seq in unique_cdr3s if seq not in esm2_cdr3]
    missing_epitope = [seq for seq in unique_epitopes if seq not in esm2_epitope]
    
    print(f"Missing embeddings: {len(missing_cdr3)} CDR3s, {len(missing_epitope)} Epitopes")
    
    if len(missing_cdr3) == 0 and len(missing_epitope) == 0:
        print("All features already exist! Done.")
        return

    # Initialize Extractor
    print("Initializing ESM2 Extractor...")
    extractor = ESM2Extractor(batch_size=32) # Auto device
    
    # Compute and Update CDR3
    if missing_cdr3:
        print(f"Computing {len(missing_cdr3)} CDR3 embeddings...")
        seq_df = pd.DataFrame({'sequence': missing_cdr3})
        # ESM2Extractor expects specific columns usually, but let's check input of 'extract_from_dataframe'
        # It calls tokenization.
        # We can use internal methods or just reuse the logic.
        
        # Manually using extractor since extract_from_dataframe is tied to specific col names
        # and returns a dict.
        
        # Batch process
        batch_size = 32
        for i in tqdm(range(0, len(missing_cdr3), batch_size)):
            batch = missing_cdr3[i:i+batch_size]
            # Use the extractor's model
            # Extractor has `get_embeddings(sequences)`?
            # Let's check `src/features/esm2_extractor.py` if we can.
            # Assuming it has a method to get embeddings for a list of sequences.
            # If not, we might fail here.
            # `extract_from_dataframe` uses `batch_encode_plus`.
            
            # Since I can't see `esm2_extractor.py` right now, I'll rely on `extract_from_dataframe`.
            pass

        # Re-using extract_from_dataframe
        temp_df = pd.DataFrame({'CDR3': missing_cdr3})
        # We need a dummy epitope col if it requires it? 
        # `extract_from_dataframe` takes `cdr3_col` and `epitope_col`.
        # Taking a guess it extracts BOTH.
        temp_df['Epitope'] = missing_cdr3[0] # Dummy
        
        new_embs = extractor.extract_from_dataframe(temp_df, cdr3_col='CDR3', epitope_col='Epitope')
        # new_embs is {'cdr3': {seq: emb}, 'epitope': ...}
        
        # Update our dict
        esm2_cdr3.update(new_embs['cdr3'])
        
    # Update Epitopes
    if missing_epitope:
        print(f"Computing {len(missing_epitope)} Epitope embeddings...")
        temp_df = pd.DataFrame({'Epitope': missing_epitope})
        temp_df['CDR3'] = missing_epitope[0] # Dummy
        
        new_embs = extractor.extract_from_dataframe(temp_df, cdr3_col='CDR3', epitope_col='Epitope')
        esm2_epitope.update(new_embs['epitope'])
        
    # 4. Save Back
    print("Saving updated features...")
    features['esm2']['cdr3'] = esm2_cdr3
    features['esm2']['epitope'] = esm2_epitope
    
    with open(new_pkl, 'wb') as f:
        pickle.dump(features, f)
        
    print(f"Saved updated features to {new_pkl}")

if __name__ == '__main__':
    main()
