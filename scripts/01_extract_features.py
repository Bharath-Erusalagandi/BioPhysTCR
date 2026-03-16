"""
GARSEF Feature Extraction Script (Day 2)

Extracts ESM2 and SaProt features for all TCR-pMHC complexes.
Caches results to BioPhysTCR/data/processed/

Usage:
    python scripts/01_extract_features.py --config configs/config.yaml
    python scripts/01_extract_features.py --esm2_only  # Just ESM2 embeddings
    python scripts/01_extract_features.py --saprot_only  # Just SaProt features
    python scripts/01_extract_features.py --debug --limit 10  # Test on 10 samples
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from features.esm2_extractor import ESM2Extractor, load_esm2_embeddings
from features.saprot_extractor import SaProtExtractor, save_saprot_features, load_saprot_features


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_splits(splits_path: Path) -> Dict:
    """Load train/val/test splits."""
    with open(splits_path, 'r') as f:
        return json.load(f)


def load_sequence_data(data_path: Path) -> pd.DataFrame:
    """Load sequence data from dataset."""
    kfold_path = data_path / "data_split" / "randomly" / "kfold.csv"
    if kfold_path.exists():
        df = pd.read_csv(kfold_path)
        print(f"Loaded {len(df)} samples from kfold.csv")
        return df
    
    test_path = data_path / "data_split" / "randomly" / "test.csv"
    if test_path.exists():
        df = pd.read_csv(test_path)
        print(f"Loaded {len(df)} samples from test.csv")
        return df
    
    raise FileNotFoundError(f"Could not find sequence data in {data_path}")


def extract_esm2_features(
    sequence_df: pd.DataFrame,
    output_dir: Path,
    batch_size: int = 32,
    pooling: str = "mean",
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Extract ESM2 embeddings for all unique sequences.

    Args:
        sequence_df: DataFrame with CDR3 and Epitope columns
        output_dir: Directory to save embeddings
        batch_size: Batch size for extraction
        pooling: Pooling method ('mean', 'max', 'none')
        device: Device to use

    Returns:
        Dictionary with 'cdr3' and 'epitope' embedding dictionaries
    """
    output_path = output_dir / "esm2_embeddings.pkl"

    if output_path.exists():
        print(f"Loading existing ESM2 embeddings from {output_path}")
        return load_esm2_embeddings(output_path)

    print("Extracting ESM2 embeddings...")
    extractor = ESM2Extractor(batch_size=batch_size, device=device)

    embeddings = extractor.extract_from_dataframe(
        sequence_df,
        cdr3_col="CDR3",
        epitope_col="Epitope",
        pooling=pooling,
        save_path=output_path
    )

    return embeddings


def extract_saprot_features(
    splits: Dict,
    data_dir: Path,
    output_dir: Path,
    model_path: str = "westlake-repl/SaProt_650M_AF2",
    foldseek_path: str = "foldseek",
    device: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Dict]:
    """
    Extract SaProt features for all structures in splits.

    Args:
        splits: Dictionary with train/val/test structure lists
        data_dir: Directory containing raw PDB files
        output_dir: Directory to save features
        model_path: Path to SaProt model
        foldseek_path: Path to foldseek binary
        device: Device to use
        limit: Optional limit on number of structures to process

    Returns:
        Dictionary mapping pdb_id to features
    """
    saprot_dir = output_dir / "saprot"
    saprot_dir.mkdir(parents=True, exist_ok=True)

    all_structures = []
    for split_name in ["train", "val", "test"]:
        if split_name in splits:
            all_structures.extend(splits[split_name])

    if limit:
        all_structures = all_structures[:limit]

    print(f"Processing {len(all_structures)} structures for SaProt features...")

    to_process = []
    for struct in all_structures:
        pdb_id = struct["pdb_id"]
        output_path = saprot_dir / f"{pdb_id}.pkl"
        if not output_path.exists():
            to_process.append(struct)

    print(f"  {len(all_structures) - len(to_process)} already processed")
    print(f"  {len(to_process)} remaining to process")

    if not to_process:
        print("All SaProt features already extracted!")
        all_features = {}
        for struct in all_structures:
            pdb_id = struct["pdb_id"]
            output_path = saprot_dir / f"{pdb_id}.pkl"
            if output_path.exists():
                all_features[pdb_id] = load_saprot_features(output_path)
        return all_features

    try:
        extractor = SaProtExtractor(
            model_path=model_path,
            foldseek_path=foldseek_path,
            device=device
        )
    except Exception as e:
        print(f"Warning: Could not initialize SaProt extractor: {e}")
        print("Skipping SaProt extraction. You may need to:")
        print("  1. Install foldseek: https://github.com/steineggerlab/foldseek")
        print("  2. Download SaProt model: westlake-repl/SaProt_650M_AF2")
        return {}

    all_features = {}
    for struct in tqdm(to_process, desc="Extracting SaProt features"):
        pdb_id = struct["pdb_id"]
        rel_path = struct["rel_path"]
        pdb_path = data_dir / rel_path

        if not pdb_path.exists():
            print(f"Warning: PDB file not found: {pdb_path}")
            continue

        try:
            embeddings, adjacency = extractor.extract_from_pdb(
                str(pdb_path),
                chain_id="D"
            )

            features = {
                "embeddings": embeddings,
                "adjacency": adjacency,
                "pdb_id": pdb_id,
            }

            output_path = saprot_dir / f"{pdb_id}.pkl"
            save_saprot_features(features, output_path)
            all_features[pdb_id] = features

        except Exception as e:
            print(f"Warning: Failed to process {pdb_id}: {e}")
            continue

    for struct in all_structures:
        pdb_id = struct["pdb_id"]
        if pdb_id not in all_features:
            output_path = saprot_dir / f"{pdb_id}.pkl"
            if output_path.exists():
                all_features[pdb_id] = load_saprot_features(output_path)

    return all_features


def create_combined_dataset(
    esm2_embeddings: Dict[str, np.ndarray],
    saprot_features: Dict[str, Dict],
    sequence_df: pd.DataFrame,
    output_dir: Path,
):
    """
    Create combined dataset with all features.

    Args:
        esm2_embeddings: ESM2 embeddings dictionary
        saprot_features: SaProt features dictionary
        sequence_df: DataFrame with sequence data
        output_dir: Directory to save combined dataset
    """
    output_path = output_dir / "combined_features.pkl"

    combined = {
        "esm2": esm2_embeddings,
        "saprot": saprot_features,
        "sequence_data": sequence_df.to_dict('records'),
    }

    with open(output_path, "wb") as f:
        pickle.dump(combined, f)

    print(f"Saved combined features to {output_path}")

    print("\n=== Feature Extraction Summary ===")
    if esm2_embeddings:
        n_cdr3 = len(esm2_embeddings.get("cdr3", {}))
        n_epitope = len(esm2_embeddings.get("epitope", {}))
        print(f"ESM2 embeddings: {n_cdr3} CDR3, {n_epitope} epitopes")

    if saprot_features:
        print(f"SaProt features: {len(saprot_features)} structures")

    print(f"Sequence data: {len(sequence_df)} samples")


def main():
    parser = argparse.ArgumentParser(description="Extract GARSEF features")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--esm2_only",
        action="store_true",
        help="Only extract ESM2 embeddings"
    )
    parser.add_argument(
        "--saprot_only",
        action="store_true",
        help="Only extract SaProt features"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode with verbose output"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for ESM2 extraction"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--foldseek_path",
        type=str,
        default="foldseek",
        help="Path to foldseek binary"
    )

    args = parser.parse_args()

    config_path = PROJECT_DIR / args.config
    data_dir = PROJECT_DIR / "data"
    output_dir = data_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    if config_path.exists():
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")
    else:
        config = {}
        print("No config file found, using defaults")

    splits_path = data_dir / "splits" / "splits_random.json"
    if splits_path.exists():
        splits = load_splits(splits_path)
        print(f"Loaded splits: {len(splits.get('train', []))} train, "
              f"{len(splits.get('val', []))} val, {len(splits.get('test', []))} test")
    else:
        splits = {}
        print("No splits file found")

    data_path = PROJECT_DIR.parent / "data" / "raw"
    if data_path.exists():
        sequence_df = load_sequence_data(data_path)
        if args.limit:
            sequence_df = sequence_df.head(args.limit)
            print(f"Limited to {len(sequence_df)} samples")
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        sequence_df = pd.DataFrame()

    esm2_embeddings = {}
    saprot_features = {}

    if not args.saprot_only:
        if len(sequence_df) > 0:
            esm2_embeddings = extract_esm2_features(
                sequence_df,
                output_dir,
                batch_size=args.batch_size,
                device=args.device,
            )
        else:
            print("Skipping ESM2 extraction: no sequence data")

    if not args.esm2_only:
        if splits:
            saprot_features = extract_saprot_features(
                splits,
                data_dir,
                output_dir,
                foldseek_path=args.foldseek_path,
                device=args.device,
                limit=args.limit,
            )
        else:
            print("Skipping SaProt extraction: no splits defined")

    if len(sequence_df) > 0:
        create_combined_dataset(
            esm2_embeddings,
            saprot_features,
            sequence_df,
            output_dir
        )

    print("\n=== Feature extraction complete! ===")


if __name__ == "__main__":
    main()
