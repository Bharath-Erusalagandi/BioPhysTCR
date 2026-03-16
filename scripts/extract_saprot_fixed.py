"""
Extract real SaProt embeddings - FIXED VERSION
Properly extracts per-residue 446-dim embeddings from full sequences.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import EsmTokenizer, EsmForMaskedLM
from Bio.PDB import PDBParser


def load_saprot_model(model_path: str, device: str = 'cuda'):
    """Load SaProt model properly for embeddings."""
    print(f"Loading SaProt model from {model_path}...")

    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForMaskedLM.from_pretrained(model_path)

    # Remove LM head - we only want embeddings
    model.lm_head = torch.nn.Sequential()

    model = model.to(device)
    model.eval()

    print(f"✓ SaProt model loaded on {device}")
    return model, tokenizer


def extract_sequence_from_pdb(pdb_file: str) -> str:
    """Extract amino acid sequence from PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', pdb_file)

    aa_map = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == ' ':  # Standard residue
                    resname = residue.resname
                    residues.append(aa_map.get(resname, 'X'))

    return ''.join(residues)


def extract_saprot_embeddings_direct(
    model,
    tokenizer,
    sequence: str,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Extract per-residue SaProt embeddings using direct model inference.

    ESM tokenizer: each amino acid = 1 token (for protein sequences)
    Output shape: [seq_len, 446] (per-residue embeddings)
    """
    try:
        # Tokenize sequence - ESM tokenizer needs spaces between amino acids!
        # "GSHS" → treated as one token
        # "G S H S" → treated as 4 tokens
        sequence_spaced = " ".join(sequence)

        inputs = tokenizer(
            sequence_spaced,
            return_tensors='pt',
            add_special_tokens=True  # Explicit - add CLS/EOS tokens
        )

        # Debug: check tokenization
        token_ids = inputs['input_ids'].squeeze().cpu().numpy()
        # print(f"  Sequence: {sequence[:20]}... (len={len(sequence)})")
        # print(f"  Tokens: {len(token_ids)} tokens")

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get embeddings from model
        with torch.no_grad():
            # Access the ESM encoder backbone directly
            outputs = model.esm(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )

            # Get the last hidden state from the encoder
            # Shape: [batch_size=1, seq_len_with_special, hidden_dim=446]
            last_hidden = outputs.last_hidden_state

            # Remove batch dimension
            embeddings = last_hidden.squeeze(0).cpu().numpy()

            # ESM tokenizer adds special tokens:
            # Token 0: <cls> (beginning)
            # Tokens 1 to seq_len: amino acid tokens (1 per residue)
            # Last token: <eos> (end)

            # We want only the amino acid embeddings (skip cls and eos)
            # If we have seq_len amino acids, we should have seq_len+2 total tokens

            if embeddings.shape[0] == len(sequence) + 2:
                # Standard case: [CLS] + seq + [EOS]
                embeddings = embeddings[1:-1, :]
            elif embeddings.shape[0] == len(sequence):
                # No special tokens (shouldn't happen with add_special_tokens=True)
                pass
            else:
                # Shape mismatch - return None to trigger error handling
                # print(f"  Shape mismatch: got {embeddings.shape[0]} tokens, expected {len(sequence)+2}")
                return None

            return embeddings.astype(np.float32)

    except Exception as e:
        print(f"Error extracting embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_structures(
    pdb_files: dict,
    model,
    tokenizer,
    output_dir: Path,
    device: str = 'cuda'
):
    """Process all PDB structures and extract SaProt embeddings."""

    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    failed = []

    print(f"\nProcessing {len(pdb_files)} structures...")

    for pdb_id, pdb_file in tqdm(pdb_files.items(), desc="Extracting SaProt"):
        try:
            # Extract sequence
            sequence = extract_sequence_from_pdb(pdb_file)
            if not sequence:
                failed.append((pdb_id, "Empty sequence"))
                continue

            # Extract SaProt embeddings
            embeddings = extract_saprot_embeddings_direct(model, tokenizer, sequence, device)

            if embeddings is None:
                failed.append((pdb_id, "Failed to extract embeddings"))
                continue

            # Verify shape matches sequence length
            if embeddings.shape[0] != len(sequence):
                failed.append((pdb_id, f"Shape mismatch: {embeddings.shape[0]} != {len(sequence)}"))
                continue

            # Save data
            data = {
                'embeddings': embeddings,
                'sequence': sequence,
                'pdb_id': pdb_id,
                'pdb_path': pdb_file,
                'embedding_dim': embeddings.shape[1],
                'seq_len': len(sequence)
            }

            output_file = output_dir / f"{pdb_id}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)

            results[pdb_id] = data

        except Exception as e:
            failed.append((pdb_id, str(e)))

    # Save summary
    summary = {
        'n_processed': len(results),
        'n_failed': len(failed),
        'pdb_ids': list(results.keys()),
        'failed': failed
    }

    with open(output_dir / 'extraction_summary.pkl', 'wb') as f:
        pickle.dump(summary, f)

    print(f"\n✓ Processed: {len(results)}/{len(pdb_files)}")
    print(f"✗ Failed: {len(failed)}")

    if failed:
        print("\nFailed structures (first 10):")
        for pdb_id, reason in failed[:10]:
            print(f"  {pdb_id}: {reason}")

    return results, failed


def find_pdb_files(base_dirs: list) -> dict:
    """Find all PDB files."""
    pdb_files = {}

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue

        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file.endswith(('.pdb', '.cif')):
                    pdb_id = file.split('.')[0]
                    file_path = os.path.join(root, file)

                    if pdb_id not in pdb_files:
                        pdb_files[pdb_id] = file_path

    return pdb_files


def update_combined_features(saprot_dir: Path, combined_features_path: Path):
    """Update combined_features.pkl with real SaProt embeddings."""

    print(f"\nUpdating {combined_features_path}...")

    # Load combined features
    with open(combined_features_path, 'rb') as f:
        combined = pickle.load(f)

    # Load new SaProt embeddings
    saprot_data = {}
    for pkl_file in saprot_dir.glob('*.pkl'):
        if pkl_file.name == 'extraction_summary.pkl':
            continue

        pdb_id = pkl_file.stem
        with open(pkl_file, 'rb') as f:
            saprot_data[pdb_id] = pickle.load(f)

    # Update combined features
    combined['saprot'] = saprot_data

    # Update metadata
    if 'metadata' in combined:
        if saprot_data:
            first_key = list(saprot_data.keys())[0]
            actual_dim = saprot_data[first_key]['embedding_dim']
            combined['metadata']['saprot_dim'] = actual_dim
            print(f"  Detected SaProt dimension: {actual_dim}")
        combined['metadata']['n_structures'] = len(saprot_data)

    # Create backup
    backup_path = combined_features_path.parent / 'combined_features_esm2_backup.pkl'
    print(f"Creating backup at {backup_path}")
    if combined_features_path.exists():
        os.rename(combined_features_path, backup_path)

    # Save updated combined features
    with open(combined_features_path, 'wb') as f:
        pickle.dump(combined, f)

    print(f"✓ Updated with {len(saprot_data)} real SaProt embeddings")
    print(f"  Embedding dimension: {combined['metadata'].get('saprot_dim', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description='Extract real SaProt embeddings (FIXED)')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to SaProt model directory')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Base data directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Find PDB files
    search_dirs = [
        str(data_dir / 'raw' / 'TCR_complexes'),
        str(data_dir / 'raw' / 'TCR_complexes_2'),
        str(data_dir / 'raw' / 'tcr_struc'),
    ]

    print("Searching for PDB files...")
    pdb_files = find_pdb_files(search_dirs)
    print(f"Found {len(pdb_files)} unique structures")

    if not pdb_files:
        print("ERROR: No PDB files found!")
        sys.exit(1)

    # Load SaProt model
    model, tokenizer = load_saprot_model(args.model_path, args.device)

    # Extract embeddings
    output_dir = data_dir / 'processed' / 'saprot_real'
    results, failed = process_structures(pdb_files, model, tokenizer, output_dir, args.device)

    # Update combined features
    combined_features_path = data_dir / 'processed' / 'combined_features.pkl'
    if combined_features_path.exists():
        update_combined_features(output_dir, combined_features_path)

    print("\n" + "="*60)
    print("SaProt extraction complete!")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Processed: {len(results)} structures")
    print(f"Failed: {len(failed)} structures")

    if len(results) > 0:
        # Show sample
        sample_id = list(results.keys())[0]
        sample = results[sample_id]
        print(f"\nSample verification ({sample_id}):")
        print(f"  Sequence length: {sample['seq_len']}")
        print(f"  Embedding shape: {sample['embeddings'].shape}")
        print(f"  Embedding dimension: {sample['embedding_dim']}")


if __name__ == '__main__':
    main()
