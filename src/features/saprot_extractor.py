"""
SaProt Embedding Extractor for GARSEF

Extracts structure-aware protein embeddings using SaProt-650M.
Requires foldseek for structure sequence generation.
Outputs 446-dimensional embeddings per residue.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import pickle
import subprocess
import tempfile
import os

from transformers import EsmTokenizer, EsmForMaskedLM, pipeline
from scipy.spatial import distance_matrix


class FoldseekRunner:
    """Wrapper for foldseek structure sequence extraction."""

    def __init__(self, foldseek_path: str = "foldseek"):
        """
        Initialize foldseek runner.

        Args:
            foldseek_path: Path to foldseek binary (or just 'foldseek' if in PATH)
        """
        self.foldseek_path = foldseek_path
        self._verify_foldseek()

    def _verify_foldseek(self):
        """Verify foldseek is available."""
        try:
            result = subprocess.run(
                [self.foldseek_path, "--help"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Warning: foldseek may not be properly installed")
        except FileNotFoundError:
            print(f"Warning: foldseek not found at {self.foldseek_path}")
            print("Please install foldseek: https://github.com/steineggerlab/foldseek")

    def get_struc_seq(
        self,
        pdb_path: str,
        chain: Optional[str] = None,
    ) -> Dict[str, Tuple[str, str, str]]:
        """
        Get structure sequence from PDB file using foldseek.

        Args:
            pdb_path: Path to PDB file
            chain: Optional specific chain to extract

        Returns:
            Dictionary mapping chain_id to (aa_seq, struc_seq, combined_seq)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "db")

            cmd = [
                self.foldseek_path, "structurealphabet",
                pdb_path, db_path,
                "--chain-name-mode", "1"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            parsed = {}
            ss_file = db_path + "_ss.tsv"

            if os.path.exists(ss_file):
                with open(ss_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            chain_id = parts[0].split('_')[-1] if '_' in parts[0] else 'A'
                            aa_seq = parts[1]
                            struc_seq = parts[2]
                            combined = ''.join([f"{aa}{ss}" for aa, ss in zip(aa_seq, struc_seq)])
                            parsed[chain_id] = (aa_seq, struc_seq, combined)

            return parsed


class SaProtExtractor:
    """Extract SaProt embeddings for protein structures."""

    def __init__(
        self,
        model_path: str = "westlake-repl/SaProt_650M_AF2",
        foldseek_path: str = "foldseek",
        device: Optional[str] = None,
        contact_threshold: float = 8.0,
    ):
        """
        Initialize SaProt extractor.

        Args:
            model_path: HuggingFace model path or local path to SaProt
            foldseek_path: Path to foldseek binary
            device: Device to run on (auto-detect if None)
            contact_threshold: Distance threshold for residue contacts (Angstroms)
        """
        self.model_path = model_path
        self.contact_threshold = contact_threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading SaProt model: {model_path}")
        print(f"Using device: {self.device}")

        self.tokenizer = EsmTokenizer.from_pretrained(model_path)
        self.model = EsmForMaskedLM.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.foldseek = FoldseekRunner(foldseek_path)

        print("SaProt model loaded successfully")

    def _get_combined_seq_simple(self, aa_sequence: str) -> str:
        """
        Generate combined sequence without foldseek (fallback).
        Uses a default structure token 'd' for each residue.

        Args:
            aa_sequence: Amino acid sequence

        Returns:
            Combined sequence with default structure tokens
        """
        return ''.join([f"{aa}d" for aa in aa_sequence])

    @torch.no_grad()
    def extract_from_sequence(
        self,
        sequence: str,
        use_structure: bool = False,
        pdb_path: Optional[str] = None,
        chain_id: Optional[str] = None,
    ) -> np.ndarray:
        """
        Extract SaProt embeddings from sequence.

        Args:
            sequence: Amino acid sequence
            use_structure: Whether to use structural information
            pdb_path: Path to PDB file (required if use_structure=True)
            chain_id: Chain ID to extract (required if use_structure=True)

        Returns:
            Embedding array of shape (seq_len, 446)
        """
        if use_structure and pdb_path:
            parsed = self.foldseek.get_struc_seq(pdb_path, chain_id)
            if chain_id and chain_id in parsed:
                combined_seq = parsed[chain_id][2]
            elif parsed:
                combined_seq = list(parsed.values())[0][2]
            else:
                combined_seq = self._get_combined_seq_simple(sequence)
        else:
            combined_seq = self._get_combined_seq_simple(sequence)

        inputs = self.tokenizer(combined_seq, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[0, 1:-1, :].cpu().numpy()

        return embeddings

    @torch.no_grad()
    def extract_from_pdb(
        self,
        pdb_path: str,
        chain_id: str,
        interface_pdb: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract SaProt embeddings and adjacency matrix from PDB.

        Args:
            pdb_path: Path to full-length PDB file
            chain_id: Chain ID to extract
            interface_pdb: Optional interface PDB for filtering residues

        Returns:
            Tuple of (embeddings, adjacency_matrix)
        """
        parsed = self.foldseek.get_struc_seq(pdb_path, chain_id)

        if chain_id in parsed:
            aa_seq, struc_seq, combined_seq = parsed[chain_id]
        elif parsed:
            chain_id = list(parsed.keys())[0]
            aa_seq, struc_seq, combined_seq = parsed[chain_id]
        else:
            raise ValueError(f"Could not extract structure sequence from {pdb_path}")

        inputs = self.tokenizer(combined_seq, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[0, 1:-1, :].cpu().numpy()

        adjacency = self._build_adjacency_from_pdb(
            interface_pdb if interface_pdb else pdb_path,
            chain_id
        )

        if interface_pdb:
            interface_indices = self._get_interface_indices(
                pdb_path, interface_pdb, chain_id
            )
            embeddings = embeddings[interface_indices]
            adjacency = adjacency[np.ix_(interface_indices, interface_indices)]

        return embeddings, adjacency

    def _build_adjacency_from_pdb(
        self,
        pdb_path: str,
        chain_id: str
    ) -> np.ndarray:
        """Build adjacency matrix from CA coordinates."""
        try:
            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("protein", pdb_path)

            ca_coords = []
            for model in structure:
                for chain in model:
                    if chain.id == chain_id:
                        for residue in chain:
                            if 'CA' in residue:
                                ca_coords.append(residue['CA'].get_coord())

            if not ca_coords:
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            if 'CA' in residue:
                                ca_coords.append(residue['CA'].get_coord())

            ca_coords = np.array(ca_coords)
            dist_matrix = distance_matrix(ca_coords, ca_coords)
            adjacency = (dist_matrix <= self.contact_threshold).astype(int)

            return adjacency

        except ImportError:
            print("Warning: BioPython not available, returning identity adjacency")
            return np.eye(1)

    def _get_interface_indices(
        self,
        full_pdb: str,
        interface_pdb: str,
        chain_id: str
    ) -> List[int]:
        """Get indices of interface residues in full structure."""
        try:
            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)

            full_struct = parser.get_structure("full", full_pdb)
            full_res_ids = []
            for model in full_struct:
                for chain in model:
                    if chain.id == chain_id:
                        for residue in chain:
                            full_res_ids.append(residue.id[1])

            int_struct = parser.get_structure("interface", interface_pdb)
            int_res_ids = set()
            for model in int_struct:
                for chain in model:
                    if chain.id == chain_id:
                        for residue in chain:
                            int_res_ids.add(residue.id[1])

            indices = [i for i, res_id in enumerate(full_res_ids) if res_id in int_res_ids]
            return indices

        except Exception as e:
            print(f"Warning: Could not get interface indices: {e}")
            return list(range(10))

    def extract_complex_features(
        self,
        complex_pdb: str,
        rec_interface_pdb: str,
        lig_interface_pdb: str,
        rec_chain: str = "D",
        lig_chain: str = "C",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features for TCR-pMHC complex.

        Args:
            complex_pdb: Path to full complex PDB
            rec_interface_pdb: Path to receptor (TCR) interface PDB
            lig_interface_pdb: Path to ligand (pMHC) interface PDB
            rec_chain: Receptor chain ID
            lig_chain: Ligand chain ID

        Returns:
            Tuple of (node_features, adjacency_matrix)
        """
        rec_emb, rec_adj = self.extract_from_pdb(
            complex_pdb, rec_chain, rec_interface_pdb
        )
        lig_emb, lig_adj = self.extract_from_pdb(
            complex_pdb, lig_chain, lig_interface_pdb
        )

        node_features = np.concatenate([rec_emb, lig_emb], axis=0)

        n_rec = rec_emb.shape[0]
        n_lig = lig_emb.shape[0]
        n_total = n_rec + n_lig

        complex_adj = np.zeros((n_total, n_total))
        complex_adj[:n_rec, :n_rec] = rec_adj
        complex_adj[n_rec:, n_rec:] = lig_adj

        crosslink = self._build_crosslink_adjacency(
            rec_interface_pdb, lig_interface_pdb
        )
        if crosslink.shape == (n_rec, n_lig):
            complex_adj[:n_rec, n_rec:] = crosslink
            complex_adj[n_rec:, :n_rec] = crosslink.T

        return node_features, complex_adj

    def _build_crosslink_adjacency(
        self,
        pdb1: str,
        pdb2: str
    ) -> np.ndarray:
        """Build cross-link adjacency between two PDB structures."""
        try:
            from Bio.PDB import PDBParser

            parser = PDBParser(QUIET=True)

            struct1 = parser.get_structure("s1", pdb1)
            struct2 = parser.get_structure("s2", pdb2)

            coords1 = []
            for model in struct1:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords1.append(residue['CA'].get_coord())

            coords2 = []
            for model in struct2:
                for chain in model:
                    for residue in chain:
                        if 'CA' in residue:
                            coords2.append(residue['CA'].get_coord())

            if coords1 and coords2:
                coords1 = np.array(coords1)
                coords2 = np.array(coords2)
                dist = distance_matrix(coords1, coords2)
                crosslink = (dist <= self.contact_threshold).astype(int)
                return crosslink

            return np.zeros((1, 1))

        except Exception as e:
            print(f"Warning: Could not build crosslink adjacency: {e}")
            return np.zeros((1, 1))


def save_saprot_features(features: Dict, path: Path):
    """Save extracted SaProt features to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(features, f)


def load_saprot_features(path: Path) -> Dict:
    """Load pre-computed SaProt features from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract SaProt embeddings")
    parser.add_argument("--pdb_path", type=str, required=True, help="Path to PDB file")
    parser.add_argument("--chain_id", type=str, default="A", help="Chain ID")
    parser.add_argument("--output_path", type=str, required=True, help="Output pickle file")
    parser.add_argument("--model_path", type=str, default="westlake-repl/SaProt_650M_AF2")
    parser.add_argument("--foldseek_path", type=str, default="foldseek")

    args = parser.parse_args()

    extractor = SaProtExtractor(
        model_path=args.model_path,
        foldseek_path=args.foldseek_path
    )

    embeddings, adjacency = extractor.extract_from_pdb(
        args.pdb_path,
        args.chain_id
    )

    result = {
        "embeddings": embeddings,
        "adjacency": adjacency
    }

    save_saprot_features(result, Path(args.output_path))
    print(f"Saved features to {args.output_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Adjacency shape: {adjacency.shape}")
