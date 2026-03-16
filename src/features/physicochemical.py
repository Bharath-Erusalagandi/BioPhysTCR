"""
Physicochemical feature extraction for protein structures.

Extracts 8-dimensional features per residue:
1. electrostatic - Estimated electrostatic potential
2. sasa - Solvent accessible surface area
3. sasa_ratio - Relative SASA (vs theoretical max)
4. bfactor - B-factor (temperature factor) from PDB
5. hydrophobicity - Kyte-Doolittle hydrophobicity
6. charge - Formal charge at pH 7
7. hbond_donor - H-bond donor capacity
8. hbond_acceptor - H-bond acceptor capacity
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

try:
    import freesasa
    FREESASA_AVAILABLE = True
except ImportError:
    FREESASA_AVAILABLE = False
    try:
        from Bio.PDB import SASA
        BIOPYTHON_SASA_AVAILABLE = True
    except ImportError:
        BIOPYTHON_SASA_AVAILABLE = False



HYDROPHOBICITY = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
    'X': 0.0, 'U': 0.0, 'B': -3.5, 'Z': -3.5, 'J': 4.15,
}

CHARGE = {
    'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
    'Q': 0, 'E': -1, 'G': 0, 'H': 0.1, 'I': 0,
    'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
    'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
    'X': 0, 'U': 0, 'B': -0.5, 'Z': -0.5, 'J': 0,
}

HBOND_DONOR = {
    'A': 1, 'R': 5, 'N': 2, 'D': 1, 'C': 1,
    'Q': 2, 'E': 1, 'G': 1, 'H': 2, 'I': 1,
    'L': 1, 'K': 3, 'M': 1, 'F': 1, 'P': 0,
    'S': 2, 'T': 2, 'W': 2, 'Y': 2, 'V': 1,
    'X': 1, 'U': 1, 'B': 1.5, 'Z': 1.5, 'J': 1,
}

HBOND_ACCEPTOR = {
    'A': 1, 'R': 1, 'N': 2, 'D': 3, 'C': 1,
    'Q': 2, 'E': 3, 'G': 1, 'H': 2, 'I': 1,
    'L': 1, 'K': 1, 'M': 1, 'F': 1, 'P': 1,
    'S': 2, 'T': 2, 'W': 1, 'Y': 2, 'V': 1,
    'X': 1, 'U': 1, 'B': 2.5, 'Z': 2.5, 'J': 1,
}

MAX_SASA = {
    'A': 129, 'R': 274, 'N': 195, 'D': 193, 'C': 167,
    'Q': 225, 'E': 223, 'G': 104, 'H': 224, 'I': 197,
    'L': 201, 'K': 236, 'M': 224, 'F': 240, 'P': 159,
    'S': 155, 'T': 172, 'W': 285, 'Y': 263, 'V': 174,
    'X': 200, 'U': 167, 'B': 194, 'Z': 224, 'J': 199,
}

FEATURE_NAMES = [
    'electrostatic', 'sasa', 'sasa_ratio', 'bfactor',
    'hydrophobicity', 'charge', 'hbond_donor', 'hbond_acceptor'
]



def get_residue_info(structure, chain_ids: Optional[List[str]] = None) -> List[Dict]:
    """Extract residue information from BioPython structure."""
    residues = []

    for model in structure:
        for chain in model:
            if chain_ids and chain.id not in chain_ids:
                continue

            for residue in chain:
                if residue.id[0] != ' ':
                    continue

                try:
                    aa_code = seq1(residue.resname)
                except:
                    aa_code = 'X'

                residues.append({
                    'chain_id': chain.id,
                    'res_id': residue.id[1],
                    'res_name': residue.resname,
                    'aa_code': aa_code,
                    'residue': residue,
                })

    return residues


def extract_bfactors(structure, chain_ids: Optional[List[str]] = None) -> Dict[Tuple, float]:
    """Extract B-factors for each residue using CA atom."""
    bfactors = {}

    for model in structure:
        for chain in model:
            if chain_ids and chain.id not in chain_ids:
                continue

            for residue in chain:
                if residue.id[0] != ' ':
                    continue

                if 'CA' in residue:
                    bfactor = residue['CA'].get_bfactor()
                else:
                    atoms = list(residue.get_atoms())
                    bfactor = np.mean([a.get_bfactor() for a in atoms]) if atoms else 0.0

                bfactors[(chain.id, residue.id[1])] = bfactor

    return bfactors


def calculate_sasa(pdb_path: Union[str, Path], chain_ids: Optional[List[str]] = None) -> Dict[Tuple, float]:
    """Calculate SASA per residue using FreeSASA or BioPython."""
    pdb_path = str(pdb_path)

    if FREESASA_AVAILABLE:
        return _calculate_sasa_freesasa(pdb_path, chain_ids)
    elif BIOPYTHON_SASA_AVAILABLE:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        return _calculate_sasa_biopython(structure, chain_ids)
    else:
        return {}


def _calculate_sasa_freesasa(pdb_path: str, chain_ids: Optional[List[str]] = None) -> Dict[Tuple, float]:
    """Calculate SASA using FreeSASA library."""
    try:
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure)

        sasa_values = {}

        for i in range(structure.nAtoms()):
            chain_id = structure.chainLabel(i)
            res_num = structure.residueNumber(i)

            if chain_ids and chain_id not in chain_ids:
                continue

            key = (chain_id, int(res_num))
            atom_area = result.atomArea(i)

            if key not in sasa_values:
                sasa_values[key] = 0.0
            sasa_values[key] += atom_area

        return sasa_values

    except Exception:
        return {}


def _calculate_sasa_biopython(structure, chain_ids: Optional[List[str]] = None) -> Dict[Tuple, float]:
    """Calculate SASA using BioPython's SASA module."""
    try:
        sasa_calc = SASA.ShrakeRupley()
        sasa_calc.compute(structure, level="R")

        sasa_values = {}

        for model in structure:
            for chain in model:
                if chain_ids and chain.id not in chain_ids:
                    continue

                for residue in chain:
                    if residue.id[0] != ' ':
                        continue

                    sasa = residue.sasa if hasattr(residue, 'sasa') else 0.0
                    sasa_values[(chain.id, residue.id[1])] = sasa

        return sasa_values

    except Exception:
        return {}


def estimate_electrostatic(aa_code: str, sasa: float, bfactor: float) -> float:
    """Estimate electrostatic potential without APBS."""
    charge = CHARGE.get(aa_code, 0.0)
    max_sasa = MAX_SASA.get(aa_code, 200.0)
    exposure = sasa / max_sasa if max_sasa > 0 else 0.5

    electrostatic = charge * (1 + exposure) * 2.5
    return electrostatic


def get_aa_features(aa_code: str) -> Dict[str, float]:
    """Get amino acid-based features for a residue."""
    return {
        'hydrophobicity': HYDROPHOBICITY.get(aa_code, 0.0),
        'charge': CHARGE.get(aa_code, 0.0),
        'hbond_donor': HBOND_DONOR.get(aa_code, 1.0),
        'hbond_acceptor': HBOND_ACCEPTOR.get(aa_code, 1.0),
        'max_sasa': MAX_SASA.get(aa_code, 200.0),
    }



class PhysicochemicalExtractor:
    """Extract physicochemical features from PDB structures."""

    def __init__(self, use_apbs: bool = False):
        """
        Initialize extractor.

        Args:
            use_apbs: Whether to use APBS for electrostatics (requires installation)
        """
        self.use_apbs = use_apbs
        self.parser = PDBParser(QUIET=True)

    def extract(
        self,
        pdb_path: Union[str, Path],
        chain_ids: Optional[List[str]] = None,
    ) -> Optional[Dict]:
        """
        Extract physicochemical features from a PDB structure.

        Args:
            pdb_path: Path to PDB file
            chain_ids: Chain IDs to process (None = all)

        Returns:
            Dict with features, residue_ids, feature_names, or None on error
        """
        pdb_path = Path(pdb_path)

        try:
            structure = self.parser.get_structure("protein", str(pdb_path))
        except Exception as e:
            return None

        residues = get_residue_info(structure, chain_ids)
        if not residues:
            return None

        bfactors = extract_bfactors(structure, chain_ids)
        sasa_values = calculate_sasa(pdb_path, chain_ids)

        features = []
        residue_ids = []

        for res_info in residues:
            chain_id = res_info['chain_id']
            res_id = res_info['res_id']
            aa_code = res_info['aa_code']

            key = (chain_id, res_id)

            bfactor = bfactors.get(key, 0.0)
            sasa = sasa_values.get(key, 0.0)

            aa_feats = get_aa_features(aa_code)

            max_sasa = aa_feats['max_sasa']
            sasa_ratio = min(sasa / max_sasa, 1.5) if max_sasa > 0 else 0.0

            electrostatic = estimate_electrostatic(aa_code, sasa, bfactor)

            feature_vec = [
                electrostatic,
                sasa,
                sasa_ratio,
                bfactor,
                aa_feats['hydrophobicity'],
                aa_feats['charge'],
                aa_feats['hbond_donor'],
                aa_feats['hbond_acceptor'],
            ]

            features.append(feature_vec)
            residue_ids.append((chain_id, res_id, aa_code))

        return {
            'features': np.array(features, dtype=np.float32),
            'residue_ids': residue_ids,
            'feature_names': FEATURE_NAMES,
        }

    def extract_chain(
        self,
        pdb_path: Union[str, Path],
        chain_id: str,
    ) -> Optional[Dict]:
        """Extract features for a single chain."""
        return self.extract(pdb_path, chain_ids=[chain_id])

    def extract_batch(
        self,
        pdb_paths: List[Union[str, Path]],
        chain_ids: Optional[List[str]] = None,
        show_progress: bool = True,
    ) -> Dict[str, Dict]:
        """
        Extract features for multiple structures.

        Returns:
            Dict mapping pdb_id -> feature dict
        """
        from tqdm import tqdm

        results = {}
        iterator = tqdm(pdb_paths, desc="Extracting features") if show_progress else pdb_paths

        for pdb_path in iterator:
            pdb_path = Path(pdb_path)
            pdb_id = pdb_path.stem.replace(".trunc.fit", "")

            result = self.extract(pdb_path, chain_ids)
            if result is not None:
                result['pdb_id'] = pdb_id
                results[pdb_id] = result

        return results



def normalize_features(
    features: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to zero mean and unit variance.

    Returns:
        Tuple of (normalized_features, mean, std)
    """
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std = features.std(axis=0)
        std[std == 0] = 1.0

    normalized = (features - mean) / std
    return normalized, mean, std


def load_physicochemical_features(path: Union[str, Path]) -> Dict:
    """Load physicochemical features from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_physicochemical_features(features: Dict, path: Union[str, Path]) -> None:
    """Save physicochemical features to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(features, f)



__all__ = [
    'PhysicochemicalExtractor',
    'FEATURE_NAMES',
    'HYDROPHOBICITY',
    'CHARGE',
    'HBOND_DONOR',
    'HBOND_ACCEPTOR',
    'MAX_SASA',
    'normalize_features',
    'load_physicochemical_features',
    'save_physicochemical_features',
]
