"""
Dataset and DataLoader utilities for BioPhysTCR.
Loads pre-extracted features and uses pre-processed graph cache for speed.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data, Batch


class GARSEFDataset(Dataset):
    """
    Dataset for BioPhysTCR TCR-pMHC binding prediction.
    Loads pre-computed features from biophystcr_features_v2.
    """

    def __init__(
        self,
        pairs_file: Union[str, Path],
        features_dir: Union[str, Path],
        transform: Optional[callable] = None
    ):
        self.features_dir = Path(features_dir)
        self.transform = transform

        self.pairs = self._load_pairs(pairs_file)
        self._load_features()

    def _load_pairs(self, pairs_file: Union[str, Path]) -> List[Dict]:
        pairs_file = Path(pairs_file)
        if pairs_file.suffix == '.json':
            with open(pairs_file, 'r') as f:
                return json.load(f)
        elif pairs_file.suffix == '.csv':
            import pandas as pd
            df = pd.read_csv(pairs_file)
            return df.to_dict('records')
        elif pairs_file.suffix == '.pkl':
            with open(pairs_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {pairs_file.suffix}")

    def _load_features(self):
        combined_path = self.features_dir / 'combined_features.pkl'
        if combined_path.exists():
            with open(combined_path, 'rb') as f:
                combined = pickle.load(f)
            self.esm2_cdr3 = combined.get('esm2', {}).get('cdr3', {})
            self.esm2_epitope = combined.get('esm2', {}).get('epitope', {})
            self.saprot = combined.get('saprot', {})
            # Fix keys (make pdb_id lowercase)
            self.saprot = {k.lower() if isinstance(k, str) else k: v for k, v in self.saprot.items()}
            
            self.physicochemical = combined.get('physicochemical', {}).get('by_structure', {})
            # Fix keys
            self.physicochemical = {k.lower() if isinstance(k, str) else k: v for k, v in self.physicochemical.items()}
            
            self.sequence_data = combined.get('sequence_data', {})
            self.metadata = combined.get('metadata', {})
        else:
            print(f"Warning: {combined_path} not found. Loading separate files.")
            self._load_features_separate()

        # Load pre-processed graph cache for faster data loading
        cache_path = self.features_dir / 'graphs_cache.pkl'
        if cache_path.exists():
            print(f"Loading graph cache from {cache_path}...")
            with open(cache_path, 'rb') as f:
                self.graphs_cache = pickle.load(f)
            # Fix keys
            self.graphs_cache = {k.lower() if isinstance(k, str) else k: v for k, v in self.graphs_cache.items()}
        else:
            print("Warning: graphs_cache.pkl not found! Loading from JSON (slow)")
            graphs_path = self.features_dir / 'complex_features' / 'complex_graphs.json'
            if graphs_path.exists():
                with open(graphs_path, 'r') as f:
                    self.graphs = json.load(f)
                # Fix keys
                self.graphs = {k.lower() if isinstance(k, str) else k: v for k, v in self.graphs.items()}
            else:
                self.graphs = {}

    def _load_features_separate(self):
        self.esm2_cdr3 = {}
        self.esm2_epitope = {}
        self.saprot = {}
        self.physicochemical = {}
        self.sequence_data = {}
        self.metadata = {}

        # Implementation of separate loading skipped for brevity as we use combined file
        pass

    def _get_esm2_embedding(self, sequence: str, seq_type: str) -> np.ndarray:
        if seq_type == 'cdr3':
            emb = self.esm2_cdr3.get(sequence)
        else:
            emb = self.esm2_epitope.get(sequence)

        if emb is None:
            return np.zeros((1280,), dtype=np.float32)
            
        if hasattr(emb, 'numpy'):  # Handle tensor
            emb = emb.numpy()
            
        if emb.ndim > 1:
            emb = emb.mean(axis=0)
        return emb.astype(np.float32)

    def _get_graph_data(self, pdb_id: str) -> Tuple[np.ndarray, np.ndarray]:
        # Normalize pdb_id
        pdb_id = pdb_id.lower()
        
        # Use pre-processed graph cache if available
        if hasattr(self, 'graphs_cache') and pdb_id in self.graphs_cache:
            cached = self.graphs_cache[pdb_id]
            
            # Get cached features
            node_features = cached['node_features']  # Has 5 dims: bfactor, resi, x, y, z
            edge_index = cached['edge_index']
            
            # We need to construct the full 1280-dim node features
            # The model expects structure_x to be [num_nodes, 1280]
            # We use SaProt embeddings as the base, and use cached features if needed
            
            saprot_emb = self._get_saprot_embedding(pdb_id)
            
            # Ensure dimensions match
            n_nodes = node_features.shape[0]
            if saprot_emb.shape[0] >= n_nodes:
                 final_node_features = saprot_emb[:n_nodes]
            else:
                # Pad with zeros if SaProt missing residues
                padding = np.zeros((n_nodes - saprot_emb.shape[0], saprot_emb.shape[1]), dtype=np.float32)
                final_node_features = np.vstack([saprot_emb, padding])
                
            return final_node_features, edge_index

        # Fall back to original processing (slow)
        graph = self.graphs.get(pdb_id, {})
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])

        if not nodes:
            return np.zeros((10, 1280), dtype=np.float32), np.array([[0, 1], [1, 0]], dtype=np.int64)

        saprot_emb = self._get_saprot_embedding(pdb_id)
        if saprot_emb.shape[0] >= len(nodes):
            node_features = saprot_emb[:len(nodes)]
        else:
            node_features = np.vstack([
                saprot_emb,
                np.zeros((len(nodes) - saprot_emb.shape[0], saprot_emb.shape[1]), dtype=np.float32)
            ])

        if edges:
            edge_list = []
            for edge in edges:
                if isinstance(edge, dict):
                    edge_list.append([edge.get('source', 0), edge.get('target', 0)])
                elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    edge_list.append([edge[0], edge[1]])
            if edge_list:
                edge_index = np.array(edge_list, dtype=np.int64).T
            else:
                n = len(nodes)
                src = list(range(n - 1)) + list(range(1, n))
                dst = list(range(1, n)) + list(range(n - 1))
                edge_index = np.array([src, dst], dtype=np.int64)
        else:
            n = len(nodes)
            src = list(range(n - 1)) + list(range(1, n))
            dst = list(range(1, n)) + list(range(n - 1))
            edge_index = np.array([src, dst], dtype=np.int64)

        return node_features, edge_index

    def _get_saprot_embedding(self, pdb_id: str) -> np.ndarray:
        pdb_id = pdb_id.lower()
        saprot_data = self.saprot.get(pdb_id)
        if saprot_data is None:
            return np.zeros((10, 1280), dtype=np.float32)

        if isinstance(saprot_data, dict):
            if 'embeddings' in saprot_data:
                embeddings = saprot_data['embeddings']
                if hasattr(embeddings, 'numpy'):
                    embeddings = embeddings.numpy()
                if isinstance(embeddings, np.ndarray):
                    return embeddings.astype(np.float32)

            embeddings = []
            for chain_id, chain_data in saprot_data.items():
                if chain_id in ['embeddings', 'adjacency', 'sequence', 'chain_id', 'pdb_id', 'embedding_dim', 'seq_len']:
                    continue
                if isinstance(chain_data, np.ndarray):
                    embeddings.append(chain_data)
                elif hasattr(chain_data, 'numpy'):
                    embeddings.append(chain_data.numpy())
                elif isinstance(chain_data, dict) and 'embedding' in chain_data:
                    emb = chain_data['embedding']
                    if hasattr(emb, 'numpy'):
                        emb = emb.numpy()
                    embeddings.append(emb)
            if embeddings:
                return np.vstack(embeddings).astype(np.float32)
        elif isinstance(saprot_data, np.ndarray):
            return saprot_data.astype(np.float32)
        elif hasattr(saprot_data, 'numpy'):
            return saprot_data.numpy().astype(np.float32)

        return np.zeros((10, 1280), dtype=np.float32)

    def _get_physicochemical(self, pdb_id: str) -> np.ndarray:
        pdb_id = pdb_id.lower()
        phys = self.physicochemical.get(pdb_id)
        if phys is None:
            return np.zeros((10, 8), dtype=np.float32)

        if isinstance(phys, np.ndarray):
            return phys.astype(np.float32)
        elif hasattr(phys, 'numpy'):
            return phys.numpy().astype(np.float32)
        elif isinstance(phys, dict):
            if 'features' in phys:
                return np.array(phys['features'], dtype=np.float32)
            features = []
            values = list(phys.values())
            # Simple heuristic: if values are dicts, it's residue data
            if values and isinstance(values[0], dict):
                 for residue_data in values:
                     if isinstance(residue_data, dict):
                         feat = [
                             residue_data.get('electrostatic', 0),
                             residue_data.get('sasa', 0),
                             residue_data.get('sasa_ratio', 0),
                             residue_data.get('bfactor', 0),
                             residue_data.get('hydrophobicity', 0),
                             residue_data.get('charge', 0),
                             residue_data.get('hbond_donor', 0),
                             residue_data.get('hbond_acceptor', 0),
                         ]
                         features.append(feat)
            if features:
                return np.array(features, dtype=np.float32)

        return np.zeros((10, 8), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]

        pdb_id = pair.get('pdb_id', pair.get('complex_id', ''))
        cdr3_seq = pair.get('cdr3', pair.get('cdr3b', ''))
        epitope_seq = pair.get('epitope', pair.get('peptide', ''))
        label = pair.get('label', pair.get('binding', 1))

        cdr3_emb = self._get_esm2_embedding(cdr3_seq, 'cdr3')
        epitope_emb = self._get_esm2_embedding(epitope_seq, 'epitope')

        # Structure features (Graph + SaProt)
        node_features, edge_index = self._get_graph_data(pdb_id)
        
        # Physicochemical features
        phys = self._get_physicochemical(pdb_id)
        
        # Ensure dimensions match node features
        if phys.shape[0] < node_features.shape[0]:
             padding = np.zeros((node_features.shape[0] - phys.shape[0], 8), dtype=np.float32)
             phys = np.vstack([phys, padding])
        else:
             phys = phys[:node_features.shape[0]]

        sample = {
            'tcr': {
                'sequence_emb': torch.tensor(cdr3_emb, dtype=torch.float32),
                'structure_x': torch.tensor(node_features, dtype=torch.float32),
                'structure_edge_index': torch.tensor(edge_index, dtype=torch.long),
                'physchem_features': torch.tensor(phys, dtype=torch.float32),
            },
            'pmhc': {
                'sequence_emb': torch.tensor(epitope_emb, dtype=torch.float32),
                'structure_x': torch.tensor(node_features, dtype=torch.float32),
                'structure_edge_index': torch.tensor(edge_index, dtype=torch.long),
                'physchem_features': torch.tensor(phys, dtype=torch.float32),
            },
            'label': torch.tensor(float(label), dtype=torch.float32),
            'metadata': {
                'pdb_id': pdb_id,
                'cdr3': cdr3_seq,
                'epitope': epitope_seq,
            }
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def collate_garsef(batch: List[Dict]) -> Dict:
    """Custom collate function for GARSEF batches."""
    tcr_data = []
    pmhc_data = []
    labels = []
    metadata = []

    for sample in batch:
        tcr_data.append(sample['tcr'])
        pmhc_data.append(sample['pmhc'])
        labels.append(sample['label'])
        metadata.append(sample['metadata'])

    def batch_graphs(data_list: List[Dict]) -> Dict:
        graphs = []
        for d in data_list:
            graph = Data(
                x=d['structure_x'],
                edge_index=d['structure_edge_index']
            )
            graphs.append(graph)

        batched_graph = Batch.from_data_list(graphs)

        return {
            'sequence_emb': torch.stack([d['sequence_emb'] for d in data_list]),
            'structure_x': batched_graph.x,
            'structure_edge_index': batched_graph.edge_index,
            'structure_batch': batched_graph.batch,
            'physchem_features': torch.stack([d['physchem_features'] for d in data_list]),
        }

    return {
        'tcr': batch_graphs(tcr_data),
        'pmhc': batch_graphs(pmhc_data),
        'label': torch.stack(labels),
        'metadata': metadata
    }


class PositiveOnlyDataset(Dataset):
    """Dataset wrapper returning only positive pairs."""

    def __init__(self, dataset: GARSEFDataset):
        self.dataset = dataset
        self.positive_indices = [
            i for i, pair in enumerate(dataset.pairs)
            if pair.get('label', pair.get('binding', 1)) == 1
        ]

    def __len__(self) -> int:
        return len(self.positive_indices)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[self.positive_indices[idx]]


class EpitopeGroupedDataset(Dataset):
    """Dataset grouped by epitope for per-epitope evaluation."""

    def __init__(self, dataset: GARSEFDataset):
        self.dataset = dataset
        self.epitope_groups = {}
        for idx, pair in enumerate(dataset.pairs):
            epitope = pair.get('epitope', pair.get('peptide', 'unknown'))
            if epitope not in self.epitope_groups:
                self.epitope_groups[epitope] = []
            self.epitope_groups[epitope].append(idx)
        self.epitopes = list(self.epitope_groups.keys())

    def get_epitope_indices(self, epitope: str) -> List[int]:
        return self.epitope_groups.get(epitope, [])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]


def create_data_loaders(
    train_file: Union[str, Path],
    val_file: Union[str, Path],
    features_dir: Union[str, Path],
    test_file: Optional[Union[str, Path]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    create_positive_loader: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create DataLoaders for GARSEF training."""
    train_dataset = GARSEFDataset(train_file, features_dir)
    val_dataset = GARSEFDataset(val_file, features_dir)

    test_loader = None
    if test_file and Path(test_file).exists():
        test_dataset = GARSEFDataset(test_file, features_dir)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_garsef,
            pin_memory=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_garsef,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_garsef,
        pin_memory=True
    )

    positive_loader = None
    if create_positive_loader:
        positive_dataset = PositiveOnlyDataset(train_dataset)
        positive_loader = DataLoader(
            positive_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_garsef,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader, positive_loader


def create_balanced_sampler(dataset: GARSEFDataset) -> WeightedRandomSampler:
    """Create weighted sampler for balanced batch sampling."""
    labels = [p.get('label', p.get('binding', 1)) for p in dataset.pairs]
    labels = np.array(labels, dtype=int)

    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

