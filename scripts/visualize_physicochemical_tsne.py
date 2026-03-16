"""
t-SNE/UMAP Visualization of Physicochemical Features

Creates 2D embedding visualization showing whether physicochemical features
can separate binding vs non-binding TCR-pMHC pairs.

Usage:
    python scripts/visualize_physicochemical_tsne.py
    python scripts/visualize_physicochemical_tsne.py --method umap
    python scripts/visualize_physicochemical_tsne.py --perplexity 50
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Try to import UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 11


def load_physicochemical_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load physicochemical features and labels.
    
    Returns:
        Tuple of (features_matrix, labels_array)
    """
    # Try to load from processed data
    processed_dir = data_dir / 'processed'
    combined_file = processed_dir / 'combined_features.pkl'
    
    if combined_file.exists():
        print(f"Loading from {combined_file}")
        with open(combined_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract physicochemical features
        phys_data = data.get('physicochemical', {})
        
        if isinstance(phys_data, dict):
            # Assume structure: {pdb_id: {'features': array}}
            features_list = []
            labels_list = []
            
            for pdb_id, feat_dict in phys_data.items():
                if 'features' in feat_dict:
                    # Average features across residues to get sample-level features
                    feat = feat_dict['features']
                    if len(feat.shape) == 2:
                        # [num_residues, 8] -> [8] by averaging
                        feat = feat.mean(axis=0)
                    features_list.append(feat)
                    
                    # Assign dummy label for now (you should replace with actual labels)
                    # In real usage, load from your dataset with binding labels
                    labels_list.append(1)  # Placeholder
            
            if features_list:
                features = np.array(features_list)
                labels = np.array(labels_list)
                return features, labels
    
    # If no data found, generate synthetic data for demonstration
    print("No physicochemical data found. Generating synthetic data...")
    return generate_synthetic_data()


def generate_synthetic_data(
    n_binders: int = 300,
    n_non_binders: int = 300,
    n_features: int = 8,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic physicochemical features for demonstration.
    
    Real usage should load actual extracted features and binding labels.
    """
    np.random.seed(seed)
    
    # Binders: higher SASA, electrostatic complementarity
    binders = np.random.randn(n_binders, n_features)
    binders[:, 1] += 1.5  # Higher SASA
    binders[:, 0] += 0.8  # Higher electrostatic
    binders[:, 2] += 1.2  # Higher SASA ratio
    
    # Non-binders: lower exposure, random features
    non_binders = np.random.randn(n_non_binders, n_features)
    non_binders[:, 1] -= 0.5  # Lower SASA
    non_binders[:, 4] += 0.5  # Different hydrophobicity distribution
    
    features = np.vstack([binders, non_binders])
    labels = np.array([1] * n_binders + [0] * n_non_binders)
    
    return features, labels


def create_tsne_visualization(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str = 'results/physicochemical_tsne.png',
    perplexity: int = 30,
    random_state: int = 42,
    dpi: int = 300
):
    """
    Create t-SNE visualization of physicochemical features.
    """
    print(f"Running t-SNE with perplexity={perplexity}...")
    
    # Normalize features
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )
    embedding = tsne.fit_transform(features_norm)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot binders and non-binders
    for label, color, name in [
        (1, '#E74C3C', 'Binding'),
        (0, '#3498DB', 'Non-Binding')
    ]:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            label=name,
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(
        't-SNE Embedding of Physicochemical Features\nTCR-pMHC Binding Pairs',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved t-SNE visualization to {output_path}")
    
    return embedding


def create_umap_visualization(
    features: np.ndarray,
    labels: np.ndarray,
    output_path: str = 'results/physicochemical_umap.png',
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    dpi: int = 300
):
    """
    Create UMAP visualization of physicochemical features.
    """
    if not UMAP_AVAILABLE:
        print("Error: UMAP not installed. Please install with: pip install umap-learn")
        return None
    
    print(f"Running UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}...")
    
    # Normalize features
    features_norm = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
    
    # Run UMAP
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        verbose=True
    )
    embedding = reducer.fit_transform(features_norm)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot
    for label, color, name in [
        (1, '#E74C3C', 'Binding'),
        (0, '#3498DB', 'Non-Binding')
    ]:
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=color,
            label=name,
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidth=0.5
        )
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=12, fontweight='bold')
    ax.set_title(
        'UMAP Embedding of Physicochemical Features\nTCR-pMHC Binding Pairs',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#FAFAFA')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved UMAP visualization to {output_path}")
    
    return embedding


def calculate_separation_metrics(embedding: np.ndarray, labels: np.ndarray):
    """Calculate how well the embedding separates binders from non-binders."""
    from sklearn.metrics import silhouette_score
    
    # Calculate silhouette score
    silhouette = silhouette_score(embedding, labels)
    
    # Calculate cluster separation
    binder_centers = embedding[labels == 1].mean(axis=0)
    non_binder_centers = embedding[labels == 0].mean(axis=0)
    center_distance = np.linalg.norm(binder_centers - non_binder_centers)
    
    print(f"\n=== Separation Metrics ===")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Cluster Center Distance: {center_distance:.4f}")
    
    return {
        'silhouette_score': silhouette,
        'center_distance': center_distance
    }


def main():
    parser = argparse.ArgumentParser(
        description='Visualize physicochemical features with t-SNE/UMAP'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['tsne', 'umap', 'both'],
        default='both',
        help='Dimensionality reduction method'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing processed data'
    )
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter'
    )
    parser.add_argument(
        '--n_neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter'
    )
    parser.add_argument(
        '--min_dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output image resolution'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots interactively'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_dir = Path(args.data_dir)
    features, labels = load_physicochemical_data(data_dir)
    
    print(f"\nLoaded {len(features)} samples with {features.shape[1]} features")
    print(f"Binders: {(labels == 1).sum()}, Non-binders: {(labels == 0).sum()}")
    
    # Run visualizations
    if args.method in ['tsne', 'both']:
        embedding = create_tsne_visualization(
            features,
            labels,
            output_path='results/physicochemical_tsne.png',
            perplexity=args.perplexity,
            dpi=args.dpi
        )
        calculate_separation_metrics(embedding, labels)
    
    if args.method in ['umap', 'both']:
        if UMAP_AVAILABLE:
            embedding = create_umap_visualization(
                features,
                labels,
                output_path='results/physicochemical_umap.png',
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                dpi=args.dpi
            )
            if embedding is not None:
                calculate_separation_metrics(embedding, labels)
        else:
            print("UMAP not available. Skipping UMAP visualization.")
    
    if args.show:
        plt.show()
    
    print("\n=== Visualization Complete ===")


if __name__ == "__main__":
    main()
