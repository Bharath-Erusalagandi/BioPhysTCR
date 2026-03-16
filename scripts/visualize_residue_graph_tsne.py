"""
Residue-Level Graph t-SNE Visualization

Creates t-SNE embedding of residue-level graph neural network features,
showing how residues cluster based on structural context and binding properties.

Usage:
    python scripts/visualize_residue_graph_tsne.py
    python scripts/visualize_residue_graph_tsne.py --color_by interface
    python scripts/visualize_residue_graph_tsne.py --perplexity 40
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 11


def load_residue_embeddings(data_dir: Path) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Load residue-level structure embeddings (e.g., from SaProt GNN).
    
    Returns:
        Tuple of (embeddings, labels, structure_ids)
    """
    processed_dir = data_dir / 'processed'
    
    # Try to load SaProt features
    saprot_dir = processed_dir / 'saprot'
    if saprot_dir.exists():
        all_embeddings = []
        all_labels = []
        all_structure_ids = []
        
        for pkl_file in saprot_dir.glob('*.pkl'):
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = data.get('embeddings', None)
            if embeddings is not None and len(embeddings.shape) == 2:
                # embeddings shape: [num_residues, embed_dim]
                all_embeddings.append(embeddings)
                
                # Assign labels (interface vs non-interface)
                # For demo: label by position (first half = interface, second half = non-interface)
                n_res = embeddings.shape[0]
                labels = ['interface'] * (n_res // 3) + ['non-interface'] * (n_res - n_res // 3)
                all_labels.extend(labels)
                
                # Track which structure each residue belongs to
                struct_id = pkl_file.stem
                all_structure_ids.extend([struct_id] * n_res)
        
        if all_embeddings:
            embeddings_array = np.vstack(all_embeddings)
            return embeddings_array, all_labels, all_structure_ids
    
    # If no data found, generate synthetic
    print("No residue embeddings found. Generating synthetic data...")
    return generate_synthetic_residue_embeddings()


def generate_synthetic_residue_embeddings(
    n_structures: int = 20,
    residues_per_struct: int = 50,
    embed_dim: int = 128,
    seed: int = 42
) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Generate synthetic residue-level embeddings for demonstration.
    
    Creates clustered embeddings representing different structural contexts.
    """
    np.random.seed(seed)
    
    all_embeddings = []
    all_labels = []
    all_structure_ids = []
    
    for struct_id in range(n_structures):
        # Each structure has interface and non-interface residues
        n_interface = residues_per_struct // 3
        n_non_interface = residues_per_struct - n_interface
        
        # Interface residues cluster together
        interface_center = np.random.randn(embed_dim) * 2
        interface_embeddings = interface_center + np.random.randn(n_interface, embed_dim) * 0.5
        
        # Non-interface residues more dispersed
        non_interface_center = np.random.randn(embed_dim) * 2
        non_interface_embeddings = non_interface_center + np.random.randn(n_non_interface, embed_dim) * 1.2
        
        all_embeddings.append(interface_embeddings)
        all_embeddings.append(non_interface_embeddings)
        
        all_labels.extend(['interface'] * n_interface)
        all_labels.extend(['non-interface'] * n_non_interface)
        
        all_structure_ids.extend([f'struct_{struct_id}'] * residues_per_struct)
    
    embeddings_array = np.vstack(all_embeddings)
    return embeddings_array, all_labels, all_structure_ids


def create_residue_tsne_visualization(
    embeddings: np.ndarray,
    labels: List[str],
    structure_ids: List[str],
    output_path: str = 'results/residue_graph_tsne.png',
    color_by: str = 'interface',
    perplexity: int = 30,
    random_state: int = 42,
    dpi: int = 300
):
    """
    Create t-SNE visualization of residue-level embeddings.
    
    Args:
        color_by: 'interface', 'structure', or 'both'
    """
    print(f"Running t-SNE on {len(embeddings)} residues...")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Normalize
    embeddings_norm = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)
    
    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        max_iter=1000,
        verbose=1
    )
    embedding_2d = tsne.fit_transform(embeddings_norm)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if color_by == 'interface':
        # Color by interface vs non-interface
        for label, color, alpha in [
            ('interface', '#E74C3C', 0.6),
            ('non-interface', '#FFA07A', 0.4)
        ]:
            mask = np.array([l == label for l in labels])
            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=color,
                label=label.replace('_', ' ').title(),
                alpha=alpha,
                s=30,
                edgecolors='white',
                linewidth=0.3
            )
        
        title = 'Residue Graph Embedding\nColored by Interface Status'
        
    elif color_by == 'structure':
        # Color by structure ID
        unique_structs = list(set(structure_ids))
        colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_structs), 20)))
        
        for i, struct_id in enumerate(unique_structs[:20]):  # Limit to 20 for visibility
            mask = np.array([s == struct_id for s in structure_ids])
            ax.scatter(
                embedding_2d[mask, 0],
                embedding_2d[mask, 1],
                c=[colors[i]],
                label=struct_id if i < 10 else None,  # Only show first 10 in legend
                alpha=0.6,
                s=30,
                edgecolors='white',
                linewidth=0.3
            )
        
        title = 'Residue Graph Embedding\nColored by Structure'
        
    else:  # both
        # Color by structure, shape by interface
        unique_structs = list(set(structure_ids))
        colors = plt.cm.Set3(np.linspace(0, 1, min(len(unique_structs), 12)))
        
        for i, struct_id in enumerate(unique_structs[:12]):
            for label, marker in [('interface', 'o'), ('non-interface', 's')]:
                mask = np.array([s == struct_id and l == label 
                               for s, l in zip(structure_ids, labels)])
                if mask.any():
                    ax.scatter(
                        embedding_2d[mask, 0],
                        embedding_2d[mask, 1],
                        c=[colors[i % 12]],
                        marker=marker,
                        label=f'{struct_id}_{label}' if i < 3 else None,
                        alpha=0.6,
                        s=40,
                        edgecolors='white',
                        linewidth=0.3
                    )
        
        title = 'Residue Graph Embedding\nStructure & Interface Status'
    
    ax.set_xlabel('t-SNE 1', fontsize=13, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    
    if color_by != 'both':
        ax.legend(
            loc='upper right',
            fontsize=10,
            framealpha=0.9,
            markerscale=1.5
        )
    
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved residue graph t-SNE to {output_path}")
    
    return embedding_2d


def main():
    parser = argparse.ArgumentParser(
        description='Visualize residue-level graph embeddings with t-SNE'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing processed data'
    )
    parser.add_argument(
        '--color_by',
        type=str,
        choices=['interface', 'structure', 'both'],
        default='interface',
        help='How to color residues'
    )
    parser.add_argument(
        '--perplexity',
        type=int,
        default=30,
        help='t-SNE perplexity parameter'
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
        help='Display plot interactively'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_dir = Path(args.data_dir)
    embeddings, labels, structure_ids = load_residue_embeddings(data_dir)
    
    print(f"\nLoaded {len(embeddings)} residues from {len(set(structure_ids))} structures")
    
    # Create visualization
    create_residue_tsne_visualization(
        embeddings,
        labels,
        structure_ids,
        output_path='results/residue_graph_tsne.png',
        color_by=args.color_by,
        perplexity=args.perplexity,
        dpi=args.dpi
    )
    
    if args.show:
        plt.show()
    
    print("\n=== Visualization Complete ===")


if __name__ == "__main__":
    main()
