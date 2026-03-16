"""
Generate Residue-Level Physicochemical Heat Map

Creates a professional heat map visualization showing the 8 physicochemical
features across TCR CDR3 residue positions.

Usage:
    python scripts/generate_physicochemical_heatmap.py
    python scripts/generate_physicochemical_heatmap.py --output results/physicochemical_heatmap.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for publication-quality figures
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 10


def generate_realistic_physicochemical_data(n_residues=15, seed=42):
    """
    Generate realistic physicochemical feature data.
    
    In real usage, replace this with actual extracted features.
    
    Args:
        n_residues: Number of residues in CDR3 region
        seed: Random seed for reproducibility
    
    Returns:
        np.ndarray: Shape (n_residues, 8) normalized feature matrix
    """
    np.random.seed(seed)
    
    # Feature names for reference
    features = [
        'Electrostatic', 'SASA', 'SASA Ratio', 'B-factor',
        'Hydrophobicity', 'Charge', 'H-donor', 'H-acceptor'
    ]
    
    # Generate realistic patterns for each feature
    data = np.zeros((n_residues, 8))
    
    # Electrostatic: varies along sequence with some periodicity
    data[:, 0] = np.sin(np.linspace(0, 4*np.pi, n_residues)) + \
                 np.random.normal(0, 0.3, n_residues)
    
    # SASA: typically higher at terminal residues (more exposed)
    u_shape = np.concatenate([
        np.linspace(1.5, -0.5, n_residues//2),
        np.linspace(-0.5, 1.5, n_residues - n_residues//2)
    ])
    data[:, 1] = u_shape + np.random.normal(0, 0.2, n_residues)
    
    # SASA Ratio: correlated with SASA
    data[:, 2] = data[:, 1] * 0.8 + np.random.normal(0, 0.3, n_residues)
    
    # B-factor: higher at flexible regions (terminal ends)
    data[:, 3] = u_shape * 1.2 + np.random.normal(0, 0.25, n_residues)
    
    # Hydrophobicity: random clusters
    data[:, 4] = np.random.choice([-1, 0, 1], n_residues) + \
                 np.random.normal(0, 0.4, n_residues)
    
    # Charge: discrete-like values
    data[:, 5] = np.random.choice([-1.5, -0.5, 0, 0.5, 1.5], n_residues) + \
                 np.random.normal(0, 0.2, n_residues)
    
    # H-donor: weakly correlated with charge
    data[:, 6] = -data[:, 5] * 0.5 + np.random.normal(0, 0.5, n_residues)
    
    # H-acceptor: complementary to H-donor
    data[:, 7] = -data[:, 6] * 0.6 + np.random.normal(0, 0.6, n_residues)
    
    # Normalize to roughly [-2, 2] range
    data = np.clip(data, -2, 2)
    
    return data


def create_physicochemical_heatmap(
    data=None,
    output_path='results/physicochemical_heatmap.png',
    dpi=300,
    figsize=(10, 6)
):
    """
    Create residue-level physicochemical heat map.
    
    Args:
        data: np.ndarray of shape (n_residues, 8) or None to use synthetic data
        output_path: Path to save figure
        dpi: Resolution for output image
        figsize: Figure size in inches
    """
    # Generate or validate data
    if data is None:
        print("Generating synthetic physicochemical data...")
        data = generate_realistic_physicochemical_data()
    
    n_residues, n_features = data.shape
    assert n_features == 8, "Expected 8 physicochemical features"
    
    # Feature labels
    feature_labels = [
        'Electrostatic',
        'SASA',
        'SASA Ratio',
        'B-factor',
        'Hydrophobicity',
        'Charge',
        'H-donor',
        'H-acceptor'
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heat map with diverging colormap
    im = ax.imshow(
        data,
        cmap='RdBu_r',  # Red-Blue reversed (Blue for negative, Red for positive)
        aspect='auto',
        vmin=-2,
        vmax=2,
        interpolation='nearest'
    )
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_features))
    ax.set_yticks(np.arange(n_residues))
    ax.set_xticklabels(feature_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([f'{i+1}' for i in range(n_residues)], fontsize=9)
    
    # Labels
    ax.set_xlabel('Physicochemical Features', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residue Position', fontsize=11, fontweight='bold')
    
    # Title
    ax.set_title(
        'Residue-Level Physicochemical Feature Profile\nTCR CDR3 Region',
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    
    # Add grid lines
    ax.set_xticks(np.arange(n_features + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_residues + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    ax.tick_params(which='minor', size=0)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        'Normalized Intensity',
        rotation=270,
        labelpad=20,
        fontsize=10,
        fontweight='bold'
    )
    cbar.ax.tick_params(labelsize=9)
    
    # Add value annotations (optional, comment out if too cluttered)
    # for i in range(n_residues):
    #     for j in range(n_features):
    #         text = ax.text(j, i, f'{data[i, j]:.1f}',
    #                       ha="center", va="center", color="black", fontsize=6)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved heat map to: {output_path}")
    
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Generate physicochemical feature heat map'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/physicochemical_heatmap.png',
        help='Output path for heat map image'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Resolution for output image'
    )
    parser.add_argument(
        '--residues',
        type=int,
        default=15,
        help='Number of CDR3 residues to visualize'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the plot interactively'
    )
    
    args = parser.parse_args()
    
    # Generate synthetic data (replace with real data loading in production)
    data = generate_realistic_physicochemical_data(n_residues=args.residues)
    
    # Create heat map
    fig, ax = create_physicochemical_heatmap(
        data=data,
        output_path=args.output,
        dpi=args.dpi
    )
    
    if args.show:
        plt.show()
    
    print("\n=== Heat Map Generation Complete ===")
    print(f"Output saved to: {args.output}")
    print("\nTo use your actual data, modify the script to load")
    print("physicochemical features from your processed dataset.")


if __name__ == "__main__":
    main()
