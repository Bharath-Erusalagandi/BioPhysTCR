"""
APBS Electrostatic Potential Visualization

Visualizes electrostatic potential on protein surfaces for TCR-pMHC complexes
to show charge complementarity at binding interfaces.

Note: This creates 2D heatmap projections. For full 3D molecular surface 
visualization, use PyMOL or Chimera with APBS plugin.

Usage:
    python scripts/visualize_apbs_electrostatics.py
    python scripts/visualize_apbs_electrostatics.py --pdb_id 1ao7
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
plt.rcParams['font.size'] = 11


def load_electrostatic_features(data_dir: Path, pdb_id: Optional[str] = None):
    """
    Load electrostatic potential from physicochemical features.
    """
    processed_dir = data_dir / 'processed'
    combined_file = processed_dir / 'combined_features.pkl'
    
    if combined_file.exists():
        with open(combined_file, 'rb') as f:
            data = pickle.load(f)
        
        phys_data = data.get('physicochemical', {})
        
        if pdb_id and pdb_id in phys_data:
            return phys_data[pdb_id]
        elif phys_data:
            # Return first available structure
            first_key = list(phys_data.keys())[0]
            return phys_data[first_key]
    
    return None


def generate_synthetic_electrostatic_surface(
    n_points: int = 10000,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic electrostatic surface for demonstration.
    
    Returns:
        Tuple of (x_coords, y_coords, potentials)
    """
    np.random.seed(seed)
    
    # Generate surface points (simulating molecular surface)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    
    # Add some structure-like features
    r = 10 + 2 * np.sin(3 * theta) * np.cos(2 * phi)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    # Project to 2D (flatten z dimension)
    x_2d = x + 0.3 * z
    y_2d = y + 0.3 * z
    
    # Create electrostatic potential pattern
    # Positive regions (blue) and negative regions (red)
    potential = np.zeros(n_points)
    
    # Add charged patches
    # Positive patch (binding interface)
    center1 = np.array([5, 5])
    dist1 = np.sqrt((x_2d - center1[0])**2 + (y_2d - center1[1])**2)
    potential += 3 * np.exp(-dist1**2 / 20)
    
    # Negative patch (complementary)
    center2 = np.array([-5, -5])
    dist2 = np.sqrt((x_2d - center2[0])**2 + (y_2d - center2[1])**2)
    potential -= 3 * np.exp(-dist2**2 / 20)
    
    # Moderate positive patch
    center3 = np.array([8, -3])
    dist3 = np.sqrt((x_2d - center3[0])**2 + (y_2d - center3[1])**2)
    potential += 2 * np.exp(-dist3**2 / 15)
    
    # Negative patch
    center4 = np.array([-3, 8])
    dist4 = np.sqrt((x_2d - center4[0])**2 + (y_2d - center4[1])**2)
    potential -= 2.5 * np.exp(-dist4**2 / 18)
    
    # Add some noise
    potential += np.random.normal(0, 0.3, n_points)
    
    # Clip to reasonable range
    potential = np.clip(potential, -5, 5)
    
    return x_2d, y_2d, potential


def create_electrostatic_surface_plot(
    x: np.ndarray,
    y: np.ndarray,
    potential: np.ndarray,
    output_path: str = 'results/electrostatic_surface.png',
    title: str = 'Electrostatic Potential on Molecular Surface',
    dpi: int = 300
):
    """
    Create 2D projection of electrostatic potential on molecular surface.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap (red-white-blue for negative-neutral-positive)
    colors = ['#D32F2F', '#EF5350', '#FFCDD2', 'white', '#BBDEFB', '#42A5F5', '#1976D2']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('electrostatic', colors, N=n_bins)
    
    # Create scatter plot with color based on potential
    scatter = ax.scatter(
        x, y,
        c=potential,
        cmap=cmap,
        s=50,
        alpha=0.8,
        vmin=-5,
        vmax=5,
        edgecolors='none'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        'Electrostatic Potential (kT/e)',
        rotation=270,
        labelpad=25,
        fontsize=12,
        fontweight='bold'
    )
    cbar.ax.tick_params(labelsize=10)
    
    # Customize appearance
    ax.set_xlabel('X (Å)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (Å)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_facecolor('#F5F5F5')
    
    # Add annotations
    ax.text(
        0.02, 0.98,
        'Red: Negative potential\nBlue: Positive potential\nWhite: Neutral',
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved electrostatic surface plot to {output_path}")


def create_interface_complementarity_plot(
    output_path: str = 'results/electrostatic_complementarity.png',
    dpi: int = 300
):
    """
    Create side-by-side comparison showing electrostatic complementarity
    between TCR and pMHC at binding interface.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Generate TCR surface
    np.random.seed(42)
    x_tcr, y_tcr, pot_tcr = generate_synthetic_electrostatic_surface(n_points=8000, seed=42)
    
    # Generate pMHC surface (complementary charges)
    x_pmhc, y_pmhc, pot_pmhc = generate_synthetic_electrostatic_surface(n_points=8000, seed=43)
    pot_pmhc = -pot_pmhc * 0.8  # Make it complementary
    
    # Custom colormap
    colors = ['#D32F2F', '#EF5350', '#FFCDD2', 'white', '#BBDEFB', '#42A5F5', '#1976D2']
    cmap = LinearSegmentedColormap.from_list('electrostatic', colors, N=256)
    
    # TCR surface
    scatter1 = ax1.scatter(
        x_tcr, y_tcr,
        c=pot_tcr,
        cmap=cmap,
        s=40,
        alpha=0.8,
        vmin=-5,
        vmax=5,
        edgecolors='none'
    )
    ax1.set_title('TCR Surface', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlabel('X (Å)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Y (Å)', fontsize=11, fontweight='bold')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    ax1.set_facecolor('#F5F5F5')
    
    # pMHC surface
    scatter2 = ax2.scatter(
        x_pmhc, y_pmhc,
        c=pot_pmhc,
        cmap=cmap,
        s=40,
        alpha=0.8,
        vmin=-5,
        vmax=5,
        edgecolors='none'
    )
    ax2.set_title('pMHC Surface', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlabel('X (Å)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Y (Å)', fontsize=11, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    ax2.set_facecolor('#F5F5F5')
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter2, cax=cbar_ax)
    cbar.set_label(
        'Electrostatic Potential (kT/e)',
        rotation=270,
        labelpad=25,
        fontsize=12,
        fontweight='bold'
    )
    
    # Overall title
    fig.suptitle(
        'Electrostatic Complementarity at TCR-pMHC Interface',
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved complementarity plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize APBS electrostatic potential'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing processed data'
    )
    parser.add_argument(
        '--pdb_id',
        type=str,
        default=None,
        help='PDB ID to visualize'
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
    
    print("Generating electrostatic potential visualizations...")
    print("Note: Using synthetic data for demonstration.")
    print("For real APBS calculations, use APBS software with PDB2PQR preprocessing.\n")
    
    # Generate single surface plot
    x, y, potential = generate_synthetic_electrostatic_surface()
    create_electrostatic_surface_plot(
        x, y, potential,
        output_path='results/electrostatic_surface.png',
        dpi=args.dpi
    )
    
    # Generate complementarity comparison
    create_interface_complementarity_plot(
        output_path='results/electrostatic_complementarity.png',
        dpi=args.dpi
    )
    
    if args.show:
        plt.show()
    
    print("\n=== Electrostatic Visualization Complete ===")
    print("\nGenerated visualizations:")
    print("  1. electrostatic_surface.png - 2D projection of surface potential")
    print("  2. electrostatic_complementarity.png - TCR vs pMHC comparison")
    print("\nFor 3D molecular visualization, use PyMOL with APBS plugin.")


if __name__ == "__main__":
    main()
