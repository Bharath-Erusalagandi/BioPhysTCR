"""
GARSEF Evaluation Script (Day 7)

Evaluates trained GARSEF model on test sets and generates metrics.

Usage:
    python scripts/03_evaluate.py --checkpoint best_model.pt --scenario 1
    python scripts/03_evaluate.py --checkpoint best_model.pt --scenario 2
    python scripts/03_evaluate.py --checkpoint best_model.pt --per_epitope

Features:
    - Overall metrics (AUC, AUPR, MCC, F1)
    - Per-epitope metrics
    - Comparison with baselines
    - t-SNE visualization of embeddings
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from models import GARSEF, GARSEFConfig
from training import MetricsCalculator, print_metrics
from utils import GARSEFDataset, collate_garsef


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_model(checkpoint_path: Path, config: Dict, device: str = 'cuda') -> GARSEF:
    """Load trained model from checkpoint."""
    model_config = GARSEFConfig(
        esm2_dim=config['model']['sequence']['input_dim'],
        sequence_hidden_dim=config['model']['sequence']['hidden_dim'],
        saprot_dim=config['model']['structure']['input_dim'],
        structure_hidden_dim=config['model']['structure']['hidden_dim'],
        structure_num_gnn_layers=config['model']['structure']['num_layers'],
        physchem_dim=config['model']['physicochemical']['input_dim'],
        physchem_hidden_dim=config['model']['physicochemical']['hidden_dim'],
        fusion_dim=config['model']['fusion']['hidden_dim'],
        dropout=config['model']['sequence']['dropout'],
        fusion_dropout=config['model']['fusion']['dropout'],
        temperature=config['training']['contrastive']['temperature'],
    )

    model = GARSEF(model_config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")

    return model


@torch.no_grad()
def evaluate_model(
    model: GARSEF,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List]:
    """
    Evaluate model and collect predictions.

    Returns:
        metrics: Dict with evaluation metrics
        y_true: True labels
        y_pred: Predicted probabilities
        metadata: Sample metadata
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_metadata = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        tcr_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['tcr'].items()}
        pmhc_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch['pmhc'].items()}
        labels = batch['label'].to(device)

        outputs = model(tcr_data, pmhc_data)

        preds = torch.sigmoid(outputs['binding_logits']).squeeze()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_metadata.extend(batch['metadata'])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    calculator = MetricsCalculator()
    metrics = calculator.compute_all(y_true, y_pred)

    return metrics, y_true, y_pred, all_metadata


@torch.no_grad()
def get_embeddings(
    model: GARSEF,
    dataloader: DataLoader,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
    """
    Extract embeddings for visualization.

    Returns:
        tcr_embeddings: TCR embeddings [N, dim]
        pmhc_embeddings: pMHC embeddings [N, dim]
        labels: Binding labels [N]
        metadata: Sample metadata
    """
    model.eval()

    tcr_embs = []
    pmhc_embs = []
    labels = []
    metadata = []

    for batch in tqdm(dataloader, desc="Extracting embeddings"):
        tcr_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['tcr'].items()}
        pmhc_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch['pmhc'].items()}

        outputs = model(tcr_data, pmhc_data)

        tcr_embs.extend(outputs['tcr_emb'].cpu().numpy())
        pmhc_embs.extend(outputs['pmhc_emb'].cpu().numpy())
        labels.extend(batch['label'].numpy())
        metadata.extend(batch['metadata'])

    return (
        np.array(tcr_embs),
        np.array(pmhc_embs),
        np.array(labels),
        metadata
    )


def compute_per_epitope_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: List[Dict]
) -> Dict[str, Dict[str, float]]:
    """Compute metrics per epitope."""
    calculator = MetricsCalculator()

    epitope_ids = [m['epitope'] for m in metadata]
    per_epitope = calculator.compute_per_epitope(y_true, y_pred, epitope_ids)

    return per_epitope


def generate_tsne_visualization(
    tcr_embs: np.ndarray,
    pmhc_embs: np.ndarray,
    labels: np.ndarray,
    output_path: Path
):
    """Generate t-SNE visualization of embeddings."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
    except ImportError:
        print("sklearn or matplotlib not available. Skipping t-SNE visualization.")
        return

    print("\nGenerating t-SNE visualization...")

    combined = np.vstack([tcr_embs, pmhc_embs])
    combined_labels = np.hstack([labels, labels])
    types = np.array(['TCR'] * len(tcr_embs) + ['pMHC'] * len(pmhc_embs))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(combined)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    colors = ['blue' if l == 0 else 'red' for l in combined_labels]
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.5, s=10)
    ax.set_title('By Binding Label (Blue=Non-binding, Red=Binding)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    ax = axes[1]
    colors = ['green' if t == 'TCR' else 'orange' for t in types]
    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, alpha=0.5, s=10)
    ax.set_title('By Type (Green=TCR, Orange=pMHC)')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"t-SNE visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate GARSEF model")

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')

    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2],
                        help='Evaluation scenario')
    parser.add_argument('--per_epitope', action='store_true',
                        help='Compute per-epitope metrics')
    parser.add_argument('--tsne', action='store_true',
                        help='Generate t-SNE visualization')

    parser.add_argument('--features_dir', type=str, default='data/processed',
                        help='Directory with processed features')
    parser.add_argument('--splits_dir', type=str, default='data/splits',
                        help='Directory with split definitions')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                        help='Directory for output files')

    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')

    args = parser.parse_args()

    config_path = PROJECT_DIR / args.config
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_DIR / checkpoint_path
    features_dir = PROJECT_DIR / args.features_dir
    splits_dir = PROJECT_DIR / args.splits_dir
    output_dir = PROJECT_DIR / args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)

    print(f"\nLoading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, config, device)

    if args.scenario == 1:
        test_csv = splits_dir / "test_random.csv"
    else:
        test_csv = splits_dir / "test_unseen_epitope.csv"

    print(f"\nLoading test data from: {test_csv}")

    if not test_csv.exists():
        print(f"Error: Test file not found: {test_csv}")
        return

    test_dataset = GARSEFDataset(test_csv, features_dir)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_garsef
    )

    print("\n" + "=" * 50)
    print(f"Evaluating on Scenario {args.scenario}")
    print("=" * 50)

    metrics, y_true, y_pred, metadata = evaluate_model(model, test_loader, device)

    print("\nOverall Metrics:")
    print(print_metrics(metrics, prefix="  "))

    if args.per_epitope:
        print("\n" + "-" * 50)
        print("Per-Epitope Metrics")
        print("-" * 50)

        per_epitope = compute_per_epitope_metrics(y_true, y_pred, metadata)

        aucs = [m['auc'] for m in per_epitope.values()]
        auprs = [m['aupr'] for m in per_epitope.values()]

        print(f"\n  Number of epitopes: {len(per_epitope)}")
        print(f"  AUC:  mean={np.mean(aucs):.4f}, std={np.std(aucs):.4f}")
        print(f"  AUPR: mean={np.mean(auprs):.4f}, std={np.std(auprs):.4f}")

        per_epitope_df = pd.DataFrame([
            {'epitope': epi, **metrics}
            for epi, metrics in per_epitope.items()
        ])
        per_epitope_path = output_dir / f"per_epitope_scenario{args.scenario}.csv"
        per_epitope_df.to_csv(per_epitope_path, index=False)
        print(f"\n  Saved to: {per_epitope_path}")

    if args.tsne:
        print("\n" + "-" * 50)
        print("Generating Embeddings")
        print("-" * 50)

        tcr_embs, pmhc_embs, labels, _ = get_embeddings(model, test_loader, device)

        tsne_path = output_dir / f"tsne_scenario{args.scenario}.png"
        generate_tsne_visualization(tcr_embs, pmhc_embs, labels, tsne_path)

    results = {
        'scenario': args.scenario,
        'checkpoint': str(checkpoint_path),
        'metrics': metrics,
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int(len(y_true) - y_true.sum()),
    }

    results_path = output_dir / f"evaluation_scenario{args.scenario}.json"
    with open(results_path, 'w') as f:
        def convert(obj):
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\nDone!")


if __name__ == '__main__':
    main()
