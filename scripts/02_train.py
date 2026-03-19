"""BioPhysTCR Training Script (Day 5)"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from models import BioPhysTCR, BioPhysTCRConfig, create_biophystcr
from training import (
    BioPhysTCRTrainer,
    TrainerConfig,
    train_biophystcr,
    MetricsCalculator,
    print_metrics
)
from utils import (
    BioPhysTCRDataset,
    PositiveOnlyDataset,
    create_data_loaders,
    collate_biophystcr
)


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_model(config: Dict) -> BioPhysTCR:
    """Create BioPhysTCR model from config."""
    model_config = BioPhysTCRConfig(
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
    return BioPhysTCR(model_config)


def create_trainer_config(config: Dict, args: argparse.Namespace) -> TrainerConfig:
    """Create trainer config from YAML config and command line args."""
    return TrainerConfig(
        epochs=args.epochs if args.epochs else config['training']['epochs'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        batch_size=args.batch_size if args.batch_size else config['training']['batch_size'],
        contrastive_weight=config['training']['contrastive']['weight'],
        binary_weight=config['training']['binary']['weight'],
        temperature=config['training']['contrastive']['temperature'],
        patience=config['training']['patience'],
        min_delta=config['training']['min_delta'],
        save_dir=args.save_dir,
        device=args.device if args.device else config['device'],
        num_workers=config['num_workers'],
        use_wandb=args.use_wandb,
        project_name=config['logging']['project_name'],
    )


def prepare_datasets(
    features_dir: Path,
    splits_dir: Path,
    scenario: int = 1,
    subset_size: Optional[int] = None
) -> tuple:
    """Prepare train/val/test datasets."""
    if scenario == 1:
        train_csv = splits_dir / "train_random.csv"
        val_csv = splits_dir / "val_random.csv"
        test_csv = splits_dir / "test_random.csv"
    else:
        train_csv = splits_dir / "train_unseen_epitope.csv"
        val_csv = splits_dir / "val_unseen_epitope.csv"
        test_csv = splits_dir / "test_unseen_epitope.csv"

    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found. Creating placeholder...")
            create_placeholder_csv(csv_path)

    return train_csv, val_csv, test_csv


def create_placeholder_csv(csv_path: Path):
    """Create a placeholder CSV file for testing."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        'tcr_id': [f'tcr_{i}' for i in range(10)],
        'pmhc_id': [f'pmhc_{i}' for i in range(10)],
        'label': [1, 0] * 5,
        'epitope': [f'epitope_{i % 3}' for i in range(10)],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    print(f"Created placeholder: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Train BioPhysTCR model")

    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')

    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2],
                        help='Evaluation scenario (1=random, 2=unseen epitopes)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device (cuda/cpu)')

    parser.add_argument('--features_dir', type=str, default='data/processed',
                        help='Directory with processed features')
    parser.add_argument('--splits_dir', type=str, default='data/splits',
                        help='Directory with split definitions')
    parser.add_argument('--save_dir', type=str, default='results/checkpoints',
                        help='Directory to save checkpoints')

    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use subset of data for testing')

    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')

    parser.add_argument('--alternating', action='store_true', default=True,
                        help='Use alternating contrastive+binary training')
    parser.add_argument('--combined', action='store_true',
                        help='Use combined loss (not alternating)')

    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    config_path = PROJECT_DIR / args.config
    features_dir = PROJECT_DIR / args.features_dir
    splits_dir = PROJECT_DIR / args.splits_dir
    save_dir = PROJECT_DIR / args.save_dir

    if args.debug:
        print("=" * 50)
        print("DEBUG MODE ENABLED")
        print("=" * 50)
        args.epochs = args.epochs or 10
        args.batch_size = args.batch_size or 8
        args.subset = args.subset or 100

    print(f"\nLoading config from: {config_path}")
    config = load_config(config_path)

    print(f"\nTraining Settings:")
    print(f"  Scenario: {args.scenario} ({'random split' if args.scenario == 1 else 'unseen epitopes'})")
    print(f"  Epochs: {args.epochs or config['training']['epochs']}")
    print(f"  Batch size: {args.batch_size or config['training']['batch_size']}")
    print(f"  Alternating training: {args.alternating and not args.combined}")
    print(f"  Features dir: {features_dir}")
    print(f"  Save dir: {save_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)

    print("\nPreparing datasets...")
    train_csv, val_csv, test_csv = prepare_datasets(
        features_dir, splits_dir, args.scenario, args.subset
    )

    print("\nCreating data loaders...")
    train_loader, val_loader, test_loader, positive_loader = create_data_loaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        features_dir=features_dir,
        batch_size=args.batch_size or config['training']['batch_size'],
        num_workers=config['num_workers'] if not args.debug else 0,
        create_positive_loader=args.alternating and not args.combined
    )

    print("\nCreating BioPhysTCR model...")
    model = create_model(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")

    trainer_config = create_trainer_config(config, args)

    trainer = BioPhysTCRTrainer(model, trainer_config)

    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    print("\n" + "=" * 50)
    print("Starting Training")
    print("=" * 50)

    start_time = time.time()

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        positive_loader=positive_loader if (args.alternating and not args.combined) else None,
        alternating=args.alternating and not args.combined
    )

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed / 60:.1f} minutes")

    print("\n" + "=" * 50)
    print("Final Evaluation")
    print("=" * 50)

    if test_loader:
        print("\nTest set evaluation:")
        test_metrics = trainer.evaluate(test_loader)
        print(print_metrics(test_metrics, prefix="  "))

        results = {
            'scenario': args.scenario,
            'test_metrics': test_metrics,
            'history': history,
            'config': config,
            'elapsed_time': elapsed,
        }

        results_path = save_dir / f"results_scenario{args.scenario}.json"
        with open(results_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
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
