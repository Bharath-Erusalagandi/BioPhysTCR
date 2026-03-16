#!/usr/bin/env python3
"""
FAST Training script for BioPhysTCR model.
Optimized for ~8 hour training window with:
- Mixed Precision (AMP) for ~2x speedup
- Large batch sizes (512-1024)
- OneCycleLR scheduler for faster convergence
- Parallel data loading
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.garsef import GARSEF, GARSEFConfig
from src.utils.data_utils import GARSEFDataset, PositiveOnlyDataset, collate_garsef
from src.training.losses import GARSEFLoss
from src.training.metrics import MetricsCalculator


def train_epoch_amp(model, train_loader, positive_loader, optimizer, scheduler,
                    scaler, criterion, device, epoch, alternating=True, accum_steps=1):
    """Train one epoch with AMP and alternating strategy."""
    model.train()

    total_loss = 0.0
    total_contrastive = 0.0
    total_binary = 0.0
    all_preds = []
    all_labels = []

    # Phase 1: Contrastive learning (if alternating)
    if alternating and positive_loader is not None:
        optimizer.zero_grad()
        for i, batch in enumerate(tqdm(positive_loader, desc=f"Epoch {epoch} Contrastive", leave=False)):
            tcr_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch['tcr'].items()}
            pmhc_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch['pmhc'].items()}

            with autocast():
                outputs = model(tcr_data, pmhc_data)
                loss = criterion.contrastive_only(outputs) / accum_steps

            scaler.scale(loss).backward()
            total_contrastive += loss.item() * accum_steps

            if (i + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

    # Phase 2: Binary classification
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Binary", leave=False)):
        tcr_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['tcr'].items()}
        pmhc_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch['pmhc'].items()}
        labels = batch['label'].to(device)

        with autocast():
            outputs = model(tcr_data, pmhc_data)
            loss = criterion.binary_only(outputs, labels) / accum_steps

        scaler.scale(loss).backward()
        total_binary += loss.item() * accum_steps

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        with torch.no_grad():
            preds = torch.sigmoid(outputs['binding_logits']).squeeze()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = MetricsCalculator.calculate_all(all_labels, all_preds)
    metrics['contrastive_loss'] = total_contrastive / max(len(positive_loader) if positive_loader else 1, 1)
    metrics['binary_loss'] = total_binary / len(train_loader)

    return metrics


@torch.no_grad()
def validate_amp(model, val_loader, criterion, device):
    """Validate with AMP."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        tcr_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch['tcr'].items()}
        pmhc_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch['pmhc'].items()}
        labels = batch['label'].to(device)

        with autocast():
            outputs = model(tcr_data, pmhc_data)
            loss = criterion.binary_only(outputs, labels)

        total_loss += loss.item()

        preds = torch.sigmoid(outputs['binding_logits']).squeeze()
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = MetricsCalculator.calculate_all(all_labels, all_preds)
    metrics['loss'] = total_loss / len(val_loader)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='FAST Train BioPhysTCR model')
    parser.add_argument('--scenario', type=int, choices=[1, 2], required=True,
                        help='Training scenario: 1=random split, 2=unseen epitopes')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size (default: 512 for speed)')
    parser.add_argument('--epochs', type=int, default=40,
                        help='Number of epochs (default: 40 with OneCycleLR)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Max learning rate (default: 3e-4)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='DataLoader workers (default: 4)')
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Gradient accumulation steps (default: 1)')

    args = parser.parse_args()

    print("="*70)
    print(f"BioPhysTCR FAST Training - Scenario {args.scenario}")
    print("="*70)
    print("Optimizations: AMP (FP16) + Large Batch + OneCycleLR")

    # Setup paths
    DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
    SPLITS_DIR = PROJECT_ROOT / 'data' / 'splits'

    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    else:
        OUTPUT_DIR = PROJECT_ROOT / 'results' / f'scenario{args.scenario}'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data splits
    print(f"\n{'='*70}")
    print("Loading Data")
    print("="*70)

    if args.scenario == 1:
        train_file = SPLITS_DIR / 'train.json'
        val_file = SPLITS_DIR / 'val.json'
    else:
        train_file = SPLITS_DIR / 'train_epitope.json'
        val_file = SPLITS_DIR / 'val_epitope.json'

    print(f"Train file: {train_file}")
    print(f"Val file: {val_file}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = GARSEFDataset(str(train_file), DATA_DIR)
    val_dataset = GARSEFDataset(str(val_file), DATA_DIR)
    positive_dataset = PositiveOnlyDataset(train_dataset)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Positive samples: {len(positive_dataset):,}")

    # Create data loaders with workers for parallel loading
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_garsef,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_garsef,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    positive_loader = DataLoader(
        positive_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_garsef,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    print(f"Positive batches: {len(positive_loader):,}")

    # Create model
    print(f"\n{'='*70}")
    print("Creating Model")
    print("="*70)

    model_config = GARSEFConfig(
        esm2_dim=1280,
        sequence_hidden_dim=256,
        sequence_num_layers=2,
        saprot_dim=1280,
        structure_hidden_dim=256,
        structure_num_gnn_layers=3,
        structure_num_attention_heads=8,
        physchem_dim=8,
        physchem_hidden_dim=64,
        physchem_num_layers=2,
        physchem_aggregation='attention',
        fusion_dim=256,
        projection_dim=128,
        use_cross_attention=True,
        dropout=0.2,
        fusion_dropout=0.3,
        temperature=0.07
    )

    model = GARSEF(model_config)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / 1e6:.1f} MB (float32)")

    # Setup training
    print(f"\n{'='*70}")
    print("Configuring Fast Training")
    print("="*70)

    CONTRASTIVE_WEIGHT = 0.5 if args.scenario == 1 else 0.7

    criterion = GARSEFLoss(
        contrastive_weight=CONTRASTIVE_WEIGHT,
        binary_weight=1.0,
        temperature=0.07,
        focal_alpha=0.25,
        focal_gamma=2.0
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # OneCycleLR - trains faster with fewer epochs
    # Account for gradient accumulation in total steps
    steps_per_epoch = (len(train_loader) + len(positive_loader)) // args.accum_steps
    total_steps = args.epochs * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos'
    )

    # Mixed precision scaler
    scaler = GradScaler()

    effective_batch = args.batch_size * args.accum_steps
    print(f"Batch size: {args.batch_size} (effective: {effective_batch})")
    print(f"Gradient accumulation: {args.accum_steps} steps")
    print(f"Epochs: {args.epochs}")
    print(f"Max LR: {args.lr}")
    print(f"Total steps: {total_steps:,}")
    print(f"Mixed Precision: Enabled (FP16)")
    print(f"Scheduler: OneCycleLR")
    print(f"Save directory: {OUTPUT_DIR}")

    # Training loop
    print(f"\n{'='*70}")
    print(f"Starting FAST Training - Scenario {args.scenario}")
    print("="*70)

    best_auc = 0.0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    history = {'train': [], 'val': []}

    start_time = time.time()

    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = train_epoch_amp(
                model, train_loader, positive_loader, optimizer, scheduler,
                scaler, criterion, device, epoch, alternating=True, accum_steps=args.accum_steps
            )

            # Validate
            val_metrics = validate_amp(model, val_loader, criterion, device)

            epoch_time = time.time() - epoch_start

            # Log progress
            print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_metrics['binary_loss']:.4f}, "
                  f"AUC: {train_metrics['auc']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                  f"AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['accuracy']:.4f}")

            history['train'].append(train_metrics)
            history['val'].append(val_metrics)

            # Check for best model
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                best_epoch = epoch
                patience_counter = 0

                # Save best model
                checkpoint_dir = OUTPUT_DIR / 'checkpoints'
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_auc,
                    'val_metrics': val_metrics,
                    'config': model_config
                }, checkpoint_dir / 'best_model.pt')
                print(f"  ★ New best model saved! (AUC: {best_auc:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

            # Estimate remaining time
            elapsed = time.time() - start_time
            epochs_done = epoch
            epochs_left = args.epochs - epoch
            eta = (elapsed / epochs_done) * epochs_left
            print(f"  ETA: {eta/3600:.1f}h remaining")

        total_time = time.time() - start_time

        print(f"\n{'='*70}")
        print("Training Completed!")
        print("="*70)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best AUC: {best_auc:.4f} (epoch {best_epoch})")
        print(f"\nResults saved to: {OUTPUT_DIR}")

        # Save training history
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        history_path = OUTPUT_DIR / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(convert_to_serializable(history), f, indent=2)

        return 0

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        return 1
    except Exception as e:
        print(f"\n\nTraining failed with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
