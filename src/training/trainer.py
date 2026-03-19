"""Training loop for BioPhysTCR model."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .losses import BioPhysTCRLoss
from .metrics import RunningMetrics, MetricsCalculator, print_metrics


@dataclass
class TrainerConfig:
    """Configuration for BioPhysTCR trainer."""

    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32

    contrastive_weight: float = 0.5
    binary_weight: float = 1.0
    temperature: float = 0.07
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    optimizer: str = 'adamw'
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    patience: int = 15
    min_delta: float = 0.001

    save_dir: str = 'checkpoints'
    save_every: int = 10

    device: str = 'cuda'
    num_workers: int = 4

    use_wandb: bool = False
    project_name: str = 'biophystcr'


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        verbose: bool = True
    ):
        """Args:"""
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        """Check if training should stop."""
        if self.mode == 'min':
            score = -score

        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop


class BioPhysTCRTrainer:
    """Trainer for BioPhysTCR model."""

    def __init__(
        self,
        model: nn.Module,
        config: Optional[TrainerConfig] = None
    ):
        """Args:"""
        if config is None:
            config = TrainerConfig()

        self.model = model
        self.config = config

        self.device = torch.device(
            config.device if torch.cuda.is_available() else 'cpu'
        )
        self.model.to(self.device)

        self.criterion = BioPhysTCRLoss(
            contrastive_weight=config.contrastive_weight,
            binary_weight=config.binary_weight,
            temperature=config.temperature,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma
        )

        self.optimizer = self._create_optimizer()

        self.scheduler = None

        self.metrics_calculator = MetricsCalculator()

        self.early_stopper = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode='max'
        )

        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': [],
            'train_aupr': [],
            'val_aupr': [],
        }

        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer.lower() == 'adam':
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler."""
        if self.config.scheduler.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=1e-6
            )
        elif self.config.scheduler.lower() == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                config=vars(self.config)
            )
        except ImportError:
            print("wandb not installed. Skipping W&B logging.")
            self.config.use_wandb = False

    def train_contrastive_epoch(
        self,
        positive_loader: DataLoader
    ) -> Dict[str, float]:
        """Train one epoch with contrastive learning (positive pairs only)."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(positive_loader, desc="Contrastive", leave=False):
            self.optimizer.zero_grad()

            tcr_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch['tcr'].items()}
            pmhc_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch['pmhc'].items()}

            outputs = self.model(tcr_data, pmhc_data)

            loss = self.criterion.contrastive_only(outputs)

            loss.backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {'contrastive_loss': total_loss / max(num_batches, 1)}

    def train_binary_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """Train one epoch with binary classification (all pairs)."""
        self.model.train()
        running_metrics = RunningMetrics()

        for batch in tqdm(train_loader, desc="Binary", leave=False):
            self.optimizer.zero_grad()

            tcr_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch['tcr'].items()}
            pmhc_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch['pmhc'].items()}
            labels = batch['label'].to(self.device)

            outputs = self.model(tcr_data, pmhc_data)

            loss = self.criterion.binary_only(outputs, labels)

            loss.backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()

            preds = torch.sigmoid(outputs['binding_logits']).squeeze()
            running_metrics.update(preds, labels, loss.item())

        return running_metrics.compute()

    def train_combined_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """Train one epoch with combined loss (contrastive + binary)."""
        self.model.train()
        running_metrics = RunningMetrics()
        total_contrastive = 0.0
        total_binary = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()

            tcr_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch['tcr'].items()}
            pmhc_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch['pmhc'].items()}
            labels = batch['label'].to(self.device)

            outputs = self.model(tcr_data, pmhc_data)

            loss_dict = self.criterion(outputs, labels, mode='combined')
            loss = loss_dict['loss']

            loss.backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.optimizer.step()

            preds = torch.sigmoid(outputs['binding_logits']).squeeze()
            running_metrics.update(preds, labels, loss.item())

            total_contrastive += loss_dict.get('contrastive_loss', 0)
            total_binary += loss_dict.get('binary_loss', 0)
            num_batches += 1

        metrics = running_metrics.compute()
        metrics['contrastive_loss'] = total_contrastive / max(num_batches, 1)
        metrics['binary_loss'] = total_binary / max(num_batches, 1)

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        running_metrics = RunningMetrics()

        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            tcr_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch['tcr'].items()}
            pmhc_data = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch['pmhc'].items()}
            labels = batch['label'].to(self.device)

            outputs = self.model(tcr_data, pmhc_data)

            loss_dict = self.criterion(outputs, labels, mode='combined')

            preds = torch.sigmoid(outputs['binding_logits']).squeeze()
            running_metrics.update(preds, labels, loss_dict['loss'].item())

        return running_metrics.compute()

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        positive_loader: Optional[DataLoader] = None,
        alternating: bool = True
    ) -> Dict[str, List[float]]:
        """Full training loop."""
        num_training_steps = len(train_loader) * self.config.epochs
        self._create_scheduler(num_training_steps)

        best_val_auc = 0.0
        best_epoch = 0

        print(f"Starting training on {self.device}")
        print(f"Train size: {len(train_loader.dataset)}")
        print(f"Val size: {len(val_loader.dataset)}")
        print(f"Alternating training: {alternating}")
        print("-" * 50)

        for epoch in range(self.config.epochs):
            start_time = time.time()

            if alternating and positive_loader is not None:
                contrastive_metrics = self.train_contrastive_epoch(positive_loader)

                binary_metrics = self.train_binary_epoch(train_loader)

                train_metrics = {**contrastive_metrics, **binary_metrics}
            else:
                train_metrics = self.train_combined_epoch(train_loader)

            val_metrics = self.evaluate(val_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['auc'])
                else:
                    self.scheduler.step()

            self.history['train_loss'].append(train_metrics.get('loss', 0))
            self.history['val_loss'].append(val_metrics.get('loss', 0))
            self.history['train_auc'].append(train_metrics.get('auc', 0))
            self.history['val_auc'].append(val_metrics.get('auc', 0))
            self.history['train_aupr'].append(train_metrics.get('aupr', 0))
            self.history['val_aupr'].append(val_metrics.get('aupr', 0))

            elapsed = time.time() - start_time
            print(f"\nEpoch {epoch + 1}/{self.config.epochs} ({elapsed:.1f}s)")
            print(f"  Train - loss: {train_metrics.get('loss', 0):.4f}, "
                  f"AUC: {train_metrics.get('auc', 0):.4f}, "
                  f"AUPR: {train_metrics.get('aupr', 0):.4f}")
            print(f"  Val   - loss: {val_metrics.get('loss', 0):.4f}, "
                  f"AUC: {val_metrics.get('auc', 0):.4f}, "
                  f"AUPR: {val_metrics.get('aupr', 0):.4f}")

            if self.config.use_wandb and self.wandb_run:
                import wandb
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics.get('loss', 0),
                    'val_loss': val_metrics.get('loss', 0),
                    'train_auc': train_metrics.get('auc', 0),
                    'val_auc': val_metrics.get('auc', 0),
                    'train_aupr': train_metrics.get('aupr', 0),
                    'val_aupr': val_metrics.get('aupr', 0),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })

            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                best_epoch = epoch + 1
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
                print(f"  New best model! AUC: {best_val_auc:.4f}")

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt', epoch, val_metrics)

            if self.early_stopper(val_metrics['auc'], epoch):
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print("-" * 50)
        print(f"Training complete! Best AUC: {best_val_auc:.4f} at epoch {best_epoch}")

        self.load_checkpoint('best_model.pt')

        return self.history

    def save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Optional[Dict] = None
    ):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        path = self.save_dir / filename
        torch.save(checkpoint, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.save_dir / filename

        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")


def train_biophystcr(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainerConfig] = None,
    positive_loader: Optional[DataLoader] = None
) -> Tuple[nn.Module, Dict]:
    """Convenience function to train BioPhysTCR model."""
    trainer = BioPhysTCRTrainer(model, config)

    history = trainer.train(
        train_loader,
        val_loader,
        positive_loader,
        alternating=(positive_loader is not None)
    )

    return model, history


__all__ = [
    'TrainerConfig',
    'BioPhysTCRTrainer',
    'EarlyStopping',
    'train_biophystcr',
]
