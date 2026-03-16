"""
Evaluation metrics for GARSEF model.

Implements standard metrics for TCR-pMHC binding prediction:
- AUC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)
- MCC (Matthews Correlation Coefficient)
- Macro-F1 Score
- Precision@k, Recall@k (recommender system metrics)

These metrics follow standard evaluation protocols for TCR-pMHC binding prediction.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)
from collections import defaultdict


def compute_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Area Under ROC Curve.

    Args:
        y_true: Binary labels [N]
        y_pred: Predicted probabilities [N]

    Returns:
        AUC score (0.5 if undefined)
    """
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError:
        return 0.5


def compute_aupr(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    """
    Compute Area Under Precision-Recall Curve.

    Better than AUC for imbalanced datasets.

    Args:
        y_true: Binary labels [N]
        y_pred: Predicted probabilities [N]

    Returns:
        AUPR score
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        return auc(recall, precision)
    except ValueError:
        return 0.0


def compute_mcc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> float:
    """
    Compute Matthews Correlation Coefficient.

    MCC is balanced even for imbalanced datasets.
    Range: [-1, 1], 0 = random, 1 = perfect, -1 = inverse

    Args:
        y_true: Binary labels [N]
        y_pred: Predicted probabilities [N]
        threshold: Classification threshold

    Returns:
        MCC score
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return matthews_corrcoef(y_true, y_pred_binary)


def compute_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    average: str = 'macro'
) -> float:
    """
    Compute F1 score.

    Args:
        y_true: Binary labels [N]
        y_pred: Predicted probabilities [N]
        threshold: Classification threshold
        average: 'macro', 'micro', 'weighted', or 'binary'

    Returns:
        F1 score
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    return f1_score(y_true, y_pred_binary, average=average, zero_division=0)


def compute_precision_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 1
) -> float:
    """
    Compute Precision@k (recommender system metric).

    For each epitope, how many of top-k predictions are correct?

    Args:
        y_true: Binary labels [N]
        y_pred: Predicted probabilities [N]
        k: Number of top predictions to consider

    Returns:
        Precision@k
    """
    top_k_indices = np.argsort(y_pred)[-k:]

    correct = y_true[top_k_indices].sum()
    return correct / k


def compute_recall_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 1
) -> float:
    """
    Compute Recall@k (recommender system metric).

    Of all positive samples, how many are in top-k predictions?

    Args:
        y_true: Binary labels [N]
        y_pred: Predicted probabilities [N]
        k: Number of top predictions to consider

    Returns:
        Recall@k (0 if no positive samples)
    """
    top_k_indices = np.argsort(y_pred)[-k:]

    total_positive = y_true.sum()
    if total_positive == 0:
        return 0.0

    correct = y_true[top_k_indices].sum()
    return correct / total_positive


class MetricsCalculator:
    """
    Calculate all evaluation metrics for GARSEF.

    Supports both overall and per-epitope metrics.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Classification threshold for binary metrics
        """
        self.threshold = threshold

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all standard metrics.

        Args:
            y_true: Binary labels [N]
            y_pred: Predicted probabilities [N]

        Returns:
            Dict with all metrics
        """
        metrics = {
            'auc': compute_auc(y_true, y_pred),
            'aupr': compute_aupr(y_true, y_pred),
            'mcc': compute_mcc(y_true, y_pred, self.threshold),
            'macro_f1': compute_f1(y_true, y_pred, self.threshold, 'macro'),
            'precision': precision_score(
                y_true, (y_pred >= self.threshold).astype(int), zero_division=0
            ),
            'recall': recall_score(
                y_true, (y_pred >= self.threshold).astype(int), zero_division=0
            ),
            'accuracy': accuracy_score(
                y_true, (y_pred >= self.threshold).astype(int)
            ),
            'precision@1': compute_precision_at_k(y_true, y_pred, 1),
            'precision@3': compute_precision_at_k(y_true, y_pred, 3),
            'recall@1': compute_recall_at_k(y_true, y_pred, 1),
            'recall@3': compute_recall_at_k(y_true, y_pred, 3),
        }

        return metrics

    def compute_per_epitope(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        epitope_ids: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics per epitope.

        Following TRAP's evaluation protocol.

        Args:
            y_true: Binary labels [N]
            y_pred: Predicted probabilities [N]
            epitope_ids: Epitope identifier for each sample [N]

        Returns:
            Dict mapping epitope_id -> metrics dict
        """
        epitope_groups = defaultdict(lambda: {'true': [], 'pred': []})

        for true, pred, epi in zip(y_true, y_pred, epitope_ids):
            epitope_groups[epi]['true'].append(true)
            epitope_groups[epi]['pred'].append(pred)

        per_epitope_metrics = {}

        for epi, data in epitope_groups.items():
            y_true_epi = np.array(data['true'])
            y_pred_epi = np.array(data['pred'])

            if len(np.unique(y_true_epi)) < 2:
                continue

            per_epitope_metrics[epi] = {
                'auc': compute_auc(y_true_epi, y_pred_epi),
                'aupr': compute_aupr(y_true_epi, y_pred_epi),
                'precision@1': compute_precision_at_k(y_true_epi, y_pred_epi, 1),
                'recall@1': compute_recall_at_k(y_true_epi, y_pred_epi, 1),
                'n_samples': len(y_true_epi),
                'n_positive': y_true_epi.sum(),
            }

        return per_epitope_metrics

    def aggregate_per_epitope(
        self,
        per_epitope_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Aggregate per-epitope metrics to overall metrics.

        Args:
            per_epitope_metrics: Output from compute_per_epitope

        Returns:
            Aggregated metrics (mean and std)
        """
        if not per_epitope_metrics:
            return {}

        aggregated = {}

        metric_names = ['auc', 'aupr', 'precision@1', 'recall@1']

        for metric in metric_names:
            values = [m[metric] for m in per_epitope_metrics.values()]
            aggregated[f'epitope_{metric}_mean'] = np.mean(values)
            aggregated[f'epitope_{metric}_std'] = np.std(values)

        return aggregated


class RunningMetrics:
    """
    Track metrics during training across batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all accumulated values."""
        self.predictions = []
        self.labels = []
        self.losses = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        loss: Optional[float] = None
    ):
        """
        Update with batch predictions.

        Args:
            predictions: Model predictions [batch_size]
            labels: True labels [batch_size]
            loss: Optional batch loss value
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        self.predictions.extend(predictions.flatten())
        self.labels.extend(labels.flatten())

        if loss is not None:
            self.losses.append(loss)

    def compute(self) -> Dict[str, float]:
        """
        Compute metrics from accumulated predictions.

        Returns:
            Dict with all metrics
        """
        y_true = np.array(self.labels)
        y_pred = np.array(self.predictions)

        calculator = MetricsCalculator()
        metrics = calculator.compute_all(y_true, y_pred)

        if self.losses:
            metrics['loss'] = np.mean(self.losses)

        return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """
    Format metrics for printing.

    Args:
        metrics: Dict of metric name -> value
        prefix: Optional prefix for each line

    Returns:
        Formatted string
    """
    lines = []

    priority = ['loss', 'auc', 'aupr', 'mcc', 'macro_f1']

    for key in priority:
        if key in metrics:
            lines.append(f"{prefix}{key}: {metrics[key]:.4f}")

    for key, value in sorted(metrics.items()):
        if key not in priority:
            lines.append(f"{prefix}{key}: {value:.4f}")

    return '\n'.join(lines)


def compute_confusion_matrix_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, int]:
    """
    Compute confusion matrix statistics.

    Args:
        y_true: Binary labels [N]
        y_pred: Predicted probabilities [N]
        threshold: Classification threshold

    Returns:
        Dict with TP, TN, FP, FN counts
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    return {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
    }


__all__ = [
    'compute_auc',
    'compute_aupr',
    'compute_mcc',
    'compute_f1',
    'compute_precision_at_k',
    'compute_recall_at_k',
    'MetricsCalculator',
    'RunningMetrics',
    'print_metrics',
    'compute_confusion_matrix_stats',
]
