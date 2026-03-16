"""
GARSEF Training Package.

Contains training utilities:
- Loss functions (InfoNCE, Focal, Combined)
- Metrics (AUC, AUPR, MCC, F1)
- Training loop with alternating phases
"""

from .losses import (
    InfoNCELoss,
    ContrastiveLoss,
    FocalLoss,
    BinaryLoss,
    GARSEFLoss,
    LabelSmoothingBCE,
    compute_accuracy,
)

from .metrics import (
    compute_auc,
    compute_aupr,
    compute_mcc,
    compute_f1,
    compute_precision_at_k,
    compute_recall_at_k,
    MetricsCalculator,
    RunningMetrics,
    print_metrics,
    compute_confusion_matrix_stats,
)

from .trainer import (
    TrainerConfig,
    GARSEFTrainer,
    EarlyStopping,
    train_garsef,
)


__all__ = [
    'InfoNCELoss',
    'ContrastiveLoss',
    'FocalLoss',
    'BinaryLoss',
    'GARSEFLoss',
    'LabelSmoothingBCE',
    'compute_accuracy',
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
    'TrainerConfig',
    'GARSEFTrainer',
    'EarlyStopping',
    'train_garsef',
]
