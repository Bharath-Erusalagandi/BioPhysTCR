"""BioPhysTCR Training Package."""

from .losses import (
    InfoNCELoss,
    ContrastiveLoss,
    FocalLoss,
    BinaryLoss,
    BioPhysTCRLoss,
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
    BioPhysTCRTrainer,
    EarlyStopping,
    train_biophystcr,
)


__all__ = [
    'InfoNCELoss',
    'ContrastiveLoss',
    'FocalLoss',
    'BinaryLoss',
    'BioPhysTCRLoss',
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
    'BioPhysTCRTrainer',
    'EarlyStopping',
    'train_biophystcr',
]
