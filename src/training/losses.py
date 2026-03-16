"""
Loss functions for GARSEF training.

Implements:
1. InfoNCE contrastive loss (CLIP-style)
2. Focal loss for binary classification (handles class imbalance)
3. Combined loss with configurable weights

Reference:
- Contrastive learning approaches for TCR-pMHC binding
- Focal Loss (Lin et al., 2017): Addressing class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss.

    Used to align TCR and pMHC representations in a shared embedding space.
    Maximizes similarity of matching pairs while minimizing similarity
    of non-matching pairs within a batch.

    This is the same contrastive loss used in CLIP and other multi-modal models.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature scaling for softmax
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        tcr_emb: torch.Tensor,
        pmhc_emb: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            tcr_emb: Normalized TCR embeddings [batch_size, embed_dim]
            pmhc_emb: Normalized pMHC embeddings [batch_size, embed_dim]
            temperature: Optional temperature override

        Returns:
            Scalar loss value
        """
        if temperature is None:
            temperature = self.temperature

        logits = torch.matmul(tcr_emb, pmhc_emb.T) / temperature

        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)

        loss_tcr = F.cross_entropy(logits, labels)
        loss_pmhc = F.cross_entropy(logits.T, labels)

        return (loss_tcr + loss_pmhc) / 2


class ContrastiveLoss(nn.Module):
    """
    Alternative contrastive loss with hard negative mining.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.5
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(
        self,
        tcr_emb: torch.Tensor,
        pmhc_emb: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss with explicit labels.

        Args:
            tcr_emb: TCR embeddings [batch_size, embed_dim]
            pmhc_emb: pMHC embeddings [batch_size, embed_dim]
            labels: Binary labels [batch_size] (1 = binding, 0 = non-binding)

        Returns:
            Scalar loss value
        """
        similarity = F.cosine_similarity(tcr_emb, pmhc_emb)

        pos_mask = labels == 1
        pos_loss = torch.mean((1 - similarity[pos_mask]) ** 2) if pos_mask.any() else 0

        neg_mask = labels == 0
        neg_sim = similarity[neg_mask]
        neg_loss = torch.mean(F.relu(neg_sim - self.margin) ** 2) if neg_mask.any() else 0

        return pos_loss + neg_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Focuses learning on hard examples by down-weighting
    well-classified examples.

    From: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    Effective for TCR-pMHC binding prediction.

    L_focal = -alpha * (1 - p_t)^gamma * log(p_t)

    where p_t = p if y=1 else 1-p
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Weighting factor for positive class (use alpha/(1+alpha) ratio)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'none', 'mean', or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            inputs: Predicted logits [batch_size] or [batch_size, 1]
            targets: Binary labels [batch_size] or [batch_size, 1]

        Returns:
            Focal loss value
        """
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()

        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BinaryLoss(nn.Module):
    """
    Binary classification loss with optional class weighting.

    Supports both standard BCE and Focal loss.
    """

    def __init__(
        self,
        loss_type: str = 'focal',
        pos_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            loss_type: 'bce' or 'focal'
            pos_weight: Weight for positive class (for BCE)
            focal_alpha: Alpha parameter for Focal loss
            focal_gamma: Gamma parameter for Focal loss
        """
        super().__init__()
        self.loss_type = loss_type

        if loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        elif loss_type == 'focal':
            self.loss_fn = FocalLoss(
                alpha=focal_alpha,
                gamma=focal_gamma
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits [batch_size, 1] or [batch_size]
            labels: Binary labels [batch_size]

        Returns:
            Binary classification loss
        """
        return self.loss_fn(logits.squeeze(-1), labels.float())


class GARSEFLoss(nn.Module):
    """
    Combined loss for GARSEF training.

    L_total = lambda_1 * L_contrastive + lambda_2 * L_binary

    Using an alternating training strategy, this loss
    can be used in different modes:
    - 'contrastive': Only contrastive loss (for positive pairs)
    - 'binary': Only binary loss (for all pairs with labels)
    - 'combined': Both losses together
    """

    def __init__(
        self,
        contrastive_weight: float = 0.5,
        binary_weight: float = 1.0,
        temperature: float = 0.07,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        """
        Args:
            contrastive_weight: Weight for contrastive loss (lambda_1)
            binary_weight: Weight for binary loss (lambda_2)
            temperature: Temperature for InfoNCE
            focal_alpha: Alpha for Focal loss
            focal_gamma: Gamma for Focal loss
        """
        super().__init__()

        self.contrastive_weight = contrastive_weight
        self.binary_weight = binary_weight

        self.contrastive_loss = InfoNCELoss(temperature)
        self.binary_loss = FocalLoss(focal_alpha, focal_gamma)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mode: str = 'combined'
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GARSEF loss.

        Args:
            outputs: Model outputs containing:
                - tcr_proj: TCR contrastive projection
                - pmhc_proj: pMHC contrastive projection
                - binding_logits: Binary prediction logits
            labels: Binary binding labels [batch_size]
            mode: 'contrastive', 'binary', or 'combined'

        Returns:
            Dict with 'loss' (total), 'contrastive_loss', 'binary_loss'
        """
        loss_dict = {}

        if mode in ['contrastive', 'combined']:
            tcr_proj = outputs['tcr_proj']
            pmhc_proj = outputs['pmhc_proj']

            contrastive = self.contrastive_loss(tcr_proj, pmhc_proj)
            loss_dict['contrastive_loss'] = contrastive

        if mode in ['binary', 'combined']:
            logits = outputs['binding_logits']
            binary = self.binary_loss(logits, labels)
            loss_dict['binary_loss'] = binary

        if mode == 'combined':
            loss_dict['loss'] = (
                self.contrastive_weight * loss_dict['contrastive_loss'] +
                self.binary_weight * loss_dict['binary_loss']
            )
        elif mode == 'contrastive':
            loss_dict['loss'] = loss_dict['contrastive_loss']
        else:
            loss_dict['loss'] = loss_dict['binary_loss']

        return loss_dict

    def contrastive_only(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute only contrastive loss (for contrastive training phase)."""
        return self.contrastive_loss(outputs['tcr_proj'], outputs['pmhc_proj'])

    def binary_only(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute only binary loss (for binary training phase)."""
        return self.binary_loss(outputs['binding_logits'], labels)


class LabelSmoothingBCE(nn.Module):
    """
    Binary Cross Entropy with label smoothing.

    Helps prevent overconfident predictions and improves generalization.
    """

    def __init__(self, smoothing: float = 0.1):
        """
        Args:
            smoothing: Label smoothing factor (0 = no smoothing)
        """
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits [batch_size]
            labels: Binary labels [batch_size]

        Returns:
            Smoothed BCE loss
        """
        labels = labels.float()

        labels = labels * (1 - self.smoothing) + 0.5 * self.smoothing

        return F.binary_cross_entropy_with_logits(logits, labels)


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Compute binary classification accuracy.

    Args:
        logits: Predicted logits [batch_size]
        labels: Binary labels [batch_size]
        threshold: Classification threshold

    Returns:
        Accuracy as a tensor
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    return (preds == labels.float()).float().mean()


__all__ = [
    'InfoNCELoss',
    'ContrastiveLoss',
    'FocalLoss',
    'BinaryLoss',
    'GARSEFLoss',
    'LabelSmoothingBCE',
    'compute_accuracy',
]
