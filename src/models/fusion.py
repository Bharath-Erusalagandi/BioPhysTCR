"""
Multi-modal Fusion module for GARSEF.

Combines sequence, structure, and physicochemical features
into unified TCR and pMHC representations.

Architecture follows GARSEF plan:
- Concatenation-based fusion
- LayerNorm for stability
- Separate projection heads for contrastive and binary tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class PhysicochemicalEncoder(nn.Module):
    """
    MLP encoder for physicochemical features.

    Processes 8-dimensional per-residue features:
    - Electrostatic potential
    - SASA (absolute and relative)
    - B-factor
    - Hydrophobicity
    - Charge
    - H-bond donor/acceptor capacity
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        aggregation: str = 'mean'
    ):
        """
        Args:
            input_dim: Number of physicochemical features per residue
            hidden_dim: Output embedding dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
            aggregation: How to aggregate residue features ('mean', 'max', 'attention')
        """
        super().__init__()

        self.aggregation = aggregation

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Physicochemical features [batch_size, num_residues, input_dim]
               or [batch_size, input_dim] if already aggregated
            mask: Valid residue mask [batch_size, num_residues]

        Returns:
            Encoded features [batch_size, hidden_dim]
        """
        if x.dim() == 2:
            return self.encoder(x)

        encoded = self.encoder(x)

        if self.aggregation == 'mean':
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                output = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                output = encoded.mean(dim=1)

        elif self.aggregation == 'max':
            if mask is not None:
                encoded = encoded.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            output, _ = encoded.max(dim=1)

        elif self.aggregation == 'attention':
            attn_scores = self.attention(encoded).squeeze(-1)

            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=1)
            output = (encoded * attn_weights.unsqueeze(-1)).sum(dim=1)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return output


class MultiModalFusion(nn.Module):
    """
    Fuses sequence, structure, and physicochemical modalities.

    GARSEF fusion architecture:
    1. Concatenate all modality embeddings
    2. Apply LayerNorm for stability
    3. MLP projection to unified dimension
    """

    def __init__(
        self,
        sequence_dim: int = 256,
        structure_dim: int = 256,
        physchem_dim: int = 64,
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()

        self.input_dim = sequence_dim + structure_dim + physchem_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        sequence_emb: torch.Tensor,
        structure_emb: torch.Tensor,
        physchem_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sequence_emb: [batch_size, sequence_dim]
            structure_emb: [batch_size, structure_dim]
            physchem_emb: [batch_size, physchem_dim]

        Returns:
            Fused embedding [batch_size, output_dim]
        """
        combined = torch.cat([sequence_emb, structure_emb, physchem_emb], dim=-1)

        return self.fusion(combined)


class ContrastiveHead(nn.Module):
    """
    Projection head for contrastive learning (InfoNCE loss).

    Projects TCR and pMHC embeddings to a shared space
    where cosine similarity indicates binding affinity.
    """

    def __init__(
        self,
        input_dim: int = 256,
        projection_dim: int = 128,
        normalize: bool = True
    ):
        super().__init__()

        self.normalize = normalize

        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embedding [batch_size, input_dim]

        Returns:
            Projected embedding [batch_size, projection_dim]
        """
        projected = self.projection(x)

        if self.normalize:
            projected = F.normalize(projected, p=2, dim=-1)

        return projected


class BinaryHead(nn.Module):
    """
    Binary classification head for binding prediction.

    Takes concatenated TCR and pMHC embeddings and predicts
    binding probability.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))

        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        tcr_emb: torch.Tensor,
        pmhc_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            tcr_emb: TCR embedding [batch_size, tcr_dim]
            pmhc_emb: pMHC embedding [batch_size, pmhc_dim]

        Returns:
            Binding logits [batch_size, 1]
        """
        combined = torch.cat([tcr_emb, pmhc_emb], dim=-1)

        return self.classifier(combined)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention based fusion between TCR and pMHC.

    Enables learning of interaction-aware representations
    where each modality attends to the other.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()

        self.tcr_to_pmhc = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.pmhc_to_tcr = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.tcr_norm = nn.LayerNorm(embed_dim)
        self.pmhc_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        tcr_emb: torch.Tensor,
        pmhc_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tcr_emb: TCR embedding [batch_size, embed_dim]
            pmhc_emb: pMHC embedding [batch_size, embed_dim]

        Returns:
            Enhanced TCR embedding [batch_size, embed_dim]
            Enhanced pMHC embedding [batch_size, embed_dim]
        """
        tcr_q = tcr_emb.unsqueeze(1)
        pmhc_q = pmhc_emb.unsqueeze(1)

        tcr_enhanced, _ = self.tcr_to_pmhc(
            query=tcr_q, key=pmhc_q, value=pmhc_q
        )
        pmhc_enhanced, _ = self.pmhc_to_tcr(
            query=pmhc_q, key=tcr_q, value=tcr_q
        )

        tcr_out = self.tcr_norm(tcr_emb + tcr_enhanced.squeeze(1))
        pmhc_out = self.pmhc_norm(pmhc_emb + pmhc_enhanced.squeeze(1))

        return tcr_out, pmhc_out


class GARSEFFusion(nn.Module):
    """
    Complete GARSEF fusion module.

    Combines:
    1. Multi-modal fusion for TCR and pMHC separately
    2. Optional cross-attention between TCR and pMHC
    3. Contrastive and binary prediction heads
    """

    def __init__(
        self,
        sequence_dim: int = 256,
        structure_dim: int = 256,
        physchem_dim: int = 64,
        fusion_dim: int = 256,
        projection_dim: int = 128,
        dropout: float = 0.3,
        use_cross_attention: bool = True
    ):
        super().__init__()

        self.use_cross_attention = use_cross_attention

        self.tcr_fusion = MultiModalFusion(
            sequence_dim, structure_dim, physchem_dim, fusion_dim, dropout
        )

        self.pmhc_fusion = MultiModalFusion(
            sequence_dim, structure_dim, physchem_dim, fusion_dim, dropout
        )

        if use_cross_attention:
            self.cross_attention = CrossAttentionFusion(
                fusion_dim, num_heads=8, dropout=dropout
            )

        self.tcr_contrastive = ContrastiveHead(fusion_dim, projection_dim)
        self.pmhc_contrastive = ContrastiveHead(fusion_dim, projection_dim)

        self.binary_head = BinaryHead(
            fusion_dim * 2, hidden_dim=128, num_layers=2, dropout=dropout
        )

    def forward(
        self,
        tcr_sequence: torch.Tensor,
        tcr_structure: torch.Tensor,
        tcr_physchem: torch.Tensor,
        pmhc_sequence: torch.Tensor,
        pmhc_structure: torch.Tensor,
        pmhc_physchem: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tcr_sequence: TCR sequence embedding [batch, seq_dim]
            tcr_structure: TCR structure embedding [batch, struct_dim]
            tcr_physchem: TCR physicochemical features [batch, phys_dim]
            pmhc_sequence: pMHC sequence embedding [batch, seq_dim]
            pmhc_structure: pMHC structure embedding [batch, struct_dim]
            pmhc_physchem: pMHC physicochemical features [batch, phys_dim]

        Returns:
            Dict with:
            - tcr_emb: TCR fused embedding
            - pmhc_emb: pMHC fused embedding
            - tcr_proj: TCR contrastive projection
            - pmhc_proj: pMHC contrastive projection
            - binding_logits: Binary binding prediction
        """
        tcr_emb = self.tcr_fusion(tcr_sequence, tcr_structure, tcr_physchem)
        pmhc_emb = self.pmhc_fusion(pmhc_sequence, pmhc_structure, pmhc_physchem)

        if self.use_cross_attention:
            tcr_emb, pmhc_emb = self.cross_attention(tcr_emb, pmhc_emb)

        tcr_proj = self.tcr_contrastive(tcr_emb)
        pmhc_proj = self.pmhc_contrastive(pmhc_emb)

        binding_logits = self.binary_head(tcr_emb, pmhc_emb)

        return {
            'tcr_emb': tcr_emb,
            'pmhc_emb': pmhc_emb,
            'tcr_proj': tcr_proj,
            'pmhc_proj': pmhc_proj,
            'binding_logits': binding_logits
        }


__all__ = [
    'PhysicochemicalEncoder',
    'MultiModalFusion',
    'ContrastiveHead',
    'BinaryHead',
    'CrossAttentionFusion',
    'GARSEFFusion',
]
