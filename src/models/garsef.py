"""
GARSEF: Graph-Augmented Residue-level Structure-Enhanced Framework

Main model combining:
- SAGE's GNN architecture for structural representation
- Contrastive learning for robust generalization
- Novel physicochemical features for binding specificity

This model predicts TCR-pMHC binding by integrating three modalities:
1. Sequence features (ESM2 embeddings)
2. Structural features (GraphSAGE on interface graphs)
3. Physicochemical features (APBS, SASA, B-factor, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from .gnn_encoder import GNNEncoder, DualGNNEncoder
from .sequence_encoder import (
    SequenceEncoder,
    SequenceEncoderWithAttention,
    TCRSequenceEncoder,
    pMHCSequenceEncoder
)
from .fusion import (
    PhysicochemicalEncoder,
    MultiModalFusion,
    ContrastiveHead,
    BinaryHead,
    CrossAttentionFusion,
    GARSEFFusion
)


@dataclass
class GARSEFConfig:
    """Configuration for GARSEF model."""

    esm2_dim: int = 1280
    sequence_hidden_dim: int = 256
    sequence_num_layers: int = 2

    saprot_dim: int = 446
    structure_hidden_dim: int = 256
    structure_num_gnn_layers: int = 3
    structure_num_attention_heads: int = 8

    physchem_dim: int = 8
    physchem_hidden_dim: int = 64
    physchem_num_layers: int = 2
    physchem_aggregation: str = 'attention'

    fusion_dim: int = 256
    projection_dim: int = 128
    use_cross_attention: bool = True

    dropout: float = 0.2
    fusion_dropout: float = 0.3

    temperature: float = 0.07


class GARSEF(nn.Module):
    """
    GARSEF: Graph-Augmented Residue-level Structure-Enhanced Framework

    Architecture:
    ============

    TCR Input:
        Sequence (ESM2) ─────────┐
        Structure (SaProt/GNN) ──┼──> TCR Fusion ──> TCR Embedding
        Physicochemical ─────────┘

    pMHC Input:
        Sequence (ESM2) ─────────┐
        Structure (SaProt/GNN) ──┼──> pMHC Fusion ──> pMHC Embedding
        Physicochemical ─────────┘

    Outputs:
        TCR Emb + pMHC Emb ──> Cross-Attention ──> Contrastive Head (InfoNCE)
                                              └──> Binary Head (BCE/Focal)
    """

    def __init__(self, config: Optional[GARSEFConfig] = None):
        super().__init__()

        if config is None:
            config = GARSEFConfig()

        self.config = config

        self.tcr_sequence_encoder = TCRSequenceEncoder(
            input_dim=config.esm2_dim,
            hidden_dim=config.sequence_hidden_dim,
            num_heads=8,
            num_layers=config.sequence_num_layers,
            dropout=config.dropout,
            share_encoder=True
        )

        self.pmhc_sequence_encoder = pMHCSequenceEncoder(
            input_dim=config.esm2_dim,
            hidden_dim=config.sequence_hidden_dim,
            num_heads=8,
            num_layers=config.sequence_num_layers,
            dropout=config.dropout,
            include_mhc=False
        )

        self.tcr_structure_encoder = GNNEncoder(
            input_dim=config.saprot_dim,
            hidden_dim=config.structure_hidden_dim,
            num_gnn_layers=config.structure_num_gnn_layers,
            num_attention_heads=config.structure_num_attention_heads,
            dropout=config.dropout
        )

        self.pmhc_structure_encoder = GNNEncoder(
            input_dim=config.saprot_dim,
            hidden_dim=config.structure_hidden_dim,
            num_gnn_layers=config.structure_num_gnn_layers,
            num_attention_heads=config.structure_num_attention_heads,
            dropout=config.dropout
        )

        self.tcr_physchem_encoder = PhysicochemicalEncoder(
            input_dim=config.physchem_dim,
            hidden_dim=config.physchem_hidden_dim,
            num_layers=config.physchem_num_layers,
            dropout=config.dropout,
            aggregation=config.physchem_aggregation
        )

        self.pmhc_physchem_encoder = PhysicochemicalEncoder(
            input_dim=config.physchem_dim,
            hidden_dim=config.physchem_hidden_dim,
            num_layers=config.physchem_num_layers,
            dropout=config.dropout,
            aggregation=config.physchem_aggregation
        )

        self.fusion = GARSEFFusion(
            sequence_dim=config.sequence_hidden_dim,
            structure_dim=config.structure_hidden_dim,
            physchem_dim=config.physchem_hidden_dim,
            fusion_dim=config.fusion_dim,
            projection_dim=config.projection_dim,
            dropout=config.fusion_dropout,
            use_cross_attention=config.use_cross_attention
        )

        self.temperature = nn.Parameter(torch.tensor(config.temperature))

    def encode_tcr(
        self,
        sequence_emb: torch.Tensor,
        structure_x: torch.Tensor,
        structure_edge_index: torch.Tensor,
        structure_batch: torch.Tensor,
        physchem_features: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
        physchem_mask: Optional[torch.Tensor] = None,
        alpha_emb: Optional[torch.Tensor] = None,
        beta_emb: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode TCR into multi-modal representation.

        Returns:
            sequence_emb: [batch_size, sequence_hidden_dim]
            structure_emb: [batch_size, structure_hidden_dim]
            physchem_emb: [batch_size, physchem_hidden_dim]
        """
        if alpha_emb is not None or beta_emb is not None:
            seq_emb = self.tcr_sequence_encoder(
                alpha_emb=alpha_emb,
                beta_emb=beta_emb
            )
        else:
            seq_emb, _ = self.tcr_sequence_encoder.encoder(sequence_emb)
            if hasattr(self.tcr_sequence_encoder, 'fusion'):
                seq_emb = self.tcr_sequence_encoder.fusion(
                    torch.cat([seq_emb, seq_emb], dim=-1)
                )

        struct_emb, _ = self.tcr_structure_encoder(
            structure_x, structure_edge_index, structure_batch
        )

        phys_emb = self.tcr_physchem_encoder(physchem_features, physchem_mask)

        return seq_emb, struct_emb, phys_emb

    def encode_pmhc(
        self,
        sequence_emb: torch.Tensor,
        structure_x: torch.Tensor,
        structure_edge_index: torch.Tensor,
        structure_batch: torch.Tensor,
        physchem_features: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
        physchem_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode pMHC into multi-modal representation.

        Returns:
            sequence_emb: [batch_size, sequence_hidden_dim]
            structure_emb: [batch_size, structure_hidden_dim]
            physchem_emb: [batch_size, physchem_hidden_dim]
        """
        seq_emb, _ = self.pmhc_sequence_encoder.epitope_encoder(
            sequence_emb, sequence_mask
        )

        struct_emb, _ = self.pmhc_structure_encoder(
            structure_x, structure_edge_index, structure_batch
        )

        phys_emb = self.pmhc_physchem_encoder(physchem_features, physchem_mask)

        return seq_emb, struct_emb, phys_emb

    def forward(
        self,
        tcr_data: Dict[str, torch.Tensor],
        pmhc_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GARSEF.

        Args:
            tcr_data: Dict containing:
                - sequence_emb: ESM2 embeddings [batch, seq_len, 1280] or [batch, 1280]
                - structure_x: Node features [num_nodes, saprot_dim]
                - structure_edge_index: Graph edges [2, num_edges]
                - structure_batch: Batch assignment [num_nodes]
                - physchem_features: Physicochemical features [batch, num_res, 8]
                - sequence_mask (optional): [batch, seq_len]
                - physchem_mask (optional): [batch, num_res]

            pmhc_data: Same structure as tcr_data

        Returns:
            Dict containing:
                - tcr_emb: TCR fused embedding [batch, fusion_dim]
                - pmhc_emb: pMHC fused embedding [batch, fusion_dim]
                - tcr_proj: TCR contrastive projection [batch, projection_dim]
                - pmhc_proj: pMHC contrastive projection [batch, projection_dim]
                - binding_logits: Binding prediction logits [batch, 1]
                - similarity: Cosine similarity matrix [batch, batch]
        """
        tcr_seq, tcr_struct, tcr_phys = self.encode_tcr(
            sequence_emb=tcr_data['sequence_emb'],
            structure_x=tcr_data['structure_x'],
            structure_edge_index=tcr_data['structure_edge_index'],
            structure_batch=tcr_data['structure_batch'],
            physchem_features=tcr_data['physchem_features'],
            sequence_mask=tcr_data.get('sequence_mask'),
            physchem_mask=tcr_data.get('physchem_mask'),
            alpha_emb=tcr_data.get('alpha_emb'),
            beta_emb=tcr_data.get('beta_emb')
        )

        pmhc_seq, pmhc_struct, pmhc_phys = self.encode_pmhc(
            sequence_emb=pmhc_data['sequence_emb'],
            structure_x=pmhc_data['structure_x'],
            structure_edge_index=pmhc_data['structure_edge_index'],
            structure_batch=pmhc_data['structure_batch'],
            physchem_features=pmhc_data['physchem_features'],
            sequence_mask=pmhc_data.get('sequence_mask'),
            physchem_mask=pmhc_data.get('physchem_mask')
        )

        outputs = self.fusion(
            tcr_sequence=tcr_seq,
            tcr_structure=tcr_struct,
            tcr_physchem=tcr_phys,
            pmhc_sequence=pmhc_seq,
            pmhc_structure=pmhc_struct,
            pmhc_physchem=pmhc_phys
        )

        tcr_proj = outputs['tcr_proj']
        pmhc_proj = outputs['pmhc_proj']
        similarity = torch.matmul(tcr_proj, pmhc_proj.T) / self.temperature

        outputs['similarity'] = similarity

        return outputs

    def predict(self, tcr_data: Dict, pmhc_data: Dict) -> torch.Tensor:
        """
        Get binding probability prediction.

        Args:
            tcr_data: TCR input data
            pmhc_data: pMHC input data

        Returns:
            Binding probabilities [batch_size]
        """
        outputs = self.forward(tcr_data, pmhc_data)
        return torch.sigmoid(outputs['binding_logits']).squeeze(-1)

    def get_embeddings(
        self,
        tcr_data: Dict,
        pmhc_data: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get fused embeddings for TCR and pMHC.

        Useful for visualization (t-SNE) and downstream tasks.
        """
        outputs = self.forward(tcr_data, pmhc_data)
        return outputs['tcr_emb'], outputs['pmhc_emb']


class GARSEFSimple(nn.Module):
    """
    Simplified GARSEF model for cases where structural data is limited.

    Uses only sequence and physicochemical features (no GNN).
    """

    def __init__(self, config: Optional[GARSEFConfig] = None):
        super().__init__()

        if config is None:
            config = GARSEFConfig()

        self.config = config

        self.tcr_sequence_encoder = SequenceEncoder(
            input_dim=config.esm2_dim,
            hidden_dim=config.sequence_hidden_dim,
            num_layers=config.sequence_num_layers,
            dropout=config.dropout
        )

        self.pmhc_sequence_encoder = SequenceEncoder(
            input_dim=config.esm2_dim,
            hidden_dim=config.sequence_hidden_dim,
            num_layers=config.sequence_num_layers,
            dropout=config.dropout
        )

        self.tcr_physchem_encoder = PhysicochemicalEncoder(
            input_dim=config.physchem_dim,
            hidden_dim=config.physchem_hidden_dim,
            num_layers=config.physchem_num_layers,
            dropout=config.dropout,
            aggregation='mean'
        )

        self.pmhc_physchem_encoder = PhysicochemicalEncoder(
            input_dim=config.physchem_dim,
            hidden_dim=config.physchem_hidden_dim,
            num_layers=config.physchem_num_layers,
            dropout=config.dropout,
            aggregation='mean'
        )

        fusion_input = config.sequence_hidden_dim + config.physchem_hidden_dim

        self.tcr_fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout)
        )

        self.pmhc_fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, config.fusion_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout)
        )

        self.contrastive_head = ContrastiveHead(
            config.fusion_dim, config.projection_dim
        )

        self.binary_head = BinaryHead(
            config.fusion_dim * 2, hidden_dim=128, num_layers=2,
            dropout=config.fusion_dropout
        )

        self.temperature = nn.Parameter(torch.tensor(config.temperature))

    def forward(
        self,
        tcr_sequence: torch.Tensor,
        tcr_physchem: torch.Tensor,
        pmhc_sequence: torch.Tensor,
        pmhc_physchem: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for simplified model.

        Args:
            tcr_sequence: TCR ESM2 embeddings [batch, 1280]
            tcr_physchem: TCR physicochemical features [batch, 8]
            pmhc_sequence: pMHC ESM2 embeddings [batch, 1280]
            pmhc_physchem: pMHC physicochemical features [batch, 8]

        Returns:
            Model outputs dict
        """
        tcr_seq = self.tcr_sequence_encoder(tcr_sequence)
        tcr_phys = self.tcr_physchem_encoder(tcr_physchem)

        pmhc_seq = self.pmhc_sequence_encoder(pmhc_sequence)
        pmhc_phys = self.pmhc_physchem_encoder(pmhc_physchem)

        tcr_emb = self.tcr_fusion(torch.cat([tcr_seq, tcr_phys], dim=-1))
        pmhc_emb = self.pmhc_fusion(torch.cat([pmhc_seq, pmhc_phys], dim=-1))

        tcr_proj = self.contrastive_head(tcr_emb)
        pmhc_proj = self.contrastive_head(pmhc_emb)

        binding_logits = self.binary_head(tcr_emb, pmhc_emb)

        similarity = torch.matmul(tcr_proj, pmhc_proj.T) / self.temperature

        return {
            'tcr_emb': tcr_emb,
            'pmhc_emb': pmhc_emb,
            'tcr_proj': tcr_proj,
            'pmhc_proj': pmhc_proj,
            'binding_logits': binding_logits,
            'similarity': similarity
        }


def create_garsef(
    config: Optional[GARSEFConfig] = None,
    use_structure: bool = True
) -> nn.Module:
    """
    Factory function to create GARSEF model.

    Args:
        config: Model configuration
        use_structure: Whether to use structural (GNN) features

    Returns:
        GARSEF or GARSEFSimple model
    """
    if use_structure:
        return GARSEF(config)
    else:
        return GARSEFSimple(config)


__all__ = [
    'GARSEF',
    'GARSEFSimple',
    'GARSEFConfig',
    'create_garsef',
]
