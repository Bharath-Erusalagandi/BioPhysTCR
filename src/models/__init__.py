"""
GARSEF Models Package.

Contains model architectures for TCR-pMHC binding prediction:
- GNN encoders (GraphSAGE-based)
- Sequence encoders (ESM2 projections)
- Multi-modal fusion modules
- Main GARSEF model
"""

from .gnn_encoder import (
    GNNEncoder,
    DualGNNEncoder,
    GraphSAGELayers,
    LSTMLayer,
    SelfAttentionLayer,
    EmbeddingLayer,
)

from .sequence_encoder import (
    SequenceEncoder,
    SequenceEncoderWithAttention,
    TCRSequenceEncoder,
    pMHCSequenceEncoder,
    PositionalEncoding,
)

from .fusion import (
    PhysicochemicalEncoder,
    MultiModalFusion,
    ContrastiveHead,
    BinaryHead,
    CrossAttentionFusion,
    GARSEFFusion,
)

from .garsef import (
    GARSEF,
    GARSEFSimple,
    GARSEFConfig,
    create_garsef,
)


__all__ = [
    'GNNEncoder',
    'DualGNNEncoder',
    'GraphSAGELayers',
    'LSTMLayer',
    'SelfAttentionLayer',
    'EmbeddingLayer',
    'SequenceEncoder',
    'SequenceEncoderWithAttention',
    'TCRSequenceEncoder',
    'pMHCSequenceEncoder',
    'PositionalEncoding',
    'PhysicochemicalEncoder',
    'MultiModalFusion',
    'ContrastiveHead',
    'BinaryHead',
    'CrossAttentionFusion',
    'GARSEFFusion',
    'GARSEF',
    'GARSEFSimple',
    'GARSEFConfig',
    'create_garsef',
]
