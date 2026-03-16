"""
GNN Encoder for GARSEF model.

GraphSAGE + LSTM + Self-Attention architecture for structure encoding
for learning structural representations of TCR-pMHC interfaces.

Multi-layer graph neural network for TCR-pMHC structure analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GraphNorm, global_max_pool
from torch_geometric.utils import to_dense_batch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional


def standardization(embedding: torch.Tensor) -> torch.Tensor:
    """Standardize embeddings to zero mean and unit variance."""
    mean = embedding.mean(dim=0, keepdim=True)
    std = embedding.std(dim=0, keepdim=True)
    return (embedding - mean) / (std + 1e-12)


class EmbeddingLayer(nn.Module):
    """Project node features to a common embedding dimension."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class GraphSAGELayers(nn.Module):
    """
    3-layer GraphSAGE for graph representation learning.

    Uses mean aggregation and GraphNorm for stable training.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout

        self.convs.append(SAGEConv(input_dim, hidden_dim, project=True, aggr='mean'))
        self.norms.append(GraphNorm(hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, project=True, aggr='mean'))
            self.norms.append(GraphNorm(hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.norms[i](x, batch)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class LSTMLayer(nn.Module):
    """
    Bidirectional LSTM for capturing sequential/spatial patterns.

    Processes graph nodes as a sequence to capture global information.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Node embeddings [num_nodes, hidden_dim]
            Hidden states (h, c)
        """
        x_dense, mask = to_dense_batch(x, batch)

        lengths = torch.bincount(batch).cpu()

        x_packed = pack_padded_sequence(
            x_dense, lengths, batch_first=True, enforce_sorted=False
        )

        output_packed, hidden = self.lstm(x_packed)

        output_dense, _ = pad_packed_sequence(output_packed, batch_first=True)

        output_dense = self.projection(output_dense)

        output = output_dense[mask]

        return output, hidden


class SelfAttentionLayer(nn.Module):
    """
    Multi-head self-attention for capturing node interactions.
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            batch: Batch assignment [num_nodes]

        Returns:
            Node embeddings [num_nodes, input_dim]
            Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        x_dense, mask = to_dense_batch(x, batch)

        attn_output, attn_weights = self.attention(
            query=x_dense,
            key=x_dense,
            value=x_dense,
            key_padding_mask=(~mask)
        )

        attn_output = self.norm(attn_output)

        output = attn_output[mask] + x

        return output, attn_weights


class GNNEncoder(nn.Module):
    """
    Complete GNN encoder: GraphSAGE + LSTM + Self-Attention.

    This architecture learns structural
    representations of protein interfaces.

    Architecture:
        1. Embedding layer to project input features
        2. 3-layer GraphSAGE for local neighborhood aggregation
        3. Bidirectional LSTM for global sequence patterns
        4. Self-attention for capturing interactions
        5. Global max pooling for graph-level representation
    """

    def __init__(
        self,
        input_dim: int = 446,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.embedding = EmbeddingLayer(input_dim, hidden_dim, dropout)

        self.gnn = GraphSAGELayers(
            hidden_dim, hidden_dim, num_gnn_layers, dropout
        )

        self.lstm = LSTMLayer(hidden_dim, hidden_dim // 2, dropout=dropout)

        self.attention = SelfAttentionLayer(
            hidden_dim // 2, num_attention_heads, dropout
        )

        self.output_projection = nn.Linear(hidden_dim // 2, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment [num_nodes]

        Returns:
            Graph embeddings [batch_size, hidden_dim]
            Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        x = self.embedding(x)

        x = self.gnn(x, edge_index, batch)

        x, _ = self.lstm(x, batch)

        x, attn_weights = self.attention(x, batch)

        graph_embedding = global_max_pool(x, batch)

        graph_embedding = self.output_projection(graph_embedding)

        return graph_embedding, attn_weights


class DualGNNEncoder(nn.Module):
    """
    Dual-level GNN encoder for processing both residue and atom graphs.

    Includes cross-attention for information exchange between levels,
    using a bi-level graph architecture.
    """

    def __init__(
        self,
        residue_input_dim: int = 446,
        atom_input_dim: int = 384,
        hidden_dim: int = 256,
        num_gnn_layers: int = 3,
        num_attention_heads: int = 8,
        dropout: float = 0.2,
        use_cross_attention: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention

        self.residue_encoder = GNNEncoder(
            residue_input_dim, hidden_dim, num_gnn_layers,
            num_attention_heads, dropout
        )

        self.atom_encoder = GNNEncoder(
            atom_input_dim, hidden_dim, num_gnn_layers,
            num_attention_heads, dropout
        )

        if use_cross_attention:
            self.res2atom_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.atom2res_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.res_norm = nn.LayerNorm(hidden_dim)
            self.atom_norm = nn.LayerNorm(hidden_dim)

        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        res_x: torch.Tensor,
        res_edge_index: torch.Tensor,
        res_batch: torch.Tensor,
        atom_x: torch.Tensor,
        atom_edge_index: torch.Tensor,
        atom_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process both residue and atom graphs with optional cross-attention.

        Args:
            res_x, res_edge_index, res_batch: Residue graph data
            atom_x, atom_edge_index, atom_batch: Atom graph data

        Returns:
            Combined graph embedding [batch_size, hidden_dim]
            Residue attention weights
            Atom attention weights
        """
        res_emb, res_attn = self.residue_encoder(res_x, res_edge_index, res_batch)
        atom_emb, atom_attn = self.atom_encoder(atom_x, atom_edge_index, atom_batch)

        if self.use_cross_attention:
            res_emb_q = res_emb.unsqueeze(1)
            atom_emb_q = atom_emb.unsqueeze(1)

            res_from_atom, _ = self.res2atom_attention(
                query=res_emb_q, key=atom_emb_q, value=atom_emb_q
            )
            atom_from_res, _ = self.atom2res_attention(
                query=atom_emb_q, key=res_emb_q, value=res_emb_q
            )

            res_emb = self.res_norm(res_emb + res_from_atom.squeeze(1))
            atom_emb = self.atom_norm(atom_emb + atom_from_res.squeeze(1))

        combined = torch.cat([res_emb, atom_emb], dim=-1)
        output = self.fusion(combined)

        return output, res_attn, atom_attn


__all__ = [
    'GNNEncoder',
    'DualGNNEncoder',
    'GraphSAGELayers',
    'LSTMLayer',
    'SelfAttentionLayer',
    'EmbeddingLayer',
]
