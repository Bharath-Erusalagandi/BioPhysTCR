"""Sequence Encoder for BioPhysTCR model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SequenceEncoder(nn.Module):
    """MLP encoder for ESM2 sequence embeddings."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_layer_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:"""
        if x.dim() == 3:
            x = x.mean(dim=1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)

        return self.encoder(x)


class SequenceEncoderWithAttention(nn.Module):
    """Sequence encoder with self-attention for variable-length sequences."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_seq_len: int = 100
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.pos_encoding = PositionalEncoding(hidden_dim, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:"""
        # Handle 2D input (already pooled embeddings)
        if x.dim() == 2:
            # Input is [batch, input_dim], add sequence dimension
            x = x.unsqueeze(1)  # [batch, 1, input_dim]

        x = self.input_projection(x)

        # Only apply positional encoding if sequence length > 1
        if x.size(1) > 1:
            x = self.pos_encoding(x)

        attn_mask = None
        if mask is not None:
            attn_mask = ~mask

        encoded = self.transformer(x, src_key_padding_mask=attn_mask)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        pooled = self.output_projection(pooled)

        return pooled, encoded


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TCRSequenceEncoder(nn.Module):
    """Specialized encoder for TCR CDR3 sequences."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        share_encoder: bool = True
    ):
        super().__init__()

        self.share_encoder = share_encoder
        self.hidden_dim = hidden_dim

        if share_encoder:
            self.encoder = SequenceEncoderWithAttention(
                input_dim, hidden_dim, num_heads, num_layers, dropout
            )
        else:
            self.alpha_encoder = SequenceEncoderWithAttention(
                input_dim, hidden_dim, num_heads, num_layers, dropout
            )
            self.beta_encoder = SequenceEncoderWithAttention(
                input_dim, hidden_dim, num_heads, num_layers, dropout
            )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        alpha_emb: Optional[torch.Tensor] = None,
        beta_emb: Optional[torch.Tensor] = None,
        alpha_mask: Optional[torch.Tensor] = None,
        beta_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args:"""
        embeddings = []

        if alpha_emb is not None:
            if self.share_encoder:
                alpha_pooled, _ = self.encoder(alpha_emb, alpha_mask)
            else:
                alpha_pooled, _ = self.alpha_encoder(alpha_emb, alpha_mask)
            embeddings.append(alpha_pooled)
        else:
            embeddings.append(torch.zeros(
                beta_emb.size(0), self.hidden_dim, device=beta_emb.device
            ))

        if beta_emb is not None:
            if self.share_encoder:
                beta_pooled, _ = self.encoder(beta_emb, beta_mask)
            else:
                beta_pooled, _ = self.beta_encoder(beta_emb, beta_mask)
            embeddings.append(beta_pooled)
        else:
            embeddings.append(torch.zeros(
                alpha_emb.size(0), self.hidden_dim, device=alpha_emb.device
            ))

        combined = torch.cat(embeddings, dim=-1)
        return self.fusion(combined)


class pMHCSequenceEncoder(nn.Module):
    """Specialized encoder for pMHC sequences."""

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
        include_mhc: bool = False
    ):
        super().__init__()

        self.include_mhc = include_mhc

        self.epitope_encoder = SequenceEncoderWithAttention(
            input_dim, hidden_dim, num_heads, num_layers, dropout
        )

        if include_mhc:
            self.mhc_encoder = SequenceEncoderWithAttention(
                input_dim, hidden_dim, num_heads, num_layers, dropout
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(
        self,
        epitope_emb: torch.Tensor,
        epitope_mask: Optional[torch.Tensor] = None,
        mhc_emb: Optional[torch.Tensor] = None,
        mhc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Args:"""
        epitope_pooled, _ = self.epitope_encoder(epitope_emb, epitope_mask)

        if self.include_mhc and mhc_emb is not None:
            mhc_pooled, _ = self.mhc_encoder(mhc_emb, mhc_mask)
            combined = torch.cat([epitope_pooled, mhc_pooled], dim=-1)
            return self.fusion(combined)

        return epitope_pooled


__all__ = [
    'SequenceEncoder',
    'SequenceEncoderWithAttention',
    'TCRSequenceEncoder',
    'pMHCSequenceEncoder',
    'PositionalEncoding',
]
