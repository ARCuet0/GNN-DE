"""Batched cross-dimension self-attention for rotation-aware dimension coupling.

Port of HyperOPT/cross_dim_attention.py to (B, N, D, ...) tensors.
Dimensions attend to each other conditioned on population state (h_out).
Reshape trick: (B, N, D, F) -> (B*N, D, F) for nn.MultiheadAttention.
"""
import torch
import torch.nn as nn
from torch import Tensor


class BatchedCrossDimAttention(nn.Module):
    """Batched cross-dimension attention: per-dim features + node embedding -> routed.

    Input:  per_dim_feats (B, N, D, 5), h_out (B, N, embed_dim)
    Output: routed (B, N, D, out_dim=96)
    """

    def __init__(self, embed_dim: int = 128, n_perdim: int = 5,
                 hidden: int = 32, attn_dim: int = 32,
                 n_attn_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dim = attn_dim

        self.per_dim_mlp = nn.Sequential(
            nn.Linear(n_perdim + embed_dim, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, attn_dim),
        )

        token_dim = attn_dim + embed_dim  # 96
        self.attn = nn.MultiheadAttention(
            token_dim, n_attn_heads, batch_first=True, dropout=0.0,
        )
        self.norm = nn.LayerNorm(token_dim)
        self.out_dim = token_dim

    def forward(self, per_dim_feats: Tensor, h_out: Tensor) -> Tensor:
        """
        Args:
            per_dim_feats: (B, N, D, 5)
            h_out:         (B, N, embed_dim)
        Returns:
            routed: (B, N, D, out_dim)
        """
        B, N, D, F = per_dim_feats.shape

        # Broadcast h_out across D: (B, N, 1, E) -> (B, N, D, E)
        h_broad = h_out.unsqueeze(2).expand(B, N, D, self.embed_dim)

        # Concat per-dim + embedding: (B, N, D, F+E)
        x = torch.cat([per_dim_feats, h_broad], dim=-1)

        # Project: (B*N*D, F+E) -> (B*N*D, attn_dim) -> (B, N, D, attn_dim)
        flat = x.reshape(-1, x.shape[-1])
        projected = self.per_dim_mlp(flat).reshape(B, N, D, self.attn_dim)

        # Form tokens: (B, N, D, attn_dim + E = 96)
        tokens = torch.cat([projected, h_broad], dim=-1)

        # Reshape to (B*N, D, 96) for MHA — each individual is a batch
        tokens_flat = tokens.reshape(B * N, D, self.out_dim)
        attn_out, _ = self.attn(tokens_flat, tokens_flat, tokens_flat)

        # Residual + norm, then reshape back
        routed_flat = self.norm(attn_out + tokens_flat)
        return routed_flat.reshape(B, N, D, self.out_dim)
