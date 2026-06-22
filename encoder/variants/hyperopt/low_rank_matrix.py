"""Batched low-rank PSD matrix factorization head.

Port of HyperOPT/low_rank_matrix.py to (B, N, D, routed_dim) tensors.
M = U U^T + diag(d) — PSD by construction, O(D * k_rank) params.
Dimension-shared projections: no D-dependent weight parameters.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BatchedLowRankMatrixHead(nn.Module):
    """Batched low-rank + diagonal PSD factorization.

    Input:  routed (B, N, D, routed_dim)
    Output: U (B, N, D, r), d (B, N, D)
    """

    def __init__(self, routed_dim: int = 160, k_rank: int = 8):
        super().__init__()
        self.k_rank = k_rank
        self.u_proj = nn.Linear(routed_dim, k_rank)
        self.d_proj = nn.Linear(routed_dim, 1)

    def forward(self, routed: Tensor) -> tuple[Tensor, Tensor]:
        """
        Args:
            routed: (B, N, D, routed_dim).
        Returns:
            U: (B, N, D, r)
            d: (B, N, D) positive via softplus.
        """
        shape = routed.shape[:-1]  # (B, N, D)
        flat = routed.reshape(-1, routed.shape[-1])  # (B*N*D, routed_dim)
        U = self.u_proj(flat).reshape(*shape, self.k_rank)  # (B, N, D, r)
        d = F.softplus(self.d_proj(flat)).reshape(*shape)    # (B, N, D)
        return U, d
