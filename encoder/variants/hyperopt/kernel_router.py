"""Batched expert-choice kernel router for K=6 heads.

Deterministic expert-choice routing: each expert scores all N nodes and
distributes weight via softmax(dim=N), scaled by N/K. No Gumbel noise.
Exploration comes from B (batch) and M (operator stochasticity).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BatchedKernelRouter(nn.Module):
    """Expert-choice deterministic router for K heads.

    Input:  h_out (B, N, embed_dim)
    Output: weights (B, N, K), soft_probs (B, N, K), lb_loss scalar
    """

    def __init__(self, embed_dim: int = 128, K: int = 6, hidden: int = 64):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, K),
        )

    def forward(self, h_out: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            h_out: (B, N, embed_dim)
        Returns:
            weights:    (B, N, K) expert-choice weights (N/K scaled)
            soft_probs: (B, N, K) softmax probs (for diagnostics)
            lb_loss:    scalar load-balance loss
        """
        B, N, _ = h_out.shape
        K = self.K

        logits = self.mlp(h_out)  # (B, N, K)

        # Expert-choice: each expert distributes over N via softmax(dim=N)
        # Transpose to (B, K, N), softmax over N, transpose back
        expert_attn = F.softmax(logits.transpose(1, 2), dim=-1)  # (B, K, N)
        weights = expert_attn.transpose(1, 2) * (N / K)  # (B, N, K)

        # Soft probs for diagnostics (standard softmax over K per node)
        soft_probs = F.softmax(logits, dim=-1)  # (B, N, K)

        # Load-balance loss (Switch Transformer style) — vmap-safe, no in-place
        top_choices = logits.argmax(dim=-1)  # (B, N)
        # One-hot then mean over N: (B, N) -> (B, N, K) -> (B, K)
        one_hot = (top_choices.unsqueeze(-1) == torch.arange(K, device=h_out.device))
        f = one_hot.float().mean(dim=1)  # (B, K)
        p = soft_probs.mean(dim=1)  # (B, K)
        lb_loss = (K * (f * p).sum(dim=-1)).mean()  # scalar

        return weights, soft_probs, lb_loss, logits
