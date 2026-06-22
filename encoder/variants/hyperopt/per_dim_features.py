"""Batched per-dimension feature computation: (B, N, D) -> (B, N, D, 5).

Port of HyperOPT/per_dim_features.py to natively batched (B, N, D) tensors.
All operations are pure tensor ops — no Python loops, no CPU transfers.
"""
import torch
from torch import Tensor


def compute_per_dim_features_batched(
    x: Tensor, fitness: Tensor,
    lb: float = -100.0, ub: float = 100.0,
) -> Tensor:
    """Compute 5 per-dimension features for all individuals, batched.

    Args:
        x:       (B, N, D) coordinates.
        fitness: (B, N) fitness values (lower = better).
        lb, ub:  search space bounds.

    Returns:
        feats: (B, N, D, 5) feature tensor.
    """
    B, N, D = x.shape
    span = ub - lb

    # f1: normalized coordinate value in [0, 1]
    f1 = (x - lb) / span  # (B, N, D)

    # f2: displacement from best individual per batch
    best_idx = fitness.argmin(dim=1)  # (B,)
    x_best = x[torch.arange(B, device=x.device), best_idx]  # (B, D)
    f2 = (x - x_best.unsqueeze(1)) / span  # (B, N, D)

    # f3: displacement from population centroid
    centroid = x.mean(dim=1)  # (B, D)
    f3 = (x - centroid.unsqueeze(1)) / span  # (B, N, D)

    # f4: population-based gradient proxy (correlation x_d with fitness)
    fit_centered = fitness - fitness.mean(dim=1, keepdim=True)  # (B, N)
    x_centered = x - x.mean(dim=1, keepdim=True)  # (B, N, D)
    # (B, N, D) * (B, N, 1) -> mean over N -> (B, D)
    cov_xf = (x_centered * fit_centered.unsqueeze(-1)).mean(dim=1)
    cov_norm = cov_xf / cov_xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    f4 = cov_norm.unsqueeze(1).expand(B, N, D)  # (B, N, D)

    # f5: placeholder (zeros)
    f5 = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)

    return torch.stack([f1, f2, f3, f4, f5], dim=-1)  # (B, N, D, 5)
