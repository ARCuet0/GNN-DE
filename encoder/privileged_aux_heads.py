"""Privileged auxiliary prediction heads for LUPI training.

Trains backbone representations to encode gradient information from
the landscape oracle.  Discarded at inference — only used during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrivilegedAuxHeads(nn.Module):
    """Predict gradient direction and magnitude from backbone embedding h.

    Args:
        h_dim: backbone embedding dimension (e.g. 128).
        D: coordinate dimensionality (e.g. 10).
    """

    def __init__(self, h_dim: int, D: int):
        super().__init__()
        self.grad_dir = nn.Linear(h_dim, D)
        self.grad_mag = nn.Linear(h_dim, 1)

    def forward(self, h: torch.Tensor):
        """h: (B, N, h_dim) -> (pred_dir (B,N,D), pred_log_mag (B,N,1))."""
        return self.grad_dir(h), self.grad_mag(h)

    def loss(self, h: torch.Tensor, grad_f: torch.Tensor) -> torch.Tensor:
        """Compute direction + magnitude prediction loss.

        Args:
            h: (B, N, h_dim) backbone embedding WITH gradient.
            grad_f: (B, N, D) landscape gradient (detached oracle target).

        Returns:
            Scalar loss (direction cosine + 0.5 * magnitude MSE).
        """
        pred_dir, pred_log_mag = self.forward(h)

        # Direction: cosine loss (safe for zero grad_f via clamp in normalize)
        target_dir = F.normalize(grad_f.float(), dim=-1)
        pred_dir_n = F.normalize(pred_dir, dim=-1)
        dir_loss = (1.0 - (pred_dir_n * target_dir).sum(dim=-1)).mean()

        # Magnitude: MSE on log scale (target detached, safe with clamp)
        target_log_mag = grad_f.float().norm(dim=-1, keepdim=True).clamp(min=1e-8).log()
        mag_loss = F.mse_loss(pred_log_mag, target_log_mag.detach())

        return dir_loss + 0.5 * mag_loss
