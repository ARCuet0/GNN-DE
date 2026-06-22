"""Shared utilities for batched operator heads."""
import torch.nn as nn


class _ParamMLP(nn.Module):
    """Shared param predictor: (B, N, in_dim) -> (B, N, n_out)."""
    def __init__(self, in_dim, n_out, hidden=None):
        super().__init__()
        hidden = hidden or in_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )
    def forward(self, x):
        return self.mlp(x)


def _make_proj(backbone_dim, embed_dim):
    """Create per-head projection + LayerNorm if backbone_dim > 0."""
    if backbone_dim > 0:
        proj = nn.Sequential(
            nn.Linear(backbone_dim, backbone_dim),
            nn.GELU(),
            nn.Linear(backbone_dim, embed_dim),
        )
        proj_norm = nn.LayerNorm(embed_dim)
        return proj, proj_norm
    return None, None
