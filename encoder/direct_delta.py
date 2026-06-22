"""
direct_delta.py — Direct δx generation from node embeddings.

Replaces classical operator formulas (DE, CMA-ES, etc.) with a learned MLP
that generates per-dimension displacements conditioned on:
  - h_out (B, N, embed_dim): WHO context from GATv2 (fitness rank, density, etc.)
  - coords (B, N, D): WHERE context (coordinate-space position)

Output: (M, B, N, D) reparameterized displacement samples.
Noise model: low-rank + diagonal (same expressiveness as CMA-ES covariance).

No .item(), no .nonzero(), no for-loops over B, no scatter ops.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedDirectDelta(nn.Module):
    """Direct δx generation via shared per-dimension MLP + reparameterized noise.

    Bridge: h_out (B, N, embed_dim) provides population context (who),
    coords (B, N, D) provides spatial context (where).
    Shared MLP processes each dimension independently with same weights.

    Noise: delta = mu + U @ eps1 + sqrt(d) * eps2
    where U is low-rank (rank r) and d is diagonal variance.
    """

    def __init__(self, embed_dim=16, head_idx=3, rank=4, backbone_dim=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.rank = rank
        self.backbone_dim = backbone_dim
        # Per-head projection from backbone
        if backbone_dim > 0:
            from encoder.batched_operators import _make_proj
            self.proj, self.proj_norm = _make_proj(backbone_dim, embed_dim)
        else:
            self.proj, self.proj_norm = None, None

        # Per-dim input: embed_dim (broadcast) + 3 coordinate features
        feat_dim = embed_dim + 3

        # Shared MLP across dimensions
        self.dim_mlp = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 16),
            nn.SiLU(),
        )

        # Output projections
        self.mu_proj = nn.Linear(16, 1)
        self.U_proj = nn.Linear(16, rank)
        self.d_proj = nn.Linear(16, 1)

        # Learned magnitude scale — init so that scale/bounds_span ≈ 0.5
        # With bounds_span=200: scale=100, delta ≈ MLP_output * 0.5
        self.log_scale = nn.Parameter(torch.tensor(math.log(100.0)))

    def get_embedding(self, h_backbone):
        if self.proj is not None:
            return self.proj_norm(self.proj(h_backbone))
        return h_backbone[..., :self.embed_dim]

    def compute_params(self, h_out, coords, fitness, adj=None,
                       route_probs=None, bounds_span=200.0,
                       knn_idx=None, h_backbone=None, **_kwargs):
        # h_out is pre-projected by the variant via get_embedding(h_backbone)
        """Compute per-dimension displacement parameters.

        Args:
            h_out:      (B, N, embed_dim) — specialized embedding for this head
            coords:     (B, N, D) — current population coordinates
            fitness:    (B, N) — fitness values
            adj:        unused (interface compat)
            route_probs: unused (interface compat)
            bounds_span: coordinate range (e.g. 200 for [-100, 100])
            knn_idx:    unused (interface compat)

        Returns:
            dict with mu (B,N,D), U (B,N,D,rank), d (B,N,D)
        """
        B, N, D = coords.shape
        half_span = bounds_span / 2

        # --- Per-dim coordinate features (all differentiable) ---
        coords_f = coords.float()

        # Normalized position [-1, 1]
        x_norm = coords_f / half_span

        # Offset from best individual per dim
        best_idx = fitness.argmin(dim=1)                    # (B,)
        x_best = coords_f[torch.arange(B, device=coords.device), best_idx]  # (B, D)
        offset_best = (coords_f - x_best.unsqueeze(1)) / half_span         # (B, N, D)

        # Offset from centroid per dim
        centroid = coords_f.mean(dim=1, keepdim=True)       # (B, 1, D)
        offset_centroid = (coords_f - centroid) / half_span  # (B, N, D)

        # Stack: (B, N, D, 3)
        dim_feats = torch.stack([x_norm, offset_best, offset_centroid], dim=-1)

        # Broadcast h_out: (B, N, 1, embed_dim) → (B, N, D, embed_dim)
        h_broadcast = h_out.unsqueeze(2).expand(-1, -1, D, -1)

        # Concatenate: (B, N, D, embed_dim + 3)
        combined = torch.cat([h_broadcast, dim_feats], dim=-1)

        # Shared MLP: (B, N, D, feat_dim) → (B, N, D, 16)
        hidden = self.dim_mlp(combined)

        # Project to mu, U, d
        mu = self.mu_proj(hidden).squeeze(-1)                    # (B, N, D)
        U = self.U_proj(hidden)                                   # (B, N, D, rank)
        d = F.softplus(self.d_proj(hidden).squeeze(-1)) + 1e-6   # (B, N, D)

        # Adaptive scale: per-dim population std (shrinks with convergence)
        # Clamp upper bound to half_span to prevent explosive deltas on random init
        pop_std = coords_f.std(dim=1, keepdim=True).clamp(min=1e-8, max=half_span)  # (B, 1, D)

        return {'mu': mu, 'U': U, 'd': d, 'pop_std': pop_std}

    def sample_batch(self, params_dict, coords, bounds_span, M):
        """Draw M reparameterized displacement samples.

        Args:
            params_dict: from compute_params (mu, U, d)
            coords:      (B, N, D) — current coords (unused, interface compat)
            bounds_span: coordinate range
            M:           number of independent samples

        Returns:
            delta: (M, B, N, D) displacement samples
        """
        mu = params_dict['mu']    # (B, N, D)
        U = params_dict['U']      # (B, N, D, rank)
        d = params_dict['d']      # (B, N, D)
        pop_std = params_dict['pop_std']  # (B, 1, D)
        B, N, D = mu.shape

        scale = self.log_scale.exp()

        # Low-rank noise: einsum(bndr, mbnr -> mbnd)
        eps1 = torch.randn(M, B, N, self.rank, device=mu.device, dtype=mu.dtype)
        low_rank = torch.einsum('bndr,mbnr->mbnd', U, eps1)

        # Diagonal noise: sqrt(d) * eps2
        eps2 = torch.randn(M, B, N, D, device=mu.device, dtype=mu.dtype)
        diag = d.sqrt().unsqueeze(0) * eps2

        # Combine: mu + structured noise, scaled by population spread
        # pop_std adapts magnitude to convergence level (shrinks as pop converges)
        delta = (mu.unsqueeze(0) + low_rank + diag) * (scale * pop_std.unsqueeze(0) / bounds_span)

        return delta
