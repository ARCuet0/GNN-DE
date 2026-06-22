"""Cauchy/Normal-parameterized F/CR head for L-SHADE-style sampling.

Replaces AdaptiveFCRBeta in BatchedDiffAttDE when fcr_mode ∈ {lshade, cauchy_neural}.
Predicts per-individual μ_F (Cauchy location) and μ_CR (Normal mean). Optionally
predicts σ_F (Cauchy scale) when `learn_sigma=True` — used by the 2026-04-28
falsification experiment (arm C) to test whether mode collapse in μ_F is a
loss-form artifact (MSE → mean) vs a structural limit. Without learn_sigma the
σ stays fixed at 0.1 per L-SHADE convention.

Initialized to output ≈ 0.5 (neutral mu) so that the trajectory under
fcr_mode='lshade' is identical to a fresh L-SHADE run at step 0; σ_F initialized
≈ 0.1 so the first training steps are equivalent to legacy fixed-σ behavior.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFCRCauchy(nn.Module):
    """Per-individual μ_F, μ_CR (and optional σ_F) predictor.

    Input: cat(h_individual, h_global) = 2*h_dim
    Outputs (legacy 2-tuple):
        mu_F:  (B, N), soft-clamped to [0.05, 1.0]. Cauchy location.
        mu_CR: (B, N), soft-clamped to [0.0, 1.0]. Normal mean.
    Outputs (learn_sigma=True, 3-tuple):
        mu_F, mu_CR, sigma_F (B, N), strictly positive (softplus-bounded).
    """

    SIGMA_BIAS_INIT = -2.30  # softplus(-2.30) ≈ 0.0951 → σ_F ≈ 0.10 at init

    def __init__(self, h_dim: int = 128, hidden: int = 64, learn_sigma: bool = False):
        super().__init__()
        self.h_dim = h_dim
        self.hidden = hidden
        self.learn_sigma = learn_sigma
        out_dim = 3 if learn_sigma else 2
        self.mlp = nn.Sequential(
            nn.Linear(2 * h_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        with torch.no_grad():
            self.mlp[-1].bias.zero_()
            if learn_sigma:
                self.mlp[-1].bias[2].fill_(self.SIGMA_BIAS_INIT)

    def expand_to_learn_sigma(self):
        """In-place expansion from 2-output (μ_F, μ_CR) to 3-output (μ_F, μ_CR, σ_F).

        Must be called AFTER ckpt resume so the trained mlp[-1] rows for
        μ_F and μ_CR are preserved. The new σ_F row gets fresh init: bias
        SIGMA_BIAS_INIT (→ σ ≈ 0.10) and the standard kaiming_uniform
        weight init that nn.Linear ships.

        No-op if `learn_sigma` is already True.
        """
        if self.learn_sigma:
            return self
        old_lin = self.mlp[-1]
        new_lin = nn.Linear(self.hidden, 3,
                            device=old_lin.weight.device,
                            dtype=old_lin.weight.dtype)
        with torch.no_grad():
            new_lin.weight.data[:2].copy_(old_lin.weight.data)
            new_lin.bias.data[:2].copy_(old_lin.bias.data)
            new_lin.bias.data[2].fill_(self.SIGMA_BIAS_INIT)
        self.mlp[-1] = new_lin
        self.learn_sigma = True
        return self

    def forward(self, h_ind: torch.Tensor, h_global: torch.Tensor):
        """
        Args:
            h_ind:    (B, N, h_dim) per-individual backbone embedding.
            h_global: (B, h_dim) population-level embedding.

        Returns:
            mu_F:  (B, N) location parameter for Cauchy F sampling.
            mu_CR: (B, N) mean for Normal CR sampling.
            sigma_F:  (B, N) scale parameter for Cauchy F sampling — only if
                      `learn_sigma=True`.
        """
        h_global_exp = h_global.unsqueeze(1).expand(-1, h_ind.size(1), -1)
        x = torch.cat([h_ind, h_global_exp], dim=-1)
        out = self.mlp(x)  # (B, N, 2 or 3)
        # Soft-clamp via sigmoid so init=0 produces neutral 0.5/0.525 outputs.
        mu_F = 0.05 + 0.95 * torch.sigmoid(out[..., 0])
        mu_CR = torch.sigmoid(out[..., 1])
        if self.learn_sigma:
            # softplus + small floor keeps sigma strictly positive and avoids
            # the Cauchy NLL singularity at sigma=0 (log p(x|mu, sigma) →
            # +∞ as sigma → 0 if x = mu, blowing the gradient).
            sigma_F = F.softplus(out[..., 2]) + 1e-3
            return mu_F, mu_CR, sigma_F
        return mu_F, mu_CR
