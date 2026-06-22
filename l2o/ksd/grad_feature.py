"""∇f-aware head wrappers for KSD ablation.

Two wrappers attach to the existing donor_selector and F/CR head without
touching upstream code:

  GeometricBiasDonorWrapper
      Adds learnable bias  geom_bias_ij = w * <g_feat[j], coords[i] - coords[j]>
      to the pbest channel of DonorSelectionGATv2. r1, r2 channels untouched.
      w is a scalar Parameter init 0.5.

  GradFeatureFCRWrapper
      Extends AdaptiveFCRBeta's first Linear from 2*h_dim → 2*h_dim + D + 1.
      First 2*h_dim columns copied from base; new D+1 zero-init. Forward at
      init is bit-exact to base.

Side-channel KSDState carries (coords, g_dir, g_mag), all detached. The
caller (measurement.py / train.py) writes `variant._ksd_state` before each
generation; wrappers read via a state_provider thunk. When state is None,
wrappers passthrough.

Sign convention: g_dir = -∇f/‖∇f‖ (descent-aligned). Then
<g_dir[j], x_i - x_j> > 0 means descent at j points toward i ⇒ j is a good
donor for i (SVGD attraction).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KSDState:
    """Side-channel for ∇f-derived features. All tensors must be detached.

    Fields:
        coords: (B, N, D) population positions, detached.
        g_dir:  (B, N, D) descent direction = -∇f/‖∇f‖, detached.
        g_mag:  (B, N, 1) log1p(‖∇f‖).clamp(min=1e-2), detached.
    """
    coords: torch.Tensor
    g_dir: torch.Tensor
    g_mag: torch.Tensor

    def assert_detached(self):
        for name in ('coords', 'g_dir', 'g_mag'):
            t = getattr(self, name)
            assert not t.requires_grad, \
                f"KSDState.{name} must be detached (got requires_grad=True)"


def sign_coherence_score(g_feat: torch.Tensor, g_raw: torch.Tensor) -> float:
    """Diagnostic: <g_feat, g_raw> mean across all elements.

    With the correct convention (g_feat = -∇f-style, g_raw = +∇f), the
    inner product per element is ≤ 0 and strongly negative on
    non-degenerate landscapes. A positive value indicates the sign
    convention is inverted.

    Soft check rather than strict assert: an assert is fragile at flat
    optima where g_raw ≈ 0 yields inner product ≈ 0 with random sign.
    """
    inner = (g_feat * g_raw).sum(-1)              # (..., N)
    return float(inner.mean().item())


# ────────────────────────────────────────────────────────────────────────────
#  Donor wrapper
# ────────────────────────────────────────────────────────────────────────────


class GeometricBiasDonorWrapper(nn.Module):
    """Wraps DonorSelectionGATv2 with a learnable geometric bias on pbest.

    bias_ij = w_geom · <g_feat[j], coords[i] - coords[j]>
    where g_feat = g_dir * g_mag (the score). Applied only to channel 0
    (pbest). Channels 1, 2 (r1, r2) pass through unchanged.

    forward signature mirrors DonorSelectionGATv2: (h, node_feat) → logits
    (B, N_q, N_c, n_roles). Both `forward` and `forward_asym` are
    overridden so the wrapper is plug-compatible.
    """

    def __init__(self, base: nn.Module,
                 state_provider: Callable[[], Optional[KSDState]],
                 w_init: float = 0.5):
        super().__init__()
        self.base = base
        self.state_provider = state_provider
        self.w_geom = nn.Parameter(torch.tensor(float(w_init)))

    def _compute_geom_bias(self, state: KSDState, B: int, N_q: int,
                            N_c: int, device, dtype) -> torch.Tensor:
        """(B, N_q, N_c) bias for the pbest channel.

        Uses coords and g_feat = g_dir * g_mag from the side-channel.
        Pads with zeros for archive slots when N_c > N_q (legacy active
        diagonal behaviour matches base donor_selector).
        """
        coords = state.coords.to(device=device, dtype=dtype)
        g_feat = (state.g_dir * state.g_mag).to(device=device, dtype=dtype)

        # Pool size N_pool may equal N_q (legacy) or be larger (archive).
        # Geometric bias is meaningful only on active pop (first N_q
        # candidates). Archive slots get bias=0.
        N_active = coords.shape[1]
        diff_active = coords.unsqueeze(2) - coords.unsqueeze(1)        # (B, N_q, N_active, D)
        # <g_feat[j], coords[i] - coords[j]>
        # broadcast: g_feat.unsqueeze(1) → (B, 1, N_active, D)
        bias_active = self.w_geom * (diff_active * g_feat.unsqueeze(1)).sum(-1)
        # bias shape (B, N_q, N_active)
        if N_c == N_q == N_active:
            return bias_active
        # General case: zero-pad to (B, N_q, N_c).
        bias = bias_active.new_zeros(B, N_q, N_c)
        n_use = min(N_active, N_c)
        bias[:, :, :n_use] = bias_active[:, :, :n_use]
        return bias

    def _add_bias_to_pbest(self, scores: torch.Tensor) -> torch.Tensor:
        """Add geom_bias to channel 0 (pbest) only, in-place safe.

        Skips the bias when the donor_selector is called with a query
        dimension different from the side-channel coords (i.e. the
        augmented surrogate forward, where N_q = N + M*N proposals). In
        that case the bias has no well-defined geometric meaning and we
        passthrough — the augmented forward is purely for surrogate
        scoring of proposals, not for selecting donors.
        """
        state = self.state_provider()
        if state is None:
            return scores
        B, N_q, N_c, R = scores.shape
        N_active = state.coords.shape[1]
        if N_q != N_active:
            # Augmented forward (surrogate scoring) — skip bias.
            return scores
        bias = self._compute_geom_bias(state, B, N_q, N_c,
                                        device=scores.device, dtype=scores.dtype)
        # Out-of-place: clone scores and add bias to channel 0. Avoids an
        # all-channel zero-tensor allocation and keeps the autograd graph
        # of `scores` (which has grad chain from the base donor_selector).
        scores = scores.clone()
        scores[..., 0] = scores[..., 0] + bias
        return scores

    def forward(self, h: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        scores = self.base(h, node_feat)
        return self._add_bias_to_pbest(scores)

    def forward_asym(self, h_query, h_cand, node_feat_query,
                      node_feat_cand, cand_mask=None):
        scores = self.base.forward_asym(h_query, h_cand, node_feat_query,
                                         node_feat_cand, cand_mask=cand_mask)
        return self._add_bias_to_pbest(scores)


# ────────────────────────────────────────────────────────────────────────────
#  F/CR wrapper
# ────────────────────────────────────────────────────────────────────────────


class GradFeatureFCRWrapper(nn.Module):
    """Wraps AdaptiveFCRBeta — extends shared Linear input by D+1 dims.

    Copy first 2*h_dim columns of weight from base, zero-init the new D+1.
    Forward at init is bit-exact to the base when fed the same
    (h_ind, h_global) and zero g_feat.
    """

    def __init__(self, base: nn.Module, D: int,
                 state_provider: Callable[[], Optional[KSDState]]):
        super().__init__()
        self.D = D
        self.state_provider = state_provider
        self.alpha_beta_max = base.alpha_beta_max

        # Reuse the f_head / cr_head from base (no changes there).
        self.f_head = base.f_head
        self.cr_head = base.cr_head

        # Extend the shared first Linear.
        old_lin: nn.Linear = base.shared[0]
        in_old = old_lin.in_features              # 2*h_dim
        out_dim = old_lin.out_features            # hidden
        new_lin = nn.Linear(in_old + D + 1, out_dim,
                            bias=(old_lin.bias is not None))
        with torch.no_grad():
            new_lin.weight[:, :in_old].copy_(old_lin.weight)
            new_lin.weight[:, in_old:].zero_()
            if old_lin.bias is not None:
                new_lin.bias.copy_(old_lin.bias)
        self.shared = nn.Sequential(new_lin, nn.SiLU())

    def forward(self, h_ind: torch.Tensor, h_global: torch.Tensor):
        h_global_exp = h_global.unsqueeze(1).expand(-1, h_ind.size(1), -1)
        h_cat = torch.cat([h_ind, h_global_exp], dim=-1)            # (B, N, 2*h_dim)
        state = self.state_provider()
        if state is None:
            # Passthrough: pad with zeros so the new Linear's columns × 0 = 0
            # (forward bit-exact to base since those weights are zero-init).
            B, N, _ = h_cat.shape
            g_feat = h_cat.new_zeros(B, N, self.D + 1)
        else:
            # cat([g_dir, g_mag]) — D + 1 features per individual.
            g_dir = state.g_dir.to(device=h_cat.device, dtype=h_cat.dtype)
            g_mag = state.g_mag.to(device=h_cat.device, dtype=h_cat.dtype)
            g_feat = torch.cat([g_dir, g_mag], dim=-1)              # (B, N, D+1)
        x = torch.cat([h_cat, g_feat], dim=-1)                      # (B, N, 2*h_dim + D + 1)
        shared = self.shared(x)
        f_raw = self.f_head(shared)
        cr_raw = self.cr_head(shared)
        f_alpha = 1 + F.softplus(f_raw[..., 0]).clamp(max=self.alpha_beta_max)
        f_beta = 1 + F.softplus(f_raw[..., 1]).clamp(max=self.alpha_beta_max)
        cr_alpha = 1 + F.softplus(cr_raw[..., 0]).clamp(max=self.alpha_beta_max)
        cr_beta = 1 + F.softplus(cr_raw[..., 1]).clamp(max=self.alpha_beta_max)
        return f_alpha, f_beta, cr_alpha, cr_beta
