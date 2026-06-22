"""
classic_k2.py — K=2 Classic variant (SHADE + LS1 Gumbel mask), fully batched.

Uses BatchedDiffLSHADE for SHADE and BatchedDiffMTSLS1 for LS1.
No Python loops over B — all ops are (B, N, ...) native.
"""
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.opt_variant import OptVariant
from encoder.batched_operators import BatchedDiffLSHADE, BatchedDiffMTSLS1
from encoder.gumbel import antithetic_gumbel_noise

log = logging.getLogger(__name__)


class ClassicK2Variant(OptVariant):
    """K=2: SHADE for all + LS1 selective via Gumbel-sigmoid mask.

    Router: cat(h * gate(h_global), h_global) → LayerNorm → MLP → logit
    Fully batched over B populations.
    """

    def __init__(self, gatv2_hidden=64, global_dim=32):
        super().__init__()
        self.shade = BatchedDiffLSHADE(embed_dim=gatv2_hidden, head_idx=0)
        self.ls1 = BatchedDiffMTSLS1(embed_dim=gatv2_hidden, head_idx=1)

        # Router: per-node LS1 logit
        self.gate = nn.Linear(global_dim, gatv2_hidden)
        self.norm = nn.LayerNorm(gatv2_hidden + global_dim)
        self.head = nn.Sequential(
            nn.Linear(gatv2_hidden + global_dim, gatv2_hidden),
            nn.ReLU(),
            nn.Linear(gatv2_hidden, 1),
        )
        self.head[-1].bias.data.fill_(0.0)

    def step(self, h, h_per_head, h_global, coords, fitness,
             cache, D, M=1, gumbel_tau=0.5, bounds_span=200.0, **kwargs):
        """SHADE + Gumbel-masked LS1, batched over B.

        Returns:
            delta: (M, B, N, D)
            extras: dict
        """
        B, N, H = h.shape

        # Router logits: (B, N)
        hg_exp = h_global.unsqueeze(1).expand(B, N, -1)  # (B, N, global_dim)
        gate = torch.sigmoid(self.gate(hg_exp))           # (B, N, H)
        x_dec = torch.cat([h * gate, hg_exp], dim=-1)     # (B, N, H+global_dim)
        logits = self.head(self.norm(x_dec)).squeeze(-1)   # (B, N)

        # Gumbel-sigmoid mask: (M, B, N)
        gumbel_noise = antithetic_gumbel_noise(M, B, N, device=h.device)

        y = (logits.unsqueeze(0) + gumbel_noise) / gumbel_tau
        mask_soft = torch.sigmoid(y)
        mask_hard = (mask_soft > 0.5).float()
        mask = (mask_hard - mask_soft).detach() + mask_soft  # STE

        # SHADE params + sample (batched)
        adj = getattr(cache, 'adj', None)
        knn_idx = getattr(cache, 'knn_idx', None)
        shade_params = self.shade.compute_params(
            h, coords, fitness.float(), adj, bounds_span=bounds_span,
            knn_idx=knn_idx)
        shade_delta = self.shade.sample_batch(shade_params, coords, bounds_span, M)  # (M, B, N, D)

        # LS1 params + sample (batched)
        ls1_params = self.ls1.compute_params(
            h, coords, fitness.float(), adj, bounds_span=bounds_span)
        ls1_delta = self.ls1.sample_batch(ls1_params, coords, bounds_span, M)  # (M, B, N, D)

        # Combine: SHADE + mask * LS1
        delta = shade_delta + mask.unsqueeze(-1) * ls1_delta

        # Entropy
        p = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()

        return delta, {
            'entropy': entropy.detach(),
            'mask_frac': mask_hard.mean().detach(),
        }
