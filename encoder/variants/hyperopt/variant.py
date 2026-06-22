"""HyperOPTK6Variant — K=6 kernel-outputting variant for the unified L2O system.

Composes: CrossDimAttention (shared) + 6 batched heads + KernelRouter.
3 structured heads (Sampling, Differential, Gradient) use learned PSD matrices.
3 discovery heads (NeighborRecomb, CoordField, Hypernet) are unconstrained.

Uses h: (B, N, gatv2_hidden) as primary input, not h_per_head.
Backbone n_heads is irrelevant to K=6.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from encoder.opt_variant import OptVariant

from .per_dim_features import compute_per_dim_features_batched
from .cross_dim_attention import BatchedCrossDimAttention
from .kernel_router import BatchedKernelRouter
from .structured_heads import (
    BatchedSamplingHead, BatchedDifferentialHead, BatchedGradientHead,
)
from .discovery_heads import (
    BatchedNeighborRecombHead, BatchedCoordFieldHead, BatchedHypernetHead,
)


class HyperOPTK6Variant(OptVariant):
    """K=6 kernel-outputting variant with expert-choice routing.

    3 structured + 3 discovery heads. All batched (B, N, D).
    """

    def __init__(self, gatv2_hidden: int = 128, K: int = 6,
                 k_rank: int = 8):
        super().__init__()
        self.K = K
        self.gatv2_hidden = gatv2_hidden

        # Shared cross-dim attention — routed_dim derived automatically
        self.cross_dim_attn = BatchedCrossDimAttention(embed_dim=gatv2_hidden)
        routed_dim = self.cross_dim_attn.out_dim  # attn_dim(32) + embed_dim

        # 6 heads: 3 structured + 3 discovery
        self.heads = nn.ModuleList([
            BatchedSamplingHead(routed_dim=routed_dim, k_rank=k_rank,
                                embed_dim=gatv2_hidden),
            BatchedDifferentialHead(routed_dim=routed_dim, k_rank=k_rank,
                                    embed_dim=gatv2_hidden),
            BatchedGradientHead(routed_dim=routed_dim, k_rank=k_rank,
                                embed_dim=gatv2_hidden),
            BatchedNeighborRecombHead(embed_dim=gatv2_hidden),
            BatchedCoordFieldHead(routed_dim=routed_dim, embed_dim=gatv2_hidden),
            BatchedHypernetHead(embed_dim=gatv2_hidden),
        ])

        # Expert-choice router
        self.router = BatchedKernelRouter(embed_dim=gatv2_hidden, K=K)

        # Phase tracking
        self._phase = 'free'

    STRUCTURED_INDICES = [0, 1, 2]   # Sampling, Differential, Gradient
    DISCOVERY_INDICES = [3, 4, 5]    # NeighborRecomb, CoordField, Hypernet

    def set_phase(self, phase: str):
        """Control staged head unfreezing.

        Phases:
            imitation:        Only structured heads + cross_dim_attn trainable.
                              Discovery heads + router frozen.
            discovery-warmup: All heads + router trainable (use routing_floor).
            free:             All trainable, no floor.
        """
        assert phase in ('imitation', 'discovery-warmup', 'free')
        self._phase = phase

        # Cross-dim attention: always trainable (shared across structured heads)
        for p in self.cross_dim_attn.parameters():
            p.requires_grad = True

        # Structured heads: always trainable
        for i in self.STRUCTURED_INDICES:
            for p in self.heads[i].parameters():
                p.requires_grad = True

        # Discovery heads + router: frozen in imitation, trainable otherwise
        discovery_trainable = phase != 'imitation'
        for i in self.DISCOVERY_INDICES:
            for p in self.heads[i].parameters():
                p.requires_grad = discovery_trainable
        for p in self.router.parameters():
            p.requires_grad = discovery_trainable

    def step(
        self,
        h: Tensor,
        h_per_head: Tensor,
        h_global: Tensor,
        coords: Tensor,
        fitness: Tensor,
        cache,
        D: int,
        M: int = 1,
        gumbel_tau: float = 1.0,
        bounds_span: float = 200.0,
        **kwargs,
    ) -> Tuple[Tensor, Dict]:
        """Produce M displacements via K=6 expert-choice routing.

        Args:
            h:          (B, N, gatv2_hidden)
            h_per_head: (B, N, n_heads, head_dim) — unused, K=6 > n_heads
            h_global:   (B, global_out)
            coords:     (B, N, D) float64
            fitness:    (B, N)
            cache:      TopologyCache with adj (B, N, N)
        Returns:
            delta: (M, B, N, D)
            extras: dict with routing_probs, entropy, lb_loss
        """
        B, N, _ = h.shape

        # 1. Per-dim features
        per_dim_feats = compute_per_dim_features_batched(
            coords, fitness
        ).to(h.dtype)  # (B, N, D, 5)

        # 2. Cross-dim attention -> shared routed repr
        routed = self.cross_dim_attn(per_dim_feats, h)  # (B, N, D, 96)

        # 3. Each head: compute_params (once, expensive)
        # Pass adj (dense) or knn_idx (sparse) depending on cache type
        adj_or_knn = getattr(cache, 'adj', None)
        if adj_or_knn is None:
            adj_or_knn = cache.knn_idx  # SparseTopologyCache

        params_list = [
            head.compute_params(routed, h, coords, fitness.float(),
                                adj_or_knn, bounds_span=bounds_span)
            for head in self.heads
        ]

        # 4. Router: deterministic expert-choice
        route_weights, soft_probs, lb_loss, logits = self.router(h)
        # route_weights: (B, N, K)

        # 5. Each head: sample_batch (M times, cheap)
        deltas_k = torch.stack([
            head.sample_batch(params, coords, bounds_span, M)
            for head, params in zip(self.heads, params_list)
        ], dim=3)  # (M, B, N, K, D)

        # 6. Weighted combination: broadcast route_weights over M
        # route_weights: (B, N, K) -> (1, B, N, K, 1) for broadcast
        w = route_weights.unsqueeze(0).unsqueeze(-1)  # (1, B, N, K, 1)
        delta = (w * deltas_k).sum(dim=3)  # (M, B, N, D)

        # Diagnostics
        entropy = -(soft_probs * soft_probs.clamp(min=1e-8).log()).sum(-1).mean()

        return delta, {
            'entropy': entropy.detach(),
            'routing_probs': soft_probs.detach(),
            'logits': logits.detach(),
            'lb_loss': lb_loss,
        }
