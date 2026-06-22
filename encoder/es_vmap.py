"""
es_vmap.py — Vectorized ES perturbations via torch.func.vmap.

Replaces the sequential M-perturbation loop with a single GPU kernel.
Works with any nn.Module that has a forward(node_feat, global_feat, cache) signature
(specifically DenseGATv2Backbone and TemporalDenseGATv2Backbone).

Key functions:
    make_perturbed_params: create M antithetic perturbations of model params
    vmapped_forward: run M forward passes in one vmap call
    es_gradient_from_returns: compute ES gradient with rank normalization
"""
from typing import Dict, Tuple

import torch
from torch.func import functional_call, vmap

from .dense_gatv2_backbone import TopologyCache


def make_perturbed_params(
    base_params: Dict[str, torch.Tensor],
    M: int,
    sigma: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Create M antithetic perturbations of base model parameters.

    Args:
        base_params: dict {name: param_tensor} — the base (unperturbed) params
        M: total number of perturbations (must be even for antithetic)
        sigma: perturbation standard deviation

    Returns:
        perturbed: dict {name: (M, *param_shape)} — stacked perturbed params
        epsilons:  dict {name: (M, *param_shape)} — noise vectors (for gradient)
    """
    assert M % 2 == 0, f"M must be even for antithetic sampling, got {M}"
    M_half = M // 2

    perturbed = {}
    epsilons = {}

    for name, p in base_params.items():
        # Antithetic: first half random, second half negated
        eps_half = torch.randn(M_half, *p.shape, device=p.device, dtype=p.dtype)
        eps = torch.cat([eps_half, -eps_half], dim=0)  # (M, *shape)

        perturbed[name] = p.unsqueeze(0) + sigma * eps  # (M, *shape)
        epsilons[name] = eps

    return perturbed, epsilons


def vmapped_forward(
    model: torch.nn.Module,
    perturbed_params: Dict[str, torch.Tensor],
    base_buffers: Dict[str, torch.Tensor],
    node_feat: torch.Tensor,
    global_feat: torch.Tensor,
    cache,
) -> torch.Tensor:
    """Run M forward passes in one vmap kernel.

    Args:
        model: nn.Module (must be in eval mode, no dropout randomness)
        perturbed_params: dict {name: (M, *shape)} — from make_perturbed_params
        base_buffers: dict {name: tensor} — model buffers (shared across M)
        node_feat: (B, N, node_in)
        global_feat: (B, global_in)
        cache: TopologyCache

    Returns:
        h_global: (M, B, global_out_dim) — global embeddings for each perturbation
    """
    # Stack buffers with M dim (vmap needs all inputs batched or marked None)
    M = next(iter(perturbed_params.values())).shape[0]
    if not base_buffers:
        stacked_buffers = {}
    else:
        # expand without clone — LayerNorm has no mutable buffers
        stacked_buffers = {n: b.unsqueeze(0).expand(M, *b.shape)
                           for n, b in base_buffers.items()}

    def single_forward(params, buffers):
        out = functional_call(model, (params, buffers),
                              (node_feat, global_feat, cache))
        # Backbone returns BackboneOutput NamedTuple; extract h_global by field.
        # (vmap can't unpack NamedTuples by attribute inside — access here.)
        if hasattr(out, 'h_global'):
            return out.h_global
        # Legacy 4-tuple fallback
        return out[3]

    h_global = vmap(single_forward)(perturbed_params, stacked_buffers)

    return h_global


def collect_combined_params(backbone, variant, M=None):
    """Collect prefixed params and buffers from backbone + variant.

    Returns (combined_params, stacked_buffers).
    If M is given, buffers are expanded to (M, *shape).
    """
    combined = {f'backbone.{n}': p.detach()
                for n, p in backbone.named_parameters()}
    combined.update({f'variant.{n}': p.detach()
                     for n, p in variant.named_parameters()})

    bufs = {f'backbone.{n}': b for n, b in backbone.named_buffers()}
    bufs.update({f'variant.{n}': b for n, b in variant.named_buffers()})

    if M is not None and bufs:
        stacked = {n: b.unsqueeze(0).expand(M, *b.shape) for n, b in bufs.items()}
    else:
        stacked = bufs

    return combined, stacked


class _CombinedForward(torch.nn.Module):
    """Thin wrapper combining backbone + variant for vmap+functional_call.

    forward() receives raw tensors (no TopologyCache), reconstructs
    a B=1 cache internally, and returns delta (N, D).

    Supports both dense (adj: N,N) and sparse (adj_or_knn: N,k) caches.
    The wrapper auto-detects based on tensor shape.
    """

    def __init__(self, backbone, variant, D, N):
        super().__init__()
        self.backbone = backbone
        self.variant = variant
        self._D = D
        self._N = N
        self._n_valid = 1  # Set per-gen before vmap call

    def forward(self, node_feat, global_feat, adj_or_knn, edge_feat,
                coords, fitness, coords_hist, fitness_hist):
        """Single-perturbation forward: raw tensors → delta (N, D).

        All inputs are unbatched (no M dim — vmap adds it).
        adj_or_knn: (N, N) bool for dense, or (N, k) long for sparse.
        edge_feat:  (N, N, E) for dense, or (N, k, E) for sparse.
        """
        # Auto-detect dense vs sparse from adj shape
        if adj_or_knn.shape[0] == adj_or_knn.shape[1]:
            # Dense: adj is (N, N)
            cache = TopologyCache(
                adj=adj_or_knn.unsqueeze(0),
                edge_feat=edge_feat.unsqueeze(0),
                B=1, N=self._N,
                node_feat=node_feat.unsqueeze(0),
                global_feat=global_feat.unsqueeze(0),
            )
        else:
            # Sparse: knn_idx is (N, k) where k < N
            from .sparse_gatv2_backbone import SparseTopologyCache
            k = adj_or_knn.shape[1]
            cache = SparseTopologyCache(
                knn_idx=adj_or_knn.unsqueeze(0),
                edge_feat=edge_feat.unsqueeze(0),
                B=1, N=self._N, k=k,
                node_feat=node_feat.unsqueeze(0),
                global_feat=global_feat.unsqueeze(0),
            )

        out = self.backbone.encode(
            node_feat.unsqueeze(0), global_feat.unsqueeze(0), cache,
            coords_hist=coords_hist, fitness_hist=fitness_hist,
            n_valid=self._n_valid)

        # BackboneOutput field access (new) with 4-tuple fallback (legacy)
        if hasattr(out, 'h_global'):
            h, h_per_head, h_global = out.h, out.h_per_head, out.h_global
            donor_logits = out.donor_logits
        else:
            h, _e, h_per_head, h_global = out[:4]
            donor_logits = None

        delta, extras = self.variant.step(
            h, h_per_head, h_global,
            coords.unsqueeze(0), fitness.unsqueeze(0),
            cache, self._D, M=1, gumbel_tau=0.5,
            donor_logits=donor_logits)

        winner = extras.get('winner', torch.zeros(1, 1, self._N, dtype=torch.long,
                                                   device=coords.device))
        return delta.squeeze(0).squeeze(0), winner.squeeze(0).squeeze(0)


def vmapped_gen_step(wrapper, perturbed_params, perturbed_buffers,
                     node_feat_M, global_feat_M, adj_M, edge_feat_M,
                     coords_M, fitness_M,
                     coords_hist_M, fitness_hist_M):
    """M parallel backbone+variant forward passes via vmap+functional_call.

    Args:
        wrapper: _CombinedForward instance (created once, reused across gens).
                 Set wrapper._n_valid before calling.
        coords_hist_M: (M, n_valid, N, D)
        fitness_hist_M: (M, n_valid, N)

    Returns:
        delta_M: (M, N, D)
    """
    def single_step(params, buffers, nf, gf, adj, ef, c, f, ch, fh):
        return functional_call(wrapper, (params, buffers),
                               (nf, gf, adj, ef, c, f, ch, fh))

    delta_M, winner_M = vmap(single_step, randomness='different')(
        perturbed_params, perturbed_buffers,
        node_feat_M, global_feat_M, adj_M, edge_feat_M,
        coords_M, fitness_M,
        coords_hist_M, fitness_hist_M)
    return delta_M, winner_M


def es_gradient_from_returns(
    returns: torch.Tensor,
    epsilons: Dict[str, torch.Tensor],
    sigma: float,
) -> Dict[str, torch.Tensor]:
    """Compute ES gradient with rank normalization (Salimans 2017).

    Args:
        returns: (M,) scalar returns per perturbation (lower = better)
        epsilons: dict {name: (M, *shape)} — noise vectors from make_perturbed_params
        sigma: perturbation std (for scaling)

    Returns:
        grad: dict {name: param_shape} — ES gradient estimate per parameter
    """
    M = returns.shape[0]

    # Rank normalization: replace returns with centered ranks
    # Ranks in [0, M-1], normalized to [-0.5, 0.5]
    ranks = returns.argsort().argsort().float()  # (M,)
    ranks = (ranks / (M - 1)) - 0.5  # centered in [-0.5, 0.5]

    # ES gradient: g = (1 / (M * sigma)) * sum(rank_i * eps_i)
    grad = {}
    for name, eps in epsilons.items():
        # ranks: (M,) -> reshape to broadcast with eps: (M, *shape)
        r = ranks.view(M, *([1] * (eps.dim() - 1)))  # (M, 1, 1, ...)
        grad[name] = (r * eps).sum(dim=0) / (M * sigma)

    return grad


def compute_es_snr(
    returns: torch.Tensor,
    epsilons: Dict[str, torch.Tensor],
) -> float:
    """Compute gradient Signal-to-Noise Ratio for an ES step.

    SNR = ||mean(ε * shaped)|| / mean(std(ε * shaped))

    Higher SNR means the gradient estimate is more reliable.
    Uses the same rank normalization as es_gradient_from_returns.

    Args:
        returns: (M,) scalar returns per perturbation
        epsilons: dict {name: (M, *shape)} — noise vectors

    Returns:
        SNR as a positive float.
    """
    M = returns.shape[0]

    # Same rank normalization as es_gradient_from_returns
    ranks = returns.argsort().argsort().float()
    shaped = (ranks / (M - 1)) - 0.5  # centered in [-0.5, 0.5]

    # Flatten all epsilons into one (M, n_params) tensor
    flat_eps = torch.cat(
        [eps.reshape(M, -1) for eps in epsilons.values()], dim=1
    )  # (M, n_params)

    with torch.no_grad():
        contrib = flat_eps * shaped.unsqueeze(1)  # (M, n_params)
        grad_mean = contrib.mean(dim=0)           # (n_params,)
        grad_std = contrib.std(dim=0).mean()      # scalar
        # Standard error of the mean decreases as 1/sqrt(M),
        # so estimation SNR = ||mean|| / (std / sqrt(M))
        snr = (grad_mean.norm() * (M ** 0.5) / (grad_std + 1e-10)).item()

    return snr
