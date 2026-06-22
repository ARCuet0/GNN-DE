"""Single-segment BPTT trajectory runner (legacy interface).

Used by older training scripts and probes. For the modern multi-segment
dovetailed interface, use run_trajectory_dovetailed from trajectory.py.
"""
import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .opt_variant import OptVariant, GenerationStep
from .similarity_graph_gpu import build_dense_graphs_gpu


def _ring_indices(gen: int, W: int) -> list:
    """Ordered indices into a ring buffer of size W at generation gen."""
    n_valid = min(gen + 1, W)
    ri = gen % W
    if n_valid >= W:
        start = (ri + 1) % W
        return [(start + i) % W for i in range(W)]
    return list(range(n_valid))


def run_trajectory(
    backbone: nn.Module,
    variant: OptVariant,
    fn,
    f_optimal: float,
    *,
    D: int,
    N: int,
    n_gens: int,
    bptt_window: int,
    bptt_segment: int,
    gumbel_tau: float = 1.0,
    lambda_fes: float = 0.01,
    lb_coeff: float = None,
    diagnostics: bool = False,
    graph_builder=None,
    device: str = 'cpu',
) -> Tuple[float, torch.Tensor, Dict]:
    """Run a full optimization trajectory with BPTT on one segment.

    Returns:
        gc_final: gap closure at end of trajectory (float, for ES)
        bptt_loss: differentiable loss from the BPTT segment (for backward)
        stats: dict with diagnostics
    """
    if graph_builder is None:
        graph_builder = build_dense_graphs_gpu

    B = 1
    bptt_start = bptt_segment * bptt_window
    bptt_end = min(bptt_start + bptt_window, n_gens)

    gen_step = GenerationStep(backbone, variant, eval_fn=fn,
                              lb=-100.0, ub=100.0)
    # CEC2017 hardcoded; init below uses `* 200 - 100`.

    coords = torch.rand(B, N, D, dtype=torch.float64, device=device) * 200 - 100
    fitness = fn(coords.squeeze(0)).unsqueeze(0)
    init_best = fitness.min().item()

    W = getattr(backbone, 'gru_window', 16)
    coords_ring = torch.zeros(W, N, D, dtype=torch.float32, device=device)
    fitness_ring = torch.zeros(W, N, dtype=torch.float32, device=device)

    if lb_coeff is not None:
        lambda_fes = lb_coeff

    bptt_losses = []
    active_fractions = []
    lb_losses = []

    diag_buf = None
    if diagnostics:
        from .diagnostics import DiagnosticBuffer
        K = getattr(variant, 'K', 4)
        diag_buf = DiagnosticBuffer(n_gens=n_gens, N=N, K=K)

    for gen in range(n_gens):
        in_bptt = bptt_start <= gen < bptt_end

        ri = gen % W
        coords_ring[ri] = coords.squeeze(0).detach().float()
        fitness_ring[ri] = fitness.squeeze(0).detach().float()
        n_valid = min(gen + 1, W)
        idx = _ring_indices(gen, W)
        coords_hist = coords_ring[idx]
        fitness_hist = fitness_ring[idx]

        with torch.no_grad():
            cache = graph_builder(
                coords.float(), fitness.float(),
                step_num=gen, max_steps=n_gens, ndim=D, k_neighbors=8)

        temporal_kw = dict(coords_hist=coords_hist,
                           fitness_hist=fitness_hist, n_valid=n_valid)

        if in_bptt:
            result = gen_step.run(
                coords=coords, fitness=fitness, cache=cache,
                f_optimal=f_optimal, M=1, gumbel_tau=gumbel_tau,
                node_feat=cache.node_feat, global_feat=cache.global_feat,
                **temporal_kw)
            bptt_losses.append(result['loss'])
            extras = result.get('extras', {})
            if 'active_fraction' in extras:
                active_fractions.append(extras['active_fraction'])
            if 'lb_loss' in extras:
                lb_losses.append(extras['lb_loss'])
            if diag_buf is not None:
                rp = extras.get('routing_probs')
                if rp is not None:
                    seg_pos = gen - bptt_start
                    diag_buf.record(rp, coords, fitness,
                                    in_bptt=True, bptt_seg_pos=seg_pos)
            coords = result['new_coords']
            fitness = result['new_fitness'].detach()
        else:
            with torch.no_grad():
                result = gen_step.run(
                    coords=coords, fitness=fitness, cache=cache,
                    f_optimal=f_optimal, M=1,
                    node_feat=cache.node_feat, global_feat=cache.global_feat,
                    **temporal_kw)
                if diag_buf is not None:
                    rp = result.get('extras', {}).get('routing_probs')
                    if rp is not None:
                        diag_buf.record(rp, coords, fitness,
                                        in_bptt=False, bptt_seg_pos=-1)
                coords = result['new_coords']
                fitness = result['new_fitness']

    # Gap closure
    final_best = fitness.min().item()
    if not math.isfinite(final_best):
        gc = 0.0
    else:
        gc = max(0.0, (init_best - final_best) / max(abs(init_best - f_optimal), 1e-8))

    if bptt_losses:
        bptt_loss = torch.stack(bptt_losses).mean()
    else:
        bptt_loss = torch.tensor(0.0, device=device, requires_grad=True)

    if active_fractions:
        bptt_loss = bptt_loss + lambda_fes * torch.stack(active_fractions).mean()
    if lb_losses:
        bptt_loss = bptt_loss + lambda_fes * torch.stack(lb_losses).mean()

    # Routing diagnostics from last BPTT gen
    routing_diag = {}
    if active_fractions or lb_losses:
        last_extras = extras
        if 'routing_probs' in last_extras:
            rp = last_extras['routing_probs']
            routing_diag['route_entropy'] = -(rp * rp.clamp(min=1e-8).log()).sum(-1).mean().item()
            routing_diag['route_max'] = rp.max(dim=-1).values.mean().item()
            routing_diag['route_argmax_counts'] = rp.argmax(dim=-1).flatten().bincount(
                minlength=rp.shape[-1]).tolist()
        if 'logits' in last_extras:
            lg = last_extras['logits']
            routing_diag['logit_mean'] = lg.mean().item()
            routing_diag['logit_std'] = lg.std().item()
            routing_diag['logit_absmax'] = lg.abs().max().item()
            routing_diag['logit_per_k_mean'] = lg.mean(dim=(0, 1)).tolist()
        if 'active_fraction' in last_extras:
            routing_diag['active_fraction'] = last_extras['active_fraction'].item()
        if 'lb_loss' in last_extras:
            routing_diag['lb_loss'] = last_extras['lb_loss'].item()

    stats = {
        'init_best': init_best, 'final_best': final_best,
        'routing': routing_diag,
    }
    if diag_buf is not None:
        stats['diag_buffer'] = diag_buf

    return gc, bptt_loss, stats
