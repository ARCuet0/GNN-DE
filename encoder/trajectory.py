"""Multi-segment dovetailed BPTT trajectory runner.

For the legacy single-segment interface, use run_trajectory from
trajectory_single.py (re-exported here for backward compat).
"""
import logging
import random
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .opt_variant import OptVariant, GenerationStep
from .similarity_graph_gpu import build_dense_graphs_gpu

# Re-export legacy interface for backward compat
from .trajectory_single import run_trajectory, _ring_indices  # noqa: F401

log = logging.getLogger(__name__)


def run_trajectory_dovetailed(
    backbone: nn.Module,
    variant: OptVariant,
    fn_batch,
    f_optimal_batch: torch.Tensor,
    *,
    D: int,
    N: int,
    B: int = 1,
    n_gens: int,
    bptt_window: int,
    n_bptt_segments: int = None,
    bptt_position: str = 'random',
    expected_gens: int = None,
    gumbel_tau: float = 1.0,
    lambda_fes: float = 0.01,
    lb_coeff: float = None,
    graph_builder=None,
    device: str = 'cpu',
    grad_checkpoint: bool = False,
) -> Tuple[float, torch.Tensor, Dict]:
    """Run B optimization trajectories with multi-segment BPTT.

    Samples n_bptt_segments random segments for gradient, rest runs no_grad.
    """
    if graph_builder is None:
        graph_builder = build_dense_graphs_gpu

    if not isinstance(f_optimal_batch, torch.Tensor):
        f_optimal_batch = torch.tensor([f_optimal_batch], device=device,
                                        dtype=torch.float64)
    if f_optimal_batch.dim() == 0:
        f_optimal_batch = f_optimal_batch.unsqueeze(0)

    _is_batched = hasattr(fn_batch, 'B')

    def _eval_flat(x):
        if not _is_batched:
            return fn_batch(x)
        K_flat = x.shape[0]
        N_per = K_flat // B
        return fn_batch(x.view(B, N_per, -1)).reshape(K_flat)

    def _eval(x):
        if _is_batched:
            return fn_batch(x)
        B_cur, N_cur, D_cur = x.shape
        return fn_batch(x.reshape(-1, D_cur)).reshape(B_cur, N_cur)

    gen_step = GenerationStep(backbone, variant, eval_fn=_eval_flat,
                              lb=-100.0, ub=100.0)
    # CEC2017 hardcoded — init at line below uses `* 200 - 100`. Plumb
    # lb/ub through the function signature if BBOB / LSGO use is needed.

    coords = torch.rand(B, N, D, dtype=torch.float64, device=device) * 200 - 100
    fitness = _eval(coords)
    init_best = fitness.min(dim=-1).values

    tau_ema = (fitness - f_optimal_batch.unsqueeze(-1)).clamp(min=0).mean().item()
    tau_ema = max(tau_ema, 1.0)

    W = getattr(backbone, 'gru_window', 16)
    coords_ring = torch.zeros(B, W, N, D, dtype=torch.float32, device=device)
    fitness_ring = torch.zeros(B, W, N, dtype=torch.float32, device=device)

    expected_total_segments = max(1, n_gens // bptt_window)
    if n_bptt_segments is not None and n_bptt_segments < expected_total_segments:
        bptt_prob = n_bptt_segments / expected_total_segments
    else:
        bptt_prob = 1.0

    if lb_coeff is not None:
        lambda_fes = lb_coeff

    segment_losses = []
    segment_fes_weights = []
    active_fractions = []
    lb_losses = []
    n_grad_segments = 0
    cumulative_fes = 0
    K_OPS = getattr(variant, 'K', 4)
    winner_counts = torch.zeros(K_OPS, device=device)
    max_fes = n_gens * N

    import numpy as np
    _conv_checkpoints = sorted(set(
        int(x) for x in np.geomspace(100, max(max_fes, 101), num=15)
    ))
    _conv_next_idx = 0
    _conv_trajectory = {}
    max_gen = min(max_fes, 4 * n_gens)
    traj_t0 = time.perf_counter()
    TRAJ_TIMEOUT_S = 300
    HEARTBEAT_EVERY = 500

    _bptt_seg_decided = {}
    _bptt_count = 0
    _bptt_max = n_bptt_segments if n_bptt_segments is not None else float('inf')
    if bptt_position == 'last':
        _expected_end = min(expected_gens if expected_gens is not None else n_gens, n_gens)
        _bptt_start_gen = max(0, _expected_end - bptt_window)
        _last_seg_id = _bptt_start_gen // bptt_window
    else:
        _last_seg_id = -1

    _gap1_fes = None
    _f_opt_scalar = f_optimal_batch[0].item()
    _pending_loss = None
    for gen in range(max_gen):
        seg_id = gen // bptt_window
        if seg_id not in _bptt_seg_decided:
            if _bptt_count >= _bptt_max:
                _bptt_seg_decided[seg_id] = False
            elif bptt_position == 'last':
                is_last = (seg_id == _last_seg_id)
                _bptt_seg_decided[seg_id] = is_last
                if is_last:
                    _bptt_count += 1
            else:
                _bptt_seg_decided[seg_id] = (random.random() < bptt_prob)
                if _bptt_seg_decided[seg_id]:
                    _bptt_count += 1
        in_bptt = _bptt_seg_decided[seg_id]

        if gen > 0 and gen % bptt_window == 0:
            coords = coords.detach()
            fitness = fitness.detach()

        ri = gen % W
        coords_ring[:, ri] = coords.detach().float()
        fitness_ring[:, ri] = fitness.detach().float()
        n_valid = min(gen + 1, W)
        idx = _ring_indices(gen, W)
        coords_hist = coords_ring[0, idx]
        fitness_hist = fitness_ring[0, idx]

        with torch.no_grad():
            cache = graph_builder(
                coords.float(), fitness.float(),
                step_num=cumulative_fes, max_steps=max_fes, ndim=D, k_neighbors=8)

        temporal_kw = dict(coords_hist=coords_hist,
                           fitness_hist=fitness_hist, n_valid=n_valid)

        if in_bptt:
            if grad_checkpoint:
                _ch = coords_hist.clone()
                _fh = fitness_hist.clone()
                _nv = torch.tensor(n_valid)

                def _checkpointed_gen(coords_t, fitness_t, ch, fh, nv_t):
                    tkw = dict(coords_hist=ch, fitness_hist=fh,
                               n_valid=nv_t.item())
                    return gen_step.run(
                        coords=coords_t, fitness=fitness_t, cache=cache,
                        f_optimal=_f_opt_scalar, M=1, gumbel_tau=gumbel_tau,
                        node_feat=cache.node_feat, global_feat=cache.global_feat,
                        tau_ema=tau_ema, **tkw)
                result = torch_checkpoint(
                    _checkpointed_gen, coords, fitness, _ch, _fh, _nv,
                    use_reentrant=False)
            else:
                result = gen_step.run(
                    coords=coords, fitness=fitness, cache=cache,
                    f_optimal=_f_opt_scalar, M=1, gumbel_tau=gumbel_tau,
                    node_feat=cache.node_feat, global_feat=cache.global_feat,
                    tau_ema=tau_ema, **temporal_kw)
            _pending_loss = result['loss']
            extras = result.get('extras', {})
            if 'active_fraction' in extras:
                active_fractions.append(extras['active_fraction'])
            if 'lb_loss' in extras:
                lb_losses.append(extras['lb_loss'])
            if gen % bptt_window == 0:
                n_grad_segments += 1
        else:
            with torch.no_grad():
                result = gen_step.run(
                    coords=coords, fitness=fitness, cache=cache,
                    f_optimal=_f_opt_scalar, M=1, gumbel_tau=gumbel_tau,
                    node_feat=cache.node_feat, global_feat=cache.global_feat,
                    tau_ema=tau_ema, **temporal_kw)
            extras = result.get('extras', {})

        if 'tau_ema' in result:
            tau_ema = result['tau_ema']

        winner = extras.get('winner')
        if winner is not None:
            flat_w = winner.flatten()
            winner_counts += flat_w.bincount(minlength=K_OPS).float()

        coords = result['new_coords']
        fitness = result['new_fitness']

        fes_this_gen = max(extras.get('fes_used', float(N)), 1.0)
        cumulative_fes += fes_this_gen

        while (_conv_next_idx < len(_conv_checkpoints)
               and cumulative_fes >= _conv_checkpoints[_conv_next_idx]):
            _conv_trajectory[_conv_checkpoints[_conv_next_idx]] = \
                fitness.min(dim=-1).values.mean().item()
            _conv_next_idx += 1

        if in_bptt and _pending_loss is not None:
            segment_losses.append(_pending_loss)
            segment_fes_weights.append(cumulative_fes / max_fes)
            _pending_loss = None

        if (fitness.min(dim=-1).values < f_optimal_batch + 1.0).all():
            if _gap1_fes is None:
                _gap1_fes = cumulative_fes
            break

        if cumulative_fes >= max_fes:
            break

        if gen > 0 and gen % HEARTBEAT_EVERY == 0:
            traj_elapsed = time.perf_counter() - traj_t0
            log.debug("HEARTBEAT gen %d/%d | fes %d/%d | %.1fs elapsed",
                      gen, max_gen, int(cumulative_fes), max_fes, traj_elapsed)

        if gen > 0 and gen % 100 == 0:
            traj_elapsed = time.perf_counter() - traj_t0
            if traj_elapsed > TRAJ_TIMEOUT_S:
                log.warning("TIMEOUT gen %d/%d after %.1fs", gen, max_gen, traj_elapsed)
                break

    # Gap closure per batch element
    final_best = fitness.min(dim=-1).values
    gc_per_b = ((init_best - final_best) /
                (init_best - f_optimal_batch).abs().clamp(min=1e-8)).clamp(min=0.0)
    gc_per_b = torch.where(torch.isfinite(gc_per_b), gc_per_b,
                           torch.zeros_like(gc_per_b))
    gc = gc_per_b.mean().item()

    if not segment_losses:
        bptt_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        losses = torch.stack(segment_losses)
        weights = torch.tensor(segment_fes_weights, device=device,
                               dtype=losses.dtype)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = torch.ones_like(weights) / len(weights)
        bptt_loss = (losses * weights).sum()
        if active_fractions:
            bptt_loss = bptt_loss + lambda_fes * torch.stack(active_fractions).mean()
        if lb_losses:
            bptt_loss = bptt_loss + lambda_fes * torch.stack(lb_losses).mean()

    # Routing diagnostics
    routing_diag = {}
    if 'routing_probs' in extras:
        rp = extras['routing_probs']
        routing_diag['route_entropy'] = -(rp * rp.clamp(min=1e-8).log()).sum(-1).mean().item()
        routing_diag['route_max'] = rp.max(dim=-1).values.mean().item()
        routing_diag['route_argmax_counts'] = rp.argmax(dim=-1).flatten().bincount(
            minlength=rp.shape[-1]).tolist()
    if 'logits' in extras:
        lg = extras['logits']
        routing_diag['logit_mean'] = lg.mean().item()
        routing_diag['logit_std'] = lg.std().item()
        routing_diag['logit_absmax'] = lg.abs().max().item()
        routing_diag['logit_per_k_mean'] = lg.mean(dim=(0, 1)).tolist()
    if 'active_fraction' in extras:
        routing_diag['active_fraction'] = extras['active_fraction'].item()
    if 'lb_loss' in extras:
        routing_diag['lb_loss'] = extras['lb_loss'].item()
    routing_diag['winner_counts'] = winner_counts.tolist()

    final_gap = (final_best - f_optimal_batch).max().item()
    if final_gap < 0:
        log.warning("FITNESS BELOW OPTIMAL: final_gap=%.4f (best=%.4f, f*=%.4f)",
                     final_gap, final_best.min().item(), _f_opt_scalar)
        oob = (coords.abs() > 100).any(dim=-1)
        if oob.any():
            for row in oob.nonzero(as_tuple=False)[:5]:
                b, n = row[0].item(), row[1].item()
                x = coords[b, n]
                log.warning("  OOB B=%d N=%d: max|x|=%.4f", b, n, x.abs().max().item())
        else:
            log.warning("  All coords within [-100, 100]^D")

    stats = {
        'n_segments': n_grad_segments,
        'gc_per_batch': gc_per_b.tolist(),
        'routing': routing_diag,
        'fes_used': cumulative_fes,
        'max_fes': max_fes,
        'convergence': _conv_trajectory,
        'n_gens': gen + 1,
        'segment_fes_weights': segment_fes_weights,
        'final_best': final_best.mean().item(),
        'final_gap': final_gap,
        'gap1_fes': _gap1_fes,
    }

    return gc, bptt_loss, stats


def bptt_step(
    backbone: nn.Module,
    variant: OptVariant,
    fn_batch,
    f_optimal_batch: torch.Tensor,
    *,
    D: int,
    N: int,
    B: int = 1,
    n_gens: int = 20,
    bptt_window: int = 5,
    n_bptt_segments: int = None,
    bptt_position: str = 'random',
    expected_gens: int = None,
    gumbel_tau: float = 1.0,
    lambda_fes: float = 0.01,
    lb_coeff: float = None,
    focal_gamma: float = 0.0,
    graph_builder=None,
    device: str = 'cpu',
    grad_checkpoint: bool = False,
) -> Dict:
    """One multi-segment BPTT training step (no ES).

    grad_checkpoint: use torch.utils.checkpoint to save VRAM (~1.8x walltime).
    """
    gc, bptt_loss, stats = run_trajectory_dovetailed(
        backbone, variant, fn_batch, f_optimal_batch,
        D=D, N=N, B=B, n_gens=n_gens, bptt_window=bptt_window,
        bptt_position=bptt_position, expected_gens=expected_gens,
        n_bptt_segments=n_bptt_segments,
        gumbel_tau=gumbel_tau, lambda_fes=lambda_fes, lb_coeff=lb_coeff,
        graph_builder=graph_builder, device=device,
        grad_checkpoint=grad_checkpoint)

    raw_bptt_loss = bptt_loss.item()
    _f_opt = f_optimal_batch[0].item() if isinstance(f_optimal_batch, torch.Tensor) else float(f_optimal_batch)
    gap = stats['final_best'] - _f_opt
    if focal_gamma > 0 and gap >= 0:
        focal_weight = min(1.0, gap / focal_gamma)
    else:
        focal_weight = 1.0
    bptt_loss = bptt_loss * focal_weight

    bptt_loss.backward()

    grad_norm_bptt = 0.0
    for p in list(backbone.parameters()) + list(variant.parameters()):
        if p.grad is not None:
            grad_norm_bptt += p.grad.norm().item() ** 2
    grad_norm_bptt = grad_norm_bptt ** 0.5

    return {
        'gc': gc,
        'bptt_loss': bptt_loss.item(),
        'raw_bptt_loss': raw_bptt_loss,
        'focal_weight': focal_weight,
        'grad_norm_bptt': grad_norm_bptt,
        'n_segments': stats['n_segments'],
        'gc_per_batch': stats['gc_per_batch'],
        'routing': stats.get('routing', {}),
        'fes_used': stats.get('fes_used', 0),
        'max_fes': stats.get('max_fes', 0),
        'n_gens': stats.get('n_gens', 0),
        'final_best': stats.get('final_best', 0),
        'final_gap': stats.get('final_gap', -1),
        'gap1_fes': stats.get('gap1_fes'),
        'convergence': stats.get('convergence', {}),
    }
