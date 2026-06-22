import logging
import random
from typing import Dict

import torch
import torch.nn as nn

from .opt_variant import OptVariant
from .es_vmap import (make_perturbed_params, es_gradient_from_returns,
                      compute_es_snr,
                      _CombinedForward, collect_combined_params)
from .similarity_graph_gpu import build_dense_graphs_gpu
from .trajectory import run_trajectory

log = logging.getLogger(__name__)


def run_trajectory_es_batched(
    backbone: nn.Module,
    variant: OptVariant,
    fn,
    f_optimal: float,
    *,
    D: int,
    N: int,
    n_gens: int,
    perturbed_params: Dict[str, torch.Tensor],
    perturbed_buffers: Dict[str, torch.Tensor],
    graph_builder=None,
    device: str = 'cpu',
    init_coords=None,
    init_fitness=None,
    init_coords_ring=None,
    init_fitness_ring=None,
    init_ring_pos: int = 0,
):
    """Run M ES trajectories in parallel (no grad).

    All M perturbations share the same function but have different model
    weights. Populations diverge after gen 0 due to different decisions.

    Args:
        perturbed_params: {name: (M, *shape)} — combined backbone+variant params
        perturbed_buffers: {name: (M, *shape)} — expanded buffers
        init_coords: optional (1, N, D) starting population (reused across M)
        init_fitness: optional (1, N) starting fitness
        init_coords_ring: optional (1, W, N, D) temporal ring buffer from previous segment
        init_fitness_ring: optional (1, W, N) temporal ring buffer from previous segment
        init_ring_pos: ring buffer write position from previous segment

    Returns tuple:
        auc_M, gc_M, actual_gens, final_coords, final_fitness,
        final_coords_ring, final_fitness_ring, final_ring_pos
    """
    from .es_vmap import vmapped_gen_step

    if graph_builder is None:
        graph_builder = build_dense_graphs_gpu

    M = next(iter(perturbed_params.values())).shape[0]

    # Create wrapper once for the whole trajectory
    wrapper = _CombinedForward(backbone, variant, D, N)

    # Init M identical populations (or reuse from previous segment)
    if init_coords is not None:
        coords_1 = init_coords.to(dtype=torch.float64, device=device)
        fitness_1 = init_fitness.to(dtype=torch.float64, device=device)
        if coords_1.dim() == 2:
            coords_1 = coords_1.unsqueeze(0)
            fitness_1 = fitness_1.unsqueeze(0)
    else:
        coords_1 = torch.rand(1, N, D, dtype=torch.float64, device=device) * 200 - 100
        fitness_1 = fn(coords_1.squeeze(0)).unsqueeze(0)
    coords_M = coords_1.expand(M, -1, -1).contiguous()
    fitness_M = fitness_1.expand(M, -1).contiguous()

    init_best_M = fitness_M.min(dim=-1).values  # (M,)

    W = getattr(backbone, 'gru_window', 16)
    ring_offset = init_ring_pos
    if init_coords_ring is not None:
        coords_ring = init_coords_ring.to(dtype=torch.float32, device=device) \
            .expand(M, -1, -1, -1).contiguous()
        fitness_ring = init_fitness_ring.to(dtype=torch.float32, device=device) \
            .expand(M, -1, -1).contiguous()
    else:
        coords_ring = torch.zeros(M, W, N, D, dtype=torch.float32, device=device)
        fitness_ring = torch.zeros(M, W, N, dtype=torch.float32, device=device)

    # AUC accumulator: sum of log(best_fitness - f* + 1) over gens per perturbation
    auc_M = torch.zeros(M, device=device)
    max_fes = n_gens * N  # FES budget
    cumulative_fes = 0
    NOOP_IDX = 3
    max_gen = max_fes  # hard ceiling: 1 gen uses at least 1 FES

    with torch.no_grad():
        for gen in range(max_gen):
            ri = (ring_offset + gen) % W
            coords_ring[:, ri] = coords_M.float()
            fitness_ring[:, ri] = fitness_M.float()
            n_valid = min(ring_offset + gen + 1, W)
            wrapper._n_valid = n_valid

            coords_hist_M = coords_ring   # (M, W, N, D)
            fitness_hist_M = fitness_ring  # (M, W, N)

            cache_M = graph_builder(
                coords_M.float(), fitness_M.float(),
                step_num=cumulative_fes, max_steps=max_fes, ndim=D, k_neighbors=8)

            adj_or_knn_M = getattr(cache_M, 'knn_idx', None)
            if adj_or_knn_M is None:
                adj_or_knn_M = cache_M.adj

            delta_M, winner_M = vmapped_gen_step(
                wrapper,
                perturbed_params, perturbed_buffers,
                cache_M.node_feat, cache_M.global_feat,
                adj_or_knn_M, cache_M.edge_feat,
                coords_M, fitness_M,
                coords_hist_M, fitness_hist_M)

            # Only evaluate non-NoOp offspring; NoOp keep parent fitness
            noop_mask_M = (winner_M == NOOP_IDX)  # (M, N)
            offspring_M = (coords_M + delta_M).clamp(-100, 100)  # (M, N, D)
            off_fit_flat = fn(offspring_M.reshape(-1, D))  # (M*N,)
            off_fitness_M = off_fit_flat.reshape(M, N)
            off_fitness_M = torch.where(noop_mask_M, fitness_M, off_fitness_M)

            # FES: count non-NoOp evals from base model (perturbation 0)
            fes_this_gen = max(int((~noop_mask_M[0]).sum().item()), 1)
            cumulative_fes += fes_this_gen

            # Greedy selection
            improved = off_fitness_M < fitness_M
            coords_M = torch.where(improved.unsqueeze(-1), offspring_M, coords_M)
            fitness_M = torch.where(improved, off_fitness_M, fitness_M)

            # Accumulate AUC: log-gap at each gen (log compresses scale)
            best_M = fitness_M.min(dim=-1).values  # (M,)
            auc_M += torch.log1p((best_M - f_optimal).abs())

            # Early stop: base model converged or FES exhausted
            if (best_M[0] - f_optimal).abs() < 0.1:
                break
            if cumulative_fes >= max_fes:
                break

    # Normalize AUC by actual gens (rank normalization in es_gradient handles scale)
    actual_gens = gen + 1
    auc_M = auc_M / max(actual_gens, 1)

    # Log-gap reduction (well-defined even near optimum, for logging only)
    final_best_M = fitness_M.min(dim=-1).values  # (M,)
    log_gap_init = torch.log1p((init_best_M - f_optimal).abs())
    log_gap_final = torch.log1p((final_best_M - f_optimal).abs())
    gc_M = (log_gap_init - log_gap_final).clamp(min=0.0)

    # Return base model's population + ring buffer state for chaining
    final_ring_pos = (ring_offset + actual_gens) % W
    return (auc_M, gc_M, actual_gens,
            coords_M[0:1].detach(), fitness_M[0:1].detach(),
            coords_ring[0:1].detach(), fitness_ring[0:1].detach(),
            final_ring_pos)


# ======================================================================
# ES-Single step: one training update
# ======================================================================

def es_single_step(
    backbone: nn.Module,
    variant: OptVariant,
    fn,
    f_optimal: float,
    *,
    D: int,
    N: int,
    n_gens: int = 20,
    bptt_window: int = 5,
    M_es: int = 16,
    sigma: float = 0.01,
    lambda_es: float = 0.1,
    gumbel_tau: float = 1.0,
    lb_coeff: float = 0.1,
    graph_builder=None,
    device: str = 'cpu',
) -> Dict:
    """One hybrid ES-Single + BPTT training step.

    1. Run base model trajectory with BPTT on random segment → BPTT gradient
    2. Run M_es perturbed trajectories (no grad) → ES gradient
    3. Combine: BPTT_grad + lambda_es * ES_grad

    Returns dict with grad norms, gc, loss, segment info.
    """
    n_segments = max(1, n_gens // bptt_window)
    bptt_seg = random.randint(0, n_segments - 1)

    # ── 1. BPTT path: base model, random segment ──
    gc_base, bptt_loss, base_stats = run_trajectory(
        backbone, variant, fn, f_optimal,
        D=D, N=N, n_gens=n_gens, bptt_window=bptt_window,
        bptt_segment=bptt_seg, gumbel_tau=gumbel_tau,
        lb_coeff=lb_coeff,
        graph_builder=graph_builder, device=device)

    # Compute BPTT gradient
    bptt_loss.backward()

    # Save BPTT gradients
    bptt_grads = {}
    all_params = list(backbone.named_parameters()) + list(variant.named_parameters())
    grad_norm_bptt = 0.0
    for name, p in all_params:
        if p.grad is not None:
            bptt_grads[name] = p.grad.clone()
            grad_norm_bptt += p.grad.norm().item() ** 2
    grad_norm_bptt = grad_norm_bptt ** 0.5

    # ── 2. ES path: M perturbed trajectories in parallel (batched) ──
    combined_params, stacked_buffers = collect_combined_params(
        backbone, variant, M=M_es)
    perturbed, epsilons = make_perturbed_params(combined_params, M=M_es, sigma=sigma)

    auc_M, gc_M, *_ = run_trajectory_es_batched(
        backbone, variant, fn, f_optimal,
        D=D, N=N, n_gens=n_gens,
        perturbed_params=perturbed,
        perturbed_buffers=stacked_buffers,
        graph_builder=graph_builder, device=device)

    returns_t = auc_M  # higher AUC = worse → ES minimizes this
    es_snr = compute_es_snr(returns_t, epsilons)
    es_grad = es_gradient_from_returns(returns_t, epsilons, sigma=sigma)

    grad_norm_es = sum(g.norm().item() ** 2 for g in es_grad.values()) ** 0.5

    # ── 3. Combine gradients (direction-normalized) ──
    # Build ordered list of (unprefixed_name, prefixed_key, param) for both
    all_named = []
    for n, p in backbone.named_parameters():
        all_named.append((n, f'backbone.{n}', p))
    for n, p in variant.named_parameters():
        all_named.append((n, f'variant.{n}', p))

    # Flatten both to compute cosine similarity and normalize
    bptt_flat = torch.cat([bptt_grads.get(n, torch.zeros_like(p)).flatten()
                           for n, _, p in all_named])
    es_flat = torch.cat([es_grad.get(k, torch.zeros_like(p)).flatten()
                         for _, k, p in all_named])

    bptt_norm = bptt_flat.norm().clamp(min=1e-10)
    es_norm = es_flat.norm().clamp(min=1e-10)

    # Cosine similarity (diagnostic — how complementary are they?)
    cos_sim = (bptt_flat @ es_flat) / (bptt_norm * es_norm)

    # Normalize to unit directions, then combine
    bptt_dir = bptt_flat / bptt_norm
    es_dir = es_flat / es_norm
    combined_flat = bptt_dir + lambda_es * es_dir

    # Write combined gradient back to params
    offset = 0
    for _, _, p in all_named:
        numel = p.numel()
        p.grad = combined_flat[offset:offset + numel].reshape(p.shape)
        offset += numel

    grad_norm_combined = combined_flat.norm().item()

    result = {
        'gc': gc_base,
        'bptt_loss': bptt_loss.item(),
        'grad_norm_bptt': grad_norm_bptt,
        'grad_norm_es': grad_norm_es,
        'grad_norm_combined': grad_norm_combined,
        'cos_sim_bptt_es': cos_sim.item(),
        'bptt_segment': bptt_seg,
        'es_gc_mean': gc_M.mean().item(),
        'es_gc_std': gc_M.std().item(),
        'es_snr': es_snr,
    }
    # Propagate routing diagnostics from run_trajectory
    if 'routing' in base_stats:
        result['routing'] = base_stats['routing']
    return result


def es_only_step(
    backbone: nn.Module,
    variant: OptVariant,
    fn,
    f_optimal: float,
    *,
    D: int,
    N: int,
    n_gens: int = 20,
    M_es: int = 16,
    sigma: float = 0.01,
    graph_builder=None,
    device: str = 'cpu',
    init_coords=None,
    init_fitness=None,
    init_coords_ring=None,
    init_fitness_ring=None,
    init_ring_pos: int = 0,
) -> Dict:
    """Pure ES-Single training step (no BPTT).

    1. Run M_es perturbed trajectories (no grad) → ES gradient
    2. Write gradient directly to params

    Pass init_coords/init_fitness/ring state to chain from a previous segment.
    Returns final state for chaining to the next.
    """
    combined_params, stacked_buffers = collect_combined_params(
        backbone, variant, M=M_es)
    perturbed, epsilons = make_perturbed_params(combined_params, M=M_es, sigma=sigma)

    (auc_M, gc_M, actual_gens,
     final_coords, final_fitness,
     final_coords_ring, final_fitness_ring,
     final_ring_pos) = run_trajectory_es_batched(
        backbone, variant, fn, f_optimal,
        D=D, N=N, n_gens=n_gens,
        perturbed_params=perturbed,
        perturbed_buffers=stacked_buffers,
        graph_builder=graph_builder, device=device,
        init_coords=init_coords, init_fitness=init_fitness,
        init_coords_ring=init_coords_ring,
        init_fitness_ring=init_fitness_ring,
        init_ring_pos=init_ring_pos)

    returns_t = auc_M  # higher AUC = worse → ES minimizes this
    es_snr = compute_es_snr(returns_t, epsilons)
    es_grad = es_gradient_from_returns(returns_t, epsilons, sigma=sigma)

    grad_norm_es = sum(g.norm().item() ** 2 for g in es_grad.values()) ** 0.5

    # Write ES gradient to params
    all_named = []
    for n, p in backbone.named_parameters():
        all_named.append((f'backbone.{n}', p))
    for n, p in variant.named_parameters():
        all_named.append((f'variant.{n}', p))

    for key, p in all_named:
        if key in es_grad:
            p.grad = es_grad[key]

    init_best = (init_fitness.min().item() if init_fitness is not None
                 else final_coords.new_tensor(float('inf')).item())

    final_best_val = final_fitness.min().item()
    final_gap = final_best_val - f_optimal

    return {
        'gc': gc_M.mean().item(),
        'bptt_loss': 0.0,
        'grad_norm_bptt': 0.0,
        'grad_norm_es': grad_norm_es,
        'es_gc_mean': gc_M.mean().item(),
        'es_gc_std': gc_M.std().item(),
        'es_auc_mean': auc_M.mean().item(),
        'es_auc_std': auc_M.std().item(),
        'es_snr': es_snr,
        'actual_gens': actual_gens,
        'final_coords': final_coords,
        'final_fitness': final_fitness,
        'final_coords_ring': final_coords_ring,
        'final_fitness_ring': final_fitness_ring,
        'final_ring_pos': final_ring_pos,
        'final_best': final_best_val,
        'final_gap': final_gap,
        'init_best': init_best,
    }


def es_bptt_step(
    backbone: nn.Module,
    variant: OptVariant,
    fn,
    f_optimal: float,
    *,
    D: int,
    N: int,
    n_gens: int = 200,
    bptt_window: int = 50,
    M_es: int = 16,
    sigma: float = 0.1,
    lambda_es: float = 0.1,
    gumbel_tau: float = 1.0,
    expected_gens: int = None,
    graph_builder=None,
    device: str = 'cpu',
) -> Dict:
    """ES + BPTT-final hybrid step.

    expected_gens: from convergence history, positions BPTT window.
    1. Run full trajectory with BPTT on LAST segment only → BPTT gradient
    2. Run M ES perturbations (full no-grad) → ES gradient
    3. Combine: direction-normalized bptt_dir + lambda_es * es_dir
    """
    from .trajectory import bptt_step

    # ── 1. BPTT path: last segment only ──
    gc_base, bptt_loss_val, bptt_stats = None, 0.0, {}
    bptt_result = bptt_step(
        backbone, variant, fn, f_optimal,
        D=D, N=N, B=1, n_gens=n_gens, bptt_window=bptt_window,
        n_bptt_segments=1, bptt_position='last',
        expected_gens=expected_gens,
        gumbel_tau=gumbel_tau,
        graph_builder=graph_builder, device=device)

    # Save BPTT gradients before ES overwrites them
    bptt_grads = {}
    all_params = list(backbone.named_parameters()) + list(variant.named_parameters())
    grad_norm_bptt = 0.0
    for name, p in all_params:
        if p.grad is not None:
            bptt_grads[name] = p.grad.clone()
            grad_norm_bptt += p.grad.norm().item() ** 2
    grad_norm_bptt = grad_norm_bptt ** 0.5

    # ── 2. ES path: M perturbed trajectories ──
    combined_params, stacked_buffers = collect_combined_params(
        backbone, variant, M=M_es)
    perturbed, epsilons = make_perturbed_params(combined_params, M=M_es, sigma=sigma)

    auc_M, gc_M, *_ = run_trajectory_es_batched(
        backbone, variant, fn, f_optimal,
        D=D, N=N, n_gens=n_gens,
        perturbed_params=perturbed,
        perturbed_buffers=stacked_buffers,
        graph_builder=graph_builder, device=device)

    returns_t = auc_M
    es_snr = compute_es_snr(returns_t, epsilons)
    es_grad = es_gradient_from_returns(returns_t, epsilons, sigma=sigma)
    grad_norm_es = sum(g.norm().item() ** 2 for g in es_grad.values()) ** 0.5

    # ── 3. Combine gradients (direction-normalized) ──
    all_named = []
    for n, p in backbone.named_parameters():
        all_named.append((n, f'backbone.{n}', p))
    for n, p in variant.named_parameters():
        all_named.append((n, f'variant.{n}', p))

    bptt_flat = torch.cat([bptt_grads.get(n, torch.zeros_like(p)).flatten()
                           for n, _, p in all_named])
    es_flat = torch.cat([es_grad.get(k, torch.zeros_like(p)).flatten()
                         for _, k, p in all_named])

    bptt_norm = bptt_flat.norm().clamp(min=1e-10)
    es_norm = es_flat.norm().clamp(min=1e-10)

    cos_sim = (bptt_flat @ es_flat) / (bptt_norm * es_norm)

    bptt_dir = bptt_flat / bptt_norm
    es_dir = es_flat / es_norm
    combined_flat = bptt_dir + lambda_es * es_dir

    # Write combined gradient to params
    offset = 0
    for _, _, p in all_named:
        numel = p.numel()
        p.grad = combined_flat[offset:offset + numel].reshape(p.shape)
        offset += numel

    # ── 4. Detailed diagnostics ──
    diagnostics = {}

    # Per-module gradient norms (BPTT vs ES)
    for name, p in all_params:
        bptt_g = bptt_grads.get(name)
        es_key = f'backbone.{name}' if name in dict(backbone.named_parameters()) else f'variant.{name}'
        es_g = es_grad.get(es_key)
        if bptt_g is not None and es_g is not None:
            diagnostics[f'grad/{name}/bptt_norm'] = bptt_g.norm().item()
            diagnostics[f'grad/{name}/es_norm'] = es_g.norm().item()
            cos = (bptt_g.flatten() @ es_g.flatten()) / (bptt_g.norm() * es_g.norm()).clamp(min=1e-10)
            diagnostics[f'grad/{name}/cos'] = cos.item()

    # Aggregate: backbone vs variant gradient contribution
    bb_bptt = sum(bptt_grads[n].norm().item() ** 2
                  for n, _ in backbone.named_parameters() if n in bptt_grads) ** 0.5
    bb_es = sum(es_grad.get(f'backbone.{n}', torch.zeros(1)).norm().item() ** 2
                for n, _ in backbone.named_parameters()) ** 0.5
    var_bptt = sum(bptt_grads[n].norm().item() ** 2
                   for n, _ in variant.named_parameters() if n in bptt_grads) ** 0.5
    var_es = sum(es_grad.get(f'variant.{n}', torch.zeros(1)).norm().item() ** 2
                 for n, _ in variant.named_parameters()) ** 0.5
    diagnostics['grad/backbone_bptt'] = bb_bptt
    diagnostics['grad/backbone_es'] = bb_es
    diagnostics['grad/variant_bptt'] = var_bptt
    diagnostics['grad/variant_es'] = var_es

    # Routing differentiation: how much does routing vary across nodes?
    routing = bptt_result.get('routing', {})
    rp = routing.get('route_argmax_counts', [])
    if rp:
        total = max(sum(rp), 1)
        diagnostics['routing/DE_pct'] = 100 * rp[0] / total
        diagnostics['routing/LS1_pct'] = 100 * rp[1] / total if len(rp) > 1 else 0
        diagnostics['routing/CMA_pct'] = 100 * rp[2] / total if len(rp) > 2 else 0
        diagnostics['routing/NoOp_pct'] = 100 * rp[3] / total if len(rp) > 3 else 0
    diagnostics['routing/entropy'] = routing.get('route_entropy', 0)
    diagnostics['routing/max_prob'] = routing.get('route_max', 0)
    diagnostics['routing/logit_absmax'] = routing.get('logit_absmax', 0)
    logit_per_k = routing.get('logit_per_k_mean', [])
    if logit_per_k:
        for k, v in enumerate(logit_per_k):
            diagnostics[f'routing/logit_k{k}_mean'] = v

    # ES perturbation quality
    diagnostics['es/auc_mean'] = auc_M.mean().item()
    diagnostics['es/auc_std'] = auc_M.std().item()
    diagnostics['es/auc_min'] = auc_M.min().item()
    diagnostics['es/auc_max'] = auc_M.max().item()
    diagnostics['es/gc_mean'] = gc_M.mean().item()
    diagnostics['es/gc_std'] = gc_M.std().item()

    return {
        'gc': bptt_result['gc'],
        'bptt_loss': bptt_result['bptt_loss'],
        'grad_norm_bptt': grad_norm_bptt,
        'grad_norm_es': grad_norm_es,
        'grad_norm_combined': combined_flat.norm().item(),
        'cos_sim_bptt_es': cos_sim.item(),
        'es_gc_mean': gc_M.mean().item(),
        'es_gc_std': gc_M.std().item(),
        'es_auc_mean': auc_M.mean().item(),
        'es_auc_std': auc_M.std().item(),
        'es_snr': es_snr,
        'routing': routing,
        'final_best': bptt_result.get('final_best', 0),
        'final_gap': bptt_result.get('final_gap', -1),
        'n_gens_actual': bptt_result.get('n_gens', n_gens),
        'diagnostics': diagnostics,
    }
