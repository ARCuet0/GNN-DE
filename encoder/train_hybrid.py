"""
train_hybrid.py — Training loop for L2O with ES-Single + BPTT.

Supports 'hybrid' (ES+BPTT) and 'bptt' (dovetailed only) modes.

Usage:
    python -m encoder.train_hybrid --device cuda --n-steps 2000
"""
import json
import logging
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from .cec2017_torch import CATEGORIES
from .opt_variant import OptVariant

# Re-export all public names so existing imports keep working
from .trajectory import (  # noqa: F401
    _ring_indices, run_trajectory,
    run_trajectory_dovetailed, bptt_step,
)
from .es_training import (  # noqa: F401
    run_trajectory_es_batched, es_single_step, es_only_step, es_bptt_step,
)
from .training_utils import (  # noqa: F401
    _sample_category_balanced, l2sp_loss,
    _make_ckpt, _build_validation_set, _run_validation,
    _auto_tune_bptt_segments, _auto_tune_es, _auto_tune_bptt_window,
    _COMBINED_BLACKLIST, _RAW_BLACKLIST,
)

log = logging.getLogger(__name__)


def train_hybrid(
    backbone: nn.Module,
    variant: OptVariant,
    *,
    n_steps: int = 2000,
    budget_mult: int = 10000,
    pop_per_dim: int = 5,
    dims: List[int] = (10,),
    M_es: int = 16,
    sigma: float = 0.01,
    bptt_window: int = 50,
    lr: float = 1e-4,
    lambda_es: float = 0.1,
    lambda_l2sp: float = 0.0,
    max_grad_norm: float = 10.0,
    graph_builder=None,
    device: str = 'cpu',
    save_dir: str = 'checkpoints',
    save_every: int = 200,
    log_every: int = 10,
    val_every: int = 50,
    patience: int = 300,
    min_steps: int = 0,
    mode: str = 'hybrid',
    batch_size: int = 1,
    n_bptt_segments: int = 0,  # 0 = auto-tune to fill VRAM
    focal_gamma: float = 0.0,
    resume_ckpt: Dict = None,
    budget_schedule: List[int] = None,
    no_augment: bool = True,
) -> List[Dict]:
    """Training loop — supports 'hybrid' (ES+BPTT) and 'bptt' (dovetailed only).

    mode='hybrid': ES-Single + random-segment BPTT (original)
    mode='bptt':   Dovetailed BPTT on ALL segments, B-batched functions (no ES)

    Returns list of per-step stats dicts.
    """
    from .augmented_cec2017 import AugmentedCEC2017

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    config = {
        'n_steps': n_steps, 'budget_mult': budget_mult,
        'M_es': M_es, 'sigma': sigma,
        'bptt_window': bptt_window, 'lr': lr, 'lambda_es': lambda_es,
    }

    if no_augment:
        aug = None
        log.info("Augmentation DISABLED — training on raw CEC2017")
    else:
        aug = AugmentedCEC2017(device=device, dims=tuple(dims))

    # Batched augmented CEC2017 for BPTT mode
    batched_aug = None
    if mode == 'bptt' and not no_augment:
        from .batched_augmented_cec2017 import BatchedAugmentedCEC2017
        batched_aug = BatchedAugmentedCEC2017(device=device, dims=tuple(dims))

    # Auto-tune n_bptt_segments if mode=bptt and n_bptt_segments=0
    if mode == 'bptt' and n_bptt_segments == 0 and device != 'cpu' and torch.cuda.is_available():
        D_tune = dims[0]
        N_tune = pop_per_dim * D_tune
        n_gens_tune = (budget_mult * D_tune) // N_tune
        if aug is not None:
            fn_tune = aug.sample()
        else:
            from .cec2017_torch import CEC2017Torch
            fn_tune = CEC2017Torch(1, D_tune, device)
        f_optimal_tune = torch.tensor([fn_tune.f_optimal], device=device,
                                       dtype=torch.float64)
        n_bptt_segments = _auto_tune_bptt_segments(
            backbone, variant, fn_tune, f_optimal_tune,
            D=D_tune, N=N_tune, B=batch_size, n_gens=n_gens_tune,
            bptt_window=bptt_window, graph_builder=graph_builder,
            device=device)
    elif mode == 'bptt' and n_bptt_segments == 0:
        n_bptt_segments = 1  # CPU fallback

    # Auto-tune ES: find best (M, sigma) for gradient quality
    if mode in ('es', 'es_bptt') and M_es == 0 and device != 'cpu' and torch.cuda.is_available():
        D_tune = dims[0]
        N_tune = pop_per_dim * D_tune
        n_gens_tune = (budget_mult * D_tune) // N_tune
        from .cec2017_torch import CEC2017Torch
        fn_tune = CEC2017Torch(1, D_tune, device)
        M_es, sigma = _auto_tune_es(
            backbone, variant, fn_tune, fn_tune.f_optimal,
            D=D_tune, N=N_tune, n_gens=n_gens_tune,
            graph_builder=graph_builder, device=device)

    # Auto-tune BPTT window for es_bptt mode
    if mode == 'es_bptt' and bptt_window == 0 and device != 'cpu' and torch.cuda.is_available():
        D_tune = dims[0]
        N_tune = pop_per_dim * D_tune
        n_gens_tune = (budget_mult * D_tune) // N_tune
        from .cec2017_torch import CEC2017Torch
        fn_tune = CEC2017Torch(1, D_tune, device)
        bptt_window = _auto_tune_bptt_window(
            backbone, variant, fn_tune, fn_tune.f_optimal,
            D=D_tune, N=N_tune, n_gens=n_gens_tune,
            graph_builder=graph_builder, device=device)

    # Budget curriculum: escalate through increasing FES levels
    _budget_queue = list(budget_schedule) if budget_schedule else []
    if _budget_queue:
        budget_mult = _budget_queue.pop(0)
    log.info("Budget curriculum: current=%d, remaining=%s", budget_mult, _budget_queue)

    all_params = list(backbone.parameters()) + list(variant.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    # Resume: restore optimizer state, reset step counter and metrics
    start_step = 0
    best_gc = 0.0
    if resume_ckpt is not None:
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            log.info("Resumed optimizer state from step %d",
                     resume_ckpt.get('step', -1))
        log.info("Weights loaded, starting fresh from step 0")

    # L2-SP anchor: snapshot pretrained weights (detached copies)
    anchor = None
    if lambda_l2sp > 0:
        anchor = {n: p.detach().clone()
                  for n, p in backbone.named_parameters()}
        log.info("L2-SP enabled: lambda=%.1e, anchoring %d backbone params",
                 lambda_l2sp, len(anchor))

    history = []
    patience_counter = 0

    # Per-fid gap EMA for adaptive curriculum (gap = final_best - f*)
    _fid_to_cat = {}
    for cat, fids in CATEGORIES.items():
        for f in fids:
            _fid_to_cat[f] = cat
    gap_ema_fid = resume_ckpt.get('gap_ema_fid', {}) if resume_ckpt else {}
    GAP_EMA_ALPHA = 0.05

    # Convergence history: EMA of actual gens per (fid, D) for BPTT window placement
    _convergence_gens = {}  # {(fid, D): ema_gens}

    # Global gap EMA for early stopping (log-normalized, across all fids)
    gap_ema_global = -1.0  # -1 = unset
    best_gap_ema = -1.0

    # Track dims that OOM so we skip them on retry
    oom_dims = set()

    # ES chaining: carry population forward per (fid, D)
    _es_populations = {}  # {(fid, D): (coords, fitness)}

    for step in range(start_step, n_steps):
        t0 = time.perf_counter()

        # Category-balanced + gap-weighted curriculum
        _bl = _RAW_BLACKLIST if no_augment else _COMBINED_BLACKLIST
        fid, D = _sample_category_balanced(dims, loss_ema_fid=gap_ema_fid, blacklist=_bl)
        if aug is not None:
            fn = aug.sample(fid=fid, D=D)
        else:
            from .cec2017_torch import CEC2017Torch
            fn = CEC2017Torch(fid, D, device)
        f_optimal = fn.f_optimal
        N = pop_per_dim * D
        n_gens = (budget_mult * D) // N

        # Skip dims that previously OOMed
        if D in oom_dims:
            log.info("step %d | SKIP D=%d (previously OOMed)", step, D)
            continue

        # Adaptive BPTT window
        if device != 'cpu' and torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(
                torch.device(device)).total_memory / (1024 ** 3)
        else:
            vram_gb = 12.0
        vram_scale = vram_gb / 12.0
        max_bptt_for_vram = max(2, int(50 * 80 * vram_scale) // max(N, 1))
        nd_factor = max(1, (N * D) // (50 * 10))
        max_bptt_for_vram = max(2, max_bptt_for_vram // max(1, nd_factor // 4))
        eff_bptt = min(bptt_window, max_bptt_for_vram)

        # Gumbel temperature warmup
        warmup_steps = 1000
        if step < warmup_steps:
            gumbel_tau = 5.0 - 4.0 * (step / warmup_steps)
        else:
            gumbel_tau = 1.0

        if device != 'cpu' and torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_res = torch.cuda.memory_reserved() / 1024**2
            if mode == 'es':
                log.info("step %d | PRE  F%02d D=%d N=%d maxfes=%d M=%d | "
                         "VRAM alloc=%.0fMB res=%.0fMB",
                         step, fid, D, N, n_gens * N, M_es,
                         mem_alloc, mem_res)
            else:
                log.info("step %d | PRE  F%02d D=%d N=%d maxfes=%d bptt=%d B=%d | "
                         "VRAM alloc=%.0fMB res=%.0fMB",
                         step, fid, D, N, n_gens * N, eff_bptt,
                         batch_size if mode == 'bptt' else M_es,
                         mem_alloc, mem_res)

        # Training step
        optimizer.zero_grad()

        try:
            if mode == 'bptt':
                if batched_aug is not None:
                    fn_batch = batched_aug.sample_batch(batch_size, fid=fid, D=D)
                else:
                    from .cec2017_torch import CEC2017Torch
                    fn_batch = CEC2017Torch(fid, D, device)
                # Use fn_batch's f_optimal for logging/curriculum (not fn's)
                f_optimal = fn_batch.f_optimal.mean().item() if hasattr(fn_batch.f_optimal, 'mean') else fn_batch.f_optimal
                step_result = bptt_step(
                    backbone, variant, fn_batch, fn_batch.f_optimal,
                    D=D, N=N, B=batch_size, n_gens=n_gens,
                    bptt_window=eff_bptt, n_bptt_segments=n_bptt_segments,
                    gumbel_tau=gumbel_tau, focal_gamma=focal_gamma,
                    graph_builder=graph_builder, device=device)
            elif mode == 'es':
                _pop_key = (fid, D)
                _pop_state = _es_populations.get(_pop_key, {})
                step_result = es_only_step(
                    backbone, variant, fn, f_optimal,
                    D=D, N=N, n_gens=n_gens,
                    M_es=M_es, sigma=sigma,
                    graph_builder=graph_builder, device=device,
                    init_coords=_pop_state.get('coords'),
                    init_fitness=_pop_state.get('fitness'),
                    init_coords_ring=_pop_state.get('coords_ring'),
                    init_fitness_ring=_pop_state.get('fitness_ring'),
                    init_ring_pos=_pop_state.get('ring_pos', 0))
                _es_populations[_pop_key] = {
                    'coords': step_result['final_coords'],
                    'fitness': step_result['final_fitness'],
                    'coords_ring': step_result['final_coords_ring'],
                    'fitness_ring': step_result['final_fitness_ring'],
                    'ring_pos': step_result['final_ring_pos'],
                }
            elif mode == 'es_bptt':
                _conv_key = (fid, D)
                _exp_gens = _convergence_gens.get(_conv_key)
                step_result = es_bptt_step(
                    backbone, variant, fn, f_optimal,
                    D=D, N=N, n_gens=n_gens, bptt_window=eff_bptt,
                    M_es=M_es, sigma=sigma,
                    lambda_es=lambda_es, gumbel_tau=gumbel_tau,
                    expected_gens=int(_exp_gens) if _exp_gens else None,
                    graph_builder=graph_builder, device=device)
                # Update convergence history
                actual = step_result.get('n_gens_actual', n_gens)
                if _exp_gens is None:
                    _convergence_gens[_conv_key] = actual
                else:
                    _convergence_gens[_conv_key] = 0.8 * _exp_gens + 0.2 * actual
            else:
                step_result = es_single_step(
                    backbone, variant, fn, f_optimal,
                    D=D, N=N, n_gens=n_gens, bptt_window=eff_bptt,
                    M_es=M_es, sigma=sigma,
                    lambda_es=lambda_es, gumbel_tau=gumbel_tau,
                    graph_builder=graph_builder, device=device)
        except torch.cuda.OutOfMemoryError:
            oom_dims.add(D)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            log.error("OOM at step %d | F%02d D=%d N=%d gens=%d — "
                      "skipping D=%d for rest of run",
                      step, fid, D, N, n_gens, D)
            continue

        # Clip BPTT gradient first, then add L2-SP (regularizer not clipped)
        raw_gn = torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

        # Skip step if gradient is NaN/Inf (prevents weight corruption)
        if not torch.isfinite(raw_gn):
            log.warning("step %d | NaN/Inf gradient (raw_gn=%s) — skipping optimizer step",
                        step, raw_gn)
            optimizer.zero_grad()
            continue

        if anchor is not None:
            # Linear decay: lambda_l2sp → 0 over 1000 steps
            l2sp_decay = max(0.0, 1.0 - step / 1000.0)
            l2sp_effective = lambda_l2sp * l2sp_decay
            if l2sp_effective > 0:
                anchor_penalty = l2sp_loss(backbone, anchor)
                (l2sp_effective * anchor_penalty).backward()

        optimizer.step()

        wall_ms = (time.perf_counter() - t0) * 1000

        if device != 'cpu' and torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**2
            mem_res = torch.cuda.memory_reserved() / 1024**2
            _post_gap = step_result.get('final_gap',
                        step_result.get('final_best', f_optimal) - f_optimal)
            log.info("step %d | POST F%02d D=%d | gap=%.2f | "
                     "VRAM alloc=%.0fMB res=%.0fMB | %.0fms",
                     step, fid, D, _post_gap,
                     mem_alloc, mem_res, wall_ms)

        step_stats = {
            'step': step, 'budget_mult': budget_mult,
            'fid': fid, 'D': D, 'N': N, 'n_gens': n_gens,
            'gc': step_result['gc'],
            'bptt_loss': step_result['bptt_loss'],
            'grad_norm_bptt': step_result['grad_norm_bptt'],
            'wall_ms': wall_ms,
            'gumbel_tau': gumbel_tau,
        }
        if mode == 'bptt':
            step_stats['B'] = batch_size
            step_stats['n_segments'] = step_result['n_segments']
        elif mode == 'es':
            step_stats['grad_norm_es'] = step_result['grad_norm_es']
            step_stats['es_gc_mean'] = step_result['es_gc_mean']
            step_stats['es_gc_std'] = step_result['es_gc_std']
            step_stats['es_auc_mean'] = step_result.get('es_auc_mean', 0)
            step_stats['es_auc_std'] = step_result.get('es_auc_std', 0)
            step_stats['final_gap'] = step_result.get('final_gap', -1)
        elif mode == 'es_bptt':
            step_stats['grad_norm_es'] = step_result['grad_norm_es']
            step_stats['grad_norm_combined'] = step_result['grad_norm_combined']
            step_stats['cos_sim_bptt_es'] = step_result['cos_sim_bptt_es']
            step_stats['es_gc_mean'] = step_result['es_gc_mean']
            step_stats['es_gc_std'] = step_result['es_gc_std']
        else:  # hybrid
            step_stats['grad_norm_es'] = step_result['grad_norm_es']
            step_stats['grad_norm_combined'] = step_result['grad_norm_combined']
            step_stats['bptt_segment'] = step_result['bptt_segment']
            step_stats['cos_sim_bptt_es'] = step_result['cos_sim_bptt_es']
            step_stats['es_gc_mean'] = step_result['es_gc_mean']
            step_stats['es_gc_std'] = step_result['es_gc_std']
        # Add routing diagnostics and convergence trajectory to JSONL
        routing = step_result.get('routing', {})
        if routing:
            step_stats['routing'] = routing
        conv = step_result.get('convergence', {})
        if conv:
            step_stats['convergence'] = {str(k): v for k, v in conv.items()}

        # Update per-fid gap EMA (drives adaptive curriculum)
        gap = abs(step_result.get('final_best', 0) - f_optimal)
        gap_norm = math.log1p(gap)
        gap_ema_fid[fid] = gap_ema_fid.get(fid, gap_norm) * (1 - GAP_EMA_ALPHA) + gap_norm * GAP_EMA_ALPHA

        # Global gap EMA (mean of per-fid EMAs, log-normalized)
        if gap_ema_fid:
            gap_ema_global = sum(gap_ema_fid.values()) / len(gap_ema_fid)
        step_stats['gap_ema'] = gap_ema_global

        history.append(step_stats)

        if step_result['gc'] > best_gc:
            best_gc = step_result['gc']

        if step % log_every == 0:
            route_str = ""
            if routing:
                route_str = (
                    f" | rmax {routing.get('route_max', 0):.2f}"
                    f" lg[{routing.get('logit_absmax', 0):.1f}]"
                )
                per_k = routing.get('logit_per_k_mean', [])
                if per_k:
                    route_str += " k_lg[" + ",".join(f"{v:.1f}" for v in per_k) + "]"
                counts = routing.get('route_argmax_counts', [])
                if counts:
                    total = max(sum(counts), 1)
                    pcts = [f"{100*c/total:.0f}" for c in counts]
                    route_str += " sel[" + ",".join(pcts) + "]%"
                acounts = routing.get('winner_counts', [])
                if acounts:
                    atotal = max(sum(acounts), 1)
                    apcts = [f"{100*c/atotal:.0f}" for c in acounts]
                    route_str += " asel[" + ",".join(apcts) + "]%"

            if mode == 'bptt':
                fes_info = step_result.get('fes_used', 0)
                fes_max = step_result.get('max_fes', n_gens * N)
                n_gens_actual = step_result.get('n_gens', 0)
                fw = step_result.get('focal_weight', 1.0)
                conv = step_result.get('convergence', {})
                if conv:
                    def _fmt_fes(f):
                        if f >= 1000: return f"{f//1000}k"
                        return str(f)
                    def _fmt_val(v, fopt):
                        gap = v - fopt
                        if abs(gap) < 0.01: return "0"
                        if gap < 0: return f"{gap:.1f}"
                        if gap < 100: return f"{gap:.1f}"
                        if gap < 1e6: return f"{gap:.0f}"
                        return f"{gap:.1e}"
                    conv_str = " conv[" + ",".join(
                        f"{_fmt_fes(fes)}:{_fmt_val(v, f_optimal)}"
                        for fes, v in sorted(conv.items())
                    ) + "]"
                else:
                    conv_str = ""
                final_gap = step_result.get('final_gap', -1)
                gap1_fes = step_result.get('gap1_fes')
                gap1_str = f" gap<1@{gap1_fes}fes" if gap1_fes is not None else ""
                log.info("step %d | F%02d D%d B=%d | gap %.2f%s | loss %.2f (raw %.2f fw %.3f) lb %.3f | "
                         "gn %.4f | segs %d | fes %d/%d gens %d%s%s | %.0fms",
                         step, fid, D, batch_size,
                         final_gap, gap1_str,
                         step_result['bptt_loss'],
                         step_result.get('raw_bptt_loss', step_result['bptt_loss']),
                         fw,
                         routing.get('lb_loss', 0),
                         step_result['grad_norm_bptt'],
                         step_result.get('n_segments', 0),
                         fes_info, fes_max, n_gens_actual,
                         route_str, conv_str, wall_ms)
            elif mode == 'es':
                log.info("step %d | F%02d D%d M=%d | gap %.2f | "
                         "auc %.2f±%.2f | gc %.3f±%.3f | "
                         "es_gn %.1f%s | %.0fms",
                         step, fid, D, M_es,
                         step_result.get('final_gap', -1),
                         step_result.get('es_auc_mean', 0),
                         step_result.get('es_auc_std', 0),
                         step_result['es_gc_mean'],
                         step_result['es_gc_std'],
                         step_result['grad_norm_es'],
                         route_str, wall_ms)
            elif mode == 'es_bptt':
                log.info("step %d | F%02d D%d M=%d W=%d | gap %.2f | "
                         "bptt_gn %.2f es_gn %.1f cos %.3f | "
                         "es_gc %.3f±%.3f%s | %.0fms",
                         step, fid, D, M_es, eff_bptt,
                         step_result.get('final_gap', -1),
                         step_result['grad_norm_bptt'],
                         step_result['grad_norm_es'],
                         step_result['cos_sim_bptt_es'],
                         step_result['es_gc_mean'],
                         step_result['es_gc_std'],
                         route_str, wall_ms)
            else:  # hybrid
                log.info("step %d | F%02d D%d | bptt_gn %.4f | es_gc %.3f±%.3f | "
                         "cos %.4f%s | %.0fms",
                         step, fid, D,
                         step_result['grad_norm_bptt'],
                         step_result['es_gc_mean'],
                         step_result['es_gc_std'],
                         step_result['cos_sim_bptt_es'],
                         route_str, wall_ms)

        # ── Early stopping based on training gap EMA ──
        if val_every > 0 and (step + 1) % val_every == 0:
            gap_improved = gap_ema_global < best_gap_ema if best_gap_ema >= 0 else True
            gap_delta = best_gap_ema - gap_ema_global if best_gap_ema >= 0 else gap_ema_global

            # Per-category EMA breakdown
            gap_ema_cat = {}
            for _f, _v in gap_ema_fid.items():
                _c = _fid_to_cat.get(_f, 'Unknown')
                gap_ema_cat.setdefault(_c, []).append(_v)
            cat_str = " ".join(f"{c}={sum(vs)/len(vs):.2f}"
                               for c, vs in sorted(gap_ema_cat.items()))
            log.info("EMA step %d | gap_ema %.4f | delta %+.4f | "
                     "patience %d/%d | %s",
                     step, gap_ema_global, gap_delta,
                     patience_counter, patience, cat_str)

            if gap_improved:
                best_gap_ema = gap_ema_global
                patience_counter = 0
                ckpt = _make_ckpt(backbone, variant, optimizer, step,
                                  best_gc, best_gap_ema, config,
                                  gap_ema_fid=gap_ema_fid)
                torch.save(ckpt, save_path / 'best_ema.pth')
            else:
                patience_counter += val_every
                if patience_counter >= patience and step >= min_steps:
                    if _budget_queue:
                        old_budget = budget_mult
                        budget_mult = _budget_queue.pop(0)
                        n_gens = (budget_mult * D) // N
                        log.info("Budget escalation at step %d: %d → %d "
                                 "(MAXFES %d → %d, remaining=%s)",
                                 step, old_budget, budget_mult,
                                 old_budget * D, budget_mult * D,
                                 _budget_queue)
                        patience_counter = 0
                        best_gap_ema = -1.0
                        gap_ema_fid.clear()
                    else:
                        log.info("Early stopping at step %d "
                                 "(patience=%d exhausted, no more budget levels)",
                                 step, patience)
                        break

        # Save periodic checkpoint
        if save_every > 0 and (step + 1) % save_every == 0:
            ckpt = _make_ckpt(backbone, variant, optimizer, step,
                              best_gc, best_gap_ema, config,
                              gap_ema_fid=gap_ema_fid)
            torch.save(ckpt, save_path / f'hybrid_step{step+1}.pth')

        # JSONL log
        with open(save_path / 'train_hybrid.jsonl', 'a') as f:
            f.write(json.dumps(step_stats) + '\n')

        # Detailed diagnostics log (es_bptt mode)
        diag = step_result.get('diagnostics')
        if diag:
            diag['step'] = step
            diag['fid'] = fid
            diag['D'] = D
            with open(save_path / 'diagnostics.jsonl', 'a') as f:
                f.write(json.dumps(diag) + '\n')

    return history


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Hybrid ES-Single + BPTT training for L2O variants')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n-steps', type=int, default=10000)
    parser.add_argument('--budget-mult', type=int, default=10000)
    parser.add_argument('--pop-per-dim', type=int, default=5)
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 30, 50])
    parser.add_argument('--M-es', type=int, default=0,
                        help='ES perturbations (0=auto-tune M and sigma for max gradient signal)')
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--bptt-window', type=int, default=80)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lambda-es', type=float, default=0.1)
    parser.add_argument('--max-grad-norm', type=float, default=10.0)
    parser.add_argument('--save-dir', default='checkpoints/k2_augmented_long')
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--val-every', type=int, default=50)
    parser.add_argument('--patience', type=int, default=300)
    parser.add_argument('--min-steps', type=int, default=0)
    parser.add_argument('--variant', choices=['k2', 'k4', 'k6',
                                              'direct_k1', 'direct_k4', 'direct_k5'],
                        default='k2')
    parser.add_argument('--phase', choices=['imitation', 'discovery-warmup', 'free'],
                        default='free',
                        help='HyperOPT K=6 training phase (ignored for k2/k4)')
    parser.add_argument('--ssl-checkpoint', type=str, default=None,
                        help='Path to SSL pretrained backbone checkpoint')
    parser.add_argument('--lambda-l2sp', type=float, default=0.0,
                        help='L2-SP anchor penalty (0=disabled)')
    parser.add_argument('--sparse', action='store_true',
                        help='Use sparse GATv2 backbone (O(N·k) instead of O(N²))')
    parser.add_argument('--topology', choices=['coordinate', 'embedding', 'learned'],
                        default='coordinate',
                        help='Topology strategy for sparse backbone')
    parser.add_argument('--k-neighbors', type=int, default=8,
                        help='Number of neighbors for sparse backbone')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--mode', choices=['hybrid', 'bptt', 'es', 'es_bptt'], default='hybrid',
                        help='Training mode: es_bptt (ES+BPTT-final, recommended), es (pure ES), bptt, hybrid')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Number of B-batched augmented functions (bptt mode)')
    parser.add_argument('--n-bptt-segments', type=int, default=0,
                        help='Number of random BPTT segments (0=auto-tune to fill VRAM)')
    parser.add_argument('--focal-gamma', type=float, default=0.0,
                        help='Focal loss exponent (0=off, 2=standard). '
                             'Weights loss by (1-gc)^gamma to suppress solved functions.')
    parser.add_argument('--budget-schedule', type=str, default=None,
                        help='Comma-separated budget_mult levels for curriculum '
                             '(e.g., "1000,2000,5000,10000"). Overrides --budget-mult.')
    parser.add_argument('--no-augment', action='store_true', default=True,
                        help='Train on raw CEC2017 (no rotation/shift/scale augmentation)')
    parser.add_argument('--augment', action='store_true',
                        help='Enable rotation/shift/scale augmentation (off by default)')
    args = parser.parse_args()

    # Parse budget schedule
    if args.budget_schedule:
        args.budget_schedule_list = [int(x) for x in args.budget_schedule.split(',')]
    else:
        args.budget_schedule_list = None

    # --augment overrides --no-augment
    if args.augment:
        args.no_augment = False

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    # ── Build backbone ──
    graph_builder = None
    if args.sparse:
        from .sparse_temporal_backbone import TemporalSparseGATv2Backbone
        from .sparse_gatv2_backbone import TopologyMode
        from .similarity_graph_gpu import build_sparse_graphs_gpu

        topo_map = {
            'coordinate': TopologyMode.COORDINATE_KNN,
            'embedding': TopologyMode.EMBEDDING_KNN,
            'learned': TopologyMode.LEARNED_SCORER,
        }
        backbone = TemporalSparseGATv2Backbone(
            d_rnn=64, d_temporal=64, gru_window=16,
            node_in=8, edge_in=4, global_in=13,
            gatv2_hidden=128, gatv2_layers=2, n_heads=8,
            global_out_dim=128, dropout=0.1,
            topology_mode=topo_map[args.topology],
            k_neighbors=args.k_neighbors,
            device=args.device,
        ).to(args.device)
        graph_builder = build_sparse_graphs_gpu
        log.info("Sparse backbone: topology=%s, k=%d", args.topology, args.k_neighbors)
    else:
        from .dense_temporal_backbone import TemporalDenseGATv2Backbone
        backbone = TemporalDenseGATv2Backbone(
            d_rnn=64, d_temporal=64, gru_window=16,
            node_in=8, edge_in=4, global_in=13,
            gatv2_hidden=128, gatv2_layers=2, n_heads=8,
            global_out_dim=128, dropout=0.1,
            device=args.device,
        ).to(args.device)

    if args.ssl_checkpoint:
        log.info("Loading SSL checkpoint: %s", args.ssl_checkpoint)
        backbone.load_ssl_checkpoint(args.ssl_checkpoint)

    if args.resume:
        log.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        backbone.load_state_dict(ckpt['backbone_state_dict'])

    # ── Build variant ──
    if args.variant == 'k2':
        from .variants.classic_k2 import ClassicK2Variant
        variant = ClassicK2Variant(
            gatv2_hidden=128, global_dim=128,
        ).to(args.device)
    elif args.variant == 'k4':
        from .variants.neural_k4 import NeuralK4Variant
        variant = NeuralK4Variant(
            K=4, head_dim=16, gatv2_hidden=128,
        ).to(args.device)
    elif args.variant == 'direct_k1':
        from .variants.neural_k4 import NeuralK4Variant
        from .direct_delta import BatchedDirectDelta
        variant = NeuralK4Variant(
            K=1, head_dim=16, gatv2_hidden=128,
            operator_classes=[BatchedDirectDelta],
        ).to(args.device)
    elif args.variant == 'direct_k4':
        from .variants.neural_k4 import NeuralK4Variant, BATCHED_OPERATOR_CLASSES_DIRECT
        variant = NeuralK4Variant(
            K=4, head_dim=16, gatv2_hidden=128,
            operator_classes=BATCHED_OPERATOR_CLASSES_DIRECT,
        ).to(args.device)
    elif args.variant == 'direct_k5':
        from .variants.neural_k4 import NeuralK4Variant, BATCHED_OPERATOR_CLASSES_K5
        variant = NeuralK4Variant(
            K=5, head_dim=16, gatv2_hidden=128,
            operator_classes=BATCHED_OPERATOR_CLASSES_K5,
        ).to(args.device)
    else:  # k6
        from .variants.hyperopt import HyperOPTK6Variant
        variant = HyperOPTK6Variant(gatv2_hidden=128).to(args.device)
        variant.set_phase(args.phase)
        log.info("HyperOPT K=6 phase: %s", args.phase)

    total_params = sum(p.numel() for p in backbone.parameters()) + \
                   sum(p.numel() for p in variant.parameters())
    log.info("Total params: %d (backbone: %d, variant: %d)",
             total_params,
             sum(p.numel() for p in backbone.parameters()),
             sum(p.numel() for p in variant.parameters()))

    if args.resume and 'variant_state_dict' in ckpt:
        variant.load_state_dict(ckpt['variant_state_dict'])

    resume_ckpt = ckpt if args.resume else None

    train_hybrid(
        backbone, variant,
        n_steps=args.n_steps,
        budget_mult=args.budget_mult,
        pop_per_dim=args.pop_per_dim,
        dims=args.dims,
        M_es=args.M_es,
        sigma=args.sigma,
        bptt_window=args.bptt_window,
        lr=args.lr,
        lambda_es=args.lambda_es,
        lambda_l2sp=args.lambda_l2sp,
        max_grad_norm=args.max_grad_norm,
        graph_builder=graph_builder,
        device=args.device,
        save_dir=args.save_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        val_every=args.val_every,
        patience=args.patience,
        min_steps=args.min_steps,
        mode=args.mode,
        batch_size=args.batch_size,
        n_bptt_segments=args.n_bptt_segments,
        focal_gamma=args.focal_gamma,
        resume_ckpt=resume_ckpt,
        budget_schedule=args.budget_schedule_list,
        no_augment=args.no_augment,
    )
