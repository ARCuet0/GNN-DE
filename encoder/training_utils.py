import logging
from typing import Dict

import torch
import torch.nn as nn

from .cec2017_torch import CATEGORIES
from .augmented_cec2017 import AugmentedCEC2017
from .trajectory import run_trajectory_dovetailed

log = logging.getLogger(__name__)

# Pre-compute combined blacklist for category-balanced sampling
_COMBINED_BLACKLIST = AugmentedCEC2017.BLACKLIST | AugmentedCEC2017.AUG_BLACKLIST
# Minimal blacklist for raw CEC2017 (no augmentation):
# F16 excluded from training — extreme coords→fitness curvature causes gradient explosions
# during BPTT. Still included in evaluation (val set does no-grad trajectories).
_RAW_BLACKLIST = {(16, D) for D in [10, 20, 30, 50, 100]}
_CAT_NAMES = list(CATEGORIES.keys())


def _sample_category_balanced(dims, loss_ema_fid=None, blacklist=None):
    """Category-balanced function sampling with PER-style curriculum.

    Stage 1: pick category uniformly (25% each).
    Stage 2: within category, pick fid proportional to max(eps, gap^α).
             α-exponent (Schaul et al. 2015) compresses the gap range to
             prevent curriculum collapse while preserving difficulty ordering.
             If loss_ema_fid is None, falls back to uniform within-category.

    Returns (fid, D) guaranteed not in combined blacklist.
    """
    if blacklist is None:
        blacklist = _COMBINED_BLACKLIST
    D = dims[torch.randint(len(dims), (1,)).item()]
    _ALPHA = 0.3   # PER prioritization exponent: 0=uniform, 1=proportional
    _EPS = 2.0     # floor weight for solved functions (gap ≈ 0)
    _UNSEEN_WEIGHT = 5.0   # priority for unexplored fids

    for _ in range(10):
        cat = _CAT_NAMES[torch.randint(len(_CAT_NAMES), (1,)).item()]
        valid_fids = [f for f in CATEGORIES[cat] if (f, D) not in blacklist]
        if not valid_fids:
            continue

        if loss_ema_fid is None or not loss_ema_fid:
            # Uniform within category (backward compat)
            fid = valid_fids[torch.randint(len(valid_fids), (1,)).item()]
        else:
            # PER-style: P(i) ∝ max(eps, |gap_i|^α) (Schaul et al. 2015)
            weights = []
            for f in valid_fids:
                if f in loss_ema_fid:
                    weights.append(max(_EPS, abs(loss_ema_fid[f]) ** _ALPHA))
                else:
                    weights.append(_UNSEEN_WEIGHT)
            w = torch.tensor(weights, dtype=torch.float32)
            w = w / w.sum()
            idx = torch.multinomial(w, 1).item()
            fid = valid_fids[idx]

        return fid, D

    # Fallback: any non-blacklisted fid
    fid = int(torch.randint(1, 30, (1,)).item())
    while (fid, D) in blacklist:
        fid = int(torch.randint(1, 30, (1,)).item())
    return fid, D


def l2sp_loss(model: nn.Module, anchor: Dict[str, torch.Tensor]) -> torch.Tensor:
    """L2-SP: penalize deviation from pretrained weights.

    Args:
        model: current model (backbone or variant)
        anchor: {param_name: pretrained_value} snapshot

    Returns:
        scalar loss: sum of ||θ - θ_pretrained||² over anchored params
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        if name in anchor:
            loss = loss + (param - anchor[name]).pow(2).sum()
    return loss


def _make_ckpt(backbone, variant, optimizer, step, best_gc, best_gap_ema, config,
               gap_ema_fid=None):
    """Build checkpoint dict for saving."""
    ckpt = {
        'backbone_state_dict': backbone.state_dict(),
        'variant_state_dict': variant.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step, 'best_gc': best_gc,
        'best_gap_ema': best_gap_ema,
        'config': config,
    }
    if gap_ema_fid is not None:
        ckpt['gap_ema_fid'] = gap_ema_fid
    return ckpt


def _build_validation_set(dims, pop_per_dim, budget_mult, device,
                          no_augment=False):
    """Build validation set: CEC2017 functions without augmentation.

    Returns val_set (list of dicts).
    """
    from .cec2017_torch import CEC2017Torch, BLACKLIST

    # Validation runs no-grad trajectories → safe to include F16
    # Only skip NaN-producing functions from BLACKLIST
    blacklist = BLACKLIST if not no_augment else set()

    val_set = []
    for D in dims:
        N = pop_per_dim * D
        n_gens = (budget_mult * D) // N
        for fid in range(1, 30):
            if (fid, D) in blacklist:
                continue
            fn = CEC2017Torch(fid, D, device)
            val_set.append({
                'fn': fn, 'D': D, 'N': N, 'n_gens': n_gens,
                'f_optimal': fn.f_optimal, 'fid': fid,
            })

    return val_set


def _run_validation(backbone, variant, val_set, bptt_window, device,
                     graph_builder=None):
    """Run validation: no-grad trajectories on raw CEC2017.

    Uses run_trajectory_dovetailed with n_bptt_segments=0 to match
    the exact training code path (FES tracking, graph_builder args)
    but without computing gradients.

    Returns (mean_gc, results_list) where each result has
    fid, D, gc, final_best, f_optimal, category.
    """
    from .cec2017_torch import FUNCTIONS
    backbone.eval()
    variant.eval()
    results = []
    with torch.no_grad():
        for v in val_set:
            gc, _, stats = run_trajectory_dovetailed(
                backbone, variant, v['fn'], v['f_optimal'],
                D=v['D'], N=v['N'], B=1, n_gens=v['n_gens'],
                bptt_window=bptt_window, n_bptt_segments=0,
                gumbel_tau=1.0,
                graph_builder=graph_builder, device=device)
            results.append({
                'fid': v['fid'], 'D': v['D'], 'gc': gc,
                'final_best': stats['final_best'],
                'f_optimal': v['f_optimal'],
                'category': FUNCTIONS[v['fid']][1],
            })
    backbone.train()
    variant.train()
    mean_gc = sum(r['gc'] for r in results) / len(results)
    mean_gap = sum(r['final_best'] - r['f_optimal'] for r in results) / len(results)
    return mean_gc, mean_gap, results


def _auto_tune_bptt_segments(backbone, variant, fn, f_optimal,
                             D, N, B, n_gens, bptt_window,
                             graph_builder, device):
    """Binary search for max n_bptt_segments that fits in available VRAM.

    Uses 85% of currently-free VRAM as target (adapts to other GPU workloads).
    """
    total_segs = max(1, n_gens // bptt_window)
    vram_free = torch.cuda.mem_get_info(torch.device(device))[0]  # bytes free
    vram_target = int(vram_free * 0.85)

    log.info("Auto-tuning n_bptt_segments: total_segs=%d, VRAM free=%.1fGB, target=%.1fGB",
             total_segs, vram_free / 1024**3, vram_target / 1024**3)

    backbone.train()
    variant.train()

    lo, hi, best = 1, total_segs, 1
    while lo <= hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated()
        try:
            gc, loss, _ = run_trajectory_dovetailed(
                backbone, variant, fn, f_optimal,
                D=D, N=N, B=B, n_gens=n_gens, bptt_window=bptt_window,
                n_bptt_segments=mid,
                graph_builder=graph_builder, device=device)
            loss.backward()
            del loss, gc, _
            peak = torch.cuda.max_memory_allocated() - baseline_mem
            log.info("  m=%d | peak=%.1fGB | %s",
                     mid, peak / 1024**3,
                     "OK" if peak <= vram_target else "over target")
            if peak <= vram_target:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.info("  m=%d | OOM", mid)
            hi = mid - 1

    # Clean up
    torch.cuda.empty_cache()
    for p in list(backbone.parameters()) + list(variant.parameters()):
        if p.grad is not None:
            p.grad = None

    log.info("Auto-tune result: n_bptt_segments=%d (of %d total)", best, total_segs)
    return best


def _auto_tune_es(backbone, variant, fn, f_optimal,
                  D, N, n_gens, graph_builder, device):
    """Find (M, sigma) for ES training by maximizing gradient SNR.

    Phase 1: Sweep M upward. For each feasible M, measure SNR with full
             n_gens trajectory. Pick M with highest SNR (respecting OOM).
    Phase 2: Sweep sigma at best M, pick highest SNR.

    Returns (best_M, best_sigma).
    """
    from .es_training import es_only_step
    import time

    vram_free = torch.cuda.mem_get_info(torch.device(device))[0]
    log.info("ES auto-tune: D=%d N=%d n_gens=%d, VRAM free=%.1fGB",
             D, N, n_gens, vram_free / 1024**3)

    backbone.eval()
    variant.eval()

    # Phase 1: sweep M, select by max SNR
    best_M = 4
    best_snr = -1.0
    for M_test in [4, 8, 16, 32, 64, 128, 256, 512]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        try:
            t0 = time.monotonic()
            result = es_only_step(
                backbone, variant, fn, f_optimal,
                D=D, N=N, n_gens=n_gens, M_es=M_test, sigma=0.1,
                graph_builder=graph_builder, device=device)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**3
            snr = result['es_snr']
            log.info("  M=%d | %.1fs (%.3fs/pert) | peak=%.1fGB | SNR=%.3f",
                     M_test, elapsed, elapsed / M_test, peak, snr)

            if snr > best_snr:
                best_snr = snr
                best_M = M_test
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.info("  M=%d | OOM", M_test)
            break

    log.info("  Best M=%d (SNR=%.3f)", best_M, best_snr)

    # Phase 2: sweep sigma at best_M, select by max SNR
    best_sigma = 0.1
    best_sigma_snr = -1.0
    for sigma_test in [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        torch.cuda.empty_cache()
        try:
            result = es_only_step(
                backbone, variant, fn, f_optimal,
                D=D, N=N, n_gens=n_gens, M_es=best_M, sigma=sigma_test,
                graph_builder=graph_builder, device=device)
            snr = result['es_snr']
            log.info("  sigma=%.3f | SNR=%.3f | gc=%.3f±%.3f",
                     sigma_test, snr,
                     result['es_gc_mean'], result['es_gc_std'])
            if snr > best_sigma_snr:
                best_sigma_snr = snr
                best_sigma = sigma_test
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.info("  sigma=%.3f | OOM", sigma_test)

    # Clean up
    torch.cuda.empty_cache()
    for p in list(backbone.parameters()) + list(variant.parameters()):
        if p.grad is not None:
            p.grad = None

    backbone.train()
    variant.train()

    log.info("ES auto-tune result: M=%d, sigma=%.3f (SNR=%.3f)",
             best_M, best_sigma, best_sigma_snr)
    return best_M, best_sigma


def _auto_tune_bptt_window(backbone, variant, fn, f_optimal,
                           D, N, n_gens, graph_builder, device):
    """Find max BPTT window that fits in VRAM and produces finite gradients.

    Tests increasing window sizes with bptt_position='last'.
    Returns best_W.
    """
    from .trajectory import bptt_step

    log.info("BPTT window auto-tune: D=%d N=%d n_gens=%d", D, N, n_gens)

    backbone.train()
    variant.train()

    best_W = 5
    for W_test in [5, 10, 20, 50, 100, 200, 500]:
        if W_test > n_gens:
            break
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated()

        # Zero grads before trial
        for p in list(backbone.parameters()) + list(variant.parameters()):
            if p.grad is not None:
                p.grad = None

        try:
            # Use n_gens=W_test: only need the BPTT segment for VRAM measurement
            result = bptt_step(
                backbone, variant, fn, f_optimal,
                D=D, N=N, B=1, n_gens=W_test, bptt_window=W_test,
                n_bptt_segments=1, bptt_position='last',
                graph_builder=graph_builder, device=device)

            peak = (torch.cuda.max_memory_allocated() - baseline_mem) / 1024**3
            gn = result['grad_norm_bptt']
            finite = gn > 0 and gn < 1e6

            log.info("  W=%d | peak=%.1fGB | grad_norm=%.4f | %s",
                     W_test, peak, gn,
                     "OK" if finite else "NaN/Inf")

            if finite:
                best_W = W_test
            else:
                log.info("  Stopping: gradient not finite at W=%d", W_test)
                break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.info("  W=%d | OOM", W_test)
            break

    # Clean up
    torch.cuda.empty_cache()
    for p in list(backbone.parameters()) + list(variant.parameters()):
        if p.grad is not None:
            p.grad = None

    log.info("BPTT window auto-tune result: W=%d", best_W)
    return best_W
