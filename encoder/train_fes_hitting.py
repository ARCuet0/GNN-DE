"""
train_fes_hitting.py — FES-budgeted trajectory + hitting_loss training.

Combines probe_fes_hitting's training loop (FES-budgeted, hitting_loss,
gradient checkpointing, spectral radius adaptive detach) with
train_hybrid's multi-function infrastructure (category-balanced sampling,
validation, checkpointing, patience).

Usage:
    python -m encoder.train_fes_hitting \
        --sparse --topology embedding \
        --variant direct_k1 --device cuda \
        --ssl-checkpoint checkpoints/k4_sparse_embed/ssl_nextstep_sparse_embed.pth \
        --n-steps 100000 --dims 10 30 50 --lr 3e-4
"""
import json
import logging
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .cec2017_torch import CEC2017Torch, CATEGORIES
from .opt_variant import GenerationStep
from .unified_loss import multi_target_hitting_loss
from .measure_jacobian import estimate_spectral_radius_from_h
from .training_utils import (
    _sample_category_balanced, _make_ckpt,
    _build_validation_set, _run_validation,
    _RAW_BLACKLIST,
)

log = logging.getLogger(__name__)

MAX_GENS = 800       # hard ceiling per trajectory
N_TARGETS = 30       # hitting_loss targets
MAX_CUM_RHO = 10.0   # spectral radius detach threshold
SOFT_MIN_BETA = 20.0  # differentiable min beta
GAP_EMA_ALPHA = 0.05  # per-fid gap EMA smoothing


def run_fes_trajectory(
    gen_step: GenerationStep,
    fn: CEC2017Torch,
    *,
    D: int,
    N: int,
    max_fes: int,
    gru_window: int,
    gumbel_tau: float = 1.0,
    bptt_w: int = 20,
    gen_count_ema: float = 50.0,
    device: str = 'cuda',
):
    """Run one FES-budgeted trajectory with hitting_loss.

    Args:
        bptt_w: BPTT window size (last W gens get gradient)
        gen_count_ema: EMA of total gens per step (for reverse-anchoring)

    Returns:
        loss: scalar tensor (differentiable)
        stats: dict with gap, total_gens, cumulative_fes, gen_count_ema, etc.
    """
    B = 1

    # Initialize population
    coords = (torch.rand(B, N, D, device=device) * 200 - 100).to(torch.float64)
    fitness = fn(coords.reshape(-1, D)).reshape(B, N)
    init_best = fitness.min().item()

    # Temporal ring buffer
    coords_ring = torch.zeros(B, gru_window, N, D, dtype=torch.float32, device=device)
    fitness_ring = torch.zeros(B, gru_window, N, dtype=torch.float32, device=device)

    cumulative_fes = 0
    bptt_best = []
    _cached_rho = 1.0
    _cum_rho = 1.0
    _n_detaches = 0
    total_gens = 0

    # Routing diagnostics
    K = gen_step.variant.K if hasattr(gen_step.variant, 'K') else 1
    all_asel_counts = torch.zeros(K)

    from .similarity_graph_gpu import build_sparse_graphs_gpu

    for gen in range(MAX_GENS):
        if cumulative_fes >= max_fes:
            break

        ri = gen % gru_window
        coords_ring[:, ri] = coords.detach().float()
        fitness_ring[:, ri] = fitness.detach().float()
        n_valid = min(gen + 1, gru_window)
        if gen < gru_window:
            idx = list(range(gen + 1))
        else:
            start = (gen + 1) % gru_window
            idx = [(start + i) % gru_window for i in range(gru_window)]
        coords_hist = coords_ring[:, idx]
        fitness_hist = fitness_ring[:, idx]

        with torch.no_grad():
            cache = build_sparse_graphs_gpu(
                coords.float(), fitness.float(),
                step_num=cumulative_fes, max_steps=max_fes, ndim=D, k_neighbors=8)

        temporal_kw = dict(coords_hist=coords_hist,
                           fitness_hist=fitness_hist, n_valid=n_valid)

        # Reverse-anchored BPTT: place window at the end of estimated trajectory.
        # gen_count_ema predicts total gens; gradient starts at (ema - W).
        # Hard cap: never exceed bptt_w gens of BPTT regardless of trajectory length.
        _bptt_start_gen = max(0, int(gen_count_ema) - bptt_w)
        in_bptt = (gen >= _bptt_start_gen) and (len(bptt_best) < bptt_w)

        if in_bptt:
            _ch = coords_hist.clone()
            _fh = fitness_hist.clone()
            _nv = torch.tensor(n_valid)

            def _ckpt_gen(coords_t, fitness_t, ch, fh, nv_t):
                tkw = dict(coords_hist=ch, fitness_hist=fh, n_valid=nv_t.item())
                return gen_step.run(
                    coords=coords_t, fitness=fitness_t, cache=cache,
                    f_optimal=fn.f_optimal, M=1, gumbel_tau=gumbel_tau,
                    node_feat=cache.node_feat, global_feat=cache.global_feat,
                    **tkw)

            result = torch_checkpoint(
                _ckpt_gen, coords, fitness, _ch, _fh, _nv,
                use_reentrant=False)
            bptt_best.append(result['best_fit'])
        else:
            with torch.no_grad():
                result = gen_step.run(
                    coords=coords, fitness=fitness, cache=cache,
                    f_optimal=fn.f_optimal, M=1, gumbel_tau=gumbel_tau,
                    node_feat=cache.node_feat, global_feat=cache.global_feat,
                    **temporal_kw)
        extras = result.get('extras', {})

        # Routing stats
        w = extras.get('winner')
        if w is not None:
            all_asel_counts += w.flatten().bincount(minlength=K).float().cpu()

        old_coords = coords
        coords = result['new_coords']
        fitness = result['new_fitness']

        if in_bptt:
            # Spectral radius adaptive detach (only in BPTT zone)
            if old_coords.requires_grad and gen % 10 == 0:
                _cached_rho = estimate_spectral_radius_from_h(
                    old_coords, coords, n_iters=3)
            _cum_rho *= _cached_rho
            if _cum_rho > MAX_CUM_RHO:
                coords = coords.detach()
                _cum_rho = 1.0
                _n_detaches += 1
        else:
            coords = coords.detach()
            fitness = fitness.detach()

        fes_this_gen = max(extras.get('fes_used', float(N)), 1.0)
        cumulative_fes += fes_this_gen
        total_gens = gen + 1

    # Update gen count EMA for next step's reverse-anchored BPTT
    GEN_EMA_ALPHA = 0.1
    gen_count_ema = (1 - GEN_EMA_ALPHA) * gen_count_ema + GEN_EMA_ALPHA * total_gens

    # Hitting loss
    if bptt_best:
        trajectory = torch.stack(bptt_best)
        loss = multi_target_hitting_loss(
            trajectory, fn.f_optimal, n_targets=N_TARGETS)
    else:
        loss = torch.tensor(0.0, device=device)

    final_gap = fitness.min().item() - fn.f_optimal
    noop_frac = 1.0 - cumulative_fes / (total_gens * N) if total_gens > 0 else 0

    # Routing percentages
    asel_total = max(all_asel_counts.sum().item(), 1)
    asel_pct = [round(100 * c.item() / asel_total, 1) for c in all_asel_counts]

    stats = {
        'gap': final_gap,
        'init_best': init_best,
        'loss': loss.item(),
        'total_gens': total_gens,
        'cumulative_fes': cumulative_fes,
        'noop_frac': round(noop_frac, 3),
        'n_detaches': _n_detaches,
        'n_bptt_gens': len(bptt_best),
        'bptt_w': bptt_w,
        'gen_count_ema': round(gen_count_ema, 1),
        'asel_pct': asel_pct,
    }

    return loss, stats


def train_fes_hitting(
    backbone: nn.Module,
    variant,
    *,
    n_steps: int = 100000,
    budget_mult: int = 10000,
    pop_per_dim: int = 5,
    dims: List[int] = (10, 30, 50),
    lr: float = 3e-4,
    max_grad_norm: float = 10.0,
    device: str = 'cuda',
    save_dir: str = 'checkpoints/fes_hitting',
    save_every: int = 500,
    log_every: int = 10,
    val_every: int = 100,
    patience: int = 300,
    resume_ckpt: Dict = None,
    graph_builder=None,
):
    """FES-hitting training loop over all CEC2017 functions.

    Combines probe_fes_hitting's FES-budgeted trajectory with hitting_loss
    and train_hybrid's multi-function rotation.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    config = {
        'n_steps': n_steps, 'budget_mult': budget_mult,
        'pop_per_dim': pop_per_dim, 'dims': list(dims),
        'lr': lr, 'max_grad_norm': max_grad_norm,
    }

    all_params = list(backbone.parameters()) + list(variant.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)

    start_step = 0
    best_gap_ema = -1.0
    gap_ema_fid = {}
    if resume_ckpt is not None:
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            log.info("Resumed optimizer from step %d", resume_ckpt.get('step', -1))
        gap_ema_fid = resume_ckpt.get('gap_ema_fid', {})

    gru_window = getattr(backbone, 'gru_window', 16)

    # Gumbel tau warmup
    warmup_steps = 1000

    # Per-fid category map
    _fid_to_cat = {}
    for cat, fids in CATEGORIES.items():
        for f in fids:
            _fid_to_cat[f] = cat

    # Validation set
    val_set = _build_validation_set(dims, pop_per_dim, budget_mult, device,
                                     no_augment=True)
    log.info("Validation set: %d entries", len(val_set))

    # Diagnostics file
    diag_file = open(save_path / 'diagnostics.jsonl', 'a')

    patience_counter = 0
    gap_ema_global = -1.0
    history = []

    # Adaptive BPTT window state (persists across steps)
    BPTT_W_INIT = 20
    BPTT_W_MAX = 60
    BPTT_W_STEP = 5  # grow by this much per curriculum advance
    _bptt_w = BPTT_W_INIT
    _gen_count_ema = 50.0
    # Per-fid curriculum level for W growth
    _fid_curriculum = {}  # {(fid, D): level}

    backbone.train()
    variant.train()

    for step in range(start_step, n_steps):
        t0 = time.perf_counter()
        optimizer.zero_grad()

        # Category-balanced function sampling
        fid, D = _sample_category_balanced(dims, loss_ema_fid=gap_ema_fid,
                                            blacklist=_RAW_BLACKLIST)
        fn = CEC2017Torch(fid, D, device)
        N = pop_per_dim * D
        max_fes = budget_mult * D

        # Adaptive BPTT window: grows with curriculum progress per function
        fid_key = (fid, D)
        curr_level = _fid_curriculum.get(fid_key, 0)
        _bptt_w = min(BPTT_W_INIT + curr_level * BPTT_W_STEP, BPTT_W_MAX)

        # Gumbel tau warmup
        if step < warmup_steps:
            gumbel_tau = 5.0 - 4.0 * (step / warmup_steps)
        else:
            gumbel_tau = 1.0

        gen_step = GenerationStep(backbone, variant, eval_fn=fn,
                                   soft_min_beta=SOFT_MIN_BETA,
                                   lb=-100.0, ub=100.0)
        # CEC2017 hardcoded.

        # Run FES-hitting trajectory
        try:
            loss, stats = run_fes_trajectory(
                gen_step, fn, D=D, N=N, max_fes=max_fes,
                gru_window=gru_window, gumbel_tau=gumbel_tau,
                bptt_w=_bptt_w, gen_count_ema=_gen_count_ema,
                device=device)

            loss.backward()
        except torch.cuda.OutOfMemoryError:
            log.warning("step %d | OOM on F%02d D=%d, skipping", step, fid, D)
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            continue

        # Gradient diagnostics
        grad_by_module = defaultdict(float)
        for prefix, model in [('backbone', backbone), ('variant', variant)]:
            for n, p in model.named_parameters():
                if p.grad is not None:
                    grad_by_module[prefix] += p.grad.norm().item() ** 2

        total_gn = sum(v for v in grad_by_module.values()) ** 0.5
        bb_gn = grad_by_module.get('backbone', 0) ** 0.5
        var_gn = grad_by_module.get('variant', 0) ** 0.5

        torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()

        elapsed = (time.perf_counter() - t0) * 1000

        # Update gen_count_ema from trajectory stats
        _gen_count_ema = stats['gen_count_ema']

        # Curriculum advancement: if gap improved significantly, grow BPTT window
        gap = stats['gap']
        key = str((fid, D))
        prev_gap = gap_ema_fid.get(key, gap)
        if key in gap_ema_fid:
            gap_ema_fid[key] = (1 - GAP_EMA_ALPHA) * gap_ema_fid[key] + GAP_EMA_ALPHA * gap
        else:
            gap_ema_fid[key] = gap
        # Advance curriculum if gap dropped by 10x or more
        if prev_gap > 0 and gap < prev_gap * 0.1:
            _fid_curriculum[fid_key] = _fid_curriculum.get(fid_key, 0) + 1

        # Global gap EMA (log-normalized)
        log_gap = math.log(max(gap, 1e-30))
        if gap_ema_global < 0:
            gap_ema_global = log_gap
        else:
            gap_ema_global = 0.99 * gap_ema_global + 0.01 * log_gap

        # Diagnostics
        diag = {
            'step': step, 'fid': fid, 'D': D,
            'gap': gap, 'loss': stats['loss'],
            'total_gens': stats['total_gens'],
            'cumulative_fes': stats['cumulative_fes'],
            'noop_frac': stats['noop_frac'],
            'n_detaches': stats['n_detaches'],
            'grad_norm': round(total_gn, 6),
            'grad_bb': round(bb_gn, 6),
            'grad_var': round(var_gn, 6),
            'asel_pct': stats['asel_pct'],
            'wall_ms': round(elapsed, 1),
            'gumbel_tau': round(gumbel_tau, 3),
        }
        diag_file.write(json.dumps(diag) + '\n')
        diag_file.flush()

        if step % log_every == 0:
            cat = _fid_to_cat.get(fid, '?')
            asel_str = '/'.join(f"{p:.0f}" for p in stats['asel_pct'])
            log.info(
                "step %d | F%02d(%s) D=%d | gap %.2e | loss %.4f | "
                "gens %d fes %d (%.0f%% noop) | gn %.4f | asel[%s] | %.0fms",
                step, fid, cat, D, gap, stats['loss'],
                stats['total_gens'], stats['cumulative_fes'],
                stats['noop_frac'] * 100, total_gn, asel_str, elapsed)

        history.append(diag)

        # Validation + checkpointing
        if step > 0 and step % val_every == 0:
            mean_gc, mean_gap, val_results = _run_validation(
                backbone, variant, val_set, bptt_window=50,
                device=device, graph_builder=graph_builder)
            backbone.train()
            variant.train()

            log.info("step %d | VAL mean_gc=%.4f mean_gap=%.2e (%d fns)",
                     step, mean_gc, mean_gap, len(val_results))

            # Best gap tracking
            if best_gap_ema < 0 or mean_gap < best_gap_ema:
                best_gap_ema = mean_gap
                patience_counter = 0
                ckpt = _make_ckpt(backbone, variant, optimizer, step,
                                   best_gc=mean_gc, best_gap_ema=best_gap_ema,
                                   config=config, gap_ema_fid=gap_ema_fid)
                torch.save(ckpt, save_path / 'best_val.pth')
                log.info("step %d | NEW BEST gap_ema=%.2e, saved", step, mean_gap)
            else:
                patience_counter += val_every
                if patience_counter >= patience:
                    log.info("step %d | Patience exhausted (%d >= %d), stopping",
                             step, patience_counter, patience)
                    break

        # Periodic checkpoint
        if step > 0 and step % save_every == 0:
            ckpt = _make_ckpt(backbone, variant, optimizer, step,
                               best_gc=0, best_gap_ema=best_gap_ema,
                               config=config, gap_ema_fid=gap_ema_fid)
            torch.save(ckpt, save_path / f'step_{step}.pth')

    # Final save
    ckpt = _make_ckpt(backbone, variant, optimizer, n_steps,
                       best_gc=0, best_gap_ema=best_gap_ema,
                       config=config, gap_ema_fid=gap_ema_fid)
    torch.save(ckpt, save_path / 'final.pth')
    diag_file.close()

    log.info("Done. %d steps, best_gap_ema=%.2e", len(history), best_gap_ema)
    return history


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='FES-hitting training for L2O variants')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n-steps', type=int, default=100000)
    parser.add_argument('--budget-mult', type=int, default=10000)
    parser.add_argument('--pop-per-dim', type=int, default=5)
    parser.add_argument('--dims', type=int, nargs='+', default=[10, 30, 50])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--max-grad-norm', type=float, default=10.0)
    parser.add_argument('--save-dir', default='checkpoints/fes_hitting')
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--val-every', type=int, default=100)
    parser.add_argument('--patience', type=int, default=300)
    parser.add_argument('--variant', choices=['k2', 'k4', 'k6',
                                              'direct_k1', 'direct_k4', 'direct_k5'],
                        default='k4')
    parser.add_argument('--ssl-checkpoint', type=str, default=None)
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--topology', choices=['coordinate', 'embedding', 'learned'],
                        default='embedding')
    parser.add_argument('--k-neighbors', type=int, default=8)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

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
        backbone.load_ssl_checkpoint(args.ssl_checkpoint)

    # ── Build variant ──
    if args.variant == 'k2':
        from .variants.classic_k2 import ClassicK2Variant
        variant = ClassicK2Variant(gatv2_hidden=128, global_dim=128).to(args.device)
    elif args.variant == 'k4':
        from .variants.neural_k4 import NeuralK4Variant
        variant = NeuralK4Variant(K=4, head_dim=16, gatv2_hidden=128).to(args.device)
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
        from .variants.hyperopt.variant import HyperOPTK6Variant
        variant = HyperOPTK6Variant(gatv2_hidden=128).to(args.device)

    total_params = sum(p.numel() for p in backbone.parameters()) + \
                   sum(p.numel() for p in variant.parameters())
    log.info("Total params: %d (backbone: %d, variant: %d)",
             total_params,
             sum(p.numel() for p in backbone.parameters()),
             sum(p.numel() for p in variant.parameters()))

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        backbone.load_state_dict(resume_ckpt['backbone_state_dict'])
        if 'variant_state_dict' in resume_ckpt:
            variant.load_state_dict(resume_ckpt['variant_state_dict'])
        log.info("Resumed from %s (step %d)", args.resume, resume_ckpt.get('step', -1))

    train_fes_hitting(
        backbone, variant,
        n_steps=args.n_steps,
        budget_mult=args.budget_mult,
        pop_per_dim=args.pop_per_dim,
        dims=args.dims,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        save_dir=args.save_dir,
        save_every=args.save_every,
        log_every=args.log_every,
        val_every=args.val_every,
        patience=args.patience,
        resume_ckpt=resume_ckpt,
        graph_builder=graph_builder,
    )
