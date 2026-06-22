"""
collect_npa_data.py — Infinite SSL data collection using augmented CEC2017.

Runs the K=4 ensemble optimizer on randomly augmented CEC2017 functions
(rotation + shift + scale) to generate unlimited trajectory data for NPA
pretraining.  Each iteration picks a random (fid, D), applies a random
affine transform, runs the ensemble for 20 epochs, and appends the
snapshots to a growing pkl file.

Usage:
    python -m encoder.collect_npa_data --out-dir DATASETS/NPA_AUGMENTED \
        --device cpu

Runs forever (Ctrl+C to stop). Data accumulates in per-dimension pkl files.
"""

import argparse
import logging
import os
import pickle
import time

import numpy as np
import torch

from .augmented_cec2017 import AugmentedCEC2017

# Import ensemble machinery
from ENSEMBLE_K4.ensemble_optimizer import (
    K_SUBPOPS, create_subpops, EnsembleOptimizer,
)
from ENSEMBLE_K4.ensemble_graph import build_ensemble_graph
from ENSEMBLE_K4.ssl_data_collector_ensemble import (
    _take_ensemble_snapshot, _add_shift_labels,
)

log = logging.getLogger(__name__)


def _make_numpy_eval(aug_fn):
    """Wrap a torch AugmentedFunction as a numpy callable for the ensemble."""
    device = aug_fn.device
    D = aug_fn.D

    def eval_fn(x):
        x_t = torch.from_numpy(x.reshape(1, D)).to(
            device=device, dtype=torch.float64)
        f = aug_fn(x_t)
        return float(f.item())

    return eval_fn


def run_one_augmented(aug, seed, n_epochs=20, budget_factor=10000):
    """Run ensemble on one random augmented function, return snapshots."""
    fn = aug.sample()
    fid, D = fn.fid, fn.D
    f_optimal = float(fn.f_optimal)

    eval_fn = _make_numpy_eval(fn)
    lb = np.full(D, -100.0)
    ub = np.full(D, 100.0)
    bounds = np.column_stack([lb, ub])
    max_evals = budget_factor * D

    # Allocation: random Dirichlet
    rng = np.random.default_rng(seed)
    alloc_fracs = rng.dirichlet(np.full(K_SUBPOPS, 1.5))

    sub_pops = create_subpops(D, lb, ub, max_evals, seed=seed)

    eval_count = [0]

    def counted_eval(x):
        eval_count[0] += 1
        val = eval_fn(x)
        if np.isnan(val):
            return 1e30
        return val

    for sp in sub_pops:
        sp.initialize(counted_eval)

    init_evals = eval_count[0]
    remaining_budget = max_evals - init_evals
    epoch_budget_base = remaining_budget // n_epochs

    best_fitness = np.inf
    for sp in sub_pops:
        bx, bf = sp.get_best()
        if bf < best_fitness:
            best_fitness = bf

    prev_best = float(best_fitness)
    snapshots = []

    opt_helper = EnsembleOptimizer(D, lb, ub, max_evals, n_epochs=n_epochs)
    prev_epoch_stats = None
    EMA_ALPHA = 0.3
    ema_efficiency = np.zeros(K_SUBPOPS, dtype=np.float64)
    improvement_ema = None
    prev_epoch_detail = None
    stagnation_counter = 0
    best_history = []
    diversity_history = []
    TEMPORAL_W = 5

    for epoch in range(n_epochs):
        remaining = max_evals - eval_count[0]
        if remaining <= 0:
            break

        epoch_budget = min(epoch_budget_base, remaining)
        budgets = opt_helper._apply_min_floor(alloc_fracs, epoch_budget)

        # Temporal features
        all_coords_temp = [sp.get_population()[0] for sp in sub_pops
                           if len(sp.get_population()[1]) > 0]
        current_div = float(np.std(np.vstack(all_coords_temp))) if all_coords_temp else 0.0

        delta_fitness = 0.0
        if len(best_history) >= 2:
            delta_fitness = np.clip(
                (best_history[0] - best_fitness) / (abs(best_history[0]) + 1e-8),
                -1, 1)

        contraction_rate_val = 0.0
        if len(diversity_history) >= 2 and diversity_history[0] > 1e-8:
            contraction_rate_val = np.clip(
                current_div / diversity_history[0] - 1.0, -1, 1)

        snapshot = _take_ensemble_snapshot(
            sub_pops=sub_pops, bounds=bounds,
            epoch=epoch, n_epochs=n_epochs,
            allocation_used=alloc_fracs.tolist(),
            epoch_stats=prev_epoch_stats,
            prev_best=prev_best,
            func_id=fid, ndim=D, seed=seed,
            evals_used=eval_count[0], max_evals=max_evals,
            allocation_strategy='dirichlet_aug',
            alloc_idx=0,
            best_fitness=best_fitness, D=D,
            ema_efficiency=ema_efficiency,
            improvement_ema=improvement_ema,
            epoch_detail=prev_epoch_detail,
            stagnation_counter=stagnation_counter,
            delta_fitness=delta_fitness,
            contraction_rate=contraction_rate_val,
        )
        if snapshot is not None:
            snapshots.append(snapshot)

        # Run sub-pops
        epoch_stats = []
        epoch_details = []
        for i, sp in enumerate(sub_pops):
            budget_i = budgets[i]
            _, fit_sp = sp.get_population()
            if budget_i < 2:
                epoch_stats.append({'skipped': True, 'name': sp.name})
                epoch_details.append({
                    'evals_used': 0, 'evals_alloc': int(budget_i),
                    'best_fit': float(np.min(fit_sp)) if len(fit_sp) > 0 else float('inf'),
                    'mean_fit': float(np.mean(fit_sp)) if len(fit_sp) > 0 else float('inf'),
                    'diversity': float(np.std(fit_sp)) if len(fit_sp) > 1 else 0.0,
                    'pop_size': len(fit_sp), 'n_gens': 0,
                })
                continue

            best_before = float(np.min(fit_sp)) if len(fit_sp) > 0 else np.inf
            evals_used, stats = sp.run_budget(counted_eval, budget_i)
            stats['name'] = sp.name
            stats['evals_allocated'] = budget_i
            stats['evals_used'] = evals_used

            _, fit_after = sp.get_population()
            best_after = float(np.min(fit_after)) if len(fit_after) > 0 else np.inf
            stats['improvement'] = max(best_before - best_after, 0.0)
            epoch_stats.append(stats)
            epoch_details.append({
                'evals_used': int(evals_used), 'evals_alloc': int(budget_i),
                'best_fit': float(np.min(fit_after)) if len(fit_after) > 0 else float('inf'),
                'mean_fit': float(np.mean(fit_after)) if len(fit_after) > 0 else float('inf'),
                'diversity': float(np.std(fit_after)) if len(fit_after) > 1 else 0.0,
                'pop_size': len(fit_after),
                'n_gens': stats.get('n_gens', 0),
            })

        prev_epoch_stats = epoch_stats

        # Update EMA
        epoch_eff = np.zeros(K_SUBPOPS, dtype=np.float64)
        for i, ep_s in enumerate(epoch_stats):
            if i >= K_SUBPOPS or not ep_s or ep_s.get('skipped', False):
                continue
            eu = ep_s.get('evals_used', 0)
            imp = max(ep_s.get('improvement', 0.0), 0.0)
            if eu > 0:
                epoch_eff[i] = imp / eu
        ema_efficiency = EMA_ALPHA * epoch_eff + (1 - EMA_ALPHA) * ema_efficiency

        # Migration
        pop_before = [len(sp.get_population()[1]) for sp in sub_pops]
        opt_helper._migrate(sub_pops)
        pop_after = [len(sp.get_population()[1]) for sp in sub_pops]
        migration_received = [max(a - b, 0) for a, b in zip(pop_after, pop_before)]

        prev_epoch_detail = {
            'per_subpop_evals_used': [d['evals_used'] for d in epoch_details],
            'per_subpop_evals_allocated': [d['evals_alloc'] for d in epoch_details],
            'per_subpop_best_fitness': [d['best_fit'] for d in epoch_details],
            'per_subpop_mean_fitness': [d['mean_fit'] for d in epoch_details],
            'per_subpop_diversity': [d['diversity'] for d in epoch_details],
            'per_subpop_pop_size': [d['pop_size'] for d in epoch_details],
            'per_subpop_n_generations': [d.get('n_gens', 0) for d in epoch_details],
            'wall_time_epoch': 0.0,
            'migration_received': migration_received,
        }

        for sp in sub_pops:
            bx, bf = sp.get_best()
            if bf < best_fitness:
                best_fitness = bf

        raw_improvement = max(prev_best - float(best_fitness), 0.0)
        norm_imp = np.clip(raw_improvement / (abs(prev_best) + 1e-8), -1, 1)
        if improvement_ema is None:
            improvement_ema = norm_imp
        else:
            improvement_ema = EMA_ALPHA * norm_imp + (1 - EMA_ALPHA) * improvement_ema

        if best_fitness < prev_best - 1e-12:
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        best_history.append(float(best_fitness))
        if len(best_history) > TEMPORAL_W:
            best_history.pop(0)
        diversity_history.append(current_div)
        if len(diversity_history) > TEMPORAL_W:
            diversity_history.pop(0)

        prev_best = float(best_fitness)

    _add_shift_labels(snapshots)
    return snapshots, fid, D


def main():
    parser = argparse.ArgumentParser(
        description="Infinite augmented CEC2017 data collection for NPA SSL")
    parser.add_argument("--out-dir", type=str, default="DATASETS/NPA_AUGMENTED")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for CEC2017Torch (cpu recommended for collection)")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--budget-factor", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=50,
                        help="Flush to disk every N runs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    os.makedirs(args.out_dir, exist_ok=True)

    aug = AugmentedCEC2017(device=args.device, dims=(10, 30, 50))

    # Accumulate per dimension
    buffers = {10: [], 30: [], 50: []}
    counters = {10: 0, 30: 0, 50: 0}
    total_runs = 0
    total_snapshots = 0
    t_start = time.time()

    log.info("Starting infinite collection → %s", args.out_dir)
    log.info("Press Ctrl+C to stop. Data saved every %d runs.", args.save_every)

    try:
        while True:
            seed = int(torch.randint(0, 2**31, (1,)).item())
            t0 = time.time()

            try:
                snaps, fid, D = run_one_augmented(
                    aug, seed, n_epochs=args.n_epochs,
                    budget_factor=args.budget_factor)
            except Exception as e:
                log.warning("Run failed (seed=%d): %s", seed, e)
                continue

            buffers[D].extend(snaps)
            counters[D] += 1
            total_runs += 1
            total_snapshots += len(snaps)
            dt = time.time() - t0

            if total_runs % 10 == 0:
                elapsed = time.time() - t_start
                rate = total_snapshots / elapsed * 3600
                log.info("Run %d | F%02d D=%d | %d snaps (%.1fs) | "
                         "total: %d snaps (%.0f/hr) | "
                         "D10=%d D30=%d D50=%d runs",
                         total_runs, fid, D, len(snaps), dt,
                         total_snapshots, rate,
                         counters[10], counters[30], counters[50])

            # Periodic flush
            if total_runs % args.save_every == 0:
                _flush(buffers, args.out_dir)

    except KeyboardInterrupt:
        log.info("Interrupted. Flushing remaining data...")
        _flush(buffers, args.out_dir)

    elapsed = time.time() - t_start
    log.info("Done: %d runs, %d snapshots in %.0fs (%.0f snaps/hr)",
             total_runs, total_snapshots, elapsed,
             total_snapshots / max(elapsed, 1) * 3600)


def _flush(buffers, out_dir):
    """Append accumulated snapshots to per-dimension pkl files."""
    for D, buf in buffers.items():
        if not buf:
            continue
        path = os.path.join(out_dir, f"augmented_d{D}.pkl")
        # Load existing + append
        existing = []
        if os.path.exists(path):
            with open(path, 'rb') as f:
                existing = pickle.load(f)
        existing.extend(buf)
        with open(path, 'wb') as f:
            pickle.dump(existing, f, protocol=pickle.HIGHEST_PROTOCOL)
        n_new = len(buf)
        buf.clear()
        log.info("  Flushed %d snapshots to %s (total: %d)",
                 n_new, path, len(existing))


if __name__ == '__main__':
    main()
