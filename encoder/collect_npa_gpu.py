"""
collect_npa_gpu.py — GPU-resident data collection for K=2 SSL.

Five MOS-style strategies (chosen uniformly at random per run):
  - shade_only:  SHADE every gen, no LS1 applied
  - mos_best:    LS1 on the single best individual (classic MOS)
  - mos_top3:    LS1 on the top 3 individuals
  - mos_top10:   LS1 on the top 10% by fitness
  - oracle:      LS1 wherever counterfactual says LS1 > SHADE

All strategies produce identical per-node oracle labels from a
counterfactual LS1 pass on ALL individuals every generation.
The strategies differ only in which LS1 results are APPLIED to the
population trajectory, creating diverse population geometries:
  - shade_only → dispersed, exploratory populations
  - mos_best   → one refined individual among dispersed cloud
  - mos_top3   → small elite cluster + dispersed majority
  - mos_top10  → gradient of refined-to-exploratory individuals
  - oracle     → trajectory under optimal switching (deployment target)

Usage:
    python -m encoder.collect_npa_gpu --device cuda \
        --out-dir DATASETS/NPA_GPU --max-snapshots 540000
"""

import argparse
import logging
import os
import pickle
import random
import time

import numpy as np
import torch

from .augmented_cec2017 import AugmentedCEC2017
from .similarity_graph_gpu import build_similarity_graph_gpu

from GNN_MOS_Classic.shared import SHADEMemoryGPU, shade_gpu, batched_mtsls1_gpu

log = logging.getLogger(__name__)

STRATEGIES = ('shade_only', 'mos_best', 'mos_top3', 'mos_top10', 'oracle')


def _to_np16(t):
    """GPU tensor → numpy float16 (compact storage)."""
    return t.cpu().to(torch.float16).numpy()


def _to_np32(t):
    """GPU tensor → numpy float32."""
    return t.cpu().numpy().astype(np.float32)


HISTORY_W = 50  # temporal window — full trajectory, slice at training time


@torch.no_grad()
def run_one(aug, device, strategy='mos_best', n_pop=100, n_gens=50,
            budget_factor=10000, ls1_evals=10, history_w=HISTORY_W):
    """Run SHADE + MOS-style LS1 on one augmented function, fully on GPU.

    Counterfactual LS1 runs on ALL individuals every generation (labels only).
    The strategy controls which counterfactual results are applied to the
    actual population trajectory.

    Returns:
        snapshots: list of dicts (one per generation)
        fid: int
        D: int
    """
    fn = aug.sample()
    fid, D = fn.fid, fn.D

    max_evals = budget_factor * D
    n_gens = min(n_gens, max_evals // max(n_pop, 1))

    fn_gpu = fn

    # Initialise population
    x = torch.rand(n_pop, D, device=device, dtype=torch.float64) * 200 - 100
    fitness = fn_gpu(x)

    shade_mem = SHADEMemoryGPU(H=10, device=device)
    total_evals = n_pop

    snapshots = []
    prev_best = fitness.min().detach()
    stagnation_counter = 0
    delta_fitness = 0.0
    contraction_rate = 0.0
    parent_info = None
    diversity_history = []  # ring buffer (W=5) for contraction_rate
    TEMPORAL_W = 5

    # NPA history ring buffer: (W, N, D) coords and (W, N) fitness on GPU
    coords_ring = torch.zeros(HISTORY_W, n_pop, D, device=device, dtype=torch.float32)
    fitness_ring = torch.zeros(HISTORY_W, n_pop, device=device, dtype=torch.float32)
    ring_len = 0  # how many valid entries (grows to HISTORY_W)

    for gen in range(n_gens):
        gen_frac = gen / max(n_gens - 1, 1)
        remaining_fes = max_evals - total_evals

        # -- Snapshot BEFORE operators (no label leakage) --
        graph_out = build_similarity_graph_gpu(
            x.float(), fitness.float(),
            step_num=gen, max_steps=n_gens, ndim=D,
            prev_best=prev_best,
            parent_info=parent_info,
            stagnation_counter=stagnation_counter,
            delta_fitness=delta_fitness,
            contraction_rate=contraction_rate,
        )
        node_feat = graph_out[0]
        edge_index = graph_out[1]
        edge_attr = graph_out[2]
        global_feat = graph_out[3]

        # -- Fitness rank (pre-operator, no leakage) --
        # Lower fitness = better → rank 0.0 = best, 1.0 = worst
        order = fitness.argsort()
        fitness_rank = torch.empty_like(fitness, dtype=torch.float32)
        fitness_rank[order] = torch.linspace(0, 1, n_pop, device=device)

        # -- SHADE: full population --
        prev_x = x.clone()
        prev_fitness_tensor = fitness.clone()

        children, F_vals, CR_vals = shade_gpu(
            x, fitness, x, shade_mem, gen_frac, lb=-100.0, ub=100.0)
        children_fit = fn_gpu(children)
        total_evals += n_pop

        shade_mem.report_child_fitness(children_fit)
        shade_improved = children_fit < fitness              # (N,)
        shade_delta_raw = torch.clamp(fitness - children_fit, min=0.0)  # (N,)

        x_after_shade = torch.where(shade_improved.unsqueeze(1), children, x)
        f_after_shade = torch.where(shade_improved, children_fit, fitness)

        # -- Counterfactual LS1: ALL individuals (labels only) --
        # Run on post-SHADE state so oracle compares "SHADE then stop" vs
        # "SHADE then LS1 refine".
        ls1_x, ls1_f, ls1_evals_used = batched_mtsls1_gpu(
            x_after_shade, f_after_shade, fn_gpu,
            lb=-100.0, ub=100.0,
            max_evals=ls1_evals, sr_frac=0.2)
        # Don't count counterfactual evals in budget (they're for labels only)

        ls1_improved = ls1_f < f_after_shade                 # (N,)
        ls1_delta_raw = torch.clamp(f_after_shade - ls1_f, min=0.0)  # (N,)

        # -- Oracle labels --
        # Myopic: raw LS1 > SHADE (ignores eval cost and trajectory effects)
        oracle_switch_myopic = (ls1_delta_raw > shade_delta_raw).to(torch.uint8)

        # Cost-adjusted: LS1 improvement PER EVAL > SHADE improvement per eval
        # LS1 costs ls1_evals per individual, SHADE costs 1.
        # This answers: "is LS1 more efficient per function evaluation?"
        ls1_eff_node = ls1_delta_raw / ls1_evals       # (N,) improvement per eval
        shade_eff_node = shade_delta_raw                # (N,) improvement per 1 eval
        oracle_switch_adjusted = (ls1_eff_node > shade_eff_node).to(torch.uint8)

        # optimal_ls1_frac: population-level budget allocation
        shade_eff = shade_delta_raw.sum()
        ls1_eff = ls1_delta_raw.sum()
        shade_eff_per_eval = shade_eff / n_pop
        ls1_eff_per_eval = ls1_eff / (n_pop * ls1_evals)
        denom = shade_eff_per_eval + ls1_eff_per_eval + 1e-30
        optimal_ls1_frac = float(ls1_eff_per_eval / denom)
        remaining_ratio = remaining_fes / max_evals
        optimal_ls1_frac_scaled = optimal_ls1_frac * remaining_ratio

        # -- NPA history: push current state into ring buffer --
        # Push BEFORE operators (same time as graph snapshot — no leakage)
        ring_idx = gen % HISTORY_W
        coords_ring[ring_idx] = x.float()
        fitness_ring[ring_idx] = fitness.float()
        ring_len = min(ring_len + 1, HISTORY_W)

        # Build contiguous (W, N, D) and (W, N) in temporal order
        history_full = ring_len >= HISTORY_W
        if ring_len >= HISTORY_W:
            start = (ring_idx + 1) % HISTORY_W
            idx_order = [(start + i) % HISTORY_W for i in range(HISTORY_W)]
        else:
            # Partial buffer: first ring_len entries in order
            idx_order = list(range(ring_len))
        coords_hist = coords_ring[idx_order]      # (n_valid, N, D)
        fitness_hist = fitness_ring[idx_order]     # (n_valid, N)

        # -- Build snapshot --
        snap = {
            # Graph features (float16 for edges/nodes, float32 for global)
            'node_feat': _to_np16(node_feat),
            'edge_index': edge_index.cpu().to(torch.int32).numpy(),
            'edge_attr': _to_np16(edge_attr),
            'global_feat': _to_np32(global_feat),
            # Raw state (float16 for coords)
            'coordinates': _to_np16(x),
            'fitness': _to_np32(fitness),
            # NPA history: only saved when ring buffer is full (gen >= W-1)
            'has_history': True,  # always present now
            'n_valid_hist': ring_len,  # how many timesteps are valid (1..W)
            'coords_hist': _to_np16(coords_hist),    # (n_valid, N, D)
            'fitness_hist': _to_np16(fitness_hist),   # (n_valid, N)
            # Per-node oracle labels
            'shade_improved': shade_improved.cpu().numpy(),
            'shade_delta': _to_np16(torch.log1p(shade_delta_raw)),
            'ls1_improved': ls1_improved.cpu().numpy(),
            'ls1_delta': _to_np16(torch.log1p(ls1_delta_raw)),
            'oracle_switch_myopic': oracle_switch_myopic.cpu().numpy(),
            'oracle_switch_adjusted': oracle_switch_adjusted.cpu().numpy(),
            'fitness_rank': _to_np16(fitness_rank),
            # Graph-level oracle labels
            'optimal_ls1_frac': np.float32(optimal_ls1_frac),
            'optimal_ls1_frac_scaled': np.float32(optimal_ls1_frac_scaled),
            'remaining_fes_ratio': np.float32(remaining_ratio),
            # Metadata
            'gen': gen,
            'n_gens': n_gens,
            'fid': fid,
            'ndim': D,
            'strategy': strategy,
        }

        # -- MOS-style LS1 application (affects trajectory) --
        x_new = x_after_shade.clone()
        f_new = f_after_shade.clone()

        if strategy == 'shade_only':
            n_apply = 0
        elif strategy == 'mos_best':
            n_apply = 1
        elif strategy == 'mos_top3':
            n_apply = 3
        elif strategy == 'mos_top10':
            n_apply = max(1, n_pop // 10)
        elif strategy == 'oracle':
            # Apply LS1 wherever cost-adjusted counterfactual says LS1 > SHADE
            apply_mask = oracle_switch_adjusted.bool() & ls1_improved
            n_apply = 0  # handled separately below
            if apply_mask.any():
                x_new[apply_mask] = ls1_x[apply_mask]
                f_new[apply_mask] = ls1_f[apply_mask]
                total_evals += int(apply_mask.sum().item()) * ls1_evals
        else:
            n_apply = 0

        if n_apply > 0:
            # Select top-n by fitness (lower = better)
            k = min(n_apply, n_pop)
            _, top_idx = torch.topk(-f_new, k)
            # Apply counterfactual LS1 results to selected individuals
            x_new[top_idx] = ls1_x[top_idx]
            f_new[top_idx] = ls1_f[top_idx]
            total_evals += k * ls1_evals

        # -- Parent info for next gen's lineage features --
        parent_info = {
            'parent_x': prev_x.float(),
            'child_x': x_new.float(),
            'parent_fitness': prev_fitness_tensor.float(),
            'child_fitness': f_new.float(),
            'parent_fit_rank': torch.zeros(n_pop, device=device),
        }

        # -- Update temporal state --
        new_best = f_new.min().detach()
        if new_best < prev_best - 1e-12:
            raw_imp = float(prev_best - new_best)
            delta_fitness = min(raw_imp / (abs(float(prev_best)) + 1e-8), 1.0)
            stagnation_counter = 0
        else:
            delta_fitness = 0.0
            stagnation_counter += 1

        # Contraction rate: diversity change over temporal window
        current_div = float(x_new.float().std())
        if len(diversity_history) >= 2 and diversity_history[0] > 1e-8:
            contraction_rate = max(-1.0, min(current_div / diversity_history[0] - 1.0, 1.0))
        else:
            contraction_rate = 0.0
        diversity_history.append(current_div)
        if len(diversity_history) > TEMPORAL_W:
            diversity_history.pop(0)

        prev_best = new_best
        x = x_new
        fitness = f_new
        snapshots.append(snap)

        if total_evals >= max_evals:
            break

    return snapshots, fid, D


def _flush(buffers, out_dir, flush_count):
    for D, buf in buffers.items():
        if not buf:
            continue
        path = os.path.join(out_dir, f"gpu_d{D}_{flush_count:04d}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(buf, f, protocol=pickle.HIGHEST_PROTOCOL)
        n = len(buf)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        buf.clear()
        log.info("  Flushed %d snapshots → %s (%.1f MB)", n, path, size_mb)


def main():
    parser = argparse.ArgumentParser(
        description="GPU-resident data collection for K=2 SSL")
    parser.add_argument("--out-dir", type=str, default="DATASETS/NPA_GPU")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-pop", type=int, default=100)
    parser.add_argument("--n-gens", type=int, default=50)
    parser.add_argument("--budget-factor", type=int, default=10000)
    parser.add_argument("--ls1-evals", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-snapshots", type=int, default=0,
                        help="Stop after this many snapshots (0=infinite)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    aug = AugmentedCEC2017(device=args.device, dims=(10, 30, 50))

    buffers = {10: [], 30: [], 50: []}
    strat_counts = {s: 0 for s in STRATEGIES}
    total_runs = 0
    total_snapshots = 0
    t_start = time.time()

    # Resume: estimate existing snapshots from file count and find max flush_count
    import glob as _glob
    existing_pkls = _glob.glob(os.path.join(args.out_dir, "gpu_d*_*.pkl"))
    flush_count = 0
    if existing_pkls:
        for p in existing_pkls:
            try:
                fc = int(os.path.basename(p).split('_')[-1].replace('.pkl', ''))
                flush_count = max(flush_count, fc)
            except ValueError:
                pass
        # Estimate snapshots: sample 10 recent files for accurate avg size
        total_bytes = sum(os.path.getsize(p) for p in existing_pkls)
        sample_files = sorted(existing_pkls)[-10:]
        sample_snaps = 0
        sample_bytes = 0
        for sp in sample_files:
            try:
                sample_bytes += os.path.getsize(sp)
                with open(sp, 'rb') as f:
                    data = pickle.load(f)
                sample_snaps += len(data) if isinstance(data, list) else 1
            except (EOFError, pickle.UnpicklingError, OSError):
                pass
        avg_snap_bytes = sample_bytes / max(sample_snaps, 1)
        total_snapshots = int(total_bytes / avg_snap_bytes)
        log.info("Resuming: ~%d snapshots estimated from %d files, %.1f GB (flush_count=%d)",
                 total_snapshots, len(existing_pkls), total_bytes / 1e9, flush_count)

    log.info("GPU-resident collection → %s", args.out_dir)
    log.info("Strategies: %s (uniform random per run)", STRATEGIES)
    if args.max_snapshots > 0:
        log.info("Target: %d snapshots (≈%.1f GB)",
                 args.max_snapshots, args.max_snapshots * 27.5 / 1024 / 1024)

    try:
        while True:
            # Pick strategy uniformly
            strategy = random.choice(STRATEGIES)
            t0 = time.time()

            try:
                snaps, fid, D = run_one(
                    aug, device, strategy=strategy,
                    n_pop=args.n_pop, n_gens=args.n_gens,
                    budget_factor=args.budget_factor,
                    ls1_evals=args.ls1_evals)
            except Exception as e:
                log.warning("Run failed: %s", e)
                continue

            buffers[D].extend(snaps)
            strat_counts[strategy] += 1
            total_runs += 1
            total_snapshots += len(snaps)
            dt = time.time() - t0

            if total_runs % 10 == 0:
                elapsed = time.time() - t_start
                rate = total_snapshots / elapsed * 3600
                sc = strat_counts
                log.info("Run %d | F%02d D=%d %s | %d snaps (%.2fs) | "
                         "total: %d (%.0f/hr) | so=%d mb=%d t3=%d t10=%d or=%d",
                         total_runs, fid, D, strategy[:5], len(snaps), dt,
                         total_snapshots, rate,
                         sc['shade_only'], sc['mos_best'], sc['mos_top3'],
                         sc['mos_top10'], sc['oracle'])

            if total_runs % args.save_every == 0:
                flush_count += 1
                _flush(buffers, args.out_dir, flush_count)

            if args.max_snapshots > 0 and total_snapshots >= args.max_snapshots:
                log.info("Reached target: %d snapshots", total_snapshots)
                break

    except KeyboardInterrupt:
        log.info("Interrupted. Flushing remaining buffer...")

    # Always flush on exit (normal or interrupt)
    flush_count += 1
    _flush(buffers, args.out_dir, flush_count)

    elapsed = time.time() - t_start
    log.info("Done: %d runs, %d snaps in %.0fs (%.0f/hr) | %s",
             total_runs, total_snapshots, elapsed,
             total_snapshots / max(elapsed, 1) * 3600, strat_counts)


if __name__ == '__main__':
    main()
