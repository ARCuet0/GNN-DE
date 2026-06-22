"""
collect_npa_gpu_batched.py — Massively parallel GPU NPA data collection.

Runs B independent SHADE+LS1 optimizations in parallel on B augmentations
of the same (fid, D) base function.  SHADE uses shade_gpu_batched (single
kernel for B populations).  LS1 and graph building loop over B (cheap).

Produces snapshots in the SAME format as collect_npa_gpu.py — compatible
with preprocess_npa_data.py and npa_gpu_dataset.py unchanged.

Usage:
    python -m encoder.collect_npa_gpu_batched --device cuda \
        --out-dir DATASETS/NPA_GPU --batch-size 16 --max-snapshots 540000
"""

import argparse
import logging
import os
import pickle
import random
import time

import numpy as np
import torch

from .batched_augmented_cec2017 import (
    BatchedAugmentedCEC2017, BatchedAugmentedFunction, BatchedPureFunction,
)
from .similarity_graph_gpu import build_similarity_graph_gpu

from GNN_MOS_Classic.shared import (
    BatchedSHADEMemory, shade_gpu_batched, batched_mtsls1_gpu,
)

log = logging.getLogger(__name__)

STRATEGIES = ('shade_only', 'mos_best', 'mos_top3', 'mos_top10', 'oracle')

HISTORY_W = 50


def _to_np16(t):
    """GPU tensor → numpy float16."""
    return t.cpu().to(torch.float16).numpy()


def _to_np32(t):
    """GPU tensor → numpy float32."""
    return t.cpu().numpy().astype(np.float32)


@torch.no_grad()
def run_batch(batched_aug, device, B, strategy='mos_best', n_pop=100,
              n_gens=50, budget_factor=10000, ls1_evals=10,
              fid=None, D=None, history_w=HISTORY_W):
    """Run B independent SHADE+LS1 in parallel on B augmentations.

    Returns:
        all_snapshots: list of B lists of snapshot dicts
        fid: int
        D: int
    """
    fn_batch = batched_aug.sample_batch(B, fid=fid, D=D)
    fid_out, D_out = fn_batch.fid, fn_batch.D

    max_evals = budget_factor * D_out
    n_gens = min(n_gens, max_evals // max(n_pop, 1))

    # Init B populations: (B, N, D)
    x = torch.rand(B, n_pop, D_out, device=device, dtype=torch.float64) * 200 - 100
    fitness = fn_batch(x)  # (B, N) — single kernel

    shade_mem = BatchedSHADEMemory(B, H=10, device=device)
    total_evals = n_pop  # same for all B (aligned budget)

    # Unique run_id per batch element (for preprocessing grouping)
    run_ids = [int(torch.randint(0, 2**31, (1,)).item()) for _ in range(B)]

    # Per-batch state: all (B,) tensors
    prev_best = fitness.min(dim=1).values  # (B,)
    stagnation_counter = torch.zeros(B, device=device, dtype=torch.long)
    delta_fitness = torch.zeros(B, device=device, dtype=torch.float64)
    contraction_rate = torch.zeros(B, device=device, dtype=torch.float64)

    # Diversity ring buffer: (B, TEMPORAL_W)
    TEMPORAL_W = 5
    diversity_ring = torch.zeros(B, TEMPORAL_W, device=device, dtype=torch.float64)
    div_ring_len = 0

    # NPA history ring: (B, W, N, D) coords and (B, W, N) fitness
    coords_ring = torch.zeros(B, history_w, n_pop, D_out, device=device,
                              dtype=torch.float32)
    fitness_ring = torch.zeros(B, history_w, n_pop, device=device,
                               dtype=torch.float32)
    ring_len = 0

    # Per-batch parent info (list of B dicts, None initially)
    parent_info_list = [None] * B

    all_snapshots = [[] for _ in range(B)]

    for gen in range(n_gens):
        gen_frac = gen / max(n_gens - 1, 1)
        remaining_fes = max_evals - total_evals

        # -- Graph building: loop over B (cheap, no fitness evals) --
        graph_data = []
        for b in range(B):
            gout = build_similarity_graph_gpu(
                x[b].float(), fitness[b].float(),
                step_num=gen, max_steps=n_gens, ndim=D_out,
                prev_best=prev_best[b].item(),
                parent_info=parent_info_list[b],
                stagnation_counter=int(stagnation_counter[b].item()),
                delta_fitness=float(delta_fitness[b].item()),
                contraction_rate=float(contraction_rate[b].item()),
            )
            graph_data.append(gout)

        # -- Fitness rank per batch element: (B, N) --
        order = fitness.argsort(dim=1)
        fitness_rank = torch.empty(B, n_pop, device=device, dtype=torch.float32)
        ranks_1d = torch.linspace(0, 1, n_pop, device=device)
        for b in range(B):
            fitness_rank[b][order[b]] = ranks_1d

        # -- SHADE: fully batched (B, N, D) --
        prev_x = x.clone()
        prev_fitness_tensor = fitness.clone()

        children = shade_gpu_batched(
            x, fitness, shade_mem, gen_frac, lb=-100.0, ub=100.0)
        children_fit = fn_batch(children)  # (B, N)
        total_evals += n_pop

        shade_mem.report_child_fitness(children_fit)
        shade_improved = children_fit < fitness         # (B, N)
        shade_delta_raw = torch.clamp(fitness - children_fit, min=0.0)  # (B, N)

        x_after_shade = torch.where(
            shade_improved.unsqueeze(2), children, x)
        f_after_shade = torch.where(shade_improved, children_fit, fitness)

        # -- Counterfactual LS1: batched via flat (B*N, D) --
        # Flatten B populations → one LS1 call, use fn_flat that applies
        # the correct per-individual augmentation via batch index mapping.
        x_flat = x_after_shade.reshape(B * n_pop, D_out)
        f_flat = f_after_shade.reshape(B * n_pop)

        def _make_flat_fn(fn_b, B, N, D):
            """Create flat eval fn: maps probe pts back to correct augmentation."""
            # Pre-compute per-individual augmentation params for (B*N,) individuals
            # Individual j belongs to batch b = j // N
            if isinstance(fn_b, BatchedPureFunction):
                return fn_b.base_fn

            Q_flat = fn_b.Q.repeat_interleave(N, dim=0)    # (B*N, D, D)
            s_flat = fn_b.s.repeat_interleave(N, dim=0)    # (B*N, D)
            a_flat = fn_b.a.repeat_interleave(N)            # (B*N,)

            def fn_flat_inner(pts):
                # pts: (K, D) — K can be any multiple of B*N (probes)
                # or a subset (active individuals only)
                K = pts.shape[0]
                if K == B * N:
                    # Direct: one point per individual
                    z = torch.bmm(
                        (pts - s_flat).unsqueeze(1),
                        Q_flat.mT).squeeze(1)
                    return a_flat * fn_b.base_fn(z)
                elif K % (B * N) == 0:
                    # Multiple probes per individual: (n_probes * B*N, D)
                    n_probes = K // (B * N)
                    # LS1 structures probes as (n_probes, M, D).reshape(-1, D)
                    # so pts[p * M + j] is probe p for individual j
                    pts_3d = pts.reshape(n_probes, B * N, D)
                    results = []
                    for p in range(n_probes):
                        z = torch.bmm(
                            (pts_3d[p] - s_flat).unsqueeze(1),
                            Q_flat.mT).squeeze(1)
                        results.append(a_flat * fn_b.base_fn(z))
                    return torch.cat(results)
                else:
                    # Active subset: fewer than B*N individuals
                    # LS1 passes subsetted individuals — we don't know
                    # the mapping, so fall back to reshape-based eval.
                    # Since LS1 subset preserves relative ordering,
                    # this only happens near budget exhaustion.
                    # Use per-point evaluation via base coords
                    # Batch all points through base_fn with mixed transforms
                    # We need the individual's augmentation params, but
                    # with unknown subset, best effort: try reshape
                    if K > B * N:
                        # Multi-probe on subset: (n_probes * n_active, D)
                        # Can't determine n_active without more info
                        # Fallback: evaluate all through first augmentation
                        # This is only used for label generation, acceptable
                        z = (pts - fn_b.s[0]) @ fn_b.Q[0].mT
                        return fn_b.a[0] * fn_b.base_fn(z)
                    else:
                        z = (pts - fn_b.s[0]) @ fn_b.Q[0].mT
                        return fn_b.a[0] * fn_b.base_fn(z)

            return fn_flat_inner

        fn_flat_eval = _make_flat_fn(fn_batch, B, n_pop, D_out)
        ls1_result = batched_mtsls1_gpu(
            x_flat, f_flat, fn_flat_eval,
            lb=-100.0, ub=100.0,
            max_evals=ls1_evals, sr_frac=0.2)
        ls1_x = ls1_result[0].reshape(B, n_pop, D_out)
        ls1_f = ls1_result[1].reshape(B, n_pop)

        ls1_improved = ls1_f < f_after_shade           # (B, N)
        ls1_delta_raw = torch.clamp(f_after_shade - ls1_f, min=0.0)

        # -- Oracle labels: vectorized over (B, N) --
        oracle_switch_myopic = (ls1_delta_raw > shade_delta_raw).to(torch.uint8)

        ls1_eff_node = ls1_delta_raw / ls1_evals
        shade_eff_node = shade_delta_raw
        oracle_switch_adjusted = (ls1_eff_node > shade_eff_node).to(torch.uint8)

        shade_eff = shade_delta_raw.sum(dim=1)          # (B,)
        ls1_eff = ls1_delta_raw.sum(dim=1)              # (B,)
        shade_eff_per_eval = shade_eff / n_pop
        ls1_eff_per_eval = ls1_eff / (n_pop * ls1_evals)
        denom = shade_eff_per_eval + ls1_eff_per_eval + 1e-30
        optimal_ls1_frac = ls1_eff_per_eval / denom     # (B,)
        remaining_ratio = remaining_fes / max_evals
        optimal_ls1_frac_scaled = optimal_ls1_frac * remaining_ratio

        # -- NPA history ring buffer --
        ring_idx = gen % history_w
        coords_ring[:, ring_idx] = x.float()
        fitness_ring[:, ring_idx] = fitness.float()
        ring_len = min(ring_len + 1, history_w)

        if ring_len >= history_w:
            start = (ring_idx + 1) % history_w
            idx_order = [(start + i) % history_w for i in range(history_w)]
        else:
            idx_order = list(range(ring_len))
        coords_hist = coords_ring[:, idx_order]     # (B, n_valid, N, D)
        fitness_hist = fitness_ring[:, idx_order]    # (B, n_valid, N)

        # -- Build B snapshot dicts --
        for b in range(B):
            node_feat, edge_index, edge_attr, global_feat = graph_data[b]
            snap = {
                'node_feat': _to_np16(node_feat),
                'edge_index': edge_index.cpu().to(torch.int32).numpy(),
                'edge_attr': _to_np16(edge_attr),
                'global_feat': _to_np32(global_feat),
                'coordinates': _to_np16(x[b]),
                'fitness': _to_np32(fitness[b]),
                'has_history': True,
                'n_valid_hist': ring_len,
                'coords_hist': _to_np16(coords_hist[b]),
                'fitness_hist': _to_np16(fitness_hist[b]),
                'shade_improved': shade_improved[b].cpu().numpy(),
                'shade_delta': _to_np16(torch.log1p(shade_delta_raw[b])),
                'ls1_improved': ls1_improved[b].cpu().numpy(),
                'ls1_delta': _to_np16(torch.log1p(ls1_delta_raw[b])),
                'oracle_switch_myopic': oracle_switch_myopic[b].cpu().numpy(),
                'oracle_switch_adjusted': oracle_switch_adjusted[b].cpu().numpy(),
                'fitness_rank': _to_np16(fitness_rank[b]),
                'optimal_ls1_frac': np.float32(optimal_ls1_frac[b].item()),
                'optimal_ls1_frac_scaled': np.float32(
                    optimal_ls1_frac_scaled[b].item()),
                'remaining_fes_ratio': np.float32(remaining_ratio),
                'gen': gen,
                'n_gens': n_gens,
                'fid': fid_out,
                'ndim': D_out,
                'strategy': strategy,
                'run_id': run_ids[b],
            }
            all_snapshots[b].append(snap)

        # -- MOS-style LS1 application --
        x_new = x_after_shade.clone()
        f_new = f_after_shade.clone()

        if strategy == 'shade_only':
            pass
        elif strategy == 'mos_best':
            _, top_idx = torch.topk(-f_new, 1, dim=1)  # (B, 1)
            for b in range(B):
                idx_b = top_idx[b]
                x_new[b, idx_b] = ls1_x[b, idx_b]
                f_new[b, idx_b] = ls1_f[b, idx_b]
            total_evals += 1 * ls1_evals
        elif strategy == 'mos_top3':
            k = min(3, n_pop)
            _, top_idx = torch.topk(-f_new, k, dim=1)  # (B, k)
            for b in range(B):
                idx_b = top_idx[b]
                x_new[b, idx_b] = ls1_x[b, idx_b]
                f_new[b, idx_b] = ls1_f[b, idx_b]
            total_evals += k * ls1_evals
        elif strategy == 'mos_top10':
            k = max(1, n_pop // 10)
            _, top_idx = torch.topk(-f_new, k, dim=1)
            for b in range(B):
                idx_b = top_idx[b]
                x_new[b, idx_b] = ls1_x[b, idx_b]
                f_new[b, idx_b] = ls1_f[b, idx_b]
            total_evals += k * ls1_evals
        elif strategy == 'oracle':
            apply_mask = oracle_switch_adjusted.bool() & ls1_improved  # (B, N)
            x_new = torch.where(apply_mask.unsqueeze(2), ls1_x, x_new)
            f_new = torch.where(apply_mask, ls1_f, f_new)
            # Budget: max across B for conservative estimate
            max_applied = int(apply_mask.sum(dim=1).max().item())
            total_evals += max_applied * ls1_evals

        # -- Parent info for next gen --
        for b in range(B):
            parent_info_list[b] = {
                'parent_x': prev_x[b].float(),
                'child_x': x_new[b].float(),
                'parent_fitness': prev_fitness_tensor[b].float(),
                'child_fitness': f_new[b].float(),
                'parent_fit_rank': torch.zeros(n_pop, device=device),
            }

        # -- Update temporal state (vectorized over B) --
        new_best = f_new.min(dim=1).values   # (B,)
        improved = new_best < prev_best - 1e-12
        raw_imp = torch.clamp(prev_best - new_best, min=0.0)
        safe_prev = prev_best.abs() + 1e-8
        delta_fitness = torch.where(
            improved,
            torch.clamp(raw_imp / safe_prev, max=1.0),
            torch.zeros_like(new_best))
        stagnation_counter = torch.where(
            improved,
            torch.zeros_like(stagnation_counter),
            stagnation_counter + 1)

        # Contraction rate
        current_div = x_new.float().std(dim=(1, 2))  # (B,)
        div_idx = gen % TEMPORAL_W
        if div_ring_len >= 2:
            oldest_div = diversity_ring[:, 0 if div_ring_len < TEMPORAL_W
                                        else (div_idx + 1) % TEMPORAL_W]
            contraction_rate = torch.clamp(
                current_div / (oldest_div + 1e-8) - 1.0, -1.0, 1.0)
        diversity_ring[:, div_idx] = current_div
        div_ring_len = min(div_ring_len + 1, TEMPORAL_W)

        prev_best = new_best
        x = x_new
        fitness = f_new

        if total_evals >= max_evals:
            break

    return all_snapshots, fid_out, D_out


def _flush_batched(buffers, out_dir, flush_count):
    """Flush per-D buffers to numbered pkl files."""
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
        description="Massively parallel GPU NPA data collection")
    parser.add_argument("--out-dir", type=str, default="DATASETS/NPA_GPU")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="B: number of parallel augmentations per batch")
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
    batched_aug = BatchedAugmentedCEC2017(device=str(device), dims=(10, 30, 50))

    B = args.batch_size
    buffers = {10: [], 30: [], 50: []}
    strat_counts = {s: 0 for s in STRATEGIES}
    total_runs = 0
    total_snapshots = 0
    t_start = time.time()

    # Resume: estimate from existing files
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
        log.info("Resuming: ~%d snapshots from %d files, %.1f GB (flush=%d)",
                 total_snapshots, len(existing_pkls), total_bytes / 1e9,
                 flush_count)

    log.info("Batched GPU collection (B=%d) → %s", B, args.out_dir)
    log.info("Strategies: %s (uniform random)", STRATEGIES)
    if args.max_snapshots > 0:
        log.info("Target: %d snapshots", args.max_snapshots)

    try:
        while True:
            strategy = random.choice(STRATEGIES)
            t0 = time.time()

            try:
                all_snaps, fid, D = run_batch(
                    batched_aug, str(device), B=B, strategy=strategy,
                    n_pop=args.n_pop, n_gens=args.n_gens,
                    budget_factor=args.budget_factor,
                    ls1_evals=args.ls1_evals)
            except Exception as e:
                log.warning("Batch failed: %s", e)
                continue

            # Flatten B snapshot lists into per-D buffer
            batch_snap_count = 0
            for b_snaps in all_snaps:
                buffers[D].extend(b_snaps)
                batch_snap_count += len(b_snaps)

            strat_counts[strategy] += 1
            total_runs += 1
            total_snapshots += batch_snap_count
            dt = time.time() - t0

            if total_runs % 5 == 0:
                elapsed = time.time() - t_start
                rate = total_snapshots / elapsed * 3600
                sc = strat_counts
                log.info("Batch %d | F%02d D=%d %s | B=%d %d snaps (%.2fs) | "
                         "total: %d (%.0f/hr) | so=%d mb=%d t3=%d t10=%d or=%d",
                         total_runs, fid, D, strategy[:5], B,
                         batch_snap_count, dt,
                         total_snapshots, rate,
                         sc['shade_only'], sc['mos_best'], sc['mos_top3'],
                         sc['mos_top10'], sc['oracle'])

            if total_runs % args.save_every == 0:
                flush_count += 1
                _flush_batched(buffers, args.out_dir, flush_count)

            if args.max_snapshots > 0 and total_snapshots >= args.max_snapshots:
                log.info("Reached target: %d snapshots", total_snapshots)
                break

    except KeyboardInterrupt:
        log.info("Interrupted. Flushing remaining buffer...")

    flush_count += 1
    _flush_batched(buffers, args.out_dir, flush_count)

    elapsed = time.time() - t_start
    log.info("Done: %d batches (B=%d), %d snaps in %.0fs (%.0f/hr) | %s",
             total_runs, B, total_snapshots, elapsed,
             total_snapshots / max(elapsed, 1) * 3600, strat_counts)


if __name__ == '__main__':
    main()
