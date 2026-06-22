"""
preprocess_npa_data.py — Convert pkl snapshots to contiguous .pt tensors.

Two-pass, parallel approach:
  Pass 1: Count samples per D (parallel across files, 16 workers)
  Pass 2: Write into memmap per D (parallel across files)

Each D is processed independently → never more than ~5 GB in RAM.

Usage:
    python -m encoder.preprocess_npa_data \
        --data-dir DATASETS/NPA_GPU --out-dir DATASETS/NPA_TENSORS \
        --window 8 --workers 16
"""

import argparse
import glob
import logging
import os
import pickle
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import torch

log = logging.getLogger(__name__)

N_POP = 100
DIMS = (10, 30, 50)


def _count_file(fpath, W):
    """Count samples per D in one pkl file. Returns {D: count}."""
    with open(fpath, 'rb') as f:
        snaps = pickle.load(f)

    runs = defaultdict(list)
    for s in snaps:
        run_key = (s['fid'], s['ndim'], s.get('strategy', '?'),
                   s.get('run_id', 0))
        runs[run_key].append(s['gen'])

    counts = defaultdict(int)
    for (fid, ndim, _), gens in runs.items():
        gens_sorted = sorted(set(gens))
        if len(gens_sorted) < W:
            continue
        for start in range(len(gens_sorted) - W + 1):
            w = gens_sorted[start:start + W]
            if w == list(range(w[0], w[0] + W)):
                counts[ndim] += 1
    return dict(counts)


def _extract_file(fpath, W, target_D):
    """Extract all windows for target_D from one pkl file.
    Returns list of (coords_hist, fitness_hist, oracle, ls1d, rank, frac, finit, fid, gen)
    as numpy arrays.
    """
    with open(fpath, 'rb') as f:
        snaps = pickle.load(f)

    runs = defaultdict(list)
    for s in snaps:
        if s['ndim'] == target_D:
            # run_id distinguishes B parallel runs with same (fid, D, strategy)
            run_key = (s['fid'], s['ndim'], s.get('strategy', '?'),
                       s.get('run_id', 0))
            runs[run_key].append(s)

    results = []
    for (fid, ndim, strategy, _run_id), run_snaps in runs.items():
        run_snaps.sort(key=lambda s: s['gen'])
        if len(run_snaps) < W:
            continue

        f_init = float(run_snaps[0]['fitness'].min())

        for start in range(len(run_snaps) - W + 1):
            end = start + W
            gens = [run_snaps[i]['gen'] for i in range(start, end)]
            if gens != list(range(gens[0], gens[0] + W)):
                continue

            last = run_snaps[end - 1]

            if last.get('has_history') and 'coords_hist' in last:
                ch = np.float16(last['coords_hist'])
                fh = np.float32(last['fitness_hist'])
            else:
                ch = np.stack([np.float16(run_snaps[start + t]['coordinates'])
                               for t in range(W)])
                fh = np.stack([np.float32(run_snaps[start + t]['fitness'])
                               for t in range(W)])

            # remaining_fes_ratio: use from snapshot if available, else approx
            fes_ratio = last.get('remaining_fes_ratio',
                                  1.0 - last['gen'] / max(last['n_gens'], 1))

            results.append((
                ch,  # (W, N, D) float16
                fh,  # (W, N) float32
                np.float32(last['oracle_switch_adjusted']),
                np.float32(last['ls1_delta']),
                np.float32(last['fitness_rank']),
                np.float32(last['optimal_ls1_frac']),
                np.float32(f_init),
                np.int16(fid),
                np.int16(last['gen']),
                np.float32(fes_ratio),
            ))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="DATASETS/NPA_GPU")
    parser.add_argument("--out-dir", default="DATASETS/NPA_TENSORS")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    W = args.window
    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data_dir, 'gpu_d*_*.pkl')))
    log.info("Found %d pkl files, using %d workers", len(files), args.workers)

    # === Pass 1: Parallel count ===
    t0 = time.time()
    log.info("Pass 1: counting samples (parallel)...")
    counts = defaultdict(int)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_count_file, f, W): f for f in files}
        done = 0
        for fut in as_completed(futures):
            for D, c in fut.result().items():
                counts[D] += c
            done += 1
            if done % 1000 == 0:
                log.info("  Counted %d/%d files (%.0fs)",
                         done, len(files), time.time() - t0)

    total = sum(counts.values())
    log.info("Pass 1 done (%.0fs): %s, total=%d",
             time.time() - t0, dict(counts), total)

    # === Pass 2: One D at a time, parallel extract → memmap ===
    for D in sorted(counts.keys()):
        S = counts[D]
        log.info("Pass 2: D=%d, %d samples", D, S)

        tmp_dir = os.path.join(args.out_dir, f'_tmp_d{D}')
        os.makedirs(tmp_dir, exist_ok=True)

        # Allocate memmaps
        mm = {
            'coords': np.memmap(os.path.join(tmp_dir, 'coords.dat'),
                                dtype=np.float16, mode='w+',
                                shape=(S, W, N_POP, D)),
            'fitness': np.memmap(os.path.join(tmp_dir, 'fitness.dat'),
                                 dtype=np.float32, mode='w+',
                                 shape=(S, W, N_POP)),
            'oracle': np.memmap(os.path.join(tmp_dir, 'oracle.dat'),
                                dtype=np.float32, mode='w+',
                                shape=(S, N_POP)),
            'ls1d': np.memmap(os.path.join(tmp_dir, 'ls1d.dat'),
                              dtype=np.float32, mode='w+',
                              shape=(S, N_POP)),
            'rank': np.memmap(os.path.join(tmp_dir, 'rank.dat'),
                              dtype=np.float32, mode='w+',
                              shape=(S, N_POP)),
            'frac': np.memmap(os.path.join(tmp_dir, 'frac.dat'),
                              dtype=np.float32, mode='w+', shape=(S,)),
            'finit': np.memmap(os.path.join(tmp_dir, 'finit.dat'),
                               dtype=np.float32, mode='w+', shape=(S,)),
            'fid': np.memmap(os.path.join(tmp_dir, 'fid.dat'),
                             dtype=np.int16, mode='w+', shape=(S,)),
            'gen': np.memmap(os.path.join(tmp_dir, 'gen.dat'),
                             dtype=np.int16, mode='w+', shape=(S,)),
            'fes_ratio': np.memmap(os.path.join(tmp_dir, 'fes_ratio.dat'),
                                    dtype=np.float32, mode='w+', shape=(S,)),
        }

        t1 = time.time()
        idx = 0
        extract_fn = partial(_extract_file, W=W, target_D=D)

        # Process files in chunks to control memory
        CHUNK = 200
        for chunk_start in range(0, len(files), CHUNK):
            chunk_files = files[chunk_start:chunk_start + CHUNK]

            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(extract_fn, f): f for f in chunk_files}
                for fut in as_completed(futures):
                    results = fut.result()
                    for (ch, fh, oracle, ls1d, rank, frac, finit,
                         fid_v, gen_v, fes_ratio) in results:
                        mm['coords'][idx] = ch
                        mm['fitness'][idx] = fh
                        mm['oracle'][idx] = oracle
                        mm['ls1d'][idx] = ls1d
                        mm['rank'][idx] = rank
                        mm['frac'][idx] = frac
                        mm['finit'][idx] = finit
                        mm['fid'][idx] = fid_v
                        mm['gen'][idx] = gen_v
                        mm['fes_ratio'][idx] = fes_ratio
                        idx += 1

            log.info("  D=%d: %d/%d files, %d/%d samples (%.0fs)",
                     D, min(chunk_start + CHUNK, len(files)), len(files),
                     idx, S, time.time() - t1)

        # Flush
        for m in mm.values():
            m.flush()

        # Move memmap dir to final location (memmap IS the format)
        final_dir = os.path.join(args.out_dir, f'd{D}')
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)

        # Save metadata
        import json
        meta = {'S': idx, 'W': W, 'N': N_POP, 'D': D}
        with open(os.path.join(final_dir, 'meta.json'), 'w') as f:
            json.dump(meta, f)

        total_mb = sum(os.path.getsize(os.path.join(final_dir, fn))
                       for fn in os.listdir(final_dir)
                       if fn.endswith('.dat')) / 1e6
        log.info("  D=%d: %d samples → %s (%.0f MB on disk, 0 MB RAM)",
                 D, idx, final_dir, total_mb)
        del mm

    log.info("Done: %d total samples in %.0fs", total, time.time() - t0)


if __name__ == '__main__':
    main()
