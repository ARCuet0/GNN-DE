"""
npa_trajectory_dataset.py — Trajectory-windowed dataset for NPA SSL pretraining.

Loads SOFT_K4_ORACLE_v2 pkl files, groups snapshots into runs (by seed,
allocation_strategy, alloc_idx), and creates sliding windows of W consecutive
generations.  All data pre-loaded into RAM; zero I/O at __getitem__ time.
"""

import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

VAL_FIDS = {3, 8, 18}


class TrajectoryDataset(Dataset):
    """Sliding-window trajectory dataset for NPA SSL pretraining.

    Each sample is a window of W consecutive (coordinates, fitness) snapshots
    from a single optimisation run, plus oracle labels from the last snapshot.
    """

    def __init__(self, data_dir: str, window: int = 8,
                 dim_filter: int = None, max_N: int = 300,
                 split: str = 'train', val_fids: set = None):
        """
        Args:
            data_dir:   root of SOFT_K4_ORACLE_v2 (contains D10/, D30/, D50/)
            window:     number of consecutive snapshots per sample
            dim_filter: if set, only load this dimensionality (10, 30, or 50)
            max_N:      pad all populations to this size
            split:      'train' or 'val'
            val_fids:   function IDs held out for validation
        """
        super().__init__()
        self.window = window
        self.max_N = max_N
        if val_fids is None:
            val_fids = VAL_FIDS

        # -- 1. Discover pkl files --
        pkl_paths = []
        for dname in sorted(os.listdir(data_dir)):
            if not dname.startswith('D'):
                continue
            dim_val = int(dname[1:])
            if dim_filter is not None and dim_val != dim_filter:
                continue
            dpath = os.path.join(data_dir, dname)
            if not os.path.isdir(dpath):
                continue
            for fname in sorted(os.listdir(dpath)):
                if fname.endswith('.pkl'):
                    pkl_paths.append(os.path.join(dpath, fname))

        log.info("TrajectoryDataset: found %d pkl files in %s (dim_filter=%s)",
                 len(pkl_paths), data_dir, dim_filter)

        # -- 2. Load all pkls, group into runs, build window index --
        self.windows = []       # list of (coords_w, fitness_w, valid_N_w, labels)
        n_runs = 0
        n_snaps_total = 0

        for pkl_path in pkl_paths:
            with open(pkl_path, 'rb') as f:
                snapshots = pickle.load(f)

            n_snaps_total += len(snapshots)

            # Group by run key
            runs = defaultdict(list)
            for snap in snapshots:
                m = snap['metadata']
                fid = m['func_id']

                # Train/val split by function ID
                is_val = fid in val_fids
                if (split == 'train' and is_val) or (split == 'val' and not is_val):
                    continue

                run_key = (m['seed'], m.get('allocation_strategy', ''),
                           m.get('alloc_idx', 0))
                runs[run_key].append(snap)

            # Sort each run by epoch, build sliding windows
            for run_key, run_snaps in runs.items():
                run_snaps.sort(key=lambda s: s['metadata']['epoch'])
                n_runs += 1

                if len(run_snaps) < window:
                    continue

                # Pre-extract numpy arrays for this run
                run_coords = []   # list of (N_t, D) arrays
                run_fitness = []  # list of (N_t,) arrays
                run_labels = []   # list of label dicts
                run_ndim = run_snaps[0]['metadata']['ndim']
                run_fid = run_snaps[0]['metadata']['func_id']

                for snap in run_snaps:
                    run_coords.append(snap['coordinates'])
                    run_fitness.append(snap['fitness'])
                    run_labels.append({
                        'oracle_alloc': snap['oracle_allocation'].astype(np.float32),
                        'subpop_eff': snap['per_subpop_efficiency'].astype(np.float32),
                        'fitness_rank': snap['fitness_rank'].astype(np.float32),
                    })

                # Sliding windows
                for start in range(len(run_snaps) - window + 1):
                    end = start + window

                    # Pad coords and fitness to max_N
                    coords_w = np.zeros((window, max_N, run_ndim), dtype=np.float32)
                    fitness_w = np.zeros((window, max_N), dtype=np.float32)
                    valid_N = np.zeros(window, dtype=np.int64)

                    for t_idx, t in enumerate(range(start, end)):
                        N_t = min(run_coords[t].shape[0], max_N)
                        coords_w[t_idx, :N_t, :] = run_coords[t][:N_t]
                        fitness_w[t_idx, :N_t] = run_fitness[t][:N_t]
                        valid_N[t_idx] = N_t

                    # Labels from LAST snapshot in window
                    last_labels = run_labels[end - 1]
                    N_last = valid_N[-1]
                    fitness_rank_pad = np.zeros(max_N, dtype=np.float32)
                    fr = last_labels['fitness_rank']
                    N_fr = min(len(fr), max_N)
                    fitness_rank_pad[:N_fr] = fr[:N_fr]

                    self.windows.append((
                        coords_w,
                        fitness_w,
                        valid_N,
                        last_labels['oracle_alloc'],
                        last_labels['subpop_eff'],
                        fitness_rank_pad,
                        run_fid,
                        run_ndim,
                    ))

        log.info("TrajectoryDataset [%s]: %d windows from %d runs "
                 "(%d total snapshots, W=%d, max_N=%d)",
                 split, len(self.windows), n_runs, n_snaps_total,
                 window, max_N)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        (coords_w, fitness_w, valid_N, oracle_alloc,
         subpop_eff, fitness_rank, fid, ndim) = self.windows[idx]

        return {
            'coords_window': torch.from_numpy(coords_w),       # (W, max_N, D)
            'fitness_window': torch.from_numpy(fitness_w),      # (W, max_N)
            'valid_N': torch.from_numpy(valid_N),               # (W,)
            'oracle_alloc': torch.from_numpy(oracle_alloc),     # (4,)
            'subpop_eff': torch.from_numpy(subpop_eff),         # (4,)
            'fitness_rank': torch.from_numpy(fitness_rank),     # (max_N,)
            'fid': fid,
            'ndim': ndim,
        }
