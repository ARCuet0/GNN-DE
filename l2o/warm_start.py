"""Warm start pool loader for L2O training.

Loads pre-computed L-SHADE trajectories and samples intermediate population
states for warm-starting the training loop. This addresses the BPTT coverage
bias where the model only sees early-stage dynamics from random init.
"""
import logging
import random
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


class WarmStartPool:
    """Lazy-loading warm start pool from .npz trajectory files.

    Each .npz contains:
        coords:         (G, N, D) population coordinates per generation
        fitness:        (G, N)    fitness values per generation
        cumulative_fes: (G,)      FES consumed up to each generation
        fid, D, N, max_evals, f_optimal: metadata
    """

    def __init__(self, pool_dir, gru_window=16):
        self.pool_dir = Path(pool_dir)
        self.gru_window = gru_window
        self._cache = {}       # fid -> list of loaded trajectory dicts
        self._index = {}       # fid -> list of .npz paths
        self._build_index()

    def _build_index(self):
        """Scan pool directory and index files by fid."""
        if not self.pool_dir.exists():
            log.warning("Warm start pool dir does not exist: %s", self.pool_dir)
            return

        for f in sorted(self.pool_dir.glob('*.npz')):
            try:
                # Parse fid from filename: {fid:02d}_seed{seed:02d}.npz
                fid = int(f.stem.split('_')[0])
            except (ValueError, IndexError):
                continue
            self._index.setdefault(fid, []).append(f)

        n_fids = len(self._index)
        n_files = sum(len(v) for v in self._index.values())
        log.info("WarmStartPool: %d files across %d functions in %s",
                 n_files, n_fids, self.pool_dir)

    def _load_trajectory(self, path):
        """Load and validate a single trajectory file."""
        data = np.load(path)
        coords = data['coords']      # (G, N, D)
        fitness = data['fitness']     # (G, N)
        fes = data['cumulative_fes']  # (G,)
        max_evals = int(data['max_evals'])

        # Skip all-inf files
        if np.all(np.isinf(fitness)):
            return None

        return {
            'coords': coords,
            'fitness': fitness,
            'cumulative_fes': fes,
            'max_evals': max_evals,
            'fid': int(data['fid']),
            'f_optimal': float(data['f_optimal']),
        }

    def _get_trajectories(self, fid):
        """Get cached trajectories for a function id."""
        if fid not in self._cache:
            paths = self._index.get(fid, [])
            trajs = []
            for p in paths:
                t = self._load_trajectory(p)
                if t is not None:
                    trajs.append(t)
            self._cache[fid] = trajs
        return self._cache[fid]

    @property
    def available_fids(self):
        """Function IDs that have valid trajectories."""
        return sorted(self._index.keys())

    def has_fid(self, fid):
        return fid in self._index and len(self._index[fid]) > 0

    def sample(self, fid, B, device, fes_frac_range=(0.05, 0.95), rng=None):
        """Sample a warm start state from the pool.

        Args:
            fid: CEC2017 function ID
            B: batch size (number of independent populations)
            device: torch device
            fes_frac_range: (lo, hi) range for fes_frac sampling
            rng: random.Random instance (for reproducibility)

        Returns:
            dict with:
                coords: (B, N, D) torch.float64
                fitness: (B, N) torch.float64
                coords_ring: (B, W, N, D) torch.float32
                fitness_ring: (B, W, N) torch.float32
                cumulative_fes: int
                step_fes: int (remaining FES budget)
                fes_frac_start: float
                gen_start: int
            or None if no valid trajectory available.
        """
        if rng is None:
            rng = random

        trajs = self._get_trajectories(fid)
        if not trajs:
            return None

        W = self.gru_window

        # Sample a random trajectory
        traj = rng.choice(trajs)
        G = len(traj['cumulative_fes'])
        max_evals = traj['max_evals']

        # Determine valid gen_start range based on fes_frac
        fes_fracs = traj['cumulative_fes'] / max_evals
        lo, hi = fes_frac_range

        valid_gens = [g for g in range(W, G - 1)
                      if lo <= fes_fracs[g] <= hi]

        if not valid_gens:
            # Fallback: any gen with enough history
            valid_gens = [g for g in range(W, G - 1)]

        if not valid_gens:
            return None

        gen_start = rng.choice(valid_gens)
        fes_frac_start = float(fes_fracs[gen_start])
        cumulative_fes = int(traj['cumulative_fes'][gen_start])

        # Current state
        coords_np = traj['coords'][gen_start]    # (N, D)
        fitness_np = traj['fitness'][gen_start]   # (N,)
        N, D = coords_np.shape

        # Build ring buffer from the W generations before gen_start
        ring_start = gen_start - W
        coords_ring_np = traj['coords'][ring_start:gen_start]     # (W, N, D)
        fitness_ring_np = traj['fitness'][ring_start:gen_start]    # (W, N)

        # Expand to batch dimension by repeating (all B populations start same)
        coords = torch.from_numpy(coords_np).unsqueeze(0).expand(B, -1, -1)
        coords = coords.clone().to(device=device, dtype=torch.float64)

        fitness = torch.from_numpy(fitness_np).unsqueeze(0).expand(B, -1)
        fitness = fitness.clone().to(device=device, dtype=torch.float64)

        coords_ring = torch.from_numpy(coords_ring_np).unsqueeze(0).expand(B, -1, -1, -1)
        coords_ring = coords_ring.clone().to(device=device, dtype=torch.float32)

        fitness_ring = torch.from_numpy(fitness_ring_np).unsqueeze(0).expand(B, -1, -1)
        fitness_ring = fitness_ring.clone().to(device=device, dtype=torch.float32)

        # Remaining FES budget
        step_fes = max_evals - cumulative_fes

        return {
            'coords': coords,
            'fitness': fitness,
            'coords_ring': coords_ring,
            'fitness_ring': fitness_ring,
            'cumulative_fes': cumulative_fes,
            'step_fes': step_fes,
            'fes_frac_start': fes_frac_start,
            'gen_start': gen_start,
            'N': N,
            'D': D,
        }
