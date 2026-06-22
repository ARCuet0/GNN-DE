"""
npa_gpu_dataset.py — Memory-mapped dataset for NPA SSL pretraining.

Reads memmap files produced by preprocess_npa_data.py.
__getitem__ is a direct memmap slice → zero RAM, zero deserialization.

Usage:
    dataset = NPAMemmapDataset('DATASETS/NPA_TENSORS', dim_filter=10)
    loader = DataLoader(dataset, batch_size=128, shuffle=True,
                        num_workers=0, pin_memory=True)
"""

import json
import logging
import os

import numpy as np
import torch

log = logging.getLogger(__name__)


class NPAMemmapDataset(torch.utils.data.Dataset):
    """Memory-mapped dataset. Zero RAM footprint beyond OS page cache."""

    def __init__(self, data_dir, dim_filter=None):
        """
        Args:
            data_dir:   path to DATASETS/NPA_TENSORS/
            dim_filter: int or None — only load specific D (10, 30, 50)
        """
        dims = [dim_filter] if dim_filter else [10, 30, 50]
        self._segments = []  # list of (mm_dict, S, D, offset)
        offset = 0

        for D in dims:
            d_dir = os.path.join(data_dir, f'd{D}')
            meta_path = os.path.join(d_dir, 'meta.json')
            if not os.path.exists(meta_path):
                log.warning("No data for D=%d at %s", D, d_dir)
                continue

            with open(meta_path) as f:
                meta = json.load(f)
            S, W, N = meta['S'], meta['W'], meta['N']

            mm = {
                'coords': np.memmap(os.path.join(d_dir, 'coords.dat'),
                                    dtype=np.float16, mode='r',
                                    shape=(S, W, N, D)),
                'fitness': np.memmap(os.path.join(d_dir, 'fitness.dat'),
                                     dtype=np.float32, mode='r',
                                     shape=(S, W, N)),
                'oracle': np.memmap(os.path.join(d_dir, 'oracle.dat'),
                                    dtype=np.float32, mode='r',
                                    shape=(S, N)),
                'ls1d': np.memmap(os.path.join(d_dir, 'ls1d.dat'),
                                  dtype=np.float32, mode='r',
                                  shape=(S, N)),
                'rank': np.memmap(os.path.join(d_dir, 'rank.dat'),
                                  dtype=np.float32, mode='r',
                                  shape=(S, N)),
                'frac': np.memmap(os.path.join(d_dir, 'frac.dat'),
                                  dtype=np.float32, mode='r', shape=(S,)),
                'finit': np.memmap(os.path.join(d_dir, 'finit.dat'),
                                   dtype=np.float32, mode='r', shape=(S,)),
                'fes_ratio': np.memmap(os.path.join(d_dir, 'fes_ratio.dat'),
                                       dtype=np.float32, mode='r', shape=(S,)),
            }

            self._segments.append((mm, S, D, offset))
            offset += S
            log.info("D=%d: %d samples loaded (memmap, 0 MB RAM)", D, S)

        self._total = offset
        self._W = W
        self._N = N
        log.info("Dataset: %d total samples, W=%d, N=%d", self._total, W, N)

    def __len__(self):
        return self._total

    def _locate(self, idx):
        """Find which segment and local index for global idx."""
        for mm, S, D, offset in self._segments:
            if idx < offset + S:
                return mm, D, idx - offset
        raise IndexError(f"idx {idx} out of range [0, {self._total})")

    def __getitem__(self, idx):
        mm, D, local = self._locate(idx)
        return {
            'coords_hist': torch.from_numpy(
                np.float32(mm['coords'][local])),      # (W, N, D)
            'fitness_hist': torch.from_numpy(
                mm['fitness'][local].copy()),            # (W, N) — may have inf, handled in log-space
            'oracle_switch': torch.from_numpy(
                mm['oracle'][local].copy()),             # (N,)
            'ls1_delta': torch.from_numpy(
                mm['ls1d'][local].copy()),               # (N,)
            'fitness_rank': torch.from_numpy(
                mm['rank'][local].copy()),                # (N,)
            'optimal_ls1_frac': torch.tensor(
                float(mm['frac'][local])),               # scalar
            'f_init': torch.tensor(
                float(mm['finit'][local])),              # scalar
            'fes_ratio': torch.tensor(
                float(mm['fes_ratio'][local])),          # scalar ∈ [0, 1]
            'ndim': D,
        }

    @staticmethod
    def collate(batch):
        """Collate with padding to max_D in the batch."""
        B = len(batch)
        W = batch[0]['coords_hist'].shape[0]
        N = batch[0]['coords_hist'].shape[1]
        max_D = max(b['coords_hist'].shape[2] for b in batch)

        # If all same D, just stack (fast path)
        if all(b['coords_hist'].shape[2] == max_D for b in batch):
            coords_hist = torch.stack([b['coords_hist'] for b in batch])
        else:
            coords_hist = torch.zeros(B, W, N, max_D)
            for i, b in enumerate(batch):
                D = b['coords_hist'].shape[2]
                coords_hist[i, :, :, :D] = b['coords_hist']

        return {
            'coords_hist': coords_hist,                                    # (B, W, N, max_D)
            'fitness_hist': torch.stack([b['fitness_hist'] for b in batch]),  # (B, W, N)
            'oracle_switch': torch.stack([b['oracle_switch'] for b in batch]),
            'ls1_delta': torch.stack([b['ls1_delta'] for b in batch]),
            'fitness_rank': torch.stack([b['fitness_rank'] for b in batch]),
            'optimal_ls1_frac': torch.stack([b['optimal_ls1_frac'] for b in batch]),
            'f_init': torch.stack([b['f_init'] for b in batch]),
            'fes_ratio': torch.stack([b['fes_ratio'] for b in batch]),
            'ndims': torch.tensor([b['ndim'] for b in batch]),
            'max_N': N,
            'max_D': max_D,
            'valid_N': torch.full((B,), N, dtype=torch.long),
        }
