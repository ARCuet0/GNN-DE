"""
diagnostics.py — Per-individual, per-generation diagnostic buffer for Block 0 analysis.

Captures routing decisions, fitness, and spatial features at every generation
during a trajectory. Activated via diagnostics=True flag in run_trajectory().

Data captured per generation:
- routing_probs: (N, K) softmax probabilities
- chosen_expert: (N,) argmax of routing probs
- fitness: (N,) raw fitness values
- fitness_rank: (N,) ordinal rank (0=best)
- dist_to_best: (N,) Euclidean distance to fittest individual
- dist_to_nearest: (N,) distance to closest neighbor
- in_bptt: bool — whether this gen is inside BPTT segment
- bptt_segment_pos: int — position within BPTT window (-1 if outside)

Supports Block 0 diagnostics:
  0.1: entropy-autocorrelation map (from routing_probs)
  0.2: Markov transition matrix (from chosen_expert)
  0.3: nop analysis (from chosen_expert + fitness_rank + distances)
  0.4: BPTT boundary effect (from in_bptt + bptt_segment_pos + routing_probs)
  0.5: hidden state memory (from fitness + routing_probs across segments)
"""
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


class DiagnosticBuffer:
    """Accumulates per-individual, per-generation diagnostics during a trajectory.

    All arrays are pre-allocated as numpy. Torch tensors are detached and
    converted on record() — zero overhead when disabled (buffer is simply None).
    """

    def __init__(self, n_gens: int, N: int, K: int):
        self.n_gens = n_gens
        self.N = N
        self.K = K
        self.gen_idx = 0

        # Pre-allocate arrays
        self.routing_probs = np.zeros((n_gens, N, K), dtype=np.float32)
        self.chosen_expert = np.zeros((n_gens, N), dtype=np.int8)
        self.fitness = np.zeros((n_gens, N), dtype=np.float64)
        self.fitness_rank = np.zeros((n_gens, N), dtype=np.int16)
        self.dist_to_best = np.zeros((n_gens, N), dtype=np.float32)
        self.dist_to_nearest = np.zeros((n_gens, N), dtype=np.float32)
        self.in_bptt = np.zeros(n_gens, dtype=bool)
        self.bptt_segment_pos = np.full(n_gens, -1, dtype=np.int16)

    def record(
        self,
        routing_probs: torch.Tensor,  # (B, N, K) or (N, K)
        coords: torch.Tensor,         # (B, N, D) or (N, D)
        fitness: torch.Tensor,        # (B, N) or (N,)
        in_bptt: bool,
        bptt_seg_pos: int,
    ):
        """Record one generation's data. Squeezes batch dim (B=1)."""
        g = self.gen_idx

        # Squeeze batch dim if present
        rp = routing_probs.detach().cpu()
        if rp.dim() == 3:
            rp = rp[0]  # (N, K)
        c = coords.detach().cpu()
        if c.dim() == 3:
            c = c[0]  # (N, D)
        f = fitness.detach().cpu()
        if f.dim() == 2:
            f = f[0]  # (N,)

        N = rp.shape[0]

        # Routing probs and argmax
        self.routing_probs[g] = rp.numpy()
        self.chosen_expert[g] = rp.argmax(dim=-1).numpy()

        # Fitness and rank
        self.fitness[g] = f.numpy()
        ranks = f.argsort().argsort()  # ordinal rank: 0=best
        self.fitness_rank[g] = ranks.numpy()

        # Distance to best individual (fittest)
        best_idx = f.argmin()
        best_coords = c[best_idx]  # (D,)
        diffs = c - best_coords.unsqueeze(0)  # (N, D)
        self.dist_to_best[g] = diffs.pow(2).sum(dim=-1).sqrt().numpy()

        # Distance to nearest neighbor
        dist_matrix = torch.cdist(c.unsqueeze(0), c.unsqueeze(0)).squeeze(0)  # (N, N)
        dist_matrix.fill_diagonal_(float('inf'))
        self.dist_to_nearest[g] = dist_matrix.min(dim=-1).values.numpy()

        # BPTT flags
        self.in_bptt[g] = in_bptt
        self.bptt_segment_pos[g] = bptt_seg_pos

        self.gen_idx += 1

    def save(self, path: Path, metadata: Optional[Dict] = None):
        """Save buffer to compressed npz. Truncates to gen_idx if incomplete."""
        g = self.gen_idx
        np.savez_compressed(
            path,
            routing_probs=self.routing_probs[:g],
            chosen_expert=self.chosen_expert[:g],
            fitness=self.fitness[:g],
            fitness_rank=self.fitness_rank[:g],
            dist_to_best=self.dist_to_best[:g],
            dist_to_nearest=self.dist_to_nearest[:g],
            in_bptt=self.in_bptt[:g],
            bptt_segment_pos=self.bptt_segment_pos[:g],
            metadata=metadata or {},
        )
