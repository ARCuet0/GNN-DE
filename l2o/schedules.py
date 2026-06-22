"""Training-loop schedules (fes_frac-dependent hyperparameters)."""
import torch


def compute_lpsr_n_target(N_init: int, N_min: int, fes_frac: float) -> int:
    """Linear Population Size Reduction schedule.

    N(t) = round(N_init - (N_init - N_min) * t), clamped to [N_min, N_init].

    Args:
        N_init:    initial population (start of trajectory).
        N_min:     final population (end of trajectory). Must satisfy N_min <= N_init.
        fes_frac:  current FES fraction in [0, 1] (auto-clamped).

    Used identically by train_distributed.py, l2o/canonical_eval.py, and
    eval_e7d_parallel.py. Centralised here to prevent the 3 implementations
    from drifting.
    """
    if N_min > N_init:
        raise ValueError(
            f'compute_lpsr_n_target: N_min ({N_min}) must be <= N_init '
            f'({N_init}); LPSR is a shrinkage schedule.')
    t = max(0.0, min(1.0, fes_frac))
    target = int(round(N_init - (N_init - N_min) * t))
    return max(N_min, min(N_init, target))


def lpsr_keep_indices(fitness: torch.Tensor, N_target: int) -> torch.Tensor:
    """Per-batch indices of the top-N_target individuals by ascending fitness.

    Args:
        fitness: (B, N) fitness values.
        N_target: count to keep (must be <= N — caller's contract).

    Returns:
        (B, N_target) int64 indices, per-row, smallest fitness first.
    """
    if N_target > fitness.shape[-1]:
        raise ValueError(
            f'lpsr_keep_indices: N_target ({N_target}) > fitness.shape[-1] '
            f'({fitness.shape[-1]}); cannot keep more individuals than exist.')
    return fitness.argsort(dim=-1)[:, :N_target]


def gather_pop(tensor: torch.Tensor, keep_idx: torch.Tensor,
               dim: int) -> torch.Tensor:
    """Per-batch gather along the population axis.

    Args:
        tensor:    arbitrary-rank tensor with N at axis `dim`. Axis 0 is
                   batch (B). Other axes (e.g. gru_W, D) are preserved.
        keep_idx:  (B, K) indices into the N axis, per row.
        dim:       axis where N lives. Must be > 0 (axis 0 is B).

    Returns:
        Contiguous tensor with axis `dim` reshaped from N to K.

    Replaces the hand-rolled `tensor.gather(dim, keep_idx.unsqueeze(...).expand(...))`
    pattern used in 3 places. The expand pattern is correct only when applied
    per-batch (one keep_idx row per B row); fancy indexing like
    `tensor[:, keep_idx]` with 1D keep_idx silently broadcasts across B and
    is wrong for B>1 — this helper makes the per-batch contract explicit.
    """
    if not 0 < dim < tensor.dim():
        raise ValueError(
            f'gather_pop: dim ({dim}) must be in [1, {tensor.dim()}); '
            f'axis 0 is batch.')
    if tensor.device != keep_idx.device:
        raise ValueError(
            f'gather_pop: device mismatch — tensor on {tensor.device}, '
            f'keep_idx on {keep_idx.device}.')
    B, K = keep_idx.shape
    if tensor.shape[0] != B:
        raise ValueError(
            f'gather_pop: tensor batch dim {tensor.shape[0]} '
            f'!= keep_idx batch dim {B}.')
    # Build idx with shape that matches tensor on every axis except `dim`.
    # Axis 0 = B (kept), axis `dim` = K (replaces N), others = 1 (broadcast).
    idx_shape = [1] * tensor.dim()
    idx_shape[0] = B
    idx_shape[dim] = K
    idx = keep_idx.view(idx_shape)
    target_shape = list(tensor.shape)
    target_shape[dim] = K
    idx = idx.expand(target_shape)
    return tensor.gather(dim, idx).contiguous()


class PopulationGenState:
    """Per-generation rolling state shared by train and eval gen loops.

    Tracks `stagnation_counters`, `delta_fitnesses`, `contraction_rates`,
    `prev_best_fit`, `prev_coords_spread` — the inputs the GAT graph builder
    consumes as global / node features. Updated in-place via `.update(coords,
    fitness)` once per generation, BEFORE building the graph for that gen.

    Centralised here so train_distributed.py, l2o/canonical_eval.py, and
    eval_e7d_parallel.py share the same arithmetic and grad semantics
    (always detached; no autograd chain across gens — important for BPTT
    in train, harmless under no_grad in eval).
    """

    def __init__(self, B: int, device):
        self.stagnation_counters = torch.zeros(B, device=device)
        self.delta_fitnesses = torch.zeros(B, device=device)
        self.contraction_rates = torch.zeros(B, device=device)
        self.prev_best_fit = None
        self.prev_coords_spread = None

    @property
    def device(self):
        return self.stagnation_counters.device

    def reset_baseline(self) -> None:
        """Clear prev_best_fit / prev_coords_spread without touching the
        rolling counters. Call this AFTER an admin operation that changes
        the population structure (e.g., LPSR-N shrink) but should NOT count
        as evolutionary stagnation: the next .update() will skip the delta
        computation, so the shrink is invisible to the stagnation signal.
        """
        self.prev_best_fit = None
        self.prev_coords_spread = None

    def update(self, coords: torch.Tensor, fitness: torch.Tensor) -> None:
        if fitness.device != self.device or coords.device != self.device:
            raise RuntimeError(
                f'PopulationGenState.update: device mismatch. '
                f'state on {self.device}, coords on {coords.device}, '
                f'fitness on {fitness.device}.')
        with torch.no_grad():
            curr_best = fitness.min(dim=1).values.detach()
            curr_spread = coords.float().std(dim=1).mean(dim=1).detach()
            if self.prev_best_fit is not None:
                _delta = self.prev_best_fit - curr_best
                self.delta_fitnesses = _delta.clamp(-1, 1)
                self.stagnation_counters = torch.where(
                    _delta.abs() < 1e-10,
                    self.stagnation_counters + 1,
                    torch.zeros_like(self.stagnation_counters))
                self.contraction_rates = (
                    (self.prev_coords_spread - curr_spread)
                    / self.prev_coords_spread.clamp(min=1e-8)
                ).clamp(-1, 1)
            self.prev_best_fit = curr_best
            self.prev_coords_spread = curr_spread


def compute_surrogate_m(m_init: int, m_final: int, fes_frac: float,
                         default_N: int) -> int:
    """LPSR-inspired linear decay of surrogate_M over the trajectory.

    Args:
        m_init:    initial value (start of trajectory). If <=0, use default_N.
        m_final:   final value (end of trajectory). If <=0, no schedule (constant m_init).
        fes_frac:  current progress in [0, 1].
        default_N: population size; fallback when m_init == 0.

    Returns:
        Current surrogate_M, rounded, clamped between min(m_init, m_final) and max(m_init, m_final).
    """
    init = m_init if m_init > 0 else default_N
    if m_final <= 0 or m_final == init:
        return init
    t = max(0.0, min(1.0, fes_frac))
    m = init * (1.0 - t) + m_final * t
    lo, hi = min(m_final, init), max(m_final, init)
    rounded = int(round(m))
    return max(lo, min(hi, rounded))
