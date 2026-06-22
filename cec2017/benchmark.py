"""
CEC2017 benchmark class and wrapper API.

Provides CEC2017Torch (the main evaluator), get_function_torch,
get_all_func_ids, and BLACKLIST.
"""

import numpy as np
import torch

from .base_functions import (
    bent_cigar, zakharov, rosenbrock, rastrigin, schaffer_f7,
    expanded_schaffer_f6, lunacek_bi_rastrigin, non_continuous_rastrigin,
    levy, modified_schwefel, elliptic, ackley, griewank, discus,
    happy_cat, hgbat,
    calculate_weight,
)
from .configs import (
    FUNCTIONS, CATEGORIES, _MIN_DIM_10,
    _HYBRID_CONFIGS, _compute_partitions,
)
from .data_loader import (
    _DATA_DIR, _opfunu_file_id,
    _load_shift, _load_shift_matrix, _load_matrix, _load_shuffle,
)

# ---------------------------------------------------------------------------
# CEC2017Torch -- main class
# ---------------------------------------------------------------------------

_cache = {}  # max 174 entries (29 funcs x 3 dims x 2 devices), acceptable


class CEC2017Torch:
    """Batched PyTorch CEC2017 benchmark function.

    Args:
        func_id: 1-29
        ndim: dimensionality (10, 30, 50, 100)
        device: torch device

    Usage:
        fn = CEC2017Torch(1, 30, 'cuda')
        x = torch.randn(100, 30, device='cuda', dtype=torch.float64)
        fitness = fn(x)  # (100,)
    """

    def __init__(self, func_id, ndim=30, device='cpu'):
        if func_id not in FUNCTIONS:
            raise ValueError(f"func_id must be 1-29, got {func_id}")
        if func_id in _MIN_DIM_10 and ndim < 10:
            raise ValueError(f"F{func_id} requires ndim >= 10, got {ndim}")

        self.func_id = func_id
        self.ndim = ndim
        self.device = device
        self.name, self.category = FUNCTIONS[func_id]
        self.f_bias = func_id * 100.0
        self.f_optimal = self.f_bias

        # Cached bounds
        self.lb = torch.full((ndim,), -100.0, device=device, dtype=torch.float64)
        self.ub = torch.full((ndim,), 100.0, device=device, dtype=torch.float64)

        if func_id <= 9:
            self._init_simple()
        elif func_id <= 19:
            self._init_hybrid()
        elif func_id <= 27:
            self._init_composition()
        else:
            self._init_composition_hybrid()

    def _to_tensor(self, arr, dtype=torch.float64):
        return torch.tensor(arr, device=self.device, dtype=dtype)

    def _init_simple(self):
        """F1-F9: shifted + rotated + single base function."""
        fid = self.func_id
        ndim = self.ndim

        shift = _load_shift(fid, ndim)
        matrix = _load_matrix(fid, ndim)

        self.shift = self._to_tensor(shift)       # (D,)
        self.matrix = self._to_tensor(matrix)      # (D, D)

        configs = {
            1: (1.0,      bent_cigar,            {}),
            2: (1.0,      zakharov,              {}),
            3: (2.048/100, rosenbrock,            {'shift': 1.0}),
            4: (1.0,      rastrigin,             {}),
            5: (0.5/100,  schaffer_f7,           {}),
            6: (6.0,      lunacek_bi_rastrigin,  {'shift': 2.5}),
            7: (5.12/100, non_continuous_rastrigin, {}),
            8: (5.12/100, levy,                  {'shift': 1.0}),
            9: (10.0,     modified_schwefel,     {}),
        }
        self.scale, self.base_fn, self.base_kwargs = configs[fid]

    def _init_hybrid(self):
        """F10-F19: shifted + rotated + partitioned into sub-functions."""
        fid = self.func_id
        ndim = self.ndim

        shift = _load_shift(fid, ndim)
        matrix = _load_matrix(fid, ndim)
        shuffle = _load_shuffle(fid, ndim)

        self.shift = self._to_tensor(shift)
        self.matrix = self._to_tensor(matrix)
        self.shuffle = self._to_tensor(shuffle, dtype=torch.long)

        props, funcs = _HYBRID_CONFIGS[fid]
        self.hybrid_funcs = funcs
        self.partition_indices = _compute_partitions(props, ndim, self.shuffle)

    def _init_composition(self):
        """F20-F27: weighted composition of shifted+rotated base functions."""
        fid = self.func_id
        ndim = self.ndim

        shift_mat = _load_shift_matrix(fid, ndim)
        matrix = _load_matrix(fid, ndim)

        self.shift_mat = self._to_tensor(shift_mat)    # (n_funcs, D)
        self.matrix_full = self._to_tensor(matrix)      # (n_funcs*D, D)

        configs = {
            20: ([10, 20, 30], [1., 1e-6, 1.], [0, 100, 200], [
                (rosenbrock, {}, 2.048/100, True),
                (elliptic, {}, 1.0, False),
                (rastrigin, {}, 1.0, False),
            ]),
            21: ([10, 20, 30], [1., 10., 1.], [0, 100, 200], [
                (rastrigin, {}, 1.0, False),
                (griewank, {}, 1.0, False),
                (modified_schwefel, {}, 10.0, False),
            ]),
            22: ([10, 20, 30, 40], [1., 10., 1., 1.], [0, 100, 200, 300], [
                (rosenbrock, {}, 2.048/100, True),
                (ackley, {}, 1.0, False),
                (modified_schwefel, {}, 1.0, False),
                (rastrigin, {}, 1.0, False),
            ]),
            23: ([10, 20, 30, 40], [10., 1e-6, 10., 1.], [0, 100, 200, 300], [
                (ackley, {}, 1.0, False),
                (elliptic, {}, 1.0, False),
                (griewank, {}, 1.0, False),
                (rastrigin, {}, 1.0, False),
            ]),
            24: ([10, 20, 30, 40, 50], [10., 1., 10., 1e-6, 1.], [0, 100, 200, 300, 400], [
                (rastrigin, {}, 1.0, False),
                (happy_cat, {}, 1.0, False),
                (ackley, {}, 1.0, False),
                (discus, {}, 1.0, False),
                (rosenbrock, {}, 2.048/100, True),
            ]),
            25: ([10, 20, 20, 30, 40], [1e-26, 10., 1e-6, 10., 5e-4], [0, 100, 200, 300, 400], [
                (expanded_schaffer_f6, {}, 1.0, True),
                (modified_schwefel, {}, 10.0, False),
                (griewank, {}, 6.0, False),
                (rosenbrock, {}, 2.048/100, True),
                (rastrigin, {}, 1.0, False),
            ]),
            26: ([10, 20, 30, 40, 50, 60], [10., 10., 2.5, 1e-26, 1e-6, 5e-4], [0, 100, 200, 300, 400, 500], [
                (hgbat, {'shift': -1.0}, 5.0/100, False),
                (rastrigin, {}, 5.12/100, False),
                (modified_schwefel, {}, 10.0, False),
                (bent_cigar, {}, 1.0, False),
                (elliptic, {}, 1.0, False),
                (expanded_schaffer_f6, {}, 1.0, True),
            ]),
            27: ([10, 20, 30, 40, 50, 60], [10., 10., 1e-6, 1., 1., 5e-4], [0, 100, 200, 300, 400, 500], [
                (ackley, {}, 1.0, False),
                (griewank, {}, 6.0, False),
                (discus, {}, 1.0, False),
                (rosenbrock, {}, 2.048/100, True),
                (happy_cat, {'shift': 0.0}, 5.0/100, False),
                (expanded_schaffer_f6, {}, 1.0, True),
            ]),
        }

        sigmas, lambdas, biases, comp_funcs = configs[fid]
        self.n_funcs = len(sigmas)
        self.sigmas = sigmas
        self.lambdas = self._to_tensor(lambdas)
        self.biases = self._to_tensor(biases)
        self.comp_funcs = comp_funcs

        self._f21_schwefel_special = (fid == 21)
        self._use_shift0 = (fid >= 24)

    def _init_composition_hybrid(self):
        """F28-F29: composition of hybrid functions."""
        fid = self.func_id
        ndim = self.ndim

        shift_mat = _load_shift_matrix(fid, ndim)
        matrix = _load_matrix(fid, ndim)

        sid = _opfunu_file_id(fid)
        shuffle_path = _DATA_DIR / f"shuffle_data_{sid}_D{ndim}.txt"
        shuffle_raw = np.genfromtxt(str(shuffle_path), dtype=int)
        shuffle_all = (shuffle_raw - 1).reshape(10, -1)

        self.shift_mat = self._to_tensor(shift_mat)
        self.matrix_full = self._to_tensor(matrix)

        configs = {
            28: ([10, 30, 50], [1., 1., 1.], [0, 100, 200], [
                (14, shuffle_all[0]),
                (15, shuffle_all[1]),
                (16, shuffle_all[2]),
            ]),
            29: ([10, 30, 50], [1., 1., 1.], [0, 100, 200], [
                (14, shuffle_all[0]),
                (17, shuffle_all[1]),
                (18, shuffle_all[2]),
            ]),
        }

        sigmas, lambdas, biases, hybrid_specs = configs[fid]
        self.n_funcs = len(sigmas)
        self.sigmas = sigmas
        self.lambdas = self._to_tensor(lambdas)
        self.biases = self._to_tensor(biases)

        self.sub_hybrids = []
        for i, (hfid, shuf) in enumerate(hybrid_specs):
            sub_matrix = self.matrix_full[i * ndim:(i + 1) * ndim, :]
            sub_shift = self.shift_mat[i]
            sub_shuffle = self._to_tensor(shuf, dtype=torch.long)
            props, funcs = _HYBRID_CONFIGS[hfid]
            partitions = _compute_partitions(props, ndim, sub_shuffle)
            self.sub_hybrids.append((sub_matrix, sub_shift, funcs, partitions))

    def __call__(self, x):
        """Evaluate. x: (B, D) -> (B,) fitness values."""
        if x.shape[1] != self.ndim:
            raise ValueError(f"Expected D={self.ndim}, got {x.shape[1]}")

        if self.func_id <= 9:
            return self._eval_simple(x)
        elif self.func_id <= 19:
            return self._eval_hybrid(x)
        elif self.func_id <= 27:
            return self._eval_composition(x)
        else:
            return self._eval_composition_hybrid(x)

    def _eval_simple(self, x):
        """F1-F9."""
        z = (self.scale * (x - self.shift)) @ self.matrix.T
        return self.base_fn(z, **self.base_kwargs) + self.f_bias

    def _eval_hybrid(self, x):
        """F10-F19."""
        mz = (x - self.shift) @ self.matrix.T
        result = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for i, (fn, kwargs) in enumerate(self.hybrid_funcs):
            idx = self.partition_indices[i]
            result = result + fn(mz[:, idx], **kwargs)
        return result + self.f_bias

    def _eval_composition(self, x):
        """F20-F27."""
        B = x.shape[0]
        D = self.ndim

        ws = torch.zeros(B, self.n_funcs, device=x.device, dtype=x.dtype)
        gs = torch.zeros(B, self.n_funcs, device=x.device, dtype=x.dtype)

        for i, (fn, kwargs, scale, add_one) in enumerate(self.comp_funcs):
            M_i = self.matrix_full[i * D:(i + 1) * D, :]

            if self._f21_schwefel_special and i == 2:
                z = 10.0 * (x - self.shift_mat[i])
            elif self._use_shift0:
                z = (scale * (x - self.shift_mat[0])) @ M_i.T
            else:
                z = (scale * (x - self.shift_mat[i])) @ M_i.T

            if add_one:
                z = z + 1.0

            gs[:, i] = self.lambdas[i] * fn(z, **kwargs) + self.biases[i]
            ws[:, i] = calculate_weight(x - self.shift_mat[i], self.sigmas[i])

        w_sum = torch.sum(ws, dim=1, keepdim=True) + 1e-30
        ws = ws / w_sum

        return torch.sum(ws * gs, dim=1) + self.f_bias

    def _eval_composition_hybrid(self, x):
        """F28-F29."""
        B = x.shape[0]

        ws = torch.zeros(B, self.n_funcs, device=x.device, dtype=x.dtype)
        gs = torch.zeros(B, self.n_funcs, device=x.device, dtype=x.dtype)

        for i, (sub_M, sub_shift, funcs, partitions) in enumerate(self.sub_hybrids):
            mz = (x - sub_shift) @ sub_M.T
            result = torch.zeros(B, device=x.device, dtype=x.dtype)
            for j, (fn, kwargs) in enumerate(funcs):
                result = result + fn(mz[:, partitions[j]], **kwargs)
            gs[:, i] = self.lambdas[i] * result + self.biases[i]
            ws[:, i] = calculate_weight(x - self.shift_mat[i], self.sigmas[i])

        w_sum = torch.sum(ws, dim=1, keepdim=True) + 1e-30
        ws = ws / w_sum
        return torch.sum(ws * gs, dim=1) + self.f_bias


# ============================================================================
# Wrapper API
# ============================================================================

def get_function_torch(func_id, ndim=30, device='cpu'):
    """Drop-in replacement for CEC2017_bench.get_function().

    Returns:
        eval_fn, lb, ub, dim, name, category, f_optimal
    """
    cache_key = (func_id, ndim, str(device))
    if cache_key not in _cache:
        _cache[cache_key] = CEC2017Torch(func_id, ndim, device)

    fn = _cache[cache_key]
    return fn, fn.lb, fn.ub, ndim, fn.name, fn.category, fn.f_optimal


# Functions that fail validation against opfunu (pre-existing mismatches).
BLACKLIST = {
    (17, 10), (29, 10),   # NaN in opfunu at D=10
    (5, 10), (5, 30), (5, 50),    # Schaffer F7: scale=0.5/100 -> spread <1.2, no selection pressure
    (19, 10), (19, 30), (19, 50),  # Hybrid 10: large rel_error vs opfunu
    (21, 10), (21, 30), (21, 50),  # Composition 2: rel_error > 1e-8
    (23, 10), (23, 30), (23, 50),  # Composition 4: rel_error > 1e-8
}


def get_all_func_ids(ndim=30):
    """Return list of valid function IDs for the given dimensionality."""
    ids = list(range(1, 30)) if ndim >= 10 else [fid for fid in range(1, 30) if fid not in _MIN_DIM_10]
    return [fid for fid in ids if (fid, ndim) not in BLACKLIST]
