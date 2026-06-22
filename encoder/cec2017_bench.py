"""
CEC2017 Benchmark wrapper.

29 functions at configurable dimensionality (D=10, 20, 30, 50, 100).
Functions grouped by type:
  - Unimodal (F1): shifted+rotated Bent Cigar
  - Multimodal (F2-F9): shifted+rotated classic functions
  - Hybrid (F10-F19): variable partitioning, mix of basic functions
  - Composition (F20-F29): weighted sums of multiple basic functions

Via opfunu (already installed).
"""
import numpy as np
from opfunu.cec_based import cec2017


FUNCTIONS = {
    1:  ('F12017',  'Bent Cigar',                  'Unimodal'),
    2:  ('F22017',  'Zakharov',                    'Multimodal'),
    3:  ('F32017',  'Rosenbrock',                  'Multimodal'),
    4:  ('F42017',  'Rastrigin',                   'Multimodal'),
    5:  ('F52017',  'Schaffer F7',                 'Multimodal'),
    6:  ('F62017',  'Lunacek Bi-Rastrigin',        'Multimodal'),
    7:  ('F72017',  'Non-Cont Rastrigin',          'Multimodal'),
    8:  ('F82017',  'Levy',                        'Multimodal'),
    9:  ('F92017',  'Schwefel',                    'Multimodal'),
    10: ('F102017', 'Hybrid 1',                    'Hybrid'),
    11: ('F112017', 'Hybrid 2',                    'Hybrid'),
    12: ('F122017', 'Hybrid 3',                    'Hybrid'),
    13: ('F132017', 'Hybrid 4',                    'Hybrid'),
    14: ('F142017', 'Hybrid 5',                    'Hybrid'),
    15: ('F152017', 'Hybrid 6',                    'Hybrid'),
    16: ('F162017', 'Hybrid 7',                    'Hybrid'),
    17: ('F172017', 'Hybrid 8',                    'Hybrid'),
    18: ('F182017', 'Hybrid 9',                    'Hybrid'),
    19: ('F192017', 'Hybrid 10',                   'Hybrid'),
    20: ('F202017', 'Composition 1',               'Composition'),
    21: ('F212017', 'Composition 2',               'Composition'),
    22: ('F222017', 'Composition 3',               'Composition'),
    23: ('F232017', 'Composition 4',               'Composition'),
    24: ('F242017', 'Composition 5',               'Composition'),
    25: ('F252017', 'Composition 6',               'Composition'),
    26: ('F262017', 'Composition 7',               'Composition'),
    27: ('F272017', 'Composition 8',               'Composition'),
    28: ('F282017', 'Composition 9',               'Composition'),
    29: ('F292017', 'Composition 10',              'Composition'),
}

CATEGORIES = {
    'Unimodal':     [1],
    'Multimodal':   [2, 3, 4, 5, 6, 7, 8, 9],
    'Hybrid':       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'Composition':  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
}

# Hybrid functions (F10-F19) and F28-F29 require ndim >= 10
_MIN_DIM_10 = set(range(10, 20)) | {28, 29}

# Cache instantiated functions keyed by (func_id, ndim)
_cache = {}


def get_function(func_id, ndim=30):
    """
    Get a CEC2017 function at specified dimensionality.

    Args:
        func_id: 1-29
        ndim: dimensionality (10, 20, 30, 50, 100)

    Returns:
        eval_fn: callable(x) -> float, x is (ndim,) float64
        lb: (ndim,) lower bounds
        ub: (ndim,) upper bounds
        dim: int
        name: str
        category: str
        f_optimal: float (global optimum value = func_id * 100)
    """
    if func_id not in FUNCTIONS:
        raise ValueError(f"func_id must be 1-29, got {func_id}")

    if func_id in _MIN_DIM_10 and ndim < 10:
        raise ValueError(
            f"F{func_id} (hybrid/composition) requires ndim >= 10, got {ndim}")

    cache_key = (func_id, ndim)
    if cache_key not in _cache:
        cls_name, name, category = FUNCTIONS[func_id]
        cls = getattr(cec2017, cls_name)
        instance = cls(ndim=ndim)
        _cache[cache_key] = instance

    instance = _cache[cache_key]
    _, name, category = FUNCTIONS[func_id]

    f_optimal = float(instance.f_global)

    def _safe_eval(x, _fn=instance.evaluate):
        val = _fn(x)
        if np.isnan(val):
            return np.inf
        return val

    return (
        _safe_eval,
        instance.lb.astype(np.float64),
        instance.ub.astype(np.float64),
        ndim,
        name,
        category,
        f_optimal,
    )


def get_all_func_ids(ndim=30):
    """Return list of valid function IDs for the given dimensionality."""
    if ndim >= 10:
        return list(range(1, 30))
    else:
        # Only unimodal + multimodal + some compositions at D<10
        return [fid for fid in range(1, 30) if fid not in _MIN_DIM_10]


if __name__ == "__main__":
    import time
    for ndim in [30, 50]:
        print(f"\nCEC2017 Benchmark — D={ndim}")
        print("=" * 70)
        valid_ids = get_all_func_ids(ndim)
        for fid in valid_ids:
            eval_fn, lb, ub, dim, name, cat, f_opt = get_function(fid, ndim)
            x = np.random.uniform(lb, ub)
            t0 = time.time()
            val = eval_fn(x)
            dt = time.time() - t0
            print(f"  F{fid:>2}: {name:<30} [{cat:<12}] "
                  f"val={val:.2e}  opt={f_opt:.1f}  {dt*1000:.2f}ms")
        print(f"\nAll {len(valid_ids)} functions OK at D={ndim}.")
