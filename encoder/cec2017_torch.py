"""
CEC2017 Benchmark -- thin re-export shim.

All implementation lives in the ``cec2017`` package.
This file preserves backward compatibility for ``from encoder.cec2017_torch import ...``.
"""

import sys
from pathlib import Path

# Ensure repo root is on sys.path so ``import cec2017`` works
# even when running from a subdirectory.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from cec2017 import (  # noqa: F401, E402
    CEC2017Torch,
    get_function_torch,
    get_all_func_ids,
    BLACKLIST,
    FUNCTIONS,
    CATEGORIES,
    bent_cigar, zakharov, rosenbrock, rastrigin, schaffer_f7,
    expanded_schaffer_f6, lunacek_bi_rastrigin, non_continuous_rastrigin,
    levy, modified_schwefel, elliptic, ackley, griewank, sphere, discus,
    happy_cat, hgbat, katsuura, weierstrass, grie_rosen_cec,
    calculate_weight,
    _opfunu_file_id, _load_shift, _load_shift_matrix,
    _load_matrix, _load_shuffle,
)
