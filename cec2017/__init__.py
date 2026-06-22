"""
CEC2017 Benchmark -- fully batched PyTorch implementation.

Re-exports all public names so that ``from cec2017 import CEC2017Torch``
works identically to the old monolithic module.
"""

from .benchmark import (  # noqa: F401
    CEC2017Torch,
    get_function_torch,
    get_all_func_ids,
    BLACKLIST,
)
from .configs import FUNCTIONS, CATEGORIES  # noqa: F401
from .base_functions import (  # noqa: F401
    bent_cigar, zakharov, rosenbrock, rastrigin, schaffer_f7,
    expanded_schaffer_f6, lunacek_bi_rastrigin, non_continuous_rastrigin,
    levy, modified_schwefel, elliptic, ackley, griewank, sphere, discus,
    happy_cat, hgbat, katsuura, weierstrass, grie_rosen_cec,
    calculate_weight,
)
from .data_loader import (  # noqa: F401
    _opfunu_file_id, _load_shift, _load_shift_matrix,
    _load_matrix, _load_shuffle,
)
