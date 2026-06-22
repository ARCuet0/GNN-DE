"""
CEC2017 function metadata, hybrid configs, and partition logic.
"""

import math

from .base_functions import (
    bent_cigar, zakharov, rosenbrock, rastrigin, schaffer_f7,
    expanded_schaffer_f6, lunacek_bi_rastrigin,
    levy, modified_schwefel, elliptic, ackley, griewank,
    discus, happy_cat, hgbat, katsuura, weierstrass, grie_rosen_cec,
)

# ---------------------------------------------------------------------------
# Function metadata
# ---------------------------------------------------------------------------
FUNCTIONS = {
    1:  ('Bent Cigar',                  'Unimodal'),
    2:  ('Zakharov',                    'Multimodal'),
    3:  ('Rosenbrock',                  'Multimodal'),
    4:  ('Rastrigin',                   'Multimodal'),
    5:  ('Schaffer F7',                 'Multimodal'),
    6:  ('Lunacek Bi-Rastrigin',        'Multimodal'),
    7:  ('Non-Cont Rastrigin',          'Multimodal'),
    8:  ('Levy',                        'Multimodal'),
    9:  ('Schwefel',                    'Multimodal'),
    10: ('Hybrid 1',                    'Hybrid'),
    11: ('Hybrid 2',                    'Hybrid'),
    12: ('Hybrid 3',                    'Hybrid'),
    13: ('Hybrid 4',                    'Hybrid'),
    14: ('Hybrid 5',                    'Hybrid'),
    15: ('Hybrid 6',                    'Hybrid'),
    16: ('Hybrid 7',                    'Hybrid'),
    17: ('Hybrid 8',                    'Hybrid'),
    18: ('Hybrid 9',                    'Hybrid'),
    19: ('Hybrid 10',                   'Hybrid'),
    20: ('Composition 1',               'Composition'),
    21: ('Composition 2',               'Composition'),
    22: ('Composition 3',               'Composition'),
    23: ('Composition 4',               'Composition'),
    24: ('Composition 5',               'Composition'),
    25: ('Composition 6',               'Composition'),
    26: ('Composition 7',               'Composition'),
    27: ('Composition 8',               'Composition'),
    28: ('Composition 9',               'Composition'),
    29: ('Composition 10',              'Composition'),
}

CATEGORIES = {
    'Unimodal':     [1],
    'Multimodal':   [2, 3, 4, 5, 6, 7, 8, 9],
    'Hybrid':       [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    'Composition':  [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
}

_MIN_DIM_10 = set(range(10, 20)) | {28, 29}

# ---------------------------------------------------------------------------
# Hybrid function configs: proportions and (func, kwargs) list.
# Shared between _init_hybrid and _init_composition_hybrid (for F28/F29).
# ---------------------------------------------------------------------------
_HYBRID_CONFIGS = {
    10: ([0.2, 0.2, 0.6], [
        (zakharov, {}), (rosenbrock, {'shift': 1.0}), (rastrigin, {})
    ]),
    11: ([0.3, 0.3, 0.4], [
        (elliptic, {}), (modified_schwefel, {}), (bent_cigar, {})
    ]),
    12: ([0.3, 0.3, 0.4], [
        (bent_cigar, {}), (rosenbrock, {'shift': 1.0}),
        (lunacek_bi_rastrigin, {'miu0': 2.5, 'd': 1.0, 'shift': 2.5})
    ]),
    13: ([0.2, 0.2, 0.2, 0.4], [
        (elliptic, {}), (ackley, {}), (schaffer_f7, {}), (rastrigin, {})
    ]),
    14: ([0.2, 0.2, 0.3, 0.3], [
        (bent_cigar, {}), (hgbat, {'shift': -1.0}),
        (rastrigin, {}), (rosenbrock, {'shift': 1.0})
    ]),
    15: ([0.2, 0.2, 0.3, 0.3], [
        (expanded_schaffer_f6, {}), (hgbat, {'shift': -1.0}),
        (rosenbrock, {'shift': 1.0}), (modified_schwefel, {})
    ]),
    16: ([0.1, 0.2, 0.2, 0.2, 0.3], [
        (katsuura, {}), (ackley, {}), (grie_rosen_cec, {}),
        (modified_schwefel, {}), (rastrigin, {})
    ]),
    17: ([0.1, 0.2, 0.2, 0.2, 0.3], [
        (elliptic, {}), (ackley, {}), (rastrigin, {}),
        (hgbat, {'shift': -1.0}), (discus, {})
    ]),
    18: ([0.2, 0.2, 0.2, 0.2, 0.2], [
        (bent_cigar, {}), (rastrigin, {}), (grie_rosen_cec, {}),
        (weierstrass, {}), (expanded_schaffer_f6, {})
    ]),
    19: ([0.1, 0.1, 0.2, 0.2, 0.2, 0.2], [
        (happy_cat, {'shift': -1.0}), (katsuura, {}), (ackley, {}),
        (rastrigin, {}), (modified_schwefel, {}), (schaffer_f7, {})
    ]),
}


def _compute_partitions(props, ndim, shuffle):
    """Compute partition index tensors from proportions and shuffle."""
    indices = []
    cumsum = 0
    for i, p in enumerate(props):
        if i < len(props) - 1:
            n = int(p * ndim)
        else:
            n = ndim - cumsum
        indices.append(shuffle[cumsum:cumsum + n])
        cumsum += n
    return indices
