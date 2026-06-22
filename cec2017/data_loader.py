"""
CEC2017 data loading utilities.

Loads shift vectors, rotation matrices, and shuffle data from opfunu's
installed data directory.
"""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Data path (opfunu installed data)
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent
# Try to find opfunu data directory
try:
    import opfunu
    _DATA_DIR = Path(opfunu.__file__).resolve().parent / "cec_based" / "data_2017"
except ImportError:
    raise ImportError("opfunu is required for loading CEC2017 shift/rotation data")


def _opfunu_file_id(func_id):
    """Opfunu uses func_id+1 for composition functions (F20-F29)."""
    return func_id + 1 if func_id >= 20 else func_id


def _load_shift(func_id, ndim):
    """Load shift vector from opfunu data. Returns (ndim,) numpy array."""
    sid = _opfunu_file_id(func_id)
    data = np.genfromtxt(str(_DATA_DIR / f"shift_data_{sid}.txt"), dtype=float)
    return data.ravel()[:ndim]


def _load_shift_matrix(func_id, ndim):
    """Load multiple shift vectors for composition functions. Returns (n, ndim) numpy array."""
    sid = _opfunu_file_id(func_id)
    data = np.genfromtxt(str(_DATA_DIR / f"shift_data_{sid}.txt"), dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :ndim]


def _load_matrix(func_id, ndim):
    """Load rotation matrix. Returns (ndim, ndim) or (n*ndim, ndim) numpy array."""
    mid = _opfunu_file_id(func_id)
    data = np.genfromtxt(str(_DATA_DIR / f"M_{mid}_D{ndim}.txt"), dtype=float)
    return data


def _load_shuffle(func_id, ndim):
    """Load shuffle data. Returns (ndim,) integer numpy array (0-indexed)."""
    sid = _opfunu_file_id(func_id)
    data = np.genfromtxt(str(_DATA_DIR / f"shuffle_data_{sid}_D{ndim}.txt"), dtype=int)
    return (data - 1).ravel()
