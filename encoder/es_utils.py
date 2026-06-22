"""
es_utils.py — Shared utilities for ES (Evolution Strategy) training loops.

Used by both GNN_MOS_Classic/train_l2o_es.py and NEURAL_META_K4/train_es_k4.py
to avoid code duplication.
"""
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


# ======================================================================
# RolloutResult — uniform return type for all rollout functions
# ======================================================================

@dataclass
class RolloutResult:
    """Uniform return type for ES rollout functions.

    All rollout functions (K=2 budget, K=4 operators, ensemble, etc.)
    must return this so the ES loop is head-agnostic.
    """
    neg_return: float       # -R for ES minimization (higher R = better optimization)
    entropy: float          # routing entropy (for diagnostics + entropy sigma mode)
    gap_closure: float      # final GC averaged over B functions
    extras: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Parameter flattening / writing
# ======================================================================

def collect_params(model, perturb_mode='all'):
    """Flatten params to perturb into one vector via model.es_param_groups().

    perturb_mode:
        'heads':     only head/routing modules
        'gat+heads': heads + backbone (GATv2, GRU, pooler, bridges)
        'all':       every parameter in the model
    """
    if perturb_mode == 'all':
        params, meta = [], []
        for name, p in model.named_parameters():
            params.append(p.data.view(-1))
            meta.append((p, name))
        if not params:
            raise ValueError("Model has no parameters")
        return torch.cat(params), meta

    groups = model.es_param_groups()

    # Map perturb_mode to which groups to collect
    if perturb_mode == 'heads':
        group_keys = ['heads']
    elif perturb_mode == 'gat+heads':
        group_keys = ['heads', 'gat']
    else:
        raise ValueError(f"Unknown perturb_mode={perturb_mode!r}")

    params, meta, seen = [], [], set()
    for key in group_keys:
        modules = groups.get(key, [])
        for mod in modules:
            if isinstance(mod, torch.nn.Parameter):
                pid = id(mod)
                if pid not in seen:
                    seen.add(pid)
                    params.append(mod.data.view(-1))
                    meta.append((mod, f'param_{pid}'))
            else:
                for pn, p in mod.named_parameters():
                    pid = id(p)
                    if pid not in seen:
                        seen.add(pid)
                        params.append(p.data.view(-1))
                        meta.append((p, pn))

    if not params:
        raise ValueError(f"No params collected for perturb_mode={perturb_mode!r} "
                         f"(groups: {list(groups.keys())})")
    return torch.cat(params), meta


def write_params(flat, meta):
    """Write flat vector back into model parameters."""
    offset = 0
    for p, _ in meta:
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view_as(p))
        offset += n


# ======================================================================
# CEC2017 function wrapper & serialization
# ======================================================================

class FnWrapper:
    """Lightweight CEC2017 function wrapper for serialization across workers."""
    __slots__ = ('_fn', 'fid', 'D', 'f_optimal')

    def __init__(self, fn, fid, D, f_optimal):
        self._fn = fn
        self.fid = fid
        self.D = D
        self.f_optimal = f_optimal

    def __call__(self, x):
        return self._fn(x)


def fn_to_info(fn):
    """Convert a CEC2017 function to a serializable tuple."""
    fid = getattr(fn, 'fid', 0)
    D = getattr(fn, 'D', 10)
    f_opt = getattr(fn, 'f_optimal', 0.0)
    if hasattr(f_opt, 'item'):
        f_opt = f_opt.item()
    elif not isinstance(f_opt, (int, float)):
        f_opt = 0.0
    return (fid, D, float(f_opt))


def make_fn(fn_info, device):
    """Reconstruct a CEC2017 function in a worker process from serializable info."""
    from encoder.cec2017_torch import get_function_torch
    fid, D, f_optimal = fn_info
    fn_tuple = get_function_torch(fid, D, device)
    return FnWrapper(fn_tuple[0], fid, D, f_optimal)


def eval_by_fn(fns, x_flat, B, N):
    """Evaluate B*N points, routing each sub-population to its function."""
    device = x_flat.device
    f_flat = torch.empty(B * N, device=device, dtype=x_flat.dtype)
    for b in range(B):
        f_flat[b * N:(b + 1) * N] = fns[b](x_flat[b * N:(b + 1) * N])
    return f_flat


# ======================================================================
# Function sampling
# ======================================================================

def sample_function(aug, device, D, dims, allowed_fids, no_augment, seed):
    """Sample one CEC2017 function (optionally augmented).

    Args:
        aug: AugmentedCEC2017 instance or None.
        device: torch device.
        D: dimensionality.
        dims: allowed dimensions (unused here, for compat).
        allowed_fids: set of allowed function IDs, or None for all.
        no_augment: if True, use raw CEC2017 (no rotation/shift).
        seed: random seed for function selection.

    Returns:
        FnWrapper instance.
    """
    from encoder.cec2017_torch import get_function_torch, get_all_func_ids

    valid = get_all_func_ids(D)
    if allowed_fids:
        valid = [f for f in valid if f in allowed_fids]

    rng = random.Random(seed)
    fid = rng.choice(valid)

    if aug is not None and not no_augment:
        fn = aug.sample(fid, D)
        f_opt = getattr(fn, 'f_optimal', 0.0)
        if hasattr(f_opt, 'item'):
            f_opt = f_opt.item()
        return FnWrapper(fn, fid, D, float(f_opt))
    else:
        fn_tuple = get_function_torch(fid, D, device)
        f_opt = fn_tuple[5] if len(fn_tuple) > 5 else 0.0
        if hasattr(f_opt, 'item'):
            f_opt = f_opt.item()
        return FnWrapper(fn_tuple[0], fid, D, float(f_opt))
