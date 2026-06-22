"""
graph_utils.py — Shared helpers for graph builders.
"""

import torch


def _temporal_to_globals(stagnation_counters, delta_fitnesses,
                         contraction_rates, P, dev):
    """Convert temporal state (tensor or list or None) to global feature slots.

    Returns stag_t, delta_fit_t, contraction_t — all (P,) float32 on dev.
    """
    def _to_t(v, default=0.0):
        if v is None:
            return torch.full((P,), default, device=dev, dtype=torch.float32)
        if torch.is_tensor(v):
            return v.to(device=dev, dtype=torch.float32)
        return torch.tensor(v, device=dev, dtype=torch.float32)
    sc = _to_t(stagnation_counters)
    df = _to_t(delta_fitnesses)
    cr = _to_t(contraction_rates)
    stag_t = (sc / 20.0).tanh() * 2 - 1
    delta_fit_t = df.clamp(-1, 1)
    contraction_t = cr.clamp(-1, 1)
    return stag_t, delta_fit_t, contraction_t
