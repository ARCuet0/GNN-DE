"""Selection modes over the augmented surrogate pool.

The augmented pool lays out parents then proposals:
  slot 0..N-1           : parents
  slot N..N+M_var*N*K-1 : proposals

Proposal flat index i ∈ [0, N_prop) where N_prop = M_var * N * K is laid
out (by `permute(1,0,2,3,4).reshape`) so that parent(i) = (i // K) % N.

Modes:
  topk           — deployed: torch.topk on full augmented scores (parents allowed).
  uniform        — uniform sample over proposals.
  exp:LAM        — over surrogate-ranked proposals, weight ∝ exp(-rank/LAM).
  weibull:K:LAM  — weight ∝ weibull_pdf(rank; shape=K, scale=LAM).
  power:ALPHA    — weight ∝ 1/(rank+1)^ALPHA.
  random_1pp     — M_sel parents sampled uniformly; 1 random proposal per parent.
  oracle_1pp     — M_sel parents sampled uniformly; BEST proposal per parent
                   (requires real fit_aug). FES accounting counts M_sel.
  top1_1pp       — M_sel parents sampled uniformly; HIGHEST-score proposal per
                   parent (deployable analogue of oracle_1pp — uses surr_scores,
                   no fitness oracle needed).

All sampled-without-replacement paths use the Gumbel-top-k trick.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch


_MODES_NO_PARAM = {'topk', 'uniform', 'random_1pp', 'oracle_1pp', 'top1_1pp',
                   'q_exploit_max_1pp', 'q_explor_max_1pp',
                   'jepa_exploit_max_1pp', 'jepa_explor_max_1pp'}

# Honest deployable default — no privileged FES info. Oracle modes exist only
# as a ceiling diagnostic and should not be used where a real optimization
# algorithm would be simulated (training / canonical eval).
DEPLOYABLE_DEFAULT_SPEC = 'random_1pp'


def parse_spec(spec: str) -> Tuple[str, Dict[str, float]]:
    parts = spec.split(':')
    mode = parts[0]
    if mode in _MODES_NO_PARAM:
        if len(parts) != 1:
            raise ValueError(f'{mode} takes no params: {spec!r}')
        return mode, {}
    if mode == 'exp':
        if len(parts) != 2:
            raise ValueError(f'exp needs :LAM, got {spec!r}')
        return 'exp', {'lam': float(parts[1])}
    if mode == 'power':
        if len(parts) != 2:
            raise ValueError(f'power needs :ALPHA, got {spec!r}')
        return 'power', {'alpha': float(parts[1])}
    if mode == 'weibull':
        if len(parts) != 3:
            raise ValueError(f'weibull needs :K:LAM, got {spec!r}')
        return 'weibull', {'k': float(parts[1]), 'lam': float(parts[2])}
    if mode == 'oracle_kpp':
        if len(parts) != 2:
            raise ValueError(f'oracle_kpp needs :K, got {spec!r}')
        k = int(parts[1])
        if k < 1:
            raise ValueError(f'oracle_kpp k must be >=1, got {k}')
        return 'oracle_kpp', {'k': k}
    if mode == 'q_exploit_softmax_1pp':
        if len(parts) != 2:
            raise ValueError(f'q_exploit_softmax_1pp needs :TAU, got {spec!r}')
        return 'q_exploit_softmax_1pp', {'tau': float(parts[1])}
    if mode == 'q_exploit_topk_random_1pp':
        if len(parts) != 2:
            raise ValueError(f'q_exploit_topk_random_1pp needs :K, got {spec!r}')
        return 'q_exploit_topk_random_1pp', {'k': int(parts[1])}
    if mode == 'jepa_exploit_softmax_1pp':
        if len(parts) != 2:
            raise ValueError(f'jepa_exploit_softmax_1pp needs :TAU, got {spec!r}')
        return 'jepa_exploit_softmax_1pp', {'tau': float(parts[1])}
    if mode == 'jepa_exploit_topk_random_1pp':
        if len(parts) != 2:
            raise ValueError(f'jepa_exploit_topk_random_1pp needs :K, got {spec!r}')
        return 'jepa_exploit_topk_random_1pp', {'k': int(parts[1])}
    raise ValueError(f'unknown selection mode: {spec!r}')


def _one_hot_mask(top_idx: torch.Tensor, N_aug: int) -> torch.Tensor:
    B, M_sel = top_idx.shape
    mask = torch.zeros(B, N_aug, dtype=torch.float32, device=top_idx.device)
    mask.scatter_(1, top_idx, 1.0)
    return mask


def _gumbel_topk(log_weights: torch.Tensor, k: int,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample k indices without replacement from categorical(exp(log_weights))."""
    # log_weights: (B, P). Returns (B, k) indices.
    u = torch.empty_like(log_weights).uniform_(
        1e-10, 1.0 - 1e-10, generator=generator)
    gumbel = -torch.log(-torch.log(u))
    perturbed = log_weights + gumbel
    return perturbed.topk(k, dim=1).indices


def _rank_weights(mode: str, params: Dict[str, float], n: int,
                  device: torch.device) -> torch.Tensor:
    """Return log-weights (n,) indexed by rank (0 = best).

    Uses log-space to stay numerically safe; callers add Gumbel noise and top-k.
    """
    r = torch.arange(n, device=device, dtype=torch.float64)
    if mode == 'uniform':
        return torch.zeros(n, device=device, dtype=torch.float64)
    if mode == 'exp':
        lam = params['lam']
        return -r / max(lam, 1e-12)
    if mode == 'power':
        alpha = params['alpha']
        return -alpha * torch.log(r + 1.0)
    if mode == 'weibull':
        k = params['k']
        lam = params['lam']
        # Weibull pdf: (k/lam)(x/lam)^(k-1) exp(-(x/lam)^k)
        # Use log form. Map rank 0 -> small x (not zero) to avoid -inf at k<1.
        x = (r + 0.5) / max(lam, 1e-12)
        # log_pdf = log(k/lam) + (k-1)*log(x) - x^k
        return (k - 1.0) * torch.log(x) - x ** k
    raise ValueError(f'_rank_weights: unsupported mode {mode}')


def select(mode: str, params: Dict[str, float], *,
           scores: torch.Tensor, fit_aug: torch.Tensor,
           N: int, M_sel: int, M_var: int, K: int,
           generator: torch.Generator | None = None,
           disen_scores: torch.Tensor | None = None,
           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (top_idx, sel_mask).

    scores       : (B, N_aug) surrogate scores (real-valued).
    fit_aug      : (B, N_aug) fitness values; only read by oracle_1pp.
    N            : parent count.
    M_sel        : selections per batch.
    M_var,K      : proposal fan-out (N_prop = M_var*N*K).
    generator    : optional torch.Generator for reproducible stochastic modes.
    disen_scores : (B, N_aug) disen-head scores (q_exploit or q_explor); used
                   only by q_exploit_*/q_explor_* modes.
    """
    B, N_aug = scores.shape
    device = scores.device
    N_prop = M_var * N * K
    assert N_aug == N + N_prop, f'N_aug={N_aug}, N={N}, N_prop={N_prop}'

    if mode == 'topk':
        top_idx = scores.topk(M_sel, dim=1).indices
        return top_idx, _one_hot_mask(top_idx, N_aug)

    if mode in ('uniform', 'exp', 'weibull', 'power'):
        # Rank proposals by surrogate score (best = rank 0).
        prop_scores = scores[:, N:]                     # (B, N_prop)
        ranked_idx = prop_scores.argsort(dim=1, descending=True)  # (B, N_prop)
        # log-weights indexed by rank → broadcast to (B, N_prop)
        log_w = _rank_weights(mode, params, N_prop, device)       # (N_prop,)
        log_w = log_w.to(scores.dtype).unsqueeze(0).expand(B, -1)
        # Gumbel-top-k over ranks, then translate rank→proposal index→aug index.
        rank_choice = _gumbel_topk(log_w, M_sel, generator=generator)  # (B, M_sel)
        prop_choice = ranked_idx.gather(1, rank_choice)
        top_idx = prop_choice + N
        return top_idx, _one_hot_mask(top_idx, N_aug)

    if mode == 'random_1pp':
        # Pick M_sel parents uniformly (without replacement), then 1 random
        # proposal-offset in [0, M_var*K) per parent.
        parent_logw = torch.zeros(B, N, device=device, dtype=scores.dtype)
        parents = _gumbel_topk(parent_logw, M_sel, generator=generator)  # (B, M_sel)
        # Random offset per selection
        offset_logw = torch.zeros(B, M_sel, M_var * K, device=device,
                                  dtype=scores.dtype)
        u = torch.empty_like(offset_logw).uniform_(
            1e-10, 1.0 - 1e-10, generator=generator)
        gumbel = -torch.log(-torch.log(u))
        offsets = gumbel.argmax(dim=-1)  # (B, M_sel) in [0, M_var*K)
        # Flat proposal index = m*N*K + n*K + k
        # offset = m*K + k; so flat = (offset // K)*N*K + n*K + (offset % K)
        m = offsets // K
        k = offsets % K
        flat = m * N * K + parents * K + k
        top_idx = flat + N
        return top_idx, _one_hot_mask(top_idx, N_aug)

    if mode == 'oracle_kpp':
        K_cap = int(params['k'])
        K_fan = M_var * K  # proposals per parent
        if K_cap > K_fan:
            K_cap = K_fan
        # Reshape proposal fitness to (B, N, K_fan) indexed by (parent, offset)
        # where offset = m*K + k_head (m in [0,M_var), k_head in [0,K)).
        # Proposal flat i = m*N*K + n*K + k_head →
        #   parent(i) = (i // K) % N, offset = (i // (N*K)) * K + (i % K).
        prop_fit = fit_aug[:, N:]  # (B, N_prop)
        # Build (B, N, K_fan) by gathering in the right order.
        m_idx = torch.arange(M_var, device=device)
        k_idx = torch.arange(K, device=device)
        n_idx = torch.arange(N, device=device)
        # flat index for (b, n, m, k_head) = m*N*K + n*K + k_head
        flat_per_parent = (m_idx.view(M_var, 1, 1) * N * K
                           + n_idx.view(1, N, 1) * K
                           + k_idx.view(1, 1, K))  # (M_var, N, K)
        flat_per_parent = flat_per_parent.permute(1, 0, 2).reshape(N, K_fan)  # (N, K_fan)
        flat_per_parent_b = flat_per_parent.unsqueeze(0).expand(B, -1, -1)  # (B, N, K_fan)
        fit_per_parent = prop_fit.gather(
            1, flat_per_parent_b.reshape(B, N * K_fan)).reshape(B, N, K_fan)
        # Sort each parent's proposals (ascending) and take top K_cap by fitness
        _, sort_idx = fit_per_parent.sort(dim=-1)  # (B, N, K_fan)
        keep_idx = sort_idx[..., :K_cap]  # (B, N, K_cap)
        # Map back to flat proposal indices
        keep_flat = flat_per_parent_b.gather(2, keep_idx)  # (B, N, K_cap)
        keep_flat2d = keep_flat.reshape(B, N * K_cap)
        keep_fit = prop_fit.gather(1, keep_flat2d)  # (B, N*K_cap)
        # Global top-M_sel by ascending fitness (best = lowest)
        _, top_within = keep_fit.topk(M_sel, dim=1, largest=False)
        chosen_flat = keep_flat2d.gather(1, top_within)  # (B, M_sel) in [0, N_prop)
        top_idx = chosen_flat + N
        return top_idx, _one_hot_mask(top_idx, N_aug)

    if mode == 'top1_1pp':
        # Sample M_sel parents uniformly. For each, find the proposal with
        # HIGHEST surrogate score among {m*N*K + n*K + k : m, k}. Same shape
        # as oracle_1pp but driven by `scores` instead of `fit_aug` — the
        # deployable variant (no fitness oracle).
        parent_logw = torch.zeros(B, N, device=device, dtype=scores.dtype)
        parents = _gumbel_topk(parent_logw, M_sel, generator=generator)  # (B, M_sel)
        m_idx = torch.arange(M_var, device=device)
        k_idx = torch.arange(K, device=device)
        n_idx = torch.arange(N, device=device).unsqueeze(1)              # (N, 1)
        mk_offsets = (m_idx.unsqueeze(1) * N * K + k_idx.unsqueeze(0)).reshape(-1)  # (M_var*K,)
        all_flat = mk_offsets.unsqueeze(0) + n_idx * K                    # (N, M_var*K)
        aug_idx_per_parent = (all_flat + N).long()                        # (N, M_var*K)
        flat_for_sel = aug_idx_per_parent[parents]                        # (B, M_sel, M_var*K)
        score_sel = scores.unsqueeze(1).expand(-1, M_sel, -1).gather(
            2, flat_for_sel)                                              # (B, M_sel, M_var*K)
        best_off = score_sel.argmax(dim=-1)                               # (B, M_sel)
        top_idx = flat_for_sel.gather(2, best_off.unsqueeze(-1)).squeeze(-1)
        return top_idx, _one_hot_mask(top_idx, N_aug)

    if mode == 'oracle_1pp':
        # Sample M_sel parents uniformly. For each, find the proposal with
        # lowest fit_aug among {m*N*K + n*K + k : m, k}.
        parent_logw = torch.zeros(B, N, device=device, dtype=scores.dtype)
        parents = _gumbel_topk(parent_logw, M_sel, generator=generator)  # (B, M_sel)
        # For each parent n, collect its proposal flat-indices in [0, N_prop).
        # proposal_flat(m, n, k) = m*N*K + n*K + k
        m_idx = torch.arange(M_var, device=device)
        k_idx = torch.arange(K, device=device)
        # (M_var, K) flat offsets relative to parent n: m*N*K + k
        mk_offsets = (m_idx.unsqueeze(1) * N * K + k_idx.unsqueeze(0)).reshape(-1)
        # per-parent proposal flat indices: (N, M_var*K)
        all_flat = parents.new_zeros(N, M_var * K)
        for n_val in range(N):
            all_flat[n_val] = mk_offsets + n_val * K
        # gather fitness for each (B, M_sel, M_var*K) proposal group
        # aug index = flat + N
        aug_idx_per_parent = (all_flat + N).long()  # (N, M_var*K)
        # Expand for batch + selections
        flat_for_sel = aug_idx_per_parent[parents]  # (B, M_sel, M_var*K)
        fit_sel = fit_aug.unsqueeze(1).expand(-1, M_sel, -1).gather(
            2, flat_for_sel)  # (B, M_sel, M_var*K)
        best_off = fit_sel.argmin(dim=-1)  # (B, M_sel) in [0, M_var*K)
        top_idx = flat_for_sel.gather(2, best_off.unsqueeze(-1)).squeeze(-1)
        return top_idx, _one_hot_mask(top_idx, N_aug)

    # ── Disen-head-based 1pp selectors (q_*) and JEPA-disen variants (jepa_*) ──
    # Both families consume `disen_scores` aligned with N_aug. The difference is
    # the SOURCE of disen_scores: q_* uses real h_aug, jepa_* uses JEPA-predicted h.
    # The selection logic is identical; only the input varies (handled upstream).
    DISEN_LIKE_MODES = (
        'q_exploit_max_1pp', 'q_explor_max_1pp',
        'q_exploit_softmax_1pp', 'q_exploit_topk_random_1pp',
        'jepa_exploit_max_1pp', 'jepa_explor_max_1pp',
        'jepa_exploit_softmax_1pp', 'jepa_exploit_topk_random_1pp',
    )
    if mode in DISEN_LIKE_MODES:
        if disen_scores is None:
            raise ValueError(f'{mode} requires disen_scores')
        # Sample M_sel parents uniformly without replacement.
        parent_logw = torch.zeros(B, N, device=device, dtype=scores.dtype)
        parents = _gumbel_topk(parent_logw, M_sel, generator=generator)  # (B, M_sel)
        m_idx = torch.arange(M_var, device=device)
        k_idx = torch.arange(K, device=device)
        n_idx = torch.arange(N, device=device).unsqueeze(1)
        mk_offsets = (m_idx.unsqueeze(1) * N * K + k_idx.unsqueeze(0)).reshape(-1)
        all_flat = mk_offsets.unsqueeze(0) + n_idx * K
        aug_idx_per_parent = (all_flat + N).long()
        flat_for_sel = aug_idx_per_parent[parents]
        ds = disen_scores.unsqueeze(1).expand(-1, M_sel, -1).gather(2, flat_for_sel)

        if 'max_1pp' in mode:  # deterministic argmax
            best_off = ds.argmax(dim=-1)
        elif 'softmax_1pp' in mode:
            tau = float(params.get('tau', 1.0))
            log_w = ds / max(tau, 1e-12)
            u = torch.empty_like(log_w).uniform_(
                1e-10, 1.0 - 1e-10, generator=generator)
            gumbel = -torch.log(-torch.log(u))
            best_off = (log_w + gumbel).argmax(dim=-1)
        elif 'topk_random_1pp' in mode:
            kk = int(params.get('k', 5))
            kk = min(kk, M_var * K)
            top_k_off = ds.topk(kk, dim=-1).indices
            r_logw = torch.zeros_like(top_k_off, dtype=scores.dtype)
            u = torch.empty_like(r_logw).uniform_(
                1e-10, 1.0 - 1e-10, generator=generator)
            gumbel = -torch.log(-torch.log(u))
            r_idx = (r_logw + gumbel).argmax(dim=-1, keepdim=True)
            best_off = top_k_off.gather(-1, r_idx).squeeze(-1)
        else:
            raise ValueError(f'unhandled disen-like mode: {mode}')
        top_idx = flat_for_sel.gather(2, best_off.unsqueeze(-1)).squeeze(-1)
        return top_idx, _one_hot_mask(top_idx, N_aug)

    raise ValueError(f'unknown mode: {mode}')
