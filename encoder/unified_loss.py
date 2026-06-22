"""
unified_loss.py — Unified loss function for all L2O training: log(f - f*).

The only loss with gradient that GROWS near the optimum:
    ∂loss/∂f = 1/(f - f*)

Gradient ratio gen_0/gen_N = 0.64 (amplifies near optimum)
vs r_t: 2.24 (decays), improvement: 1.95 (decays), log_gap: 739 (catastrophic).

Verified numerically on F1 sphere D=10, 200 generations, fixed absolute step.
Decision: 2026-03-27.
"""
import math
import torch


def log_gap_loss(
    f_child: torch.Tensor,
    f_optimal: float,
    reduce: str = 'mean',
    delta: float = 1e-2,
) -> torch.Tensor:
    """Compute log(f - f* + delta) loss.

    Stabilized: gradient saturates at 1/delta near optimum instead of exploding.
    Max gradient = 1/delta = 100 (with default delta=1e-2).

    Args:
        f_child:   (...) fitness values of offspring
        f_optimal: known global optimum (CEC2017 provides this)
        reduce:    'mean' (scalar), 'none' (per-element)
        delta:     stability floor — max gradient = 1/delta

    Returns:
        loss: scalar if reduce='mean', same shape as f_child if 'none'
    """
    gap = (f_child - f_optimal).clamp(min=0.0) + delta
    loss = torch.log(gap)

    if reduce == 'mean':
        return loss.mean()
    elif reduce == 'none':
        return loss
    else:
        raise ValueError(f"Unknown reduce={reduce!r}")


def importance_weighted_log_gap(
    f_child: torch.Tensor,
    f_optimal: float,
    delta: float = 1e-2,
) -> torch.Tensor:
    """Log-gap loss with importance weighting over M dimension.

    Weights M samples by advantage (better samples contribute more).
    Variance-reduced via adaptive temperature and ESS fallback.

    Args:
        f_child: (M, N) or (M, B, N) fitness values
        f_optimal: global optimum
        delta: stability floor — max gradient = 1/delta

    Returns:
        loss: scalar
    """
    gap = (f_child - f_optimal).clamp(min=0.0) + delta
    log_gap = torch.log(gap)  # (M, ...)

    M = log_gap.shape[0]

    if M < 3:
        return log_gap.mean()

    # Flatten all dims except M for weighting
    flat = log_gap.view(M, -1)  # (M, N_total)

    with torch.no_grad():
        # Per-sample mean loss (lower = better)
        sample_loss = flat.mean(dim=1)  # (M,)
        baseline = sample_loss.mean()
        advantage = sample_loss - baseline  # (M,)

        # Adaptive temperature
        temp = advantage.std().clamp(min=0.1)
        weights = torch.softmax(-advantage / temp, dim=0)  # (M,) — better = higher weight

        # ESS check: fallback to uniform if weights too concentrated
        ess = 1.0 / (weights ** 2).sum()
        if ess < 3.0:
            weights = torch.ones(M, device=f_child.device) / M

    # Weighted mean over M, then mean over elements
    # weights: (M,) -> (M, 1)
    weighted = (weights.unsqueeze(1) * flat).sum(dim=0)  # (N_total,)
    return weighted.mean()


def adaptive_log1p_loss(
    f_child: torch.Tensor,
    f_optimal: float,
    tau_ema: float,
    momentum: float = 0.9,
    reduce: str = 'mean',
    eval_mask: torch.Tensor = None,
) -> tuple[torch.Tensor, float]:
    """Soft-log loss: log(1 + gap/τ) with adaptive τ.

    Solves the gradient spike problem of log(f - f*):
    - Far from optimum (gap >> τ): behaves like log(gap/τ) — scale-invariant
    - Near optimum (gap << τ): behaves like gap/τ — linear, gradient bounded at 1/τ
    - Smooth transition, no discontinuity

    Gradient: 1/(τ + gap).  Max = 1/τ (bounded).

    Args:
        f_child:   (...) fitness values of offspring
        f_optimal: known global optimum
        tau_ema:   current EMA of τ (scale parameter)
        momentum:  EMA momentum for τ update (1.0 = no update)
        reduce:    'mean' (scalar), 'none' (per-element)
        eval_mask: bool tensor, same shape as f_child. True = actually evaluated
                   (non-NoOp). Only evaluated offspring contribute to τ update.

    Returns:
        (loss, tau_new): loss tensor and updated τ
    """
    gap = (f_child - f_optimal).clamp(min=0.0)

    # Update τ only from evaluated offspring (exclude NoOp copies of parent fitness)
    with torch.no_grad():
        if eval_mask is not None and eval_mask.any():
            gap_mean = gap[eval_mask].mean().item()
        else:
            gap_mean = gap.mean().item()
        tau_new = momentum * tau_ema + (1.0 - momentum) * gap_mean

    # Prevent division by zero
    tau_safe = max(tau_new, 1e-30)

    loss = torch.log1p(gap / tau_safe)

    if reduce == 'mean':
        return loss.mean(), tau_new
    elif reduce == 'none':
        return loss, tau_new
    else:
        raise ValueError(f"Unknown reduce={reduce!r}")


def multi_target_hitting_loss(
    trajectory: torch.Tensor,
    f_optimal: float,
    n_targets: int = 30,
    scale: float = 1.0,
    **_kwargs,
) -> torch.Tensor:
    """Clamped hinge loss in log-space with fixed targets.

    K targets at gap = 10^x, x uniform in [0, 10]. Each target contributes
    min(clamp, max(0, log(gap/target))). The clamp (= spacing between targets
    in log-space) ensures only the nearest unmet target has active gradient.

    Properties:
    - Monotonically decreasing as gap decreases (0 violations)
    - Loss range [0, ~0.79] × scale — bounded O(1)
    - Gradient = 1/(gap × K) per active target — never saturates
    - Works for any gap scale (10^0 to 10^10+)

    Args:
        trajectory: (T,) fitness values over time (must carry gradient)
        f_optimal:  known global optimum
        n_targets:  number of log-spaced targets (default 30)
        scale:      multiply loss for larger range (default 10)

    Returns:
        loss: scalar — lower means faster convergence
    """
    gaps = (trajectory - f_optimal).clamp(min=1e-8)  # (T,)

    # K targets at gap = 10^x, x in [0, 10], equispaced
    exponents = torch.linspace(0, 10, n_targets, device=trajectory.device)
    log_targets = exponents * math.log(10)  # log(10^x) = x * ln(10)

    # Clamp = spacing between consecutive targets in log-space
    clamp_val = (10.0 / (n_targets - 1)) * math.log(10)

    log_gaps = gaps.log()  # (T,) — differentiable: d/d(gap) = 1/gap

    # Shortfall: how far each gen's gap is above each target, clamped to 1 step
    # (T, K): 0 when gap ≤ target (reached), ramps up when gap > target, saturates at clamp_val
    shortfall = (log_gaps.unsqueeze(1) - log_targets.unsqueeze(0)).clamp(
        min=0, max=clamp_val)

    loss = shortfall.mean() * scale

    return loss
