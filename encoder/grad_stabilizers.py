"""Gradient stabilizers for full-trajectory BPTT.

Tools to prevent gradient explosion over 230+ generation recurrences:

1. soft_min_scalar — logsumexp soft-min that distributes gradient across
   near-best individuals instead of concentrating 50:1 on a single argmin.

2. GradNormBarrier — (legacy) clips backward gradient norm per generation.
   Suffers from compounding attenuation over long chains.

3. register_spectral_clip — clips the Jacobian spectral norm (not the gradient).
   Only intervenes when ρ(J_g) > 1, transparent otherwise.
   Uses power iteration via existing estimate_spectral_radius_from_h.
"""
import math
import torch

from .measure_jacobian import estimate_spectral_radius_from_h


def soft_min_scalar(values: torch.Tensor, beta: float = 20.0,
                    normalize: bool = True) -> torch.Tensor:
    """Differentiable soft-min returning a scalar (flattens all dims).

    Uses logsumexp: softmin(x) = -logsumexp(-beta * x) / beta

    Args:
        values: any shape
        beta: sharpness. Higher = closer to hard min.
        normalize: if True, min-max normalize before logsumexp to
                   spread gradient across all individuals.
    """
    flat = values.reshape(-1)
    if normalize:
        v_min = flat.min()
        v_range = (flat.max() - v_min).clamp(min=1e-8)
        normed = (flat - v_min) / v_range
        soft_idx = -torch.logsumexp(-beta * normed, dim=0) / beta
        return v_min + soft_idx * v_range
    return -torch.logsumexp(-beta * flat, dim=0) / beta


def soft_min(values: torch.Tensor, beta: float = 20.0, dim: int = -1,
             normalize: bool = True) -> torch.Tensor:
    """Differentiable soft-min along a dimension, preserving other dims.

    Args:
        values: any shape
        beta: sharpness. Higher = closer to hard min.
        dim: dimension to reduce over
        normalize: if True, min-max normalize values to [0,1] before
                   logsumexp, then map result back to original scale.
                   This prevents gradient starvation: without it,
                   exp(-beta*gap) underflows to 0 for gaps > ~1/beta,
                   starving all but 1-2 individuals. With normalize=True
                   and N=50, ~20-25 individuals receive meaningful gradient.
    Returns:
        tensor with `dim` reduced
    """
    if normalize:
        v_min = values.min(dim=dim, keepdim=True).values
        v_range = (values.max(dim=dim, keepdim=True).values - v_min).clamp(min=1e-8)
        normed = (values - v_min) / v_range  # [0, 1]
        soft_idx = -torch.logsumexp(-beta * normed, dim=dim) / beta
        # soft_idx may undershoot 0 slightly (≈ -1/(beta*N)), mapping to
        # a value below v_min. This is negligible vs the gap to f* and
        # any clamp would kill gradient flow. Leave unclamped.
        return v_min.squeeze(dim) + soft_idx * v_range.squeeze(dim)
    return -torch.logsumexp(-beta * values, dim=dim) / beta


def hard_min_ste(values: torch.Tensor, beta: float = 20.0, dim: int = -1) -> torch.Tensor:
    """Hard min forward, soft_min (normalized) backward via straight-through estimator.

    Forward:  exact hard min along `dim` (no undershoot).
    Backward: gradient from normalized soft_min (spreads across ~20 individuals).

    This avoids the billion-scale undershoot of normalized soft_min while
    preserving its broad gradient distribution.
    """
    hard = values.min(dim=dim).values.detach()   # exact, no undershoot
    soft = soft_min(values, beta, dim=dim, normalize=True)  # gradient source
    return hard + (soft - soft.detach())         # STE: forward=hard, backward=soft


def signed_log1p(x: torch.Tensor) -> torch.Tensor:
    """Monotone transform defined for all reals.  Approx log(x) for x >> 1.

    signed_log1p(x) = sign(x) * log(1 + |x|)

    Properties: odd function, C¹ everywhere, gradient = 1/(1+|x|).
    Used in hitting loss to replace log(gap.clamp(1e-8)) so that
    negative gaps (soft_min undershoot below f*) still carry gradient.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def log1p_linear(x: torch.Tensor, knee: float = 1e4) -> torch.Tensor:
    """log1p for |x| < knee, linear tail for |x| >= knee.

    Gradient is 1/(1+|x|) below the knee (same as log1p) and a constant
    1/(1+knee) above it.  This prevents gradient from vanishing at large
    values while keeping log-scale sensitivity near zero.

    Odd function: log1p_linear(-x) = -log1p_linear(x).
    """
    ax = torch.abs(x)
    slope = 1.0 / (1.0 + knee)
    log_knee = math.log1p(knee)
    below = torch.log1p(ax)
    above = log_knee + slope * (ax - knee)
    return torch.sign(x) * torch.where(ax <= knee, below, above)


class GradNormBarrier(torch.autograd.Function):
    """Identity forward, gradient norm clipping backward.

    Inserted on coords at each BPTT generation boundary to cap the
    per-step gradient amplification from the recurrence Jacobian.

    With max_norm=1.0, gradient norms cannot grow regardless of how
    many generations are backpropagated through.
    """

    @staticmethod
    def forward(ctx, x, max_norm):
        ctx.max_norm = max_norm
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        gn = grad.norm()
        scale = torch.where(
            gn > ctx.max_norm,
            ctx.max_norm / gn,
            torch.ones_like(gn),
        )
        return grad * scale, None


def make_gen_clip_hook(max_norm=100.0):
    """Create a backward hook that clips gradient norm per BPTT generation.

    Prevents gradient direction monopolization: without per-gen clipping,
    a single generation with 7000× amplification causes the final
    clip_grad_norm_ to scale the ENTIRE gradient by 10/7000 ≈ 0.001,
    starving all other generations' contributions.

    With max_norm=100, the hook is identity for normal gradient flow
    (amplification 1-5×) and only fires on pathological generations.
    Compounding attenuation is negligible when the hook fires on < 10%
    of generations.

    Fully GPU-resident: no Python branching on GPU tensors, no implicit
    .item() sync.  Stats (fires/total) are collected on GPU and read
    once after backward.

    Args:
        max_norm: maximum gradient norm allowed at this generation boundary.

    Returns:
        (hook, stats) where stats has 'norms' (list of GPU scalar tensors).
        Call ``sum(n.item() > max_norm for n in stats['norms'])`` after
        backward to get fire count (single sync point).
    """
    stats = {'norms': []}

    def hook(grad):
        norm = grad.norm()
        stats['norms'].append(norm.detach())
        scale = torch.clamp(max_norm / norm, max=1.0)
        return grad * scale

    return hook, stats


def register_spectral_clip(new_coords, old_coords, n_iters=3, max_rho=1.0):
    """Clip Jacobian spectral norm via backward hook on new_coords.

    Estimates ρ = σ_max(∂new_coords/∂old_coords) via power iteration.
    If ρ > max_rho, registers a hook on new_coords that scales the
    incoming gradient by max_rho/ρ BEFORE the Jacobian is applied
    in backward. This ensures ||J^T × scaled_g|| ≤ max_rho × ||g||.

    Unlike GradNormBarrier, this only clips the TRANSITION (Jacobian),
    not the accumulated signal. Transparent when ρ < max_rho.

    Args:
        new_coords: output of gen step (coords_{g+1})
        old_coords: input to gen step (coords_g, must have requires_grad)
        n_iters: power iteration steps (3 is enough for well-separated σ)
        max_rho: maximum allowed spectral norm (1.0 = no amplification)

    Returns:
        rho: estimated spectral radius (float, for diagnostics)
    """
    if not old_coords.requires_grad:
        return 0.0

    rho = estimate_spectral_radius_from_h(old_coords, new_coords, n_iters=n_iters)

    if rho > max_rho:
        scale = max_rho / rho
        new_coords.register_hook(lambda grad, s=scale: grad * s)

    return rho


def scale_gradient(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Forward: identity. Backward: grad *= scale.

    Uses the algebraic trick: x*s + x.detach()*(1-s) = x in forward,
    but only the x*s term carries gradient, so backward sees grad*s.

    Useful for tiered loss routing: scale=0 fully detaches a path from
    upstream parameters while keeping the forward computation intact.
    """
    if scale == 1.0 or not x.requires_grad:
        return x
    return x * scale + x.detach() * (1.0 - scale)
