"""
measure_jacobian.py — Estimate Jacobian spectral radius ρ for adaptive BPTT.

ρ = spectral_radius(∂h_{t+1}/∂h_t) measures how much one generation's
backbone output influences the next. Low ρ (< 0.5) means short BPTT
windows suffice. High ρ (> 0.9) requires long windows or dovetailing.

Uses power iteration on the Jacobian-vector product (no full Jacobian
materialization — works for h of any size).

Based on: Aicher, Foti & Fox (UAI 2019) — "Adaptively Truncating BPTT
to Control Gradient Bias."
"""
import math

import torch


def estimate_spectral_radius_from_h(
    h_t: torch.Tensor,
    h_tp1: torch.Tensor,
    n_iters: int = 10,
) -> float:
    """Estimate top singular value of J = ∂h_{t+1}/∂h_t via power iteration.

    h_t must have requires_grad=True and h_tp1 must be a differentiable
    function of h_t (connected computation graph).

    Args:
        h_t:     (B, N, H) input state (with grad)
        h_tp1:   (B, N, H) output state (differentiable from h_t)
        n_iters: power iteration steps

    Returns:
        rho: estimated spectral radius (float)
    """
    flat_dim = h_t.numel()
    device = h_t.device

    # Random initial vector
    v = torch.randn(flat_dim, device=device)
    v = v / v.norm()

    for _ in range(n_iters):
        # Jacobian-vector product: J @ v
        # J = ∂h_tp1/∂h_t, so J @ v = ∂(h_tp1 · u)/∂h_t where u = v reshaped
        h_tp1_flat = h_tp1.reshape(-1)
        jvp = torch.autograd.grad(
            h_tp1_flat, h_t,
            grad_outputs=v.reshape(h_tp1_flat.shape) if v.shape == h_tp1_flat.shape else v[:h_tp1_flat.numel()],
            create_graph=False, retain_graph=True,
        )[0].reshape(-1)

        # J^T @ (J @ v) for power iteration on J^T J
        # But we want singular value, not eigenvalue of J^T J
        # σ_max = ‖Jv‖ when v is the right singular vector
        norm = jvp.norm().item()
        if norm < 1e-12:
            return 0.0
        v = jvp / jvp.norm()

    # σ_max ≈ ‖Jv‖ from the last iteration (no extra autograd call needed)
    return norm


def estimate_recurrence_rho(
    old_coords: torch.Tensor,
    new_coords: torch.Tensor,
    n_iters: int = 3,
) -> float:
    """Estimate spectral radius of the FULL recurrence Jacobian ∂new_coords/∂old_coords.

    Unlike estimate_spectral_radius_from_h which measures ∂h/∂h, this measures
    the actual coordinate recurrence including all paths (heads, backbone, graph).

    Args:
        old_coords: (B, N, D) previous generation coords (must have requires_grad)
        new_coords: (B, N, D) current generation coords (differentiable from old_coords)
        n_iters: power iteration steps (3 is usually sufficient)

    Returns:
        rho: estimated spectral radius (float)
    """
    flat_dim = old_coords.numel()
    device = old_coords.device

    v = torch.randn(flat_dim, device=device)
    v = v / v.norm()

    for _ in range(n_iters):
        nc_flat = new_coords.reshape(-1)
        jvp = torch.autograd.grad(
            nc_flat, old_coords,
            grad_outputs=v[:nc_flat.numel()].reshape(nc_flat.shape),
            create_graph=False, retain_graph=True,
        )[0].reshape(-1)

        norm = jvp.norm().item()
        if norm < 1e-12:
            return 0.0
        v = jvp / jvp.norm()

    return norm


def min_safe_bptt(rho: float, tolerance: float = 0.05, max_L: int = 5000) -> int:
    """Compute minimum BPTT window for given spectral radius and bias tolerance.

    The truncation bias at window L is approximately:
        bias ≈ ρ^L / (1 - ρ) × |∂loss/∂h|

    We want bias / |full_gradient| < tolerance, so:
        ρ^L / (1 - ρ) < tolerance
        L > log(tolerance × (1 - ρ)) / log(ρ)

    Args:
        rho: spectral radius (0 < ρ < 1 for stable systems)
        tolerance: acceptable bias fraction (default 5%)
        max_L: cap for ρ ≈ 1

    Returns:
        L: minimum safe BPTT window length
    """
    if rho <= 0:
        return 1
    if rho >= 1.0:
        return max_L

    target = tolerance * (1.0 - rho)
    if target <= 0:
        return max_L

    L = math.log(target) / math.log(rho)
    return min(max_L, max(1, int(math.ceil(L))))
