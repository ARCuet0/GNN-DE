"""Analytic SVGD step φ* — implementation correctness check (brief §3.2 #4).

φ*(x_i) = (1/N) Σ_j [k(x_j, x_i) s(x_j) + ∇_{x_j} k(x_j, x_i)]

Used in the alignment test to verify -∂KSD/∂X is structurally consistent
with the SVGD update direction. NOT a research metric — purely a sanity
check on KSD's implementation.

Single-scale RBF here (the canonical SVGD form). The matching test in
test_svgd_alignment.py uses the multi-scale ksd_loss against single-scale
φ*, so cosine alignment is approximate (>0.5 threshold) rather than exact.
"""
import torch

from l2o.ksd.score import compute_score


def svgd_phi_analytic(X: torch.Tensor, eval_fn, h: torch.Tensor,
                       T: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Analytic SVGD step φ*(x_i) for each particle.

    Args:
        X: (N, D) population. Detached internally — this is purely diagnostic.
        eval_fn: callable (N, D) -> (N,) fitness.
        h: scalar bandwidth (e.g. h_new returned by ksd_loss).
        T: temperature.
        eps: numerical floor.

    Returns:
        phi: (N, D) SVGD update direction per particle.
    """
    X_d = X.detach().clone()
    # Compute ∇f(X). No create_graph — this is a one-shot analytic step.
    X_leaf = X_d.clone().requires_grad_(True)
    f_sum = eval_fn(X_leaf).sum()
    g = torch.autograd.grad(f_sum, X_leaf)[0]
    s = compute_score(g.detach(), T=T, eps=eps)               # (N, D)

    diff = X_d.unsqueeze(1) - X_d.unsqueeze(0)                 # (N, N, D)
    sq_dist = (diff ** 2).sum(-1)                              # (N, N)
    if torch.is_tensor(h):
        h_safe = h.detach() + eps
    else:
        h_safe = torch.tensor(float(h), dtype=X_d.dtype,
                              device=X_d.device) + eps

    K = torch.exp(-sq_dist / h_safe)                           # (N, N)

    # ∇_{x_j} k(x_j, x_i) where diff[i,j] = X[i] - X[j].
    # ∂k/∂x_j = +(2/h) (X[i] - X[j]) k = (2/h) diff[i,j] K[i,j]
    grad_xj_k = (2.0 / h_safe) * diff * K.unsqueeze(-1)        # (N, N, D)

    # phi[i] = (1/N) Σ_j [K[i,j] s(x_j) + ∇_{x_j} k(x_j, x_i)]
    phi = (K.unsqueeze(-1) * s.unsqueeze(0) + grad_xj_k).mean(dim=1)
    return phi
