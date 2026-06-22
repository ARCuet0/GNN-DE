"""Score normalisation for KSD (brief §2.1).

Converts ∇f(X) into the score s(x) = ∇log p*(x) of the Gibbs target
p* ∝ exp(-f/T), with magnitude in log-space (preserves direction, tames
F09-style outliers where ‖∇f‖ ~ 1e5).

s = -(g/T) / ‖g/T‖ · log1p(‖g/T‖).clamp(min=1e-2)

The clamp at 1e-2 ensures flat regions still contribute (silent zeros
would mask bugs). The choice of log1p over τ-soft (g / (τ + ‖g‖)) is
deliberate: log1p is NOT scale-invariant; τ-soft would be — see brief
§2.1 caveat. If scale invariance becomes load-bearing later, swap here.
"""
import torch


def compute_score(grad_f_X: torch.Tensor, T: float = 1.0,
                  eps: float = 1e-8) -> torch.Tensor:
    """Normalised score from a gradient tensor.

    Args:
        grad_f_X: (..., D) tensor of ∇f at X. Must allow autograd to flow
                  if KSD will backward through the score (i.e. NOT detached).
        T: temperature for the Gibbs target.
        eps: numerical floor for the norm divisor.

    Returns:
        s: (..., D) score, same shape as input.
    """
    g = grad_f_X / T
    norm = torch.linalg.norm(g, dim=-1, keepdim=True) + eps
    s_dir = -g / norm
    s_mag = torch.log1p(norm).clamp(min=1e-2)
    return s_dir * s_mag
