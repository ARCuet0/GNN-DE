"""Multi-scale RBF kernel + bandwidth EMA (brief §2.2-2.3).

Single-scale RBF collapses in high D (concentration of measure). At D=30
default, the multi-scale average over (h, h/4, 4h) keeps at least one
scale informative. Apply identically to gradients of the kernel — chain
rule preserves the linear combination.

Bandwidth uses median heuristic with EMA smoothing across BPTT generations
to avoid brittle per-step bandwidth changes. EMA state is external and
detached.
"""
import torch


def multi_scale_kernel(sq_dist: torch.Tensor, h: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Average of three RBFs at scales (h, h/4, 4h).

    Args:
        sq_dist: (..., N, N) squared distances. May contain zero diagonal.
        h: scalar bandwidth (must be positive).
        eps: numerical floor.

    Returns:
        K: same shape as sq_dist. K[i, i] = 1.
    """
    h_safe = h + eps
    K1 = torch.exp(-sq_dist / h_safe)
    K2 = torch.exp(-sq_dist / (h_safe / 4))
    K3 = torch.exp(-sq_dist / (h_safe * 4))
    return (K1 + K2 + K3) / 3


def bandwidth_with_ema(sq_dist: torch.Tensor,
                        h_ema_prev: torch.Tensor | None,
                        alpha: float = 0.9,
                        N: int | None = None,
                        eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Median-heuristic bandwidth with EMA smoothing.

    h_current = median(sq_dist) / log(N + 1)   (detached)
    h         = alpha * h_ema_prev + (1-alpha) * h_current

    The clamp(min=eps) prevents h from collapsing to 0 when sq_dist's
    median is dominated by the zero diagonal — this would otherwise
    produce inf in 1/h.

    Args:
        sq_dist: (N, N) pairwise squared distances. Detached internally.
        h_ema_prev: previous EMA state, or None for first call.
        alpha: EMA decay (0.9 default per brief).
        N: population size. If None, inferred from sq_dist.shape[-1].
        eps: numerical floor.

    Returns:
        (h, h_new): h to use in current call (detached), h_new to pass
                    forward as the next h_ema_prev. Both equal numerically
                    on the first call.
    """
    if N is None:
        N = sq_dist.shape[-1]
    h_current = (torch.median(sq_dist.detach())
                 / torch.log(torch.tensor(N + 1.0,
                                          dtype=sq_dist.dtype,
                                          device=sq_dist.device))
                ).clamp(min=eps)

    if h_ema_prev is None:
        h = h_current
    else:
        if not torch.is_tensor(h_ema_prev):
            h_ema_prev = torch.tensor(float(h_ema_prev),
                                      dtype=h_current.dtype,
                                      device=h_current.device)
        h = alpha * h_ema_prev + (1.0 - alpha) * h_current
    h = h.detach()
    return h, h
