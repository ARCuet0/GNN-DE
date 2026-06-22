"""KSD² loss with multi-scale RBF.

KSD² = (1/N²) Σ_ij k_p(x_i, x_j)

where the Stein kernel is built from score s(x) = -∇f/T and an RBF k:

  k_p(x, y) = sᵀs · k(x,y)
            + s(x)ᵀ ∇_y k(x, y)
            + s(y)ᵀ ∇_x k(x, y)
            + tr(∇_x ∇_y k(x, y))

For RBF k(x,y) = exp(-‖x-y‖²/h):
  ∇_x k =  -(2/h)(x-y) k
  ∇_y k =  +(2/h)(x-y) k         (= -∇_x k)
  tr(∇_x ∇_y k) = (2D/h - 4‖x-y‖²/h²) k

Multi-scale: each operation (k, grad, trace) is the average over scales
{h, h/4, 4h}. Chain rule preserves the linear combination.

Two entry points:
    `ksd_loss(X, ...)`         — X has shape (N, D). Single-batch form.
    `ksd_loss_batched(X, ...)` — X has shape (B, N, D). Vectorised over
                                 the batch dim; per-batch independent
                                 bandwidth EMA state. Returns mean over B.

The score uses `create_graph=True` so ∂KSD²/∂X exists. The ∇f(X) computed
inside MUST NOT be detached — that is the silent-failure mode that
breaks the meta-objective.
"""
import torch

from l2o.ksd.score import compute_score
from l2o.ksd.kernel import bandwidth_with_ema


# Per-scale factors of the multi-scale RBF average. Tuple is iterated
# inline to keep the kernel computation explicit and autograd-friendly.
_SCALE_FACTORS = (1.0, 0.25, 4.0)


def ksd_loss(X: torch.Tensor, eval_fn,
             T: float = 1.0,
             h_ema: torch.Tensor | None = None,
             alpha: float = 0.9,
             eps: float = 1e-8,
             return_terms: bool = False):
    """KSD² between empirical(X) and Gibbs target p* ∝ exp(-f/T).

    Single-batch form. Equivalent to ksd_loss_batched(X.unsqueeze(0))
    minus the batch reduction. Kept as a thin wrapper around the batched
    implementation so unit tests can exercise the (N, D) contract.

    Args:
        X: (N, D) population. Must have requires_grad=True for autograd
           to flow through ∇f and back to X.
        eval_fn: callable accepting (..., D) and returning (...) fitness.
        T, h_ema, alpha, eps, return_terms: see ksd_loss_batched.

    Returns:
        loss, h_new                          (default)
        loss, h_new, t1, t2, t3, t4          (return_terms=True)
    """
    Xb = X.unsqueeze(0)                              # (1, N, D)
    h_list = [h_ema] if h_ema is not None else None
    out = ksd_loss_batched(Xb, eval_fn, T=T, h_ema=h_list, alpha=alpha,
                            eps=eps, return_terms=return_terms)
    if return_terms:
        loss, h_new_list, t1, t2, t3, t4 = out
        return loss, h_new_list[0], t1, t2, t3, t4
    loss, h_new_list = out
    return loss, h_new_list[0]


def ksd_loss_batched(X: torch.Tensor, eval_fn,
                      T: float = 1.0,
                      h_ema: list | None = None,
                      alpha: float = 0.9,
                      eps: float = 1e-8,
                      return_terms: bool = False):
    """Per-batch vectorised KSD² with mean reduction over the batch dim.

    Replaces an explicit Python loop over B in the caller with a single
    autograd graph that batches the score, kernel, and Stein terms.

    Args:
        X: (B, N, D) population, requires_grad=True.
        eval_fn: callable accepting (..., D) and returning (...) fitness.
        T: temperature for p* ∝ exp(-f/T).
        h_ema: optional list of length B with previous EMA bandwidths.
               Each entry is a detached scalar tensor or None. When None
               (or any entry None), the median heuristic is used for
               that batch slice on the current call.
        alpha: EMA decay (default 0.9).
        eps: numerical floor.
        return_terms: also return per-term means (averaged over B, N, N).

    Returns:
        loss, h_new_list                      (default)
        loss, h_new_list, t1, t2, t3, t4      (return_terms=True)

        h_new_list is a list of B detached scalars ready for the next call.
    """
    if X.dim() != 3:
        raise ValueError(f"ksd_loss_batched expects (B, N, D); got shape {tuple(X.shape)}")
    B, N, D = X.shape
    dtype, device = X.dtype, X.device

    # ── 1. Score: ∇f / T → log1p-normalised. NO DETACH.
    f_sum = eval_fn(X.reshape(B * N, D)).sum()
    g = torch.autograd.grad(f_sum, X, create_graph=True)[0]   # (B, N, D)
    s = compute_score(g, T=T, eps=eps)                         # (B, N, D)

    # ── 2. Pairwise differences and squared distances per batch.
    diff = X.unsqueeze(2) - X.unsqueeze(1)                     # (B, N, N, D)
    sq_dist = (diff ** 2).sum(-1)                              # (B, N, N)

    # ── 3. Per-batch bandwidth via shared scalar helper. Carries EMA.
    h_list, h_new_list = [], []
    for b in range(B):
        h_prev = None if (h_ema is None or h_ema[b] is None) else h_ema[b]
        h_b, h_new_b = bandwidth_with_ema(sq_dist[b], h_ema_prev=h_prev,
                                           alpha=alpha, N=N, eps=eps)
        h_list.append(h_b)
        h_new_list.append(h_new_b)
    h = torch.stack(h_list).to(dtype=dtype, device=device).view(B, 1, 1)  # (B, 1, 1)

    # Per-scale bandwidths broadcast across (N, N) and (N, N, D).
    h_scales = [(h * f + eps) for f in _SCALE_FACTORS]         # list of (B, 1, 1)

    # ── 4. Per-scale K, ∇_x K, and trace term, summed then averaged.
    K_acc = torch.zeros_like(sq_dist)
    grad_x_k_acc = torch.zeros_like(diff)
    trace_acc = torch.zeros_like(sq_dist)
    for h_s in h_scales:
        K_s = torch.exp(-sq_dist / h_s)                         # (B, N, N)
        K_acc = K_acc + K_s
        # ∇_x k = -(2/h_s)(x-y) k.
        grad_x_k_acc = grad_x_k_acc + (-2.0 / h_s.unsqueeze(-1)) * diff * K_s.unsqueeze(-1)
        # tr(∇_x ∇_y k) = (2D/h_s - 4 sq_dist/h_s²) k.
        trace_acc = trace_acc + (2.0 * D / h_s - 4.0 * sq_dist / (h_s ** 2)) * K_s

    inv_S = 1.0 / len(_SCALE_FACTORS)
    K = K_acc * inv_S                                           # (B, N, N)
    grad_x_k = grad_x_k_acc * inv_S                             # (B, N, N, D)
    grad_y_k = -grad_x_k                                        # RBF symmetry
    trace_term = trace_acc * inv_S                              # (B, N, N)

    # ── 5. Stein kernel terms.
    s_i = s.unsqueeze(2)               # (B, N, 1, D), indexed by i
    s_j = s.unsqueeze(1)               # (B, 1, N, D), indexed by j

    term1 = (s_i * s_j).sum(-1) * K                             # sᵀs · k
    term2 = (s_i * grad_y_k).sum(-1)                            # s(x_i)ᵀ ∇_y k
    term3 = (s_j * grad_x_k).sum(-1)                            # s(x_j)ᵀ ∇_x k
    term4 = trace_term                                          # tr(∇_x ∇_y k)

    ksd_per_batch = (term1 + term2 + term3 + term4).mean(dim=(1, 2))  # (B,)
    loss = ksd_per_batch.mean()

    if return_terms:
        return (loss, h_new_list,
                term1.mean(), term2.mean(), term3.mean(), term4.mean())
    return loss, h_new_list
