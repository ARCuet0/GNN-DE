"""Landscape gradient oracle for LUPI training.

Computes df/dx in a separate forward/backward pass, fully detached from
the main computation graph.  Used as privileged targets for geometric
auxiliary losses and auxiliary prediction heads.
"""
import torch
import torch.nn.functional as F


def compute_grad_f(coords: torch.Tensor, eval_fn) -> torch.Tensor:
    """Compute landscape gradient df/dx detached from the main graph.

    Args:
        coords: (B, N, D) population coordinates.
        eval_fn: callable  (B*N, D) -> (B*N,) fitness values.

    Returns:
        (B, N, D) gradient tensor, same dtype as coords, no grad_fn.
    """
    B, N, D = coords.shape
    c = coords.detach().requires_grad_(True)
    f = eval_fn(c.reshape(-1, D)).reshape(B, N)
    (g,) = torch.autograd.grad(f.sum(), c)
    return g.detach()


def compute_alignment_target(
    coords: torch.Tensor,
    grad_f: torch.Tensor,
    diag_mask: torch.Tensor,
    tau: float = 0.5,
) -> torch.Tensor:
    """Gradient-alignment softmax target for attention-based geo losses.

    For each pair (i, j), computes how well "moving from i toward j"
    aligns with the negative gradient at i.

    Args:
        coords: (B, N, D) population coordinates (detached float).
        grad_f: (B, N, D) landscape gradient (detached).
        diag_mask: (1, N, N) or (B, N, N) bool mask for self-exclusion.
        tau: softmax temperature (lower = more peaked).

    Returns:
        (B, N, N) softmax-normalized alignment target.
    """
    direction = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, D)
    neg_grad = -grad_f.unsqueeze(2)  # (B, N, 1, D)
    alignment = F.cosine_similarity(
        direction, neg_grad.expand_as(direction), dim=-1)  # (B, N, N)
    alignment = alignment.masked_fill(diag_mask, -1e9)
    return torch.softmax(alignment / tau, dim=-1)
