"""
_quadratic_guard.py — context manager that detects O(N^2) ops at runtime.

Used by D1000/tests/test_no_quadratic_in_pipeline.py to enforce the lema:
no torch.cdist call, no (..., N, N, ...) intermediate via torch.bmm /
torch.matmul anywhere in the deployed pipeline when D1000 flags are set.

Usage:

    with quadratic_guard(N=200, allow_under=64) as guard:
        # ... run forward pass ...
    assert not guard.violations, guard.violations
"""
from contextlib import contextmanager

import torch


class QuadraticOpForbidden(RuntimeError):
    """Raised when a torch.cdist call is detected inside quadratic_guard."""


def _has_NxN_dim(shape, N: int) -> bool:
    """Detect adjacent (..., N, N, ...) pattern. Strict equality on BOTH dims
    avoids false positives when an unrelated dim (e.g. gatv2_hidden) happens
    to be >= N — a (B, N, gatv2_hidden) Linear output is not an N² intermediate.
    """
    s = list(shape)
    for i in range(len(s) - 1):
        if s[i] == N and s[i + 1] == N:
            return True
    return False


@contextmanager
def quadratic_guard(N: int, allow_under: int = 64,
                    raise_on_cdist: bool = True):
    """Detect O(N^2) compute / memory inside a forward pass.

    Args:
        N:           population size at issue (N^2 patterns are flagged when
                     a tensor has two adjacent dims both >= min(N, allow_under)
                     and one of them == N).
        allow_under: tensors with both adjacent dims < this threshold are
                     ignored (e.g., k=8 for sparse edge features).
        raise_on_cdist: if True, torch.cdist raises QuadraticOpForbidden when
                     called. If False, it's recorded in `violations` and
                     allowed to proceed. Default True.

    Yields a `Guard` object with:
        guard.violations: list[str] — descriptions of detected ops.
    """
    class Guard:
        violations: list

    g = Guard()
    g.violations = []

    orig_cdist = torch.cdist
    orig_bmm = torch.bmm
    orig_matmul = torch.matmul

    def cdist_intercept(*a, **kw):
        msg = f"torch.cdist called inside quadratic_guard"
        g.violations.append(msg)
        if raise_on_cdist:
            raise QuadraticOpForbidden(msg)
        return orig_cdist(*a, **kw)

    def bmm_intercept(a, b, **kw):
        out = orig_bmm(a, b, **kw)
        if hasattr(out, 'shape') and _has_NxN_dim(out.shape, N):
            g.violations.append(
                f"torch.bmm produced NxN-shaped output: {tuple(out.shape)}")
        return out

    def matmul_intercept(a, b, **kw):
        out = orig_matmul(a, b, **kw)
        if (hasattr(out, 'shape') and out.dim() >= 2
                and _has_NxN_dim(out.shape, N)):
            g.violations.append(
                f"torch.matmul produced NxN-shaped output: "
                f"{tuple(out.shape)}")
        return out

    torch.cdist = cdist_intercept
    torch.bmm = bmm_intercept
    torch.matmul = matmul_intercept
    try:
        yield g
    finally:
        torch.cdist = orig_cdist
        torch.bmm = orig_bmm
        torch.matmul = orig_matmul
