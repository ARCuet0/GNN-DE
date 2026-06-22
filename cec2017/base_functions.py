"""
CEC2017 base mathematical functions.

All functions accept (B, D) tensors and return (B,) tensors.
Fully differentiable for use with autograd.
"""

import math
import torch


def bent_cigar(x):
    return x[:, 0] ** 2 + 1e6 * torch.sum(x[:, 1:] ** 2, dim=1)


def zakharov(x):
    t = torch.sum(0.5 * x, dim=1)
    return torch.sum(x ** 2, dim=1) + t ** 2 + t ** 4


def rosenbrock(x, shift=0.0):
    z = x + shift
    t1 = 100.0 * (z[:, :-1] ** 2 - z[:, 1:]) ** 2
    t2 = (z[:, :-1] - 1.0) ** 2
    return torch.sum(t1 + t2, dim=1)


def rastrigin(x):
    return torch.sum(x ** 2 - 10.0 * torch.cos(2.0 * math.pi * x) + 10.0, dim=1)


def schaffer_f7(x):
    B, D = x.shape
    sum_sq = x[:, :-1] ** 2 + x[:, 1:] ** 2 + 1e-30  # eps avoids NaN grad at s=0
    s = torch.sqrt(sum_sq)                              # s_i = sqrt(x_i^2 + x_{i+1}^2)
    vals = torch.sqrt(s) * (torch.sin(50.0 * s ** 0.2) + 1.0)
    return (torch.sum(vals, dim=1) / max(D - 1, 1)) ** 2


def expanded_schaffer_f6(x):
    x1 = x
    x2 = torch.roll(x, -1, dims=1)
    sum_sq = x1 ** 2 + x2 ** 2
    sin_val = torch.sin(torch.sqrt(sum_sq + 1e-30)) ** 2
    denom = (1.0 + 0.001 * sum_sq) ** 2
    return torch.sum(0.5 + (sin_val - 0.5) / denom, dim=1)


def lunacek_bi_rastrigin(x, miu0=2.5, d=1.0, shift=0.0):
    z = x + shift
    B, D = z.shape
    s = 1.0 - 1.0 / (2.0 * math.sqrt(D + 20) - 8.2)
    miu1 = -math.sqrt((miu0 ** 2 - d) / s)
    delta = z - miu0
    term1 = torch.sum(delta ** 2, dim=1)
    term2 = s * torch.sum((z - miu1) ** 2, dim=1) + d * D
    return torch.min(term1, term2) + 10.0 * (D - torch.sum(torch.cos(2.0 * math.pi * delta), dim=1))


def _rounder(x):
    """Non-continuous rounding -- exact match of opfunu's rounder(x, abs(x)).

    Uses trunc+fractional decomposition, NOT torch.round (which does
    banker's rounding and disagrees at half-integers).
    For negative x: dec is in (-1, 0], so dec < 0.5 is always true -> result = inter (truncation).
    For positive x: dec >= 0.5 -> inter + 1 (ceil), dec < 0.5 -> inter (floor).
    """
    condition = torch.abs(x)
    temp_2x = 2.0 * x
    inter = torch.trunc(temp_2x)
    dec = temp_2x - inter
    result = torch.where(dec < 0.5, inter, inter + 1.0)
    return torch.where(condition < 0.5, x, result / 2.0)


def non_continuous_rastrigin(x):
    """Matches opfunu: pairs (y_i, y_{i+1}) then rastrigin on 2D elements."""
    y = _rounder(x)
    y_shift = torch.roll(y, -1, dims=1)
    pairs = torch.stack([y, y_shift], dim=2)  # (B, D, 2)
    flat = pairs.reshape(x.shape[0], -1)       # (B, 2D)
    return rastrigin(flat)


def levy(x, shift=0.0):
    z = x + shift
    w = 1.0 + (z - 1.0) / 4.0
    t1 = torch.sin(math.pi * w[:, 0]) ** 2
    t2 = torch.sum((w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * w[:, :-1] + 1.0) ** 2), dim=1)
    t3 = (w[:, -1] - 1.0) ** 2 * (1.0 + torch.sin(2.0 * math.pi * w[:, -1]) ** 2)
    return t1 + t2 + t3


def modified_schwefel(x):
    z = x + 4.209687462275036e+002
    B, D = z.shape

    abs_z = torch.abs(z)
    mod_val = torch.fmod(abs_z, 500.0)

    mask_pos = z > 500.0
    mask_neg = z < -500.0
    # -500 <= z <= 500 (base case)
    fx = -(z * torch.sin(torch.sqrt(abs_z + 1e-30)))

    # z > 500: CEC2017 technical report spec (opfunu has sign bug in multiplier)
    fx = torch.where(mask_pos,
                     -((500.0 - mod_val) * torch.sin(torch.sqrt(torch.clamp(500.0 - mod_val, min=1e-30)))
                       - ((z - 500.0) / 100.0) ** 2 / D),
                     fx)

    # z < -500
    fx = torch.where(mask_neg,
                     -((mod_val - 500.0) * torch.sin(torch.sqrt(torch.clamp(500.0 - mod_val, min=1e-30)))
                       - ((z + 500.0) / 100.0) ** 2 / D),
                     fx)

    return torch.sum(fx, dim=1) + 4.189828872724338e+002 * D


def elliptic(x):
    B, D = x.shape
    idx = torch.arange(0, D, device=x.device, dtype=x.dtype)
    coeff = 10.0 ** (6.0 * idx / max(D - 1, 1))
    return torch.sum(coeff * x ** 2, dim=1)


def ackley(x):
    B, D = x.shape
    t1 = torch.sum(x ** 2, dim=1)
    t2 = torch.sum(torch.cos(2.0 * math.pi * x), dim=1)
    return -20.0 * torch.exp(-0.2 * torch.sqrt(t1 / D)) - torch.exp(t2 / D) + 20.0 + math.e


def griewank(x):
    B, D = x.shape
    idx = torch.arange(1, D + 1, device=x.device, dtype=x.dtype)
    t1 = torch.sum(x ** 2, dim=1) / 4000.0
    cos_terms = torch.cos(x / torch.sqrt(idx))
    t2 = torch.prod(cos_terms, dim=1)
    return t1 - t2 + 1.0


def sphere(x):
    return torch.sum(x ** 2, dim=1)


def discus(x):
    return 1e6 * x[:, 0] ** 2 + torch.sum(x[:, 1:] ** 2, dim=1)


def happy_cat(x, shift=0.0):
    z = x + shift
    B, D = z.shape
    t1 = torch.sum(z, dim=1)
    t2 = torch.sum(z ** 2, dim=1)
    smooth_abs = torch.sqrt((t2 - D) ** 2 + 1e-30)
    return smooth_abs ** 0.25 + (0.5 * t2 + t1) / D + 0.5


def hgbat(x, shift=0.0):
    z = x + shift
    B, D = z.shape
    t1 = torch.sum(z, dim=1)
    t2 = torch.sum(z ** 2, dim=1)
    smooth_abs = torch.sqrt((t2 ** 2 - t1 ** 2) ** 2 + 1e-30)
    return smooth_abs ** 0.5 + (0.5 * t2 + t1) / D + 0.5


def _soft_round(x):
    """Differentiable approximation of round(x) using sin. Exact at integers."""
    return x - torch.sin(2 * math.pi * x) / (2 * math.pi)


def katsuura(x):
    B, D = x.shape
    j = torch.arange(1, 33, device=x.device, dtype=x.dtype)  # (32,)
    # x: (B, D) -> (B, D, 1), j: (32,) -> (1, 1, 32)
    two_j = (2.0 ** j).unsqueeze(0).unsqueeze(0)  # (1, 1, 32)
    x_exp = x.unsqueeze(2)  # (B, D, 1)
    val = two_j * x_exp  # (B, D, 32)
    temp = torch.sum(torch.abs(val - _soft_round(val)) / two_j, dim=2)  # (B, D)
    idx = torch.arange(1, D + 1, device=x.device, dtype=x.dtype)  # (D,)
    factors = (1.0 + idx * temp) ** (10.0 / D ** 1.2)  # (B, D)
    # log-sum-exp for stable gradients through product
    return (torch.exp(torch.sum(torch.log(factors.clamp(min=1e-30)), dim=1)) - 1.0) * 10.0 / D ** 2


def weierstrass(x, a=0.5, b=3.0, k_max=20):
    B, D = x.shape
    k = torch.arange(0, k_max + 1, device=x.device, dtype=x.dtype)  # (k_max+1,)
    a_k = a ** k  # (k_max+1,)
    b_k = b ** k  # (k_max+1,)
    # x: (B, D) -> (B, D, 1)
    x_exp = x.unsqueeze(2)  # (B, D, 1)
    cos_vals = a_k * torch.cos(2.0 * math.pi * b_k * (x_exp + 0.5))  # (B, D, k_max+1)
    result = torch.sum(cos_vals, dim=(1, 2))  # (B,)
    # Subtract baseline: D * sum(a^k * cos(pi * b^k))
    baseline = D * torch.sum(a_k * torch.cos(math.pi * b_k))
    return result - baseline


def grie_rosen_cec(x):
    """Expanded Griewank-Rosenbrock (CEC version)."""
    z = x + 1.0
    # Pairs: (z_i, z_{i+1}) with wrap-around
    z1 = z
    z2 = torch.roll(z, -1, dims=1)
    tmp1 = (z1 ** 2 - z2) ** 2
    tmp2 = (z1 - 1.0) ** 2
    temp = 100.0 * tmp1 + tmp2
    return torch.sum(temp ** 2 / 4000.0 - torch.cos(temp) + 1.0, dim=1)


def calculate_weight(dx, sigma=1.0):
    """Composition weight. dx: (B, D), returns (B,)."""
    sum_sq = torch.sum(dx ** 2, dim=1)  # (B,)
    D = dx.shape[1]
    w = torch.where(
        sum_sq != 0,
        torch.sqrt(1.0 / sum_sq.clamp(min=1e-30)) * torch.exp(-sum_sq / (2.0 * D * sigma ** 2)),
        torch.full_like(sum_sq, 1e99)
    )
    return w
