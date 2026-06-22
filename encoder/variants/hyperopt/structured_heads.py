"""Batched structured heads: Sampling, Differential, Gradient.

Each uses LowRankMatrixHead for PSD factorization M = UU^T + diag(d).
All ops are natively batched (B, N, D) — no loops over B, no scatter.
Interface: compute_params() → dict, sample_batch() → (M, B, N, D).

Convention: sample_batch() returns delta / bounds_span (fractional displacement),
matching batched_operators.py K4 convention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .low_rank_matrix import BatchedLowRankMatrixHead


def _apply_low_rank_batched(U: Tensor, vec: Tensor, d: Tensor) -> Tensor:
    """Apply M @ vec = U(U^T vec) + d*vec in factored form.

    Args:
        U:   (B, N, D, r)
        vec: (B, N, D)
        d:   (B, N, D)
    Returns:
        (B, N, D)
    """
    # U^T @ vec: (B, N, r) via einsum over D
    Ut_vec = torch.einsum('bndr,bnd->bnr', U, vec)
    # U @ (U^T vec): (B, N, D) via einsum over r
    low_rank = torch.einsum('bndr,bnr->bnd', U, Ut_vec)
    return low_rank + d * vec


_SOFTPLUS_MAX = 10.0


class BatchedSamplingHead(nn.Module):
    """CMA-ES analog with direction/magnitude separation.

    Reparameterized Gaussian → L2-normalize direction → scale by step_size.
    """

    def __init__(self, routed_dim: int = 160, k_rank: int = 8,
                 embed_dim: int = 128):
        super().__init__()
        self.matrix = BatchedLowRankMatrixHead(routed_dim, k_rank)
        self.mu_proj = nn.Linear(routed_dim, 1)
        self.step_size = nn.Linear(embed_dim, 1)

    def compute_params(self, routed, h_out, coords, fitness, adj,
                       bounds_span=200.0):
        U, d = self.matrix(routed)  # (B,N,D,r), (B,N,D)
        flat = routed.reshape(-1, routed.shape[-1])
        mu = self.mu_proj(flat).reshape(*routed.shape[:3])  # (B, N, D)
        step_scale = F.softplus(self.step_size(h_out)).clamp(max=_SOFTPLUS_MAX)
        return {'mu': mu, 'U': U, 'd': d, 'step_scale': step_scale}

    def sample_batch(self, params, coords, bounds_span, M):
        """Sample M displacements. Returns (M, B, N, D) / bounds_span."""
        mu, U, d = params['mu'], params['U'], params['d']
        step_scale = params['step_scale']  # (B, N, 1)
        B, N, D, r = U.shape

        eps1 = torch.randn(M, B, N, r, device=U.device, dtype=U.dtype)
        eps2 = torch.randn(M, B, N, D, device=U.device, dtype=U.dtype)

        low_rank = torch.einsum('bndr,mbnr->mbnd', U, eps1)
        raw_delta = mu.unsqueeze(0) + low_rank + d.sqrt().unsqueeze(0) * eps2

        # Normalize direction, scale by learned step_size
        dir_norm = raw_delta / (raw_delta.norm(dim=-1, keepdim=True) + 1e-8)

        noise = torch.randn(M, *step_scale.shape, device=step_scale.device,
                            dtype=step_scale.dtype) * 0.1
        scale_noisy = F.softplus(
            torch.log(step_scale.unsqueeze(0).clamp(min=1e-6)) + noise
        ).clamp(max=_SOFTPLUS_MAX)

        delta = scale_noisy * dir_norm
        return delta / bounds_span


class BatchedDifferentialHead(nn.Module):
    """L-SHADE analog: delta = F_scale * M @ (x_pbest - x_i) / bounds_span.

    Matrix-weighted differential mutation with learned PSD matrix.
    """

    def __init__(self, routed_dim: int = 160, k_rank: int = 8,
                 embed_dim: int = 128, p_best: float = 0.1):
        super().__init__()
        self.matrix = BatchedLowRankMatrixHead(routed_dim, k_rank)
        self.scale = nn.Linear(embed_dim, 1)
        self.p_best = p_best

    def compute_params(self, routed, h_out, coords, fitness, adj,
                       bounds_span=200.0):
        U, d = self.matrix(routed)
        F_scale = torch.sigmoid(self.scale(h_out))  # (B, N, 1)

        B, N, D = coords.shape
        p = max(2, int(self.p_best * N))

        _, topk_idx = fitness.topk(p, dim=1, largest=False)
        rand_idx = torch.randint(p, (B, N), device=fitness.device)
        pbest_idx = topk_idx.gather(1, rand_idx)

        x_pbest = coords.gather(1, pbest_idx.unsqueeze(-1).expand(B, N, D))
        diff = (x_pbest - coords).to(U.dtype)

        raw = _apply_low_rank_batched(U, diff, d)
        # L2-normalize direction (like discovery heads)
        delta_base = raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)

        return {'F_scale': F_scale, 'delta_base': delta_base}

    def sample_batch(self, params, coords, bounds_span, M):
        """Returns (M, B, N, D) / bounds_span."""
        F_scale = params['F_scale']
        delta_base = params['delta_base']

        noise = torch.randn(M, *F_scale.shape, device=F_scale.device,
                            dtype=F_scale.dtype) * 0.1
        F_noisy = torch.sigmoid(
            torch.logit(F_scale.unsqueeze(0).clamp(1e-4, 1 - 1e-4)) + noise
        )

        delta = F_noisy * delta_base.unsqueeze(0)
        return delta / bounds_span


class BatchedGradientHead(nn.Module):
    """MTS / quasi-Newton analog: delta = -alpha * norm(M @ grad) / bounds_span.

    Direction L2-normalized, magnitude via clamped softplus.
    """

    def __init__(self, routed_dim: int = 160, k_rank: int = 8,
                 embed_dim: int = 128):
        super().__init__()
        self.matrix = BatchedLowRankMatrixHead(routed_dim, k_rank)
        self.lr_scale = nn.Linear(embed_dim, 1)

    def compute_params(self, routed, h_out, coords, fitness, adj_or_knn,
                       bounds_span=200.0):
        U, d = self.matrix(routed)
        alpha = F.softplus(self.lr_scale(h_out)).clamp(max=_SOFTPLUS_MAX)

        if adj_or_knn.dim() == 3 and adj_or_knn.dtype == torch.long:
            grad = self._estimate_gradient_sparse(coords, fitness, adj_or_knn)
        else:
            grad = self._estimate_gradient_dense(coords, fitness, adj_or_knn)
        grad = grad.to(U.dtype)

        raw_direction = -_apply_low_rank_batched(U, grad, d)
        # L2-normalize direction (like discovery heads)
        direction = raw_direction / (raw_direction.norm(dim=-1, keepdim=True) + 1e-8)

        return {'alpha': alpha, 'direction': direction}

    def _estimate_gradient_dense(self, coords, fitness, adj):
        """Estimate gradient from neighbor fitness diffs using dense adj."""
        B, N, D = coords.shape
        coord_diff = coords.unsqueeze(1) - coords.unsqueeze(2)
        fit_diff = fitness.unsqueeze(1) - fitness.unsqueeze(2)
        dist_sq = (coord_diff ** 2).sum(dim=-1).clamp(min=1e-8)

        edge_grad = (fit_diff / dist_sq).unsqueeze(-1) * coord_diff

        adj_f = adj.float().unsqueeze(-1)
        degree = adj.float().sum(dim=2, keepdim=True).clamp(min=1)

        grad = (adj_f * edge_grad).sum(dim=2) / degree
        return grad

    def _estimate_gradient_sparse(self, coords, fitness, knn_idx):
        """Estimate gradient from neighbor fitness diffs using knn_idx (B,N,k)."""
        B, N, D = coords.shape
        k = knn_idx.shape[2]

        # Gather neighbor coords and fitness: (B, N, k, D) and (B, N, k)
        idx_expand = knn_idx.unsqueeze(-1).expand(B, N, k, D)
        neigh_coords = coords.gather(1, idx_expand.reshape(B, N * k, D)).reshape(B, N, k, D)
        neigh_fitness = fitness.gather(1, knn_idx.reshape(B, N * k)).reshape(B, N, k)

        coord_diff = neigh_coords - coords.unsqueeze(2)  # (B, N, k, D)
        fit_diff = neigh_fitness - fitness.unsqueeze(2)    # (B, N, k)
        dist_sq = (coord_diff ** 2).sum(dim=-1).clamp(min=1e-8)  # (B, N, k)

        edge_grad = (fit_diff / dist_sq).unsqueeze(-1) * coord_diff  # (B, N, k, D)
        grad = edge_grad.mean(dim=2)  # (B, N, D) — k neighbors, uniform weight
        return grad

    def sample_batch(self, params, coords, bounds_span, M):
        """Returns (M, B, N, D) / bounds_span."""
        alpha = params['alpha']
        direction = params['direction']

        noise = torch.randn(M, *alpha.shape, device=alpha.device,
                            dtype=alpha.dtype) * 0.1
        alpha_noisy = F.softplus(
            torch.log(alpha.unsqueeze(0).clamp(min=1e-6)) + noise
        ).clamp(max=_SOFTPLUS_MAX)

        delta = alpha_noisy * direction.unsqueeze(0)
        return delta / bounds_span
