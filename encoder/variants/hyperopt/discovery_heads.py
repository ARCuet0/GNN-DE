"""Batched discovery heads: NeighborRecomb, CoordField, Hypernet.

Genuinely novel operators that can learn mixing outside known algorithm vocab.
All ops natively batched (B, N, D) — no loops over B, no scatter.
Interface: compute_params() -> dict, sample_batch() -> (M, B, N, D).

Convention: sample_batch() returns delta / bounds_span (fractional displacement),
matching batched_operators.py K4 convention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .per_dim_features import compute_per_dim_features_batched

_SOFTPLUS_MAX = 10.0


def _normalize_and_scale(direction: Tensor, step_scale: Tensor) -> Tensor:
    """Normalize direction to unit norm, scale by learned step size.

    Args:
        direction: (B, N, D) unnormalized displacement
        step_scale: (B, N, 1) positive scalar
    Returns:
        (B, N, D)
    """
    dir_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    return step_scale * dir_norm


def _knn_to_attn_mask(knn_idx, N):
    """Convert knn_idx (B, N, k) to attention mask (B, N, N) bool.

    Returns mask where True = masked (non-neighbor), matching ~adj convention.
    """
    B, N_nodes, k = knn_idx.shape
    # Build adj from knn_idx, then invert
    adj = torch.zeros(B, N_nodes, N_nodes, dtype=torch.bool, device=knn_idx.device)
    batch_idx = torch.arange(B, device=knn_idx.device).unsqueeze(1).unsqueeze(2).expand_as(knn_idx)
    node_idx = torch.arange(N_nodes, device=knn_idx.device).unsqueeze(0).unsqueeze(2).expand_as(knn_idx)
    adj[batch_idx, node_idx, knn_idx] = True
    return ~adj


class BatchedNeighborRecombHead(nn.Module):
    """Population-conditioned displacement via adj-masked cross-attention.

    Uses dense adj as attention mask — no edge_index, no scatter.
    Query = each node, Key/Value = all nodes masked by adjacency.
    """

    def __init__(self, embed_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, n_heads, batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dir_proj = nn.Linear(embed_dim, 1)
        self.step_size = nn.Linear(embed_dim, 1)

    def compute_params(self, routed, h_out, coords, fitness, adj_or_knn,
                       bounds_span=200.0):
        """
        Args:
            routed:     (B, N, D, routed_dim) [unused — uses h_out directly]
            h_out:      (B, N, embed_dim)
            adj_or_knn: (B, N, N) bool adj OR (B, N, k) long knn_idx
        Returns:
            dict with direction (B,N,D), step_scale (B,N,1)
        """
        B, N, E = h_out.shape
        D = coords.shape[-1]

        # Build attention mask: True = masked (non-neighbor)
        if adj_or_knn.dim() == 3 and adj_or_knn.dtype == torch.long:
            attn_mask = _knn_to_attn_mask(adj_or_knn, N)
        else:
            attn_mask = ~adj_or_knn
        # MHA with batch_first expects: (B, tgt_len, src_len) for attn_mask
        # But MHA only supports 2D (tgt, src) or 3D (B*nheads, tgt, src)
        n_heads = self.cross_attn.num_heads
        attn_mask_expanded = attn_mask.unsqueeze(1).expand(
            B, n_heads, N, N
        ).reshape(B * n_heads, N, N)

        # Cross-attend: each node attends to neighbors
        attn_out, _ = self.cross_attn(
            h_out, h_out, h_out,
            attn_mask=attn_mask_expanded,
        )  # (B, N, E)
        combined = self.norm(attn_out + h_out)  # (B, N, E)

        # Project to per-dimension displacement
        combined_broad = combined.unsqueeze(2).expand(B, N, D, E)
        flat = combined_broad.reshape(B * N * D, E)
        direction = self.dir_proj(flat).reshape(B, N, D)  # (B, N, D)

        step_scale = F.softplus(self.step_size(h_out)).clamp(max=_SOFTPLUS_MAX)

        return {'direction': direction, 'step_scale': step_scale}

    def sample_batch(self, params, coords, bounds_span, M):
        """Returns (M, B, N, D) / bounds_span."""
        direction = params['direction']
        step_scale = params['step_scale']

        noise = torch.randn(M, *step_scale.shape, device=step_scale.device,
                            dtype=step_scale.dtype) * 0.1
        scale_noisy = F.softplus(
            torch.log(step_scale.unsqueeze(0).clamp(min=1e-6)) + noise
        ).clamp(max=_SOFTPLUS_MAX)

        dir_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        delta = scale_noisy * dir_norm.unsqueeze(0)
        return delta / bounds_span


class BatchedCoordFieldHead(nn.Module):
    """Dimension-aware neural field on routed cross-dim representations.

    Each dimension's routed repr (96-d) through shared MLP -> direction.
    Trivially batchable — all dense ops.
    """

    def __init__(self, routed_dim: int = 160, hidden: int = 64,
                 embed_dim: int = 128):
        super().__init__()
        self.field_mlp = nn.Sequential(
            nn.Linear(routed_dim, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, 1),
        )
        self.step_size = nn.Linear(embed_dim, 1)

    def compute_params(self, routed, h_out, coords, fitness, adj,
                       bounds_span=200.0):
        """
        Args:
            routed: (B, N, D, routed_dim)
            h_out:  (B, N, embed_dim)
        Returns:
            dict with direction (B,N,D), step_scale (B,N,1)
        """
        shape = routed.shape[:3]  # (B, N, D)
        flat = routed.reshape(-1, routed.shape[-1])
        direction = self.field_mlp(flat).reshape(*shape)  # (B, N, D)
        step_scale = F.softplus(self.step_size(h_out)).clamp(max=_SOFTPLUS_MAX)
        return {'direction': direction, 'step_scale': step_scale}

    def sample_batch(self, params, coords, bounds_span, M):
        """Returns (M, B, N, D) / bounds_span."""
        direction = params['direction']
        step_scale = params['step_scale']

        noise = torch.randn(M, *step_scale.shape, device=step_scale.device,
                            dtype=step_scale.dtype) * 0.1
        scale_noisy = F.softplus(
            torch.log(step_scale.unsqueeze(0).clamp(min=1e-6)) + noise
        ).clamp(max=_SOFTPLUS_MAX)

        dir_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        delta = scale_noisy * dir_norm.unsqueeze(0)
        return delta / bounds_span


class BatchedHypernetHead(nn.Module):
    """Per-individual hypernetwork: generates W1, b1, W2, b2 from h_out.

    Applied to per-dim features via batched matmul. Spectral norm on generators.
    """

    def __init__(self, embed_dim: int = 128, n_perdim: int = 5,
                 bottleneck: int = 16):
        super().__init__()
        self.bottleneck = bottleneck
        self.n_perdim = n_perdim

        # Plain linears — spectral_norm is vmap-incompatible (aten::div.out).
        # Weight magnitude is controlled by small init + optimizer clipping.
        self.w1_gen = nn.Linear(embed_dim, n_perdim * bottleneck)
        self.b1_gen = nn.Linear(embed_dim, bottleneck)
        self.w2_gen = nn.Linear(embed_dim, bottleneck)
        # Small init to prevent hypernet explosion at start
        for lin in [self.w1_gen, self.b1_gen, self.w2_gen]:
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.zeros_(lin.bias)
        self.b2_gen = nn.Linear(embed_dim, 1)
        self.step_size = nn.Linear(embed_dim, 1)

    def compute_params(self, routed, h_out, coords, fitness, adj,
                       bounds_span=200.0):
        """
        Args:
            routed: (B, N, D, routed_dim) [unused — uses h_out + coords]
            h_out:  (B, N, embed_dim)
            coords: (B, N, D)
            fitness: (B, N)
        Returns:
            dict with direction (B,N,D), step_scale (B,N,1)
        """
        B, N, E = h_out.shape
        D = coords.shape[-1]

        # Per-dim features
        per_dim = compute_per_dim_features_batched(
            coords, fitness
        ).to(h_out.dtype)  # (B, N, D, 5)

        # Generate per-individual weights: flatten B*N
        h_flat = h_out.reshape(B * N, E)
        W1 = self.w1_gen(h_flat).view(B * N, self.bottleneck, self.n_perdim)
        b1 = self.b1_gen(h_flat).unsqueeze(1)  # (B*N, 1, bot)
        W2 = self.w2_gen(h_flat).unsqueeze(-1)  # (B*N, bot, 1)
        b2 = self.b2_gen(h_flat).unsqueeze(1)  # (B*N, 1, 1)

        # Apply generated net per dimension via bmm
        per_dim_flat = per_dim.reshape(B * N, D, self.n_perdim)  # (B*N, D, 5)
        h1 = torch.bmm(per_dim_flat, W1.transpose(1, 2))  # (B*N, D, bot)
        h1 = F.leaky_relu(h1 + b1, 0.1)
        direction = (torch.bmm(h1, W2) + b2).squeeze(-1)  # (B*N, D)
        direction = direction.reshape(B, N, D)

        step_scale = F.softplus(self.step_size(h_out)).clamp(max=_SOFTPLUS_MAX)

        return {'direction': direction, 'step_scale': step_scale}

    def sample_batch(self, params, coords, bounds_span, M):
        """Returns (M, B, N, D) / bounds_span."""
        direction = params['direction']
        step_scale = params['step_scale']

        noise = torch.randn(M, *step_scale.shape, device=step_scale.device,
                            dtype=step_scale.dtype) * 0.1
        scale_noisy = F.softplus(
            torch.log(step_scale.unsqueeze(0).clamp(min=1e-6)) + noise
        ).clamp(max=_SOFTPLUS_MAX)

        dir_norm = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        delta = scale_noisy * dir_norm.unsqueeze(0)
        return delta / bounds_span
