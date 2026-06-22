"""
topology_strategies.py — Three graph topology strategies for sparse GATv2.

All strategies produce knn_idx (B, N, k) — neighbor indices per node.
Self-loops are excluded. Chunked computation for large N.

Strategies:
    CoordinateKNN:  kNN in coordinate space (cdist + topk)
    EmbeddingKNN:   kNN in learned embedding space (zero parameters)
    LearnedScorer:  q/k projections → attention scores → topk
"""
import math

import torch
import torch.nn as nn


def _exclude_self_diagonal(matrix: torch.Tensor, start: int, end: int,
                           fill_value: float) -> None:
    """Vectorized self-exclusion for a chunk of rows [start, end)."""
    device = matrix.device
    local_idx = torch.arange(end - start, device=device)
    global_idx = torch.arange(start, end, device=device)
    matrix[:, local_idx, global_idx] = fill_value


def _chunked_knn(data: torch.Tensor, k: int, chunk_size: int = 512) -> torch.Tensor:
    """Compute kNN indices with chunked cdist to limit peak memory.

    Args:
        data: (B, N, D) — points in some space
        k: number of neighbors

    Returns:
        knn_idx: (B, N, k) long — neighbor indices (self excluded)
    """
    B, N, D = data.shape
    device = data.device

    if N <= chunk_size:
        dist = torch.cdist(data, data)  # (B, N, N)
        _exclude_self_diagonal(dist, 0, N, float('inf'))
        _, idx = dist.topk(k, dim=-1, largest=False)
        return idx

    knn_idx = torch.empty(B, N, k, dtype=torch.long, device=device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        dist_chunk = torch.cdist(data[:, start:end], data)  # (B, chunk, N)
        _exclude_self_diagonal(dist_chunk, start, end, float('inf'))
        _, idx_chunk = dist_chunk.topk(k, dim=-1, largest=False)
        knn_idx[:, start:end] = idx_chunk

    return knn_idx


class DistanceKNN(nn.Module):
    """kNN via Euclidean distance in any space. Zero learnable parameters."""

    def __init__(self, k: int = 8, chunk_size: int = 512):
        super().__init__()
        self.k = k
        self.chunk_size = chunk_size

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Args: data (B, N, D) — coordinates or embeddings. Returns: knn_idx (B, N, k)."""
        return _chunked_knn(data, self.k, self.chunk_size)


# Semantic aliases — same implementation, different intent
CoordinateKNN = DistanceKNN
EmbeddingKNN = DistanceKNN


class LearnedScorer(nn.Module):
    """Learned edge scorer with query-key projections.

    Computes compatibility scores via q·k dot product, then takes top-k.
    The topology is learned end-to-end (gradients flow through q_proj, k_proj
    via the attention scores used in downstream layers, not through topk itself).
    """

    def __init__(self, d_in: int, k: int = 8, d_k: int = 16,
                 chunk_size: int = 512):
        super().__init__()
        self.k = k
        self.d_k = d_k
        self.chunk_size = chunk_size
        self.q_proj = nn.Linear(d_in, d_k)
        self.k_proj = nn.Linear(d_in, d_k)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Args: h (B, N, d_in). Returns: knn_idx (B, N, k)."""
        B, N, _ = h.shape
        q = self.q_proj(h)  # (B, N, d_k)
        k_out = self.k_proj(h)  # (B, N, d_k)
        scale = math.sqrt(self.d_k)
        device = h.device

        if N <= self.chunk_size:
            scores = torch.bmm(q, k_out.transpose(1, 2)) / scale  # (B, N, N)
            _exclude_self_diagonal(scores, 0, N, float('-inf'))
            _, idx = scores.topk(self.k, dim=-1)
            return idx

        knn_idx = torch.empty(B, N, self.k, dtype=torch.long, device=device)
        for start in range(0, N, self.chunk_size):
            end = min(start + self.chunk_size, N)
            scores_chunk = torch.bmm(
                q[:, start:end], k_out.transpose(1, 2)) / scale
            _exclude_self_diagonal(scores_chunk, start, end, float('-inf'))
            _, idx_chunk = scores_chunk.topk(self.k, dim=-1)
            knn_idx[:, start:end] = idx_chunk

        return knn_idx


def build_topology(mode, *, k: int = 8, d_in: int = 0, d_k: int = 16,
                   chunk_size: int = 512, knn_n_iters: int = 3,
                   knn_fallback_n: int = 64, knn_seed: int = 0):
    """Factory function for topology strategies."""
    from .sparse_gatv2_backbone import TopologyMode

    if mode == TopologyMode.COORDINATE_KNN:
        return CoordinateKNN(k=k, chunk_size=chunk_size)
    elif mode == TopologyMode.EMBEDDING_KNN:
        return EmbeddingKNN(k=k, chunk_size=chunk_size)
    elif mode == TopologyMode.LEARNED_SCORER:
        return LearnedScorer(d_in=d_in, k=k, d_k=d_k, chunk_size=chunk_size)
    elif mode == TopologyMode.TORCH_KNN:
        # D1000 line: strict-O(N*k) approximate kNN via NN-Descent.
        from .topology_strategies_torch_knn import TorchNNDescentKNN
        return TorchNNDescentKNN(k=k, n_iters=knn_n_iters,
                                 fallback_n=knn_fallback_n, seed=knn_seed)
    else:
        raise ValueError(f"Unknown topology mode: {mode}")
