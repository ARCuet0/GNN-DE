"""
graph_builder_sparse_delta.py — augmented-pop graph via parent inheritance.

Problem this solves: surrogate scoring (`opt_variant.py:_run_surrogate`)
rebuilds the graph on (B, N + M*N*K, D) augmented coordinates. Even with the
sparse path that's still an O(N_aug^2) cdist via _chunked_knn (chunked but
O(N²) total compute).

Solution under the lema: each proposal `p = (m, n, k_idx)` is a small delta
from parent `n`; its kNN neighborhood is dominated by parent `n` and parent
`n`'s kNN. So we:

  * Inherit parent's kNN: knn_idx[N + p] = [n, knn_idx[n][:k-1]].
  * Inherit parent's edge_feat / node_feat by gather (proxies; edge e0 can be
    rescaled by ||delta|| if needed).

Cost: O(N_aug * k) — strictly O(N_aug). Parent rows are unchanged; new rows
are populated via N gather + concat.

This is an APPROXIMATION. Phase 0 / 1 microbench validates the cost; phase 2
parity gate (D=10/D=30 vs cdist-sparse) validates whether the approximation
hurts downstream scoring quality. Fallback path under flag
`surrogate_augment_strategy='rebuild'` uses TorchNNDescentKNN on the full
N_aug — still O(N_aug * k) per the lema, but a real kNN.
"""
import torch

from .sparse_gatv2_backbone import SparseTopologyCache


def augment_sparse_cache(cache: SparseTopologyCache,
                         deltas: torch.Tensor,
                         coords: torch.Tensor,
                         fitness: torch.Tensor) -> SparseTopologyCache:
    """Augment a SparseTopologyCache with proposal rows via parent inheritance.

    Args:
        cache:    parent SparseTopologyCache with knn_idx (B,N,k),
                  edge_feat (B,N,k,4), node_feat (B,N,8), global_feat (B,16).
        deltas:   (M, B, N, K, D) proposal deltas.
        coords:   (B, N, D) parent coordinates.
        fitness:  (B, N) parent fitness.

    Returns:
        SparseTopologyCache with N_aug = N + M*N*K rows.

    Layout convention (must match opt_variant.py:_run_surrogate's flat
    layout): for new-row position p ∈ [0, M*N*K),
        m, rest = divmod(p, N*K); n, k_idx = divmod(rest, K)
    so the parent of new-row (N + p) is `n`.
    """
    B, N, k = cache.knn_idx.shape
    M, _, _, K, D = deltas.shape
    device = cache.knn_idx.device
    N_prop = M * N * K
    N_aug = N + N_prop

    # Parent index per new row, in the flat layout (matches opt_variant.py).
    # parent_of_new[p] = (p // K) % N for p in [0, M*N*K).
    parent_idx = (
        torch.arange(N_prop, device=device) // K) % N            # (N_prop,)
    parent_idx_b = parent_idx.view(1, N_prop, 1).expand(B, N_prop, 1)  # (B,N_prop,1)

    # Inherit parent's kNN (drop last neighbor to make room for parent itself).
    # Gather parent rows from cache.knn_idx. shape: (B, N_prop, k)
    parent_knn = cache.knn_idx.gather(
        1, parent_idx.view(1, N_prop, 1).expand(B, N_prop, k))     # (B, N_prop, k)
    # Replace position 0 with parent index (so parent is always the first NN
    # of its proposal); shift others right.
    inherited_knn = torch.cat(
        [parent_idx_b, parent_knn[..., :k - 1]], dim=-1)           # (B, N_prop, k)

    new_knn = torch.cat([cache.knn_idx, inherited_knn], dim=1)     # (B, N_aug, k)

    # Inherit edge_feat from parent (cheap proxy).
    edge_dim = cache.edge_feat.shape[-1]
    parent_edge = cache.edge_feat.gather(
        1, parent_idx.view(1, N_prop, 1, 1).expand(B, N_prop, k, edge_dim))
    new_edge = torch.cat([cache.edge_feat, parent_edge], dim=1)

    # Inherit node_feat from parent.
    node_dim = cache.node_feat.shape[-1]
    parent_node = cache.node_feat.gather(
        1, parent_idx.view(1, N_prop, 1).expand(B, N_prop, node_dim))
    new_node = torch.cat([cache.node_feat, parent_node], dim=1)

    # Global features: unchanged (population-level).
    return SparseTopologyCache(
        knn_idx=new_knn,
        edge_feat=new_edge,
        B=B, N=N_aug, k=k,
        node_feat=new_node,
        global_feat=cache.global_feat,
        alive=cache.alive,
    )
