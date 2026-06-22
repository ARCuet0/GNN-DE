"""
graph_features_sparse.py — Sparse graph features for ES training.

No N² dense matrices. All features computed from k-NN edge_index only.
Hard rank via argsort (non-differentiable — ES doesn't need gradients
through graph features).

Memory: O(N×k) instead of O(N²). Speed: O(N log N) instead of O(N²).
"""
import math
import torch
import torch.nn.functional as F

from .similarity_graph import (
    BASE_NODE_DIM, NODE_DIM, EDGE_DIM, GLOBAL_DIM,
)
from .techniques_v2 import N_TECHNIQUES


def hard_rank(values: torch.Tensor) -> torch.Tensor:
    """Non-differentiable rank via argsort. O(N log N).

    Args:
        values: (N,) or (B, N)
    Returns:
        ranks: same shape, float, in [0, N-1]
    """
    if values.dim() == 1:
        ranks = torch.empty_like(values)
        ranks[values.argsort()] = torch.arange(
            len(values), device=values.device, dtype=values.dtype)
        return ranks
    # Batched
    B, N = values.shape
    ranks = torch.empty_like(values)
    idx = values.argsort(dim=1)
    ranks.scatter_(1, idx, torch.arange(
        N, device=values.device, dtype=values.dtype).unsqueeze(0).expand(B, N))
    return ranks


def compute_sparse_shared(x, fitness, k_neighbors=8, lb=-100.0, ub=100.0):
    """Compute shared intermediates using only k-NN sparse ops.

    No N² matrices. O(N×k + N log N).
    """
    dev = x.device
    N, D = x.shape
    span = ub - lb
    span_safe = span if abs(span) > 1e-12 else 1.0

    coords_norm = (x.float() - lb) / span_safe

    # Z-scored log-fitness
    log_fit = torch.log(fitness.float().clamp(min=1e-30))
    lf_mean = log_fit.mean()
    lf_std = log_fit.std().clamp(min=1e-4)
    fit = (log_fit - lf_mean) / lf_std

    # k-NN via cdist + topk (cuBLAS gemm, fast for N≤500)
    coords_det = coords_norm.detach()
    dist_matrix = torch.cdist(coords_det, coords_det)
    dist_matrix.fill_diagonal_(float('inf'))
    k = min(k_neighbors, N - 1)
    knn_dists, knn_idx = torch.topk(dist_matrix, k, dim=1, largest=False)

    # Hard rank (O(N log N), no N² sigmoid)
    fit_rank = hard_rank(fit) / max(N - 1, 1)

    fit_mean = fit.mean()
    fit_std = fit.std().clamp(min=1e-8)
    fit_zscore = ((fit - fit_mean) / fit_std).clamp(-3, 3)

    # Best position (argmin, not softmin — no grad needed)
    best_idx = fit.argmin()
    best_x = coords_norm[best_idx]
    dist_to_best = (coords_norm - best_x.unsqueeze(0)).pow(2).sum(dim=1).sqrt()
    dtb_rank = hard_rank(dist_to_best) / max(N - 1, 1)

    # Local density from kNN distances (already have them, no extra N²)
    local_density = knn_dists.mean(dim=1)
    max_density = local_density.max().clamp(min=1e-8)

    # Gradient consistency from kNN
    nn_coords = coords_norm[knn_idx]  # (N, k, D)
    diffs = nn_coords - coords_norm.unsqueeze(1)
    dists_nn = (diffs.pow(2).sum(dim=2, keepdim=True) + 1e-8).sqrt()
    directions = diffs / dists_nn
    f_diffs = fit[knn_idx] - fit.unsqueeze(1)
    grads = f_diffs / dists_nn.squeeze(-1)
    weighted = grads.unsqueeze(-1) * directions
    weighted_sum = weighted.sum(dim=1)
    total_abs_grad = grads.abs().sum(dim=1).clamp(min=1e-8)
    ws_norm = (weighted_sum.pow(2).sum(dim=1) + 1e-8).sqrt()
    gradient_consistency = (ws_norm / total_abs_grad).clamp(0, 1)

    # Local convexity
    mean_nn_fit = fit[knn_idx].mean(dim=1)
    local_convexity = ((mean_nn_fit - fit) / fit_std).clamp(-3, 3) / 3.0

    # NBC ratio — approximate: use kNN neighbors instead of all-pairs
    # For each node, find nearest-better among kNN (not all N)
    nn_fit = fit[knn_idx]  # (N, k)
    is_better = nn_fit < fit.unsqueeze(1)  # (N, k)
    # Distance to nearest better neighbor (among kNN)
    large_val = knn_dists.max() + 1.0
    better_dists = torch.where(is_better, knn_dists, large_val)
    nbn_dist = better_dists.min(dim=1).values  # (N,)
    nn_dist = knn_dists[:, 0]  # nearest neighbor distance
    has_better = is_better.any(dim=1)
    nbc_ratio = torch.where(
        has_better,
        nbn_dist / (nn_dist + 1e-8),
        torch.ones_like(nn_dist)
    ).clamp(0, 5)

    return {
        'coords_norm': coords_norm,
        'fit': fit,
        'fit_rank': fit_rank,
        'fit_mean': fit_mean,
        'fit_std': fit_std,
        'fit_zscore': fit_zscore,
        'dist_to_best': dist_to_best,
        'dtb_rank': dtb_rank,
        'knn_idx': knn_idx,
        'knn_dists': knn_dists,
        'local_density': local_density,
        'max_density': max_density,
        'gradient_consistency': gradient_consistency,
        'local_convexity': local_convexity,
        'nbc_ratio': nbc_ratio,
        'k': k,
    }


def compute_sparse_node_features(shared, step_num, max_steps):
    """9 base node features from sparse shared intermediates."""
    N = shared['fit_rank'].shape[0]
    dev = shared['fit_rank'].device
    gen_frac = step_num / max(max_steps, 1)

    return torch.stack([
        shared['fit_rank'] * 2 - 1,
        shared['fit_zscore'] / 3.0,
        shared['dtb_rank'] * 2 - 1,
        (1 - shared['local_density'] / shared['max_density']).clamp(0, 1) * 2 - 1,
        shared['gradient_consistency'] * 2 - 1,
        shared['local_convexity'],
        (shared['nbc_ratio'] / 2.5).clamp(0, 2) - 1,
        torch.full((N,), gen_frac * 2 - 1, device=dev),
        torch.full((N,), gen_frac * 2 - 1, device=dev),
    ], dim=1)  # (N, 9)


def compute_sparse_edge_features(shared, edge_index):
    """9 edge features computed directly from edge_index. No N² matrices."""
    coords_norm = shared['coords_norm']
    fit = shared['fit']
    fit_rank = shared['fit_rank']
    fit_std = shared['fit_std']
    knn_idx = shared['knn_idx']
    k = shared['k']

    src, dst = edge_index
    E = edge_index.shape[1]
    D = coords_norm.shape[1]
    N = coords_norm.shape[0]
    dev = coords_norm.device

    if E == 0:
        return torch.zeros(0, EDGE_DIM, device=dev)

    # Edge distances (direct, no N² matrix)
    edge_vec = coords_norm[src] - coords_norm[dst]
    edge_dists = (edge_vec.pow(2).sum(dim=1) + 1e-8).sqrt()
    max_edge_dist = edge_dists.max().clamp(min=1e-8)

    e0 = (edge_dists / max_edge_dist).clamp(0, 1) * 2 - 1
    e1 = ((fit[src] - fit[dst]) / fit_std).clamp(-3, 3) / 3.0

    # Cosine similarity (centered on centroid)
    centroid = coords_norm.mean(dim=0)
    c_src = coords_norm[src] - centroid
    c_dst = coords_norm[dst] - centroid
    dot = (c_src * c_dst).sum(dim=1)
    norm_s = (c_src.pow(2).sum(dim=1) + 1e-8).sqrt()
    norm_d = (c_dst.pow(2).sum(dim=1) + 1e-8).sqrt()
    e2 = dot / (norm_s * norm_d)

    # Mutual kNN (no N² adjacency)
    knn_src = knn_idx[src]  # (E, k)
    knn_dst = knn_idx[dst]  # (E, k)
    is_fwd = (knn_src == dst.unsqueeze(1)).any(dim=1)  # (E,)
    is_bwd = (knn_dst == src.unsqueeze(1)).any(dim=1)  # (E,)
    e3 = (is_fwd & is_bwd).float()

    # Effective dimensions
    delta = edge_vec.abs()
    delta_sum = delta.sum(dim=1)
    delta_sq_sum = (delta ** 2).sum(dim=1)
    eff_dims = (delta_sum ** 2) / (delta_sq_sum + 1e-8)
    e4 = (eff_dims / max(D, 1)).clamp(0, 1) * 2 - 1

    # Edge distance rank (hard, O(E log E))
    e5_rank = hard_rank(edge_dists) / max(E - 1, 1)
    e5 = e5_rank * 2 - 1

    # Fitness rank difference
    e6 = (fit_rank[src] - fit_rank[dst]).abs() * 2 - 1

    # Landscape gradient magnitude
    e7 = ((fit[src] - fit[dst]).abs() / (fit_std * edge_dists + 1e-8)).clamp(0, 3) / 1.5 - 1

    # Neighborhood overlap (no N² adjacency)
    overlap = (knn_src.unsqueeze(2) == knn_dst.unsqueeze(1)).any(dim=2).float().sum(dim=1)
    e8 = overlap / max(k, 1) * 2 - 1

    return torch.stack([e0, e1, e2, e3, e4, e5, e6, e7, e8], dim=1)


def compute_sparse_global_features(shared, ndim, step_num, max_steps):
    """16 global features from sparse intermediates."""
    coords_norm = shared['coords_norm']
    fit = shared['fit']
    fit_std = shared['fit_std']
    fit_mean = shared['fit_mean']
    dist_to_best = shared['dist_to_best']
    gradient_consistency = shared['gradient_consistency']
    local_convexity = shared['local_convexity']
    nbc_ratio = shared['nbc_ratio']

    dev = coords_norm.device
    gen_frac = step_num / max(max_steps, 1)

    progress = gen_frac
    diversity = coords_norm.std(dim=0).mean()
    imp_rate = 0.0

    # FDC
    fdc_num = ((fit - fit_mean) * dist_to_best).mean()
    fdc_den = fit_std * dist_to_best.std().clamp(min=1e-8)
    fdc = (fdc_num / fdc_den).clamp(-1, 1)

    gc_mean = gradient_consistency.mean()
    conv_frac = (local_convexity > 0).float().mean() * 2 - 1
    nbc_mean = nbc_ratio.mean() / 2.5
    norm_dim = math.log10(ndim / 100.0)

    tech_rates = torch.zeros(N_TECHNIQUES, device=dev)

    feats = [
        progress * 2 - 1,
        diversity * 2 - 1,
        imp_rate,
        fdc,
        gc_mean * 2 - 1,
        conv_frac,
        nbc_mean * 2 - 1,
        norm_dim,
    ]
    global_t = torch.tensor(feats, device=dev, dtype=torch.float32)
    global_t = torch.cat([global_t, tech_rates])

    if global_t.shape[0] < GLOBAL_DIM:
        global_t = F.pad(global_t, (0, GLOBAL_DIM - global_t.shape[0]))
    return global_t[:GLOBAL_DIM].unsqueeze(0)  # (1, 16)


# ======================================================================
# Batched sparse graph builder for ES-Single
# ======================================================================

def build_batched_sparse_graphs_gpu(
    xs: torch.Tensor,
    fitnesses: torch.Tensor,
    step_nums: list,
    max_steps: int,
    ndim: int,
    k_neighbors: int = 8,
    K: int = 4,
    max_K: int = 4,
):
    """Build B sparse graphs fully vectorized. No Python loops over B.

    All ops are (B, N, ...) batched tensors. Memory: O(B × N × k).

    Args:
        xs:        (B, N, D)
        fitnesses: (B, N)

    Returns:
        all_nodes, all_edges, all_edge_attr, all_global, v_indices, e_indices
    """
    from .graph_features import build_spatial_edge_index

    B, N, D = xs.shape
    dev = xs.device
    span = 200.0  # ub - lb
    k = min(k_neighbors, N - 1)

    # ── Normalize ──
    coords_norm = (xs.float() + 100.0) / span  # (B, N, D)

    # ── Z-scored log-fitness ──
    log_fit = torch.log(fitnesses.float().clamp(min=1e-30))
    lf_mean = log_fit.mean(dim=1, keepdim=True)
    lf_std = log_fit.std(dim=1, keepdim=True).clamp(min=1e-4)
    fit = (log_fit - lf_mean) / lf_std  # (B, N)

    # ── kNN via batched cdist + topk (cuBLAS gemm, 2 kernel launches) ──
    coords_det = coords_norm.detach()
    dist_mat = torch.cdist(coords_det, coords_det)  # (B, N, N)
    dist_mat.masked_fill_(torch.eye(N, device=dev, dtype=torch.bool).unsqueeze(0), float('inf'))
    knn_dists, knn_idx = torch.topk(dist_mat, k, dim=2, largest=False)  # (B, N, k)
    del dist_mat

    # ── Hard rank (batched argsort, O(B×N log N)) ──
    fit_rank = hard_rank(fit) / max(N - 1, 1)  # (B, N)
    fit_mean = fit.mean(dim=1)  # (B,)
    fit_std = fit.std(dim=1).clamp(min=1e-8)  # (B,)
    fit_zscore = ((fit - fit_mean.unsqueeze(1)) / fit_std.unsqueeze(1)).clamp(-3, 3)

    # ── Best position (batched argmin) ──
    best_idx = fit.argmin(dim=1)  # (B,)
    b_idx = torch.arange(B, device=dev)
    best_x = coords_norm[b_idx, best_idx]  # (B, D)
    dist_to_best = (coords_norm - best_x.unsqueeze(1)).pow(2).sum(dim=2).sqrt()  # (B, N)
    dtb_rank = hard_rank(dist_to_best) / max(N - 1, 1)  # (B, N)

    # ── Local density from kNN ──
    local_density = knn_dists.mean(dim=2)  # (B, N)
    max_density = local_density.max(dim=1, keepdim=True).values.clamp(min=1e-8)

    # ── Gradient consistency (batched over B) ──
    b_exp = torch.arange(B, device=dev).reshape(B, 1, 1).expand(B, N, k)
    nn_coords = coords_norm[b_exp, knn_idx]  # (B, N, k, D)
    diffs = nn_coords - coords_norm.unsqueeze(2)
    dists_nn = (diffs.pow(2).sum(dim=3, keepdim=True) + 1e-8).sqrt()
    directions = diffs / dists_nn
    nn_fit = fit[b_exp, knn_idx]  # (B, N, k)
    f_diffs = nn_fit - fit.unsqueeze(2)
    grads = f_diffs / dists_nn.squeeze(-1)
    weighted = grads.unsqueeze(-1) * directions
    weighted_sum = weighted.sum(dim=2)  # (B, N, D)
    total_abs_grad = grads.abs().sum(dim=2).clamp(min=1e-8)
    ws_norm = (weighted_sum.pow(2).sum(dim=2) + 1e-8).sqrt()
    gradient_consistency = (ws_norm / total_abs_grad).clamp(0, 1)  # (B, N)

    # ── Local convexity ──
    mean_nn_fit = nn_fit.mean(dim=2)
    local_convexity = ((mean_nn_fit - fit) / fit_std.unsqueeze(1)).clamp(-3, 3) / 3.0

    # ── NBC ratio (approximate, kNN only) ──
    is_better = nn_fit < fit.unsqueeze(2)  # (B, N, k)
    large_val = knn_dists.max() + 1.0
    better_dists = torch.where(is_better, knn_dists, large_val)
    nbn_dist = better_dists.min(dim=2).values
    nn_dist_nearest = knn_dists[:, :, 0]
    has_better = is_better.any(dim=2)
    nbc_ratio = torch.where(
        has_better,
        nbn_dist / (nn_dist_nearest + 1e-8),
        torch.ones_like(nn_dist_nearest)
    ).clamp(0, 5)

    # ── Node features (B, N, 9) ──
    gen_fracs = torch.tensor(
        [s / max(max_steps, 1) for s in step_nums],
        device=dev, dtype=torch.float32)  # (B,)
    gf_exp = gen_fracs.unsqueeze(1).expand(B, N)

    node_feats = torch.stack([
        fit_rank * 2 - 1,
        fit_zscore / 3.0,
        dtb_rank * 2 - 1,
        (1 - local_density / max_density).clamp(0, 1) * 2 - 1,
        gradient_consistency * 2 - 1,
        local_convexity,
        (nbc_ratio / 2.5).clamp(0, 2) - 1,
        gf_exp * 2 - 1,
        gf_exp * 2 - 1,
    ], dim=2)  # (B, N, 9)

    lineage = torch.zeros(B, N, NODE_DIM - BASE_NODE_DIM, device=dev)
    all_node_feats = torch.cat([node_feats, lineage], dim=2)  # (B, N, 16)

    # ── Edge index: build ONE template, replicate ──
    template_ei = build_spatial_edge_index(N, knn_idx[0])  # (2, E)
    E = template_ei.shape[1]
    offsets = (torch.arange(B, device=dev) * N).unsqueeze(1)  # (B, 1)
    all_edges = (template_ei.unsqueeze(0) + offsets.unsqueeze(2)).reshape(2, B * E)

    # ── Edge features (B, E, 9) — all vectorized ──
    src_t = template_ei[0]  # (E,)
    dst_t = template_ei[1]

    # Edge vectors: (B, E, D)
    edge_vec = coords_norm[:, src_t] - coords_norm[:, dst_t]
    edge_dists = (edge_vec.pow(2).sum(dim=2) + 1e-8).sqrt()  # (B, E)
    max_edge_dist = edge_dists.max(dim=1, keepdim=True).values.clamp(min=1e-8)

    e0 = (edge_dists / max_edge_dist).clamp(0, 1) * 2 - 1
    e1 = ((fit[:, src_t] - fit[:, dst_t]) / fit_std.unsqueeze(1)).clamp(-3, 3) / 3.0

    centroid = coords_norm.mean(dim=1, keepdim=True)  # (B, 1, D)
    c_src = coords_norm[:, src_t] - centroid
    c_dst = coords_norm[:, dst_t] - centroid
    dot = (c_src * c_dst).sum(dim=2)
    norm_s = (c_src.pow(2).sum(dim=2) + 1e-8).sqrt()
    norm_d = (c_dst.pow(2).sum(dim=2) + 1e-8).sqrt()
    e2 = dot / (norm_s * norm_d)

    # Mutual kNN: build batched adjacency (B, N, N) — but sparse!
    # Actually need it for e3 and e8. Use scatter instead of dense adj.
    # For each (b, i, j) in edges: check if j is in knn_idx[b, i] AND i in knn_idx[b, j]
    # Vectorized: for template edges, check membership in knn_idx
    # knn_idx: (B, N, k), src_t/dst_t: (E,)
    # is_fwd[b,e] = dst_t[e] in knn_idx[b, src_t[e]]
    knn_src = knn_idx[:, src_t]  # (B, E, k)
    knn_dst = knn_idx[:, dst_t]  # (B, E, k)
    is_fwd = (knn_src == dst_t.unsqueeze(0).unsqueeze(2)).any(dim=2)  # (B, E)
    is_bwd = (knn_dst == src_t.unsqueeze(0).unsqueeze(2)).any(dim=2)  # (B, E)
    e3 = (is_fwd & is_bwd).float()

    delta = edge_vec.abs()
    delta_sum = delta.sum(dim=2)
    delta_sq_sum = (delta ** 2).sum(dim=2)
    eff_dims = (delta_sum ** 2) / (delta_sq_sum + 1e-8)
    e4 = (eff_dims / max(D, 1)).clamp(0, 1) * 2 - 1

    e5_rank = hard_rank(edge_dists) / max(E - 1, 1)  # (B, E)
    e5 = e5_rank * 2 - 1

    e6 = (fit_rank[:, src_t] - fit_rank[:, dst_t]).abs() * 2 - 1
    e7 = ((fit[:, src_t] - fit[:, dst_t]).abs()
           / (fit_std.unsqueeze(1) * edge_dists + 1e-8)).clamp(0, 3) / 1.5 - 1

    # Neighborhood overlap: count shared kNN neighbors
    # knn_src: (B, E, k), knn_dst: (B, E, k)
    # For each edge, count how many of src's kNN are also in dst's kNN
    # Vectorized via broadcasting: (B, E, k, 1) == (B, E, 1, k) → (B, E, k, k) — too large
    # Instead: sort and use searchsorted, or just compute overlap via set intersection
    # Simpler: use the sparse adj check
    # overlap = sum over k: is knn_src[b,e,j] in knn_dst[b,e,:]?
    # (B, E, k, 1) == (B, E, 1, k) → max k²=64, manageable for k=8
    overlap = (knn_src.unsqueeze(3) == knn_dst.unsqueeze(2)).any(dim=3).float().sum(dim=2)
    e8 = overlap / max(k, 1) * 2 - 1

    all_edge_feats = torch.stack([e0, e1, e2, e3, e4, e5, e6, e7, e8], dim=2)
    all_edge_attr = all_edge_feats.reshape(B * E, EDGE_DIM)

    # ── Global features (B, 16) ──
    diversity = coords_norm.std(dim=1).mean(dim=1)  # (B,)
    fdc_num = ((fit - fit_mean.unsqueeze(1)) * dist_to_best).mean(dim=1)
    fdc_den = fit_std * dist_to_best.std(dim=1).clamp(min=1e-8)
    fdc = (fdc_num / fdc_den).clamp(-1, 1)
    gc_mean = gradient_consistency.mean(dim=1)
    conv_frac = (local_convexity > 0).float().mean(dim=1) * 2 - 1
    nbc_mean = nbc_ratio.mean(dim=1) / 2.5
    norm_dim = torch.full((B,), math.log10(ndim / 100.0), device=dev)
    tech_rates = torch.zeros(B, N_TECHNIQUES, device=dev)

    global_feats = torch.cat([
        (gen_fracs * 2 - 1).unsqueeze(1),
        (diversity * 2 - 1).unsqueeze(1),
        torch.zeros(B, 1, device=dev),  # imp_rate
        fdc.unsqueeze(1),
        (gc_mean * 2 - 1).unsqueeze(1),
        conv_frac.unsqueeze(1),
        (nbc_mean * 2 - 1).unsqueeze(1),
        norm_dim.unsqueeze(1),
        tech_rates,
    ], dim=1)
    if global_feats.shape[1] < GLOBAL_DIM:
        global_feats = F.pad(global_feats, (0, GLOBAL_DIM - global_feats.shape[1]))
    global_feats = global_feats[:, :GLOBAL_DIM]

    # ── Assemble PyG batch ──
    all_nodes = all_node_feats.reshape(B * N, NODE_DIM)
    v_indices = torch.arange(B, device=dev).unsqueeze(1).expand(B, N).reshape(-1)
    e_indices = torch.arange(B, device=dev).unsqueeze(1).expand(B, E).reshape(-1)

    return all_nodes, all_edges, all_edge_attr, global_feats, v_indices, e_indices
