"""
graph_features.py — Shared differentiable graph feature kernels.

Extracted from similarity_graph_gpu.py and temporal_graph.py to eliminate
duplication.  All computations are GPU-resident and differentiable w.r.t.
x and fitness.  Topology (edge_index/knn_idx) is computed from detached
coordinates for stable k-NN sparsity.
"""

import math
import torch
import torch.nn.functional as F

from .similarity_graph import (
    BASE_NODE_DIM, LINEAGE_DIM, NODE_DIM, EDGE_DIM, GLOBAL_DIM,
)


# ======================================================================
# Differentiable helpers
# ======================================================================

def soft_rank(values: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
    """Differentiable rank via pairwise sigmoid.

    For each element i, rank_i = sum_j sigmoid(beta * (v_j - v_i)).
    Returns ranks in [0, N-1] (continuous).

    Args:
        values: (N,) or (B, N) tensor, may require grad.
        beta:   sharpness of the sigmoid approximation.

    Returns:
        (*batch, N) soft ranks in [0, N-1].
    """
    if values.dim() == 1:
        diff = values.unsqueeze(0) - values.unsqueeze(1)  # (N, N)
        scale = beta / diff.abs().mean().clamp(min=1e-4)
        scores = torch.sigmoid(scale * diff)
        mask = 1.0 - torch.eye(scores.shape[0], device=scores.device, dtype=scores.dtype)
        scores = scores * mask
        return scores.sum(dim=1)

    # Batched: values is (B, N)
    B, N = values.shape
    diff = values.unsqueeze(1) - values.unsqueeze(2)  # (B, N, N)
    scale = beta / diff.abs().mean(dim=(1, 2), keepdim=True).clamp(min=1e-4)
    scores = torch.sigmoid(scale * diff)
    mask = 1.0 - torch.eye(N, device=scores.device, dtype=scores.dtype)
    scores = scores * mask.unsqueeze(0)
    return scores.sum(dim=2)  # (B, N)


# ======================================================================
# Lineage feature augmentation (Fork A)
# ======================================================================

def augment_node_features_lineage(
    node_attr: torch.Tensor,
    parent_info: dict = None,
) -> torch.Tensor:
    """Append 3 lineage features to node_attr, expanding (N, 5) -> (N, 8).

    Features: displacement_mag, improvement_ratio, parent_fitness_rank.
    All differentiable — gradients flow through displacement and improvement
    back to x/fitness.

    Args:
        node_attr:   (N, BASE_NODE_DIM=6) base node features.
        parent_info: dict with keys (all tensors on same device as node_attr):
            'parent_x':        (N, D) parent coordinates
            'child_x':         (N, D) current coordinates
            'parent_fitness':  (N,) parent fitness values
            'child_fitness':   (N,) current fitness values
            'parent_fit_rank': (N,) soft-rank of parent in previous gen [0,1]
            If None, uses generation-0 defaults (zeros).

    Returns:
        (N, NODE_DIM=9) augmented node features.
    """
    N = node_attr.shape[0]
    dev = node_attr.device

    if parent_info is None:
        displacement_mag = torch.zeros(N, device=dev)
        improvement_ratio = torch.zeros(N, device=dev)
        parent_rank = torch.zeros(N, device=dev)
    else:
        diff = parent_info['child_x'] - parent_info['parent_x']
        D = diff.shape[1]
        displacement_mag = (diff.pow(2).sum(dim=1) + 1e-8).sqrt() / (D ** 0.5)

        f_parent = parent_info['parent_fitness']
        f_child = parent_info['child_fitness']
        log_parent = torch.log(f_parent.float().clamp(min=1e-30))
        log_child = torch.log(f_child.float().clamp(min=1e-30))
        improvement_ratio = (log_parent - log_child).clamp(-3, 3)

        parent_rank = parent_info['parent_fit_rank']

    # Soft-rank within graph: guarantees uniform variance [0,1] regardless of scale
    displacement_scaled = soft_rank(displacement_mag, beta=10.0) / max(N - 1, 1) * 2 - 1
    improvement_scaled = soft_rank(improvement_ratio, beta=10.0) / max(N - 1, 1) * 2 - 1
    parent_rank_scaled = parent_rank * 2 - 1

    lineage = torch.stack([
        displacement_scaled,            # (N,)
        improvement_scaled,             # (N,)
        parent_rank_scaled,             # (N,)
    ], dim=1)                           # (N, 3)

    return torch.cat([node_attr, lineage], dim=1)


# ======================================================================
# Shared intermediates
# ======================================================================

def compute_shared_intermediates(
    x: torch.Tensor,
    fitness: torch.Tensor,
    k_neighbors: int = 8,
    lb: float = -100.0,
    ub: float = 100.0,
    beta: float = 10.0,
):
    """Compute shared intermediates used by node, edge, and global features.

    Fitness is preprocessed as z-scored log-fitness for scale invariance.

    Returns a dict with all shared tensors needed by feature kernels.
    """
    dev = x.device
    N, D = x.shape
    span = ub - lb
    span_safe = span if abs(span) > 1e-12 else 1.0

    # Normalize coordinates to [0, 1]
    coords_norm = (x.float() - lb) / span_safe

    # Z-scored log-fitness: scale-invariant
    _log_fit = torch.log(fitness.float().clamp(min=1e-30))
    _lf_mean = _log_fit.mean()
    _lf_std = _log_fit.std().clamp(min=1e-4)
    fit = (_log_fit - _lf_mean) / _lf_std

    # Pairwise distances
    dist_matrix_diff = torch.cdist(coords_norm, coords_norm)
    dist_matrix_topo = dist_matrix_diff.detach().clone()
    dist_matrix_topo.fill_diagonal_(float('inf'))

    k = min(k_neighbors, N - 1)
    _, knn_idx = torch.topk(dist_matrix_topo, k, dim=1, largest=False)

    # Adjacency matrix
    row_expand = torch.arange(N, device=dev).unsqueeze(1).expand(N, k)
    adj_float = torch.zeros(N, N, device=dev)
    adj_float[row_expand.reshape(-1), knn_idx.reshape(-1)] = 1.0

    # Fitness rank
    fit_rank = soft_rank(fit, beta=beta) / max(N - 1, 1)
    fit_mean = fit.mean()
    fit_std = fit.std().clamp(min=1e-8)
    fit_zscore = ((fit - fit_mean) / fit_std).clamp(-3, 3)

    centroid = coords_norm.mean(dim=0)

    # Softmin-weighted best position
    softmin_weights = F.softmax(-fit / fit_std, dim=0)
    best_x_soft = (softmin_weights.unsqueeze(1) * coords_norm).sum(0)
    _d2b = coords_norm - best_x_soft.unsqueeze(0)
    dist_to_best = (_d2b.pow(2).sum(dim=1) + 1e-8).sqrt()

    # Safe distance matrix (diagonal masked)
    diag_mask = torch.eye(N, device=dev, dtype=torch.bool)
    max_dist_est = dist_matrix_diff.detach().max() + 1.0
    dist_matrix_safe = torch.where(
        diag_mask, max_dist_est * torch.ones_like(dist_matrix_diff),
        dist_matrix_diff
    )

    # Local density
    knn_dists_diff = dist_matrix_diff[
        row_expand.reshape(-1), knn_idx.reshape(-1)
    ].reshape(N, k)
    local_density = knn_dists_diff.mean(dim=1)
    max_density = local_density.max().clamp(min=1e-8)

    # Gradient consistency
    nn_coords = coords_norm[knn_idx]
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

    # NBC ratio — kNN-only, O(N·k) instead of O(N²)
    # dists_nn: (N, k, 1) already computed above; squeeze to (N, k)
    knn_dists_flat = dists_nn.squeeze(-1)                          # (N, k)
    nn_dist = knn_dists_flat.min(dim=1).values                     # nearest ANY neighbor
    knn_fit = fit[knn_idx]                                         # (N, k)
    better_mask = knn_fit < fit.unsqueeze(1)                       # (N, k)
    nbn_dists = torch.where(better_mask, knn_dists_flat,
                            torch.tensor(float('inf'), device=dev))
    nbn_dist = nbn_dists.min(dim=1).values                         # (N,)
    # Fallback: if no better neighbor in kNN, ratio = 1.0 (neutral)
    nbn_dist = torch.where(nbn_dist.isinf(), nn_dist, nbn_dist)
    nbc_ratio = (nbn_dist / (nn_dist + 1e-8)).clamp(0, 5)

    return {
        'coords_norm': coords_norm,
        'fit': fit,
        'fit_rank': fit_rank,
        'fit_mean': fit_mean,
        'fit_std': fit_std,
        'fit_zscore': fit_zscore,
        'centroid': centroid,
        'dist_to_best': dist_to_best,
        'dist_matrix_diff': dist_matrix_diff,
        'dist_matrix_safe': dist_matrix_safe,
        'knn_idx': knn_idx,
        'adj_float': adj_float,
        'local_density': local_density,
        'max_density': max_density,
        'gradient_consistency': gradient_consistency,
        'local_convexity': local_convexity,
        'nbc_ratio': nbc_ratio,
        'diag_mask': diag_mask,
        'k': k,
        '_lf_mean': _lf_mean,
        '_lf_std': _lf_std,
    }


# ======================================================================
# Node features (9 base)
# ======================================================================

def compute_base_node_features(
    shared: dict,
    step_num: int,
    max_steps: int,
    generation_ids: torch.Tensor = None,
) -> torch.Tensor:
    """Compute 5 base node features from shared intermediates.

    Returns: (N, BASE_NODE_DIM=5) tensor on same device.
    """
    fit_rank = shared['fit_rank']
    dist_to_best = shared['dist_to_best']
    local_density = shared['local_density']
    max_density = shared['max_density']
    gradient_consistency = shared['gradient_consistency']
    coords_norm = shared['coords_norm']

    dev = fit_rank.device
    N = fit_rank.shape[0]
    gen_frac = step_num / max(max_steps, 1)
    beta = 10.0

    dtb_rank = soft_rank(dist_to_best, beta=beta) / max(N - 1, 1)

    # Distance to population centroid (rank-normalized, per-node)
    centroid = shared['centroid']
    dist_to_centroid = (coords_norm - centroid).pow(2).sum(dim=1).sqrt()
    dtc_rank = soft_rank(dist_to_centroid, beta=beta) / max(N - 1, 1)

    return torch.stack([
        fit_rank * 2 - 1,                                         # 0: fit_rank
        dtb_rank * 2 - 1,                                         # 1: dist_to_best_rank
        (1 - local_density / max_density).clamp(0, 1) * 2 - 1,    # 2: local_density_inv
        gradient_consistency * 2 - 1,                              # 3: gradient_consistency
        dtc_rank * 2 - 1,                                          # 4: dist_to_centroid_rank
    ], dim=1)


# ======================================================================
# Edge features (4)
# ======================================================================

def compute_edge_features(
    shared: dict,
    edge_index: torch.Tensor,
    beta: float = 10.0,
) -> torch.Tensor:
    """Compute 4 edge features for given edge_index using shared intermediates.

    Returns: (E, EDGE_DIM=4) tensor on same device.
    """
    coords_norm = shared['coords_norm']
    fit_rank = shared['fit_rank']
    centroid = shared['centroid']
    dist_matrix_diff = shared['dist_matrix_diff']
    adj_float = shared['adj_float']

    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.shape[1]
    dev = coords_norm.device

    if E == 0:
        return torch.zeros(0, EDGE_DIM, device=dev)

    edge_dists = dist_matrix_diff[src, dst]

    # 0: distance percentile (soft rank among all edges)
    dist_ranks_soft = soft_rank(edge_dists, beta=beta)
    e0 = dist_ranks_soft / max(E - 1, 1) * 2 - 1

    # 1: fitness rank difference
    e1 = (fit_rank[src] - fit_rank[dst]).abs() * 2 - 1

    # 2: cosine similarity (centered coordinates)
    c_src = coords_norm[src] - centroid
    c_dst = coords_norm[dst] - centroid
    dot = (c_src * c_dst).sum(dim=1)
    norm_s = (c_src.pow(2).sum(dim=1) + 1e-8).sqrt()
    norm_d = (c_dst.pow(2).sum(dim=1) + 1e-8).sqrt()
    e2 = dot / (norm_s * norm_d)

    # 3: mutual k-NN
    e3 = adj_float[src, dst] * adj_float[dst, src]

    return torch.stack([e0, e1, e2, e3], dim=1)


def compute_edge_features_dense(shared: dict) -> torch.Tensor:
    """Compute 4 edge features for ALL N×N pairs.

    Returns: (N, N, EDGE_DIM=4) tensor on same device.
    """
    coords_norm = shared['coords_norm']   # (N, D)
    fit_rank = shared['fit_rank']          # (N,)
    centroid = shared['centroid']          # (D,)
    dist_matrix_diff = shared['dist_matrix_diff']  # (N, N)
    adj_float = shared['adj_float']       # (N, N) — k-NN adjacency

    N = coords_norm.shape[0]
    dev = coords_norm.device

    # 0: distance percentile (rank of each pairwise distance among all N² pairs)
    flat_dists = dist_matrix_diff.reshape(-1)
    ranks = flat_dists.argsort().argsort().float()  # O(N² log N²) but N≤275
    e0 = (ranks / max(N * N - 1, 1) * 2 - 1).reshape(N, N)

    # 1: fitness rank difference (absolute)
    e1 = (fit_rank.unsqueeze(1) - fit_rank.unsqueeze(0)).abs() * 2 - 1  # (N, N)

    # 2: cosine similarity (centered)
    c = coords_norm - centroid  # (N, D)
    norms = (c.pow(2).sum(dim=1, keepdim=True) + 1e-8).sqrt()  # (N, 1)
    c_normed = c / norms
    e2 = c_normed @ c_normed.T  # (N, N) — cosine similarity matrix

    # 3: mutual k-NN (binary)
    e3 = adj_float * adj_float.T  # (N, N)

    return torch.stack([e0, e1, e2, e3], dim=-1)  # (N, N, 4)


# ======================================================================
# Spatial edge construction (bidirectional k-NN, deduplicated)
# ======================================================================

def build_spatial_edge_index(N: int, knn_idx: torch.Tensor) -> torch.Tensor:
    """Build bidirectional, deduplicated edge_index from knn_idx.

    Args:
        N:       number of nodes
        knn_idx: (N, k) k-NN indices

    Returns:
        edge_index: (2, E) long tensor, deduplicated bidirectional edges
    """
    dev = knn_idx.device
    k = knn_idx.shape[1]
    row_expand = torch.arange(N, device=dev).unsqueeze(1).expand(N, k)
    src_fwd = row_expand.reshape(-1)
    dst_fwd = knn_idx.reshape(-1)
    all_src = torch.cat([src_fwd, dst_fwd])
    all_dst = torch.cat([dst_fwd, src_fwd])
    all_edges = torch.stack([all_src, all_dst], dim=0)
    return torch.unique(all_edges, dim=1)


# ======================================================================
# Global features (11)
# ======================================================================

def compute_global_features(
    shared: dict,
    ndim: int,
    step_num: int,
    max_steps: int,
    prev_best: float = None,
    eval_budget_frac: float = None,
    improvement_ema: float = None,
    stagnation_counter: int = 0,
    prev_x: torch.Tensor = None,
    prev_fitness: torch.Tensor = None,
    parent_info: dict = None,
    beta: float = 10.0,
    delta_fitness: float = 0.0,
    contraction_rate: float = 0.0,
) -> torch.Tensor:
    """Compute 13 global features from shared intermediates.

    Returns: (1, GLOBAL_DIM=13) tensor on same device.
    """
    coords_norm = shared['coords_norm']
    fit = shared['fit']
    fit_rank = shared['fit_rank']
    fit_std = shared['fit_std']
    fit_mean = shared['fit_mean']
    dist_to_best = shared['dist_to_best']
    gradient_consistency = shared['gradient_consistency']
    local_convexity = shared['local_convexity']
    nbc_ratio = shared['nbc_ratio']
    local_density = shared['local_density']
    _lf_mean = shared['_lf_mean']
    _lf_std = shared['_lf_std']

    dev = coords_norm.device
    N = coords_norm.shape[0]
    gen_frac = step_num / max(max_steps, 1)

    pop_diversity = (coords_norm.var(dim=0) + 1e-8).sqrt().mean()

    if improvement_ema is not None:
        if isinstance(improvement_ema, torch.Tensor):
            improvement_rate = improvement_ema.float().clamp(-1, 1)
        else:
            improvement_rate = max(-1.0, min(float(improvement_ema), 1.0))
    elif prev_best is not None:
        curr_best = fit.min()
        if isinstance(prev_best, torch.Tensor):
            prev_best_t = prev_best.float().clamp(min=1e-30)
        else:
            prev_best_t = torch.tensor(max(float(prev_best), 1e-30), device=dev)
        zprev = (prev_best_t.log() - _lf_mean) / _lf_std
        improvement_rate = (zprev - curr_best).clamp(-1, 1)
    else:
        improvement_rate = 0.0

    # FDC
    dtb_std = dist_to_best.std().clamp(min=1e-8)
    dtb_centered = dist_to_best - dist_to_best.mean()
    fit_centered = fit - fit_mean
    denom = (dtb_std * fit_std).clamp(min=1e-2)
    fdc = ((dtb_centered * fit_centered).mean() / denom).clamp(-1, 1)
    fdc = torch.where(torch.isfinite(fdc), fdc, torch.zeros_like(fdc))

    mean_grad_con = gradient_consistency.mean()
    conv_frac = torch.sigmoid(50.0 * local_convexity).mean()
    mean_nbc = nbc_ratio.mean().clamp(0, 5)

    progress = (min(max(eval_budget_frac, 0.0), 1.0) * 2 - 1
                if eval_budget_frac is not None
                else gen_frac * 2 - 1)

    # ---- Stagnation counter (normalized) ----
    stag_norm = math.tanh(stagnation_counter / 20.0) * 2 - 1

    # ---- Front vs tail improvement ----
    front_vs_tail = torch.tensor(0.0, device=dev)
    if parent_info is not None:
        f_parent = parent_info['parent_fitness']
        f_child = parent_info['child_fitness']
        log_imp = (torch.log(f_parent.float().clamp(min=1e-30))
                   - torch.log(f_child.float().clamp(min=1e-30)))
        # Top 20% by fitness rank (low rank = better)
        elite_mask = (fit_rank < 0.2).float()
        tail_mask = (fit_rank > 0.8).float()
        elite_sum = elite_mask.sum().clamp(min=1)
        tail_sum = tail_mask.sum().clamp(min=1)
        elite_imp = (log_imp * elite_mask).sum() / elite_sum
        tail_imp = (log_imp * tail_mask).sum() / tail_sum
        front_vs_tail = (elite_imp - tail_imp).clamp(-3, 3) / 3.0

    # ---- Direction consensus ----
    # Cosine similarity of displacement vectors among top-20% individuals.
    # Only meaningful when parent_info provides child_x - parent_x (same coord space).
    direction_consensus = torch.tensor(0.0, device=dev)
    if parent_info is not None:
        disp = parent_info['child_x'] - parent_info['parent_x']  # (N, D) raw
        top_k = max(int(N * 0.2), 2)
        _, top_idx = torch.topk(fit_rank, top_k, largest=False)
        top_disp = disp[top_idx]
        top_norms = (top_disp.pow(2).sum(dim=1, keepdim=True) + 1e-8).sqrt()
        top_dirs = top_disp / top_norms
        cos_matrix = top_dirs @ top_dirs.T
        eye_mask = 1.0 - torch.eye(top_k, device=dev)
        direction_consensus = (cos_matrix * eye_mask).sum() / eye_mask.sum().clamp(min=1)

    # ---- Density-quality correlation ----
    density_rank = soft_rank(local_density, beta=beta) / max(N - 1, 1)
    dr_centered = density_rank - density_rank.mean()
    fr_centered = fit_rank - fit_rank.mean()
    dr_std = density_rank.std().clamp(min=1e-8)
    fr_std = fit_rank.std().clamp(min=1e-8)
    dq_corr = ((dr_centered * fr_centered).mean() / (dr_std * fr_std)).clamp(-1, 1)
    dq_corr = torch.where(torch.isfinite(dq_corr), dq_corr, torch.zeros_like(dq_corr))

    # ---- Assemble ----
    g_list = [
        torch.tensor(progress, device=dev, dtype=torch.float32),       # 0: gen_frac
        (pop_diversity / 0.3).clamp(0, 1).float() * 2 - 1,            # 1: diversity
    ]
    if isinstance(improvement_rate, torch.Tensor):
        g_list.append(improvement_rate.float())                         # 2: improvement_rate
    else:
        g_list.append(torch.tensor(float(improvement_rate), device=dev))
    if isinstance(fdc, torch.Tensor):
        g_list.append(fdc.float())                                      # 3: population_fdc
    else:
        g_list.append(torch.tensor(float(fdc), device=dev))
    g_list.append((mean_grad_con * 2 - 1).float())                     # 4: mean_grad_consistency
    g_list.append((conv_frac * 2 - 1).float())                         # 5: convexity_fraction
    g_list.append((mean_nbc / 2.5).clamp(0, 2).float() - 1)            # 6: mean_nbc_ratio
    g_list.append(torch.tensor(stag_norm, device=dev, dtype=torch.float32))  # 7: stagnation_counter
    g_list.append(front_vs_tail.float())                                # 8: front_vs_tail
    g_list.append(direction_consensus.float())                          # 9: direction_consensus
    g_list.append(dq_corr.float())                                      # 10: density_quality_corr
    g_list.append(torch.tensor(float(max(-1, min(delta_fitness, 1))),
                               device=dev, dtype=torch.float32))        # 11: delta_fitness
    g_list.append(torch.tensor(float(max(-1, min(contraction_rate, 1))),
                               device=dev, dtype=torch.float32))        # 12: contraction_rate

    return torch.stack(g_list).unsqueeze(0)


# ======================================================================
# Degenerate graph helper
# ======================================================================

def make_degenerate_global(
    N: int,
    ndim: int,
    step_num: int,
    max_steps: int,
    device: torch.device,
    eval_budget_frac: float = None,
    stagnation_counter: int = 0,
) -> torch.Tensor:
    """Build global features for degenerate graphs (N<=1 or E=0)."""
    gen_frac = step_num / max(max_steps, 1)
    progress = (min(max(eval_budget_frac, 0.0), 1.0) * 2 - 1
                if eval_budget_frac is not None
                else gen_frac * 2 - 1)
    stag_norm = math.tanh(stagnation_counter / 20.0) * 2 - 1
    g = [progress, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0,
         stag_norm, 0.0, 0.0, 0.0, 0.0, 0.0]
    return torch.tensor(g, device=device, dtype=torch.float32).unsqueeze(0)
