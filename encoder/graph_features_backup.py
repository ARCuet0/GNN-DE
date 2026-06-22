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
from .techniques_v2 import N_TECHNIQUES


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
    K: int = 4,
    max_K: int = 4,
) -> torch.Tensor:
    """Append 7 lineage features to node_attr, expanding (N, 9) -> (N, 16).

    All 7 features are differentiable — gradients flow through soft_probs,
    displacement magnitude, and improvement ratio back to the network's
    parameters and to x/fitness.

    For K < max_K, the operator one-hot is zero-padded to max_K width
    so that NODE_DIM stays constant regardless of K (PNA compat).

    Args:
        node_attr:   (N, BASE_NODE_DIM) base node features.
        parent_info: dict with keys (all tensors on same device as node_attr):
            'soft_probs':      (N, K) Gumbel-softmax operator probabilities
            'parent_x':        (N, D) parent coordinates
            'child_x':         (N, D) current coordinates
            'parent_fitness':  (N,) parent fitness values
            'child_fitness':   (N,) current fitness values
            'parent_fit_rank': (N,) soft-rank of parent in previous gen [0,1]
            If None, uses generation-0 defaults (uniform op, zeros).
        K: number of operators (for uniform default).
        max_K: pad one-hot to this width (default 4, for PNA compat).

    Returns:
        (N, NODE_DIM) augmented node features.
    """
    N = node_attr.shape[0]
    dev = node_attr.device

    if parent_info is None:
        op_onehot = torch.full((N, K), 1.0 / K, device=dev)
        displacement_mag = torch.zeros(N, device=dev)
        improvement_ratio = torch.zeros(N, device=dev)
        parent_rank = torch.full((N,), 0.0, device=dev)
    else:
        op_onehot = parent_info['soft_probs']
        diff = parent_info['child_x'] - parent_info['parent_x']
        D = diff.shape[1]
        displacement_mag = (diff.pow(2).sum(dim=1) + 1e-8).sqrt() / (D ** 0.5)

        f_parent = parent_info['parent_fitness']
        f_child = parent_info['child_fitness']
        log_parent = torch.log(f_parent.float().clamp(min=1e-30))
        log_child = torch.log(f_child.float().clamp(min=1e-30))
        improvement_ratio = (log_parent - log_child).clamp(-3, 3)

        parent_rank = parent_info['parent_fit_rank']

    # Pad one-hot to max_K for PNA input dimension compatibility
    if K < max_K:
        op_onehot = torch.nn.functional.pad(op_onehot, (0, max_K - K))

    displacement_scaled = torch.tanh(displacement_mag) * 2 - 1
    improvement_scaled = improvement_ratio / 3.0
    parent_rank_scaled = parent_rank * 2 - 1

    lineage = torch.cat([
        op_onehot,                                  # (N, K=4)
        displacement_scaled.unsqueeze(1),           # (N, 1)
        improvement_scaled.unsqueeze(1),            # (N, 1)
        parent_rank_scaled.unsqueeze(1),            # (N, 1)
    ], dim=1)

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

    coords_detached = coords_norm.detach()
    dist_matrix_topo = torch.cdist(coords_detached, coords_detached)
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

    # NBC ratio — softmin over distances weighted by better_prob (matches CPU nearest-better semantics)
    nn_dist = dist_matrix_safe.min(dim=1).values
    fit_diff_ij = fit.unsqueeze(1) - fit.unsqueeze(0)
    better_prob = torch.sigmoid(beta * fit_diff_ij)
    better_prob_masked = better_prob * (~diag_mask).float()
    has_better = better_prob_masked.sum(dim=1) > 0.5
    # Softmin: penalize non-better neighbors with large distance offset
    large_val = dist_matrix_safe.max().detach() + 1.0
    masked_dist = dist_matrix_safe + (1.0 - better_prob_masked) * large_val
    dist_temp = masked_dist.std(dim=1).clamp(min=1e-4).unsqueeze(1)
    nbn_dist_softmin = (F.softmax(-masked_dist / dist_temp, dim=1) * masked_dist).sum(dim=1)
    # Fallback for best individual (no fitter neighbors): use nn_dist
    nbn_dist = torch.where(has_better, nbn_dist_softmin, nn_dist)
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
    """Compute 9 base node features from shared intermediates.

    Returns: (N, BASE_NODE_DIM=9) tensor on same device.
    """
    fit_rank = shared['fit_rank']
    fit_zscore = shared['fit_zscore']
    dist_to_best = shared['dist_to_best']
    local_density = shared['local_density']
    max_density = shared['max_density']
    gradient_consistency = shared['gradient_consistency']
    local_convexity = shared['local_convexity']
    nbc_ratio = shared['nbc_ratio']

    dev = fit_rank.device
    N = fit_rank.shape[0]
    gen_frac = step_num / max(max_steps, 1)
    beta = 10.0  # same default as compute_shared_intermediates

    dtb_rank = soft_rank(dist_to_best, beta=beta) / max(N - 1, 1)

    if generation_ids is not None:
        tenure = ((step_num - generation_ids.float()) / max(max_steps, 1)).clamp(0, 1)
    else:
        tenure = torch.full((N,), gen_frac, device=dev).clamp(0, 1)

    return torch.stack([
        fit_rank * 2 - 1,                                         # 0
        fit_zscore / 3.0,                                          # 1
        dtb_rank * 2 - 1,                                         # 2
        (1 - local_density / max_density).clamp(0, 1) * 2 - 1,    # 3
        gradient_consistency * 2 - 1,                              # 4
        local_convexity,                                           # 5
        (nbc_ratio / 2.5).clamp(0, 2) - 1,                        # 6
        tenure * 2 - 1,                                            # 7
        torch.full((N,), gen_frac * 2 - 1, device=dev),           # 8
    ], dim=1)


# ======================================================================
# Edge features (9)
# ======================================================================

def compute_edge_features(
    shared: dict,
    edge_index: torch.Tensor,
    beta: float = 10.0,
) -> torch.Tensor:
    """Compute 9 edge features for given edge_index using shared intermediates.

    Returns: (E, EDGE_DIM=9) tensor on same device.
    """
    coords_norm = shared['coords_norm']
    fit = shared['fit']
    fit_rank = shared['fit_rank']
    fit_std = shared['fit_std']
    centroid = shared['centroid']
    dist_matrix_diff = shared['dist_matrix_diff']
    adj_float = shared['adj_float']
    k = shared['k']

    src = edge_index[0]
    dst = edge_index[1]
    E = edge_index.shape[1]
    D = coords_norm.shape[1]
    dev = coords_norm.device

    if E == 0:
        return torch.zeros(0, EDGE_DIM, device=dev)

    edge_dists = dist_matrix_diff[src, dst]
    max_edge_dist = edge_dists.max().clamp(min=1e-8)

    e0 = (edge_dists / max_edge_dist).clamp(0, 1) * 2 - 1
    e1 = ((fit[src] - fit[dst]) / fit_std).clamp(-3, 3) / 3.0

    c_src = coords_norm[src] - centroid
    c_dst = coords_norm[dst] - centroid
    dot = (c_src * c_dst).sum(dim=1)
    norm_s = (c_src.pow(2).sum(dim=1) + 1e-8).sqrt()
    norm_d = (c_dst.pow(2).sum(dim=1) + 1e-8).sqrt()
    e2 = dot / (norm_s * norm_d)

    e3 = adj_float[src, dst] * adj_float[dst, src]

    edge_delta = (coords_norm[src] - coords_norm[dst]).abs()
    delta_sum = edge_delta.sum(dim=1)
    delta_sq_sum = (edge_delta ** 2).sum(dim=1)
    eff_dims = (delta_sum ** 2) / (delta_sq_sum + 1e-8)
    e4 = (eff_dims / max(D, 1)).clamp(0, 1) * 2 - 1

    dist_ranks_soft = soft_rank(edge_dists, beta=beta)
    e5 = dist_ranks_soft / max(E - 1, 1) * 2 - 1

    e6 = (fit_rank[src] - fit_rank[dst]).abs() * 2 - 1
    e7 = ((fit[src] - fit[dst]).abs() / (fit_std * edge_dists + 1e-8)).clamp(0, 3) / 1.5 - 1

    overlap = (adj_float[src] * adj_float[dst]).sum(dim=1) / max(k, 1) * 2 - 1
    e8 = overlap

    return torch.stack([e0, e1, e2, e3, e4, e5, e6, e7, e8], dim=1)


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
# Global features (16)
# ======================================================================

def compute_global_features(
    shared: dict,
    ndim: int,
    step_num: int,
    max_steps: int,
    technique_success_rates: torch.Tensor = None,
    prev_best: float = None,
    eval_budget_frac: float = None,
    improvement_ema: float = None,
    n_techniques: int = N_TECHNIQUES,
    beta: float = 10.0,
) -> torch.Tensor:
    """Compute 16 global features from shared intermediates.

    Returns: (1, GLOBAL_DIM=16) tensor on same device.
    """
    coords_norm = shared['coords_norm']
    fit = shared['fit']
    fit_std = shared['fit_std']
    fit_mean = shared['fit_mean']
    dist_to_best = shared['dist_to_best']
    gradient_consistency = shared['gradient_consistency']
    local_convexity = shared['local_convexity']
    nbc_ratio = shared['nbc_ratio']
    _lf_mean = shared['_lf_mean']
    _lf_std = shared['_lf_std']

    dev = coords_norm.device
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

    g_list = [
        torch.tensor(progress, device=dev, dtype=torch.float32),
        (pop_diversity / 0.3).clamp(0, 1).float() * 2 - 1,
    ]
    if isinstance(improvement_rate, torch.Tensor):
        g_list.append(improvement_rate.float())
    else:
        g_list.append(torch.tensor(float(improvement_rate), device=dev))
    if isinstance(fdc, torch.Tensor):
        g_list.append(fdc.float())
    else:
        g_list.append(torch.tensor(float(fdc), device=dev))
    g_list.append((mean_grad_con * 2 - 1).float())
    g_list.append((conv_frac * 2 - 1).float())
    g_list.append((mean_nbc / 2.5).clamp(0, 2).float() - 1)
    g_list.append(torch.tensor(math.log10(ndim / 100.0),
                               device=dev, dtype=torch.float32))

    for i in range(n_techniques):
        if technique_success_rates is not None and i < len(technique_success_rates):
            g_list.append(technique_success_rates[i].float() * 2 - 1)
        else:
            g_list.append(torch.tensor(0.0, device=dev))

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
    technique_success_rates: torch.Tensor = None,
    eval_budget_frac: float = None,
    n_techniques: int = N_TECHNIQUES,
) -> torch.Tensor:
    """Build global features for degenerate graphs (N<=1 or E=0)."""
    gen_frac = step_num / max(max_steps, 1)
    progress = (min(max(eval_budget_frac, 0.0), 1.0) * 2 - 1
                if eval_budget_frac is not None
                else gen_frac * 2 - 1)
    g = [progress, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0,
         math.log10(ndim / 100.0)]
    for i in range(n_techniques):
        if technique_success_rates is not None and i < len(technique_success_rates):
            g.append(float(technique_success_rates[i]) * 2 - 1)
        else:
            g.append(0.0)
    return torch.tensor(g, device=device, dtype=torch.float32).unsqueeze(0)
