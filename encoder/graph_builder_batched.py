"""
graph_builder_batched.py — Batched-uniform graph builder (same N for all B).

Contains:
  - build_batched_uniform_graphs_gpu (uniform N, PyG batch format)
"""

import torch

from .similarity_graph import (
    LINEAGE_DIM, NODE_DIM, EDGE_DIM, GLOBAL_DIM,
)
from .graph_features import soft_rank
from .graph_utils import _temporal_to_globals


def build_batched_uniform_graphs_gpu(
    xs: torch.Tensor,
    fitnesses: torch.Tensor,
    step_nums: list,
    max_steps: int,
    ndim: int,
    prev_bests: torch.Tensor = None,
    k_neighbors: int = 8,
    beta: float = 10.0,
    lb: float = -100.0,
    ub: float = 100.0,
    stagnation_counters: list = None,
    delta_fitnesses: list = None,
    contraction_rates: list = None,
):
    """Build B similarity graphs in parallel when all have the same N.

    All heavy computation (cdist, soft_rank, kNN, features) is batched
    as (B, N, ...) tensors -- one kernel per operation, not B kernels.

    Args:
        xs:        (B, N, D) coordinates
        fitnesses: (B, N) fitness values
        step_nums: list of B ints
        max_steps: int
        ndim:      int (same for all)
        prev_bests: (B,) or None
        stagnation_counters: list of B ints or None
        delta_fitnesses: list of B floats or None
        contraction_rates: list of B floats or None

    Returns:
        all_nodes, all_edges, all_edge_attr, all_global, v_indices, e_indices
        (same format as build_batched_similarity_graphs_gpu)
    """
    B, N, D = xs.shape
    dev = xs.device
    span = ub - lb
    span_safe = span if abs(span) > 1e-12 else 1.0

    stag_t, delta_fit_t, contraction_t = _temporal_to_globals(
        stagnation_counters, delta_fitnesses, contraction_rates, B, dev)

    # -- Normalize coordinates --
    coords_norm = (xs.float() - lb) / span_safe  # (B, N, D)

    # -- Z-scored log-fitness --
    log_fit = torch.log(fitnesses.float().clamp(min=1e-30))  # (B, N)
    lf_mean = log_fit.mean(dim=1, keepdim=True)
    lf_std = log_fit.std(dim=1, keepdim=True).clamp(min=1e-4)
    fit = (log_fit - lf_mean) / lf_std  # (B, N)

    # -- Batched pairwise distances --
    dist_matrix = torch.cdist(coords_norm, coords_norm)  # (B, N, N)
    dist_topo = dist_matrix.detach().clone()
    diag_mask = torch.eye(N, device=dev, dtype=torch.bool).unsqueeze(0)
    dist_topo = dist_topo.masked_fill(diag_mask, float('inf'))

    # -- Batched kNN --
    k = min(k_neighbors, N - 1)
    _, knn_idx = torch.topk(dist_topo, k, dim=2, largest=False)  # (B, N, k)

    # -- Batched soft_rank (fitness) --
    fit_rank = soft_rank(fit, beta=beta) / max(N - 1, 1)  # (B, N)
    fit_mean_b = fit.mean(dim=1)  # (B,)
    fit_std_b = fit.std(dim=1).clamp(min=1e-8)  # (B,)

    # -- Batched best position (softmin) --
    softmin_w = torch.softmax(-fit / fit_std_b.unsqueeze(1), dim=1)
    best_x = (softmin_w.unsqueeze(2) * coords_norm).sum(dim=1)  # (B, D)
    d2b = coords_norm - best_x.unsqueeze(1)
    dist_to_best = (d2b.pow(2).sum(dim=2) + 1e-8).sqrt()
    dtb_rank = soft_rank(dist_to_best, beta=beta) / max(N - 1, 1)

    # -- Batched local density --
    row_exp = torch.arange(N, device=dev).unsqueeze(1).expand(N, k)
    knn_dists = dist_matrix[
        torch.arange(B, device=dev).reshape(B, 1, 1).expand(B, N, k),
        row_exp.unsqueeze(0).expand(B, N, k),
        knn_idx
    ]
    local_density = knn_dists.mean(dim=2)
    max_density = local_density.max(dim=1, keepdim=True).values.clamp(min=1e-8)

    # -- Batched gradient consistency --
    b_idx = torch.arange(B, device=dev).reshape(B, 1, 1).expand(B, N, k)
    nn_coords = coords_norm[b_idx, knn_idx]
    diffs = nn_coords - coords_norm.unsqueeze(2)
    dists_nn = (diffs.pow(2).sum(dim=3, keepdim=True) + 1e-8).sqrt()
    directions = diffs / dists_nn
    nn_fit = fit.gather(1, knn_idx.reshape(B, -1)).reshape(B, N, k)
    f_diffs = nn_fit - fit.unsqueeze(2)
    grads = f_diffs / dists_nn.squeeze(-1)
    weighted = grads.unsqueeze(-1) * directions
    weighted_sum = weighted.sum(dim=2)
    total_abs_grad = grads.abs().sum(dim=2).clamp(min=1e-8)
    ws_norm = (weighted_sum.pow(2).sum(dim=2) + 1e-8).sqrt()
    gradient_consistency = (ws_norm / total_abs_grad).clamp(0, 1)

    # -- Node features (B, N, 5) -- base only --
    centroid = coords_norm.mean(dim=1, keepdim=True)  # (B, 1, D)
    dist_to_centroid = ((coords_norm - centroid).pow(2).sum(dim=2) + 1e-8).sqrt()
    dtc_rank = torch.stack([
        soft_rank(dist_to_centroid[b], beta=10.0) / max(N - 1, 1)
        for b in range(B)
    ])

    node_feats = torch.stack([
        fit_rank * 2 - 1,
        dtb_rank * 2 - 1,
        (1 - local_density / max_density).clamp(0, 1) * 2 - 1,
        gradient_consistency * 2 - 1,
        dtc_rank * 2 - 1,
    ], dim=2)  # (B, N, 5)

    # Lineage: zeros (no parent info in batched mode)
    lineage = torch.zeros(B, N, LINEAGE_DIM, device=dev)
    lineage[:, :, 0] = -1.0
    lineage[:, :, 2] = -1.0
    all_node_feats = torch.cat([node_feats, lineage], dim=2)  # (B, N, 9)

    # -- Per-graph edge index + features --
    row_exp = torch.arange(N, device=dev).unsqueeze(1).expand(N, k)
    src_fwd = row_exp.reshape(-1)
    dst_fwd = knn_idx.reshape(B, N * k)
    src_all = torch.cat([
        src_fwd.unsqueeze(0).expand(B, -1),
        dst_fwd
    ], dim=1)
    dst_all = torch.cat([
        dst_fwd,
        src_fwd.unsqueeze(0).expand(B, -1)
    ], dim=1)
    E = 2 * N * k

    offsets = torch.arange(B, device=dev).unsqueeze(1) * N
    all_edges = torch.stack([
        (src_all + offsets).reshape(-1),
        (dst_all + offsets).reshape(-1)
    ], dim=0)

    edge_dists = dist_matrix[
        torch.arange(B, device=dev).unsqueeze(1).expand(B, E),
        src_all, dst_all]

    e0_rank = soft_rank(edge_dists, beta=beta)
    e0 = e0_rank / max(E - 1, 1) * 2 - 1
    e1 = (fit_rank.gather(1, src_all) - fit_rank.gather(1, dst_all)).abs() * 2 - 1

    centroid = coords_norm.mean(dim=1, keepdim=True)
    c_src = coords_norm.gather(1, src_all.unsqueeze(2).expand(-1, -1, D)) - centroid
    c_dst = coords_norm.gather(1, dst_all.unsqueeze(2).expand(-1, -1, D)) - centroid
    dot = (c_src * c_dst).sum(dim=2)
    norm_s = (c_src.pow(2).sum(dim=2) + 1e-8).sqrt()
    norm_d = (c_dst.pow(2).sum(dim=2) + 1e-8).sqrt()
    e2 = dot / (norm_s * norm_d)

    adj = torch.zeros(B, N, N, device=dev)
    adj.scatter_(2, knn_idx, 1.0)
    e3 = torch.zeros(B, E, device=dev)
    for b_idx_loop in range(B):
        e3[b_idx_loop] = (adj[b_idx_loop, src_all[b_idx_loop], dst_all[b_idx_loop]]
                          * adj[b_idx_loop, dst_all[b_idx_loop], src_all[b_idx_loop]])

    all_edge_feats = torch.stack([e0, e1, e2, e3], dim=2)
    all_edge_attr = all_edge_feats.reshape(B * E, EDGE_DIM)

    # -- Global features (B, GLOBAL_DIM=13) --
    gen_fracs = torch.tensor([s / max(max_steps, 1) for s in step_nums],
                             device=dev, dtype=torch.float32)
    progress = gen_fracs
    diversity = coords_norm.std(dim=1).mean(dim=1)
    imp_rate = torch.zeros(B, device=dev)

    fdc_num = ((fit - fit_mean_b.unsqueeze(1)) * dist_to_best).mean(dim=1)
    fdc_den = fit_std_b * dist_to_best.std(dim=1).clamp(min=1e-8)
    fdc = (fdc_num / fdc_den).clamp(-1, 1)

    gc_mean = gradient_consistency.mean(dim=1)

    mean_nn_fit = nn_fit.mean(dim=2)
    local_convexity = ((mean_nn_fit - fit) / fit_std_b.unsqueeze(1)).clamp(-3, 3) / 3.0
    conv_frac = (local_convexity > 0).float().mean(dim=1) * 2 - 1

    nn_dist_min = dist_topo.min(dim=2).values
    fit_diff_ij = fit.unsqueeze(1) - fit.unsqueeze(2)
    better_prob = torch.sigmoid(beta * fit_diff_ij)
    better_prob_masked = better_prob * (~diag_mask).float()
    has_better = better_prob_masked.sum(dim=2) > 0.5
    max_dist = dist_matrix.detach().amax(dim=(1, 2), keepdim=True) + 1.0
    dist_safe = torch.where(diag_mask, max_dist, dist_matrix)
    masked_dist = dist_safe + (1.0 - better_prob_masked) * max_dist
    dist_temp = masked_dist.std(dim=2).clamp(min=1e-4).unsqueeze(2)
    nbn_softmin = (torch.softmax(-masked_dist / dist_temp, dim=2) * masked_dist).sum(dim=2)
    nbc_ratio = torch.where(has_better, nbn_softmin / (nn_dist_min + 1e-8),
                            torch.ones_like(nn_dist_min)).clamp(0, 5)
    nbc_mean = nbc_ratio.mean(dim=1) / 2.5

    density_rank = soft_rank(local_density, beta=beta) / max(N - 1, 1)
    dr_centered = density_rank - density_rank.mean(dim=1, keepdim=True)
    fr_centered = fit_rank - fit_rank.mean(dim=1, keepdim=True)
    dr_std = density_rank.std(dim=1).clamp(min=1e-8)
    fr_std = fit_rank.std(dim=1).clamp(min=1e-8)
    dq_corr = ((dr_centered * fr_centered).mean(dim=1) / (dr_std * fr_std)).clamp(-1, 1)

    global_feats = torch.stack([
        progress * 2 - 1,
        diversity * 2 - 1,
        imp_rate,
        fdc,
        gc_mean * 2 - 1,
        conv_frac,
        nbc_mean * 2 - 1,
        stag_t,
        torch.zeros(B, device=dev),
        torch.zeros(B, device=dev),
        dq_corr,
        delta_fit_t,
        contraction_t,
    ], dim=1)

    # -- Assemble into PyG batch format --
    all_nodes = all_node_feats.reshape(B * N, NODE_DIM)
    v_indices = torch.arange(B, device=dev).unsqueeze(1).expand(B, N).reshape(-1)
    e_indices = torch.arange(B, device=dev).unsqueeze(1).expand(B, E).reshape(-1)

    return all_nodes, all_edges, all_edge_attr, global_feats, v_indices, e_indices
