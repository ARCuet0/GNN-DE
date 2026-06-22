"""
graph_builder_dense.py — Dense graph builder producing TopologyCache.

Contains:
  - build_dense_graphs_gpu (dense adjacency → TopologyCache)
"""

import torch

from .similarity_graph import LINEAGE_DIM, EDGE_DIM
from .graph_features import soft_rank
from .graph_utils import _temporal_to_globals


def build_dense_graphs_gpu(
    xs: torch.Tensor,
    fitnesses: torch.Tensor,
    step_num: int,
    max_steps: int,
    ndim: int,
    k_neighbors: int = 8,
    lb: float = -100.0,
    ub: float = 100.0,
    beta: float = 10.0,
    stagnation_counters: list = None,
    delta_fitnesses: list = None,
    contraction_rates: list = None,
):
    """Build P dense similarity graphs -- one per isolated population.

    Produces TopologyCache directly (no PyG sparse format, no v_indices).

    Args:
        xs:        (P, N, D) float32 coordinates
        fitnesses: (P, N) float32 fitness values
        step_num:  current generation (int, same for all P)
        max_steps: total generations
        ndim:      problem dimensionality D
        k_neighbors: number of nearest neighbors

    Returns:
        TopologyCache with adj (P,N,N), edge_feat (P,N,N,4),
        node_feat (P,N,8), global_feat (P,16), B=P, N=N.
    """
    from .dense_gatv2_backbone import TopologyCache

    P, N, D = xs.shape
    dev = xs.device
    span = ub - lb
    span_safe = span if abs(span) > 1e-12 else 1.0

    stag_t, delta_fit_t, contraction_t = _temporal_to_globals(
        stagnation_counters, delta_fitnesses, contraction_rates, P, dev)

    coords_norm = (xs.float() - lb) / span_safe

    log_fit = torch.log(fitnesses.float().clamp(min=1e-30))
    lf_mean = log_fit.mean(dim=1, keepdim=True)
    lf_std = log_fit.std(dim=1, keepdim=True).clamp(min=1e-4)
    fit = (log_fit - lf_mean) / lf_std

    dist_matrix = torch.cdist(coords_norm, coords_norm)
    dist_topo = dist_matrix.detach().clone()
    diag_mask = torch.eye(N, device=dev, dtype=torch.bool).unsqueeze(0)
    dist_topo = dist_topo.masked_fill(diag_mask, float('inf'))

    k = min(k_neighbors, N - 1)
    _, knn_idx = torch.topk(dist_topo, k, dim=2, largest=False)

    # -- Dense adjacency (bidirectional k-NN) --
    adj = torch.zeros(P, N, N, device=dev, dtype=torch.bool)
    adj.scatter_(2, knn_idx, True)
    adj = adj | adj.transpose(1, 2)
    adj = adj & ~diag_mask

    # -- Node features (P, N, 5 base + 3 lineage = 8) --
    fit_rank = soft_rank(fit, beta=beta) / max(N - 1, 1)
    fit_std_b = fit.std(dim=1).clamp(min=1e-8)

    softmin_w = torch.softmax(-fit / fit_std_b.unsqueeze(1), dim=1)
    best_x = (softmin_w.unsqueeze(2) * coords_norm).sum(dim=1)
    d2b = coords_norm - best_x.unsqueeze(1)
    dist_to_best = (d2b.pow(2).sum(dim=2) + 1e-8).sqrt()
    dtb_rank = soft_rank(dist_to_best, beta=beta) / max(N - 1, 1)

    row_exp = torch.arange(N, device=dev).unsqueeze(1).expand(N, k)
    b_idx = torch.arange(P, device=dev).reshape(P, 1, 1).expand(P, N, k)
    knn_dists = dist_matrix[b_idx, row_exp.unsqueeze(0).expand(P, N, k), knn_idx]
    local_density = knn_dists.mean(dim=2)
    max_density = local_density.max(dim=1, keepdim=True).values.clamp(min=1e-8)

    nn_coords = coords_norm[b_idx, knn_idx]
    diffs = nn_coords - coords_norm.unsqueeze(2)
    dists_nn = (diffs.pow(2).sum(dim=3, keepdim=True) + 1e-8).sqrt()
    directions = diffs / dists_nn
    nn_fit = fit.gather(1, knn_idx.reshape(P, -1)).reshape(P, N, k)
    f_diffs = nn_fit - fit.unsqueeze(2)
    grads = f_diffs / dists_nn.squeeze(-1)
    weighted = grads.unsqueeze(-1) * directions
    weighted_sum = weighted.sum(dim=2)
    total_abs_grad = grads.abs().sum(dim=2).clamp(min=1e-8)
    ws_norm = (weighted_sum.pow(2).sum(dim=2) + 1e-8).sqrt()
    gradient_consistency = (ws_norm / total_abs_grad).clamp(0, 1)

    centroid = coords_norm.mean(dim=1, keepdim=True)
    dist_to_centroid = ((coords_norm - centroid).pow(2).sum(dim=2) + 1e-8).sqrt()
    dtc_rank = soft_rank(dist_to_centroid, beta=beta) / max(N - 1, 1)

    node_feats_base = torch.stack([
        fit_rank * 2 - 1,
        dtb_rank * 2 - 1,
        (1 - local_density / max_density).clamp(0, 1) * 2 - 1,
        gradient_consistency * 2 - 1,
        dtc_rank * 2 - 1,
    ], dim=2)

    lineage = torch.zeros(P, N, LINEAGE_DIM, device=dev)
    lineage[:, :, 0] = -1.0
    lineage[:, :, 2] = -1.0
    node_feat_out = torch.cat([node_feats_base, lineage], dim=2)

    # -- Dense edge features (P, N, N, 4) --
    adj_float = adj.float()
    n_edges_per_pop = adj_float.sum(dim=(1, 2)).clamp(min=1)

    edge_dists = dist_matrix * adj_float
    flat_dists = edge_dists.reshape(P, -1)
    flat_adj = adj.reshape(P, -1)
    rank_dists = flat_dists.masked_fill(~flat_adj, float('inf'))
    dist_ranks = rank_dists.argsort(dim=1).argsort(dim=1).float()
    dist_ranks = dist_ranks.reshape(P, N, N)
    e0 = (dist_ranks / n_edges_per_pop.reshape(P, 1, 1)).clamp(0, 1) * 2 - 1
    e0 = e0 * adj_float

    fr_i = fit_rank.unsqueeze(2).expand(P, N, N)
    fr_j = fit_rank.unsqueeze(1).expand(P, N, N)
    e1 = ((fr_i - fr_j).abs() * 2 - 1) * adj_float

    centered = coords_norm - centroid
    dot = torch.bmm(centered, centered.transpose(1, 2))
    norms = (centered.pow(2).sum(dim=2) + 1e-8).sqrt()
    norm_ij = norms.unsqueeze(2) * norms.unsqueeze(1)
    e2 = (dot / norm_ij) * adj_float

    knn_adj = torch.zeros(P, N, N, device=dev, dtype=torch.bool)
    knn_adj.scatter_(2, knn_idx, True)
    mutual = (knn_adj & knn_adj.transpose(1, 2)).float()
    e3 = mutual * adj_float

    edge_feat_dense = torch.stack([e0, e1, e2, e3], dim=3)

    # -- Global features (P, 16) --
    gen_frac = step_num / max(max_steps, 1)
    diversity = coords_norm.std(dim=1).mean(dim=1)

    fit_mean_b = fit.mean(dim=1)
    fdc_num = ((fit - fit_mean_b.unsqueeze(1)) * dist_to_best).mean(dim=1)
    fdc_den = fit_std_b * dist_to_best.std(dim=1).clamp(min=1e-8)
    fdc = (fdc_num / fdc_den).clamp(-1, 1)

    gc_mean = gradient_consistency.mean(dim=1)

    mean_nn_fit = nn_fit.mean(dim=2)
    local_convexity = ((mean_nn_fit - fit) / fit_std_b.unsqueeze(1)).clamp(-3, 3) / 3.0
    conv_frac = (local_convexity > 0).float().mean(dim=1) * 2 - 1

    nn_dist = knn_dists[:, :, 0]
    better_mask_d = nn_fit < fit.unsqueeze(2)
    _inf = torch.tensor(float('inf'), device=dev)
    nbn_dists_d = torch.where(better_mask_d, knn_dists, _inf)
    nbn_dist_d = nbn_dists_d.min(dim=2).values
    nbn_dist_d = torch.where(nbn_dist_d.isinf(), nn_dist, nbn_dist_d)
    nbc_ratio = (nbn_dist_d / (nn_dist + 1e-8)).clamp(0, 5)
    nbc_mean = (nbc_ratio.mean(dim=1) / 2.5).clamp(0, 2)

    density_rank = soft_rank(local_density, beta=beta) / max(N - 1, 1)
    dr_c = density_rank - density_rank.mean(dim=1, keepdim=True)
    fr_c = fit_rank - fit_rank.mean(dim=1, keepdim=True)
    dq_corr = ((dr_c * fr_c).mean(dim=1) /
               (density_rank.std(dim=1).clamp(min=1e-8) *
                fit_rank.std(dim=1).clamp(min=1e-8))).clamp(-1, 1)

    # -- Landscape features (dense path) --
    _fit_centered_d = fit - fit.mean(dim=1, keepdim=True)
    _fit_std_d = fit.std(dim=1).clamp(min=1e-8)
    fit_kurt_d = (_fit_centered_d.pow(4).mean(dim=1) / _fit_std_d.pow(4).clamp(min=1e-8)) - 3.0
    fit_kurt_d = (fit_kurt_d / 10.0).clamp(-1, 1)

    _fit_diff_d = (fit.unsqueeze(2) - nn_fit).abs()
    ruggedness_d = (_fit_diff_d / knn_dists.clamp(min=1e-8)).mean(dim=(1, 2))
    ruggedness_d = (ruggedness_d.log1p() / 2.5).clamp(-1, 1)

    _dtb_rank_d = dist_to_best.argsort(dim=1).argsort(dim=1).float() / max(N - 1, 1)
    _fr_hard_d = fit.argsort(dim=1).argsort(dim=1).float() / max(N - 1, 1)
    _dr_c_d = _dtb_rank_d - _dtb_rank_d.mean(dim=1, keepdim=True)
    _fr_c_d = _fr_hard_d - _fr_hard_d.mean(dim=1, keepdim=True)
    spearman_d = ((_dr_c_d * _fr_c_d).sum(dim=1) /
                  (_dr_c_d.norm(dim=1) * _fr_c_d.norm(dim=1)).clamp(min=1e-8)).clamp(-1, 1)

    global_feat_out = torch.stack([
        torch.full((P,), gen_frac * 2 - 1, device=dev),
        diversity * 2 - 1,
        local_convexity.mean(dim=1),
        fdc,
        gc_mean * 2 - 1,
        conv_frac,
        nbc_mean * 2 - 1,
        stag_t,
        torch.zeros(P, device=dev),
        torch.zeros(P, device=dev),
        dq_corr,
        delta_fit_t,
        contraction_t,
        fit_kurt_d,
        ruggedness_d,
        spearman_d,
    ], dim=1)

    return TopologyCache(
        adj=adj,
        edge_feat=edge_feat_dense,
        B=P, N=N,
        node_feat=node_feat_out,
        global_feat=global_feat_out,
    )
