"""
graph_builder_sparse.py — Sparse graph builder with O(N*k) edge features.

Contains:
  - build_sparse_graphs_gpu (sparse kNN → SparseTopologyCache)
"""

import torch

from .similarity_graph import LINEAGE_DIM
from .graph_features import soft_rank
from .graph_utils import _temporal_to_globals


def _positivity_shift(*fits):
    """Per-population additive shift making min(fitness) >= 1, so ``log()`` is
    valid for non-positive fitness (e.g. BBOB instances with f_opt < 0). Returns
    a (P, 1) shift. NO-OP (shift == 0) whenever the population min is already
    >= 1, hence bit-identical on CEC2017 (fvals >= 100): the deployed checkpoint's
    CEC2017 behavior is unchanged, only non-positive (BBOB OOD) populations are
    affected. See finding_bbob_fitness_blindness_clamp_2026_06_05.
    """
    f_min = fits[0].min(dim=1, keepdim=True).values
    for f in fits[1:]:
        f_min = torch.minimum(f_min, f.min(dim=1, keepdim=True).values)
    return (1.0 - f_min).clamp(min=0.0)


def _safe_log_fitness(*fits):
    """``log()`` of each fitness tensor under a SHARED per-population positivity
    shift, so log-ratios across the inputs stay valid (the lineage / improvement
    features compare prev vs curr). Valid for non-positive fitness; bit-identical
    to ``log(f.clamp(min=1e-30))`` on CEC2017 where the shift is 0. Returns one
    shifted log per input. See finding_bbob_fitness_blindness_clamp_2026_06_05.
    """
    shift = _positivity_shift(*fits)
    return tuple(torch.log((f + shift).clamp(min=1e-30)) for f in fits)


def build_sparse_graphs_gpu(
    xs: torch.Tensor,
    fitnesses: torch.Tensor,
    step_num: int,
    max_steps: int,
    ndim: int,
    k_neighbors: int = 8,
    knn_idx: torch.Tensor = None,
    lb: float = -100.0,
    ub: float = 100.0,
    beta: float = 10.0,
    stagnation_counters: list = None,
    delta_fitnesses: list = None,
    contraction_rates: list = None,
    alive: torch.Tensor = None,
    prev_coords: torch.Tensor = None,
    prev_fitnesses: torch.Tensor = None,
):
    """Build P sparse similarity graphs with O(N*k) edge features.

    Same node/global features as build_dense_graphs_gpu.
    Edge features computed only for k neighbors via gather.

    Args:
        xs:        (P, N, D) float32 coordinates
        fitnesses: (P, N) float32 fitness values
        knn_idx:   (P, N, k) long -- precomputed neighbor indices (optional)
                   If None, computes kNN from coordinate space.

    Returns:
        SparseTopologyCache with knn_idx (P,N,k), edge_feat (P,N,k,4),
        node_feat (P,N,8), global_feat (P,16).
    """
    from .sparse_gatv2_backbone import SparseTopologyCache

    P, N, D = xs.shape
    dev = xs.device
    span = ub - lb
    span_safe = span if abs(span) > 1e-12 else 1.0

    stag_t, delta_fit_t, contraction_t = _temporal_to_globals(
        stagnation_counters, delta_fitnesses, contraction_rates, P, dev)

    coords_norm = (xs.float() - lb) / span_safe

    log_fit, = _safe_log_fitness(fitnesses.float())
    lf_mean = log_fit.mean(dim=1, keepdim=True)
    lf_std = log_fit.std(dim=1, keepdim=True).clamp(min=1e-4)
    fit = (log_fit - lf_mean) / lf_std
    converged = lf_std.squeeze(-1) < 1e-3
    fit = fit.masked_fill(converged.unsqueeze(-1), 0.0)

    # kNN computation (if not provided) -- chunked to avoid O(N^2) memory
    k = min(k_neighbors, N - 1)
    if knn_idx is None:
        from .topology_strategies import _chunked_knn
        knn_idx = _chunked_knn(coords_norm, k, chunk_size=512)

    # -- Node features (same as dense, P, N, 8) --
    fit_rank = soft_rank(fit, beta=beta) / max(N - 1, 1)
    fit_std_b = fit.std(dim=1).clamp(min=1e-8)

    softmin_w = torch.softmax(-fit / fit_std_b.unsqueeze(1), dim=1)
    best_x = (softmin_w.unsqueeze(2) * coords_norm).sum(dim=1)
    d2b = coords_norm - best_x.unsqueeze(1)
    dist_to_best = (d2b.pow(2).sum(dim=2) + 1e-8).sqrt()
    dtb_rank = soft_rank(dist_to_best, beta=beta) / max(N - 1, 1)

    b_idx = torch.arange(P, device=dev).reshape(P, 1, 1).expand(P, N, k)
    knn_coords = coords_norm[b_idx, knn_idx]  # (P, N, k, D)
    knn_diffs = knn_coords - coords_norm.unsqueeze(2)
    knn_dists = (knn_diffs.pow(2).sum(dim=3) + 1e-8).sqrt()  # (P, N, k)
    local_density = knn_dists.mean(dim=2)
    max_density = local_density.max(dim=1, keepdim=True).values.clamp(min=1e-8)

    dists_nn = knn_dists.unsqueeze(-1)  # (P, N, k, 1)
    directions = knn_diffs / dists_nn
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

    if prev_coords is not None and prev_fitnesses is not None:
        disp = ((xs.float() - prev_coords.float()).pow(2).sum(dim=2) + 1e-8).sqrt() / (D ** 0.5)
        disp_rank = soft_rank(disp, beta=beta) / max(N - 1, 1) * 2 - 1
        log_prev, log_curr = _safe_log_fitness(prev_fitnesses.float(), fitnesses.float())
        improv = (log_prev - log_curr).clamp(-3, 3)
        improv_rank = soft_rank(improv, beta=beta) / max(N - 1, 1) * 2 - 1
        prev_fr = soft_rank(log_prev, beta=beta) / max(N - 1, 1) * 2 - 1
        lineage = torch.stack([disp_rank, improv_rank, prev_fr], dim=2)
    else:
        lineage = torch.zeros(P, N, LINEAGE_DIM, device=dev)
        lineage[:, :, 0] = -1.0
        lineage[:, :, 2] = -1.0
    node_feat_out = torch.cat([node_feats_base, lineage], dim=2)

    # -- Sparse edge features (P, N, k, 4) via gather --
    dist_ranks = soft_rank(knn_dists.reshape(-1, k), beta=beta).reshape(P, N, k)
    e0 = (dist_ranks / max(k - 1, 1)) * 2 - 1

    fr_j = fit_rank.gather(1, knn_idx.reshape(P, -1)).reshape(P, N, k)
    e1 = (fit_rank.unsqueeze(2) - fr_j).abs() * 2 - 1

    centered = coords_norm - centroid
    c_j = knn_coords - centroid.unsqueeze(2)
    dot = (centered.unsqueeze(2) * c_j).sum(dim=-1)
    norm_i = (centered.pow(2).sum(dim=-1) + 1e-8).sqrt()
    norm_j = (c_j.pow(2).sum(dim=-1) + 1e-8).sqrt()
    e2 = dot / (norm_i.unsqueeze(2) * norm_j)

    knn_set = knn_idx
    i_idx = torch.arange(N, device=dev).unsqueeze(1).expand(N, k)
    i_idx = i_idx.unsqueeze(0).expand(P, -1, -1)
    j_neighbors = knn_set[b_idx, knn_idx]  # (P, N, k, k)
    e3 = (j_neighbors == i_idx.unsqueeze(-1)).any(dim=-1).float()

    edge_feat_out = torch.stack([e0, e1, e2, e3], dim=3)

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

    knn_dists_sorted = knn_dists.sort(dim=2).values
    nn_dist = knn_dists_sorted[:, :, 0]
    better_mask = nn_fit < fit.unsqueeze(2)
    _inf = torch.tensor(float('inf'), device=dev)
    nbn_dists = torch.where(better_mask, knn_dists, _inf)
    nbn_dist = nbn_dists.min(dim=2).values
    nbn_dist = torch.where(nbn_dist.isinf(), nn_dist, nbn_dist)
    nbc_ratio = (nbn_dist / (nn_dist + 1e-8)).clamp(0, 5)
    nbc_mean = (nbc_ratio.mean(dim=1) / 2.5).clamp(0, 2)

    density_rank = soft_rank(local_density, beta=beta) / max(N - 1, 1)
    dr_c = density_rank - density_rank.mean(dim=1, keepdim=True)
    fr_c = fit_rank - fit_rank.mean(dim=1, keepdim=True)
    dq_corr = ((dr_c * fr_c).mean(dim=1) /
               (density_rank.std(dim=1).clamp(min=1e-8) *
                fit_rank.std(dim=1).clamp(min=1e-8))).clamp(-1, 1)

    # -- Fix previously-zeroed features (2, 8, 9) --
    mean_local_convexity = local_convexity.mean(dim=1)

    front_vs_tail = torch.zeros(P, device=dev)
    if prev_fitnesses is not None:
        _lp, _lc = _safe_log_fitness(prev_fitnesses.float(), fitnesses.float())
        _log_imp = _lp - _lc
        _elite_mask = (fit_rank < 0.2).float()
        _tail_mask = (fit_rank > 0.8).float()
        _elite_imp = (_log_imp * _elite_mask).sum(dim=1) / _elite_mask.sum(dim=1).clamp(min=1)
        _tail_imp = (_log_imp * _tail_mask).sum(dim=1) / _tail_mask.sum(dim=1).clamp(min=1)
        front_vs_tail = ((_elite_imp - _tail_imp) / 3.0).clamp(-1, 1)

    direction_consensus = torch.zeros(P, device=dev)
    if prev_coords is not None:
        _disp = (xs - prev_coords).float()
        _top_k = max(int(N * 0.2), 2)
        _, _top_idx = torch.topk(fit_rank, _top_k, dim=1, largest=False)
        _top_disp = _disp.gather(1, _top_idx.unsqueeze(-1).expand(-1, -1, D))
        _norms = _top_disp.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-8).sqrt()
        _dirs = _top_disp / _norms
        _cos_mat = torch.bmm(_dirs, _dirs.transpose(1, 2))
        _eye = torch.eye(_top_k, device=dev).unsqueeze(0)
        _mask = 1.0 - _eye
        direction_consensus = ((_cos_mat * _mask).sum(dim=(1, 2))
                               / _mask.sum().clamp(min=1)).clamp(-1, 1)

    # -- New landscape features (13, 14, 15) --
    _fit_centered = fit - fit_mean_b.unsqueeze(1)
    fit_kurt = (_fit_centered.pow(4).mean(dim=1) / fit_std_b.pow(4).clamp(min=1e-8)) - 3.0
    fit_kurt = (fit_kurt / 10.0).clamp(-1, 1)

    _fit_diff = (fit.unsqueeze(2) - nn_fit).abs()
    ruggedness = (_fit_diff / knn_dists.clamp(min=1e-8)).mean(dim=(1, 2))
    ruggedness = (ruggedness.log1p() / 2.5).clamp(-1, 1)

    _dtb_rank = dist_to_best.argsort(dim=1).argsort(dim=1).float() / max(N - 1, 1)
    _fr_hard = fit.argsort(dim=1).argsort(dim=1).float() / max(N - 1, 1)
    _dr_c = _dtb_rank - _dtb_rank.mean(dim=1, keepdim=True)
    _fr_c_sp = _fr_hard - _fr_hard.mean(dim=1, keepdim=True)
    spearman = ((_dr_c * _fr_c_sp).sum(dim=1) /
                (_dr_c.norm(dim=1) * _fr_c_sp.norm(dim=1)).clamp(min=1e-8)).clamp(-1, 1)

    global_feat_out = torch.stack([
        torch.full((P,), gen_frac * 2 - 1, device=dev),
        diversity * 2 - 1,
        mean_local_convexity,
        fdc,
        gc_mean * 2 - 1,
        conv_frac,
        nbc_mean * 2 - 1,
        stag_t,
        front_vs_tail,
        direction_consensus,
        dq_corr,
        delta_fit_t,
        contraction_t,
        fit_kurt,
        ruggedness,
        spearman,
    ], dim=1)

    return SparseTopologyCache(
        knn_idx=knn_idx,
        edge_feat=edge_feat_out,
        B=P, N=N, k=k,
        node_feat=node_feat_out,
        global_feat=global_feat_out,
        alive=alive,
    )


def build_dense_edge_attr_gpu(
    coords: torch.Tensor,
    fitness: torch.Tensor,
    lb: float = -100.0,
    ub: float = 100.0,
    beta: float = 10.0,
) -> torch.Tensor:
    """Dense all-to-all edge attributes for the B2 (set-attn + edge bias) arm.

    Mirrors the per-pair formulas of `build_sparse_graphs_gpu` but over ALL
    pairs (B, N, N) — used by `TemporalSetAttentionEdgeBackbone` to inject an
    edge bias into the all-to-all attention logits. The reciprocity feature
    is dropped (degenerate when every pair is connected); 3 features remain:

      e0_dense: distance soft-rank per source row, scaled to [-1, 1].
                Per row i, softrank over the N pair-distances to j.
                Self-pair has distance 0 → smallest rank → -1.
      e1_dense: |fit_rank_i - fit_rank_j| * 2 - 1 ∈ [-1, 1].
                Self-pair → -1 (zero magnitude).
      e2_dense: cosine of (i - centroid) and (j - centroid) ∈ [-1, 1].
                Self-pair → +1.

    Args:
        coords:  (B, N, D) raw search-space coordinates.
        fitness: (B, N) fitness values (positive).
        lb, ub:  search box bounds; coords get normalized to [0, 1] via
                 (x - lb) / (ub - lb). Default = CEC2017 box.
        beta:    soft-rank sigmoid sharpness (same default as the sparse builder).

    Returns:
        (B, N, N, 3) float32 dense edge attribute tensor.
    """
    from .graph_features import soft_rank

    B, N, D = coords.shape
    dev = coords.device
    span = ub - lb
    span_safe = span if abs(span) > 1e-12 else 1.0
    coords_norm = (coords.float() - lb) / span_safe

    # Standardized log-fitness, same as the sparse builder.
    log_fit = torch.log(fitness.float().clamp(min=1e-30))
    lf_mean = log_fit.mean(dim=1, keepdim=True)
    lf_std = log_fit.std(dim=1, keepdim=True).clamp(min=1e-4)
    fit = (log_fit - lf_mean) / lf_std
    converged = lf_std.squeeze(-1) < 1e-3
    fit = fit.masked_fill(converged.unsqueeze(-1), 0.0)

    # fit_rank in [0, 1].
    fit_rank = soft_rank(fit, beta=beta) / max(N - 1, 1)

    # --- e0_dense: row-normalized pairwise distance in [-1, 1] ---
    # The sparse builder uses a soft-rank of distances; the dense version
    # would have to apply soft_rank over (B, N, N) input which costs
    # O(B*N*N*N) memory for the internal pairwise-diff (17 GB at N≈1050 in
    # the surrogate-augmented pop). Use a row min-max instead — both
    # constructs are monotone in distance with the same sign convention:
    # closest pair → +1, farthest pair → -1 (matches the sparse builder's
    # soft_rank convention, which gives the largest rank to the smallest
    # value and is scaled to +1).
    diffs = coords_norm.unsqueeze(2) - coords_norm.unsqueeze(1)   # (B, N, N, D)
    dist = (diffs.pow(2).sum(dim=3) + 1e-8).sqrt()                # (B, N, N)
    row_min = dist.amin(dim=2, keepdim=True)                      # (B, N, 1)
    row_max = dist.amax(dim=2, keepdim=True)
    row_span = (row_max - row_min).clamp(min=1e-8)
    e0 = 1 - 2 * (dist - row_min) / row_span                      # close → +1

    # --- e1_dense: dense |fit_rank_i - fit_rank_j| * 2 - 1 ---
    fit_rank_i = fit_rank.unsqueeze(2)                            # (B, N, 1)
    fit_rank_j = fit_rank.unsqueeze(1)                            # (B, 1, N)
    e1 = (fit_rank_i - fit_rank_j).abs() * 2 - 1                  # (B, N, N)

    # --- e2_dense: cosine of (i - centroid) and (j - centroid) ---
    centroid = coords_norm.mean(dim=1, keepdim=True)              # (B, 1, D)
    centered = coords_norm - centroid                             # (B, N, D)
    norm = (centered.pow(2).sum(dim=-1) + 1e-8).sqrt()            # (B, N)
    dot = torch.bmm(centered, centered.transpose(1, 2))           # (B, N, N)
    e2 = dot / (norm.unsqueeze(2) * norm.unsqueeze(1))            # (B, N, N)

    return torch.stack([e0, e1, e2], dim=3)                       # (B, N, N, 3)
