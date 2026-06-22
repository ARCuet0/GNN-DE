"""
similarity_graph_gpu.py — GPU-resident similarity graph builder.

Pure PyTorch reimplementation of build_similarity_graph() from
similarity_graph.py.  Zero CPU<->GPU transfers: all inputs and outputs
stay on the same device as `x`.

Differentiable version: all node and edge FEATURES carry gradients back
to `x` and `fitness`.  Topology (edge_index) remains discrete, computed
from detached coordinates so that the k-NN sparsity pattern is stable
for PNA message passing.

Feature parity: same 9 base node + 7 lineage node + 9 edge + 16 global
features, same scaling to ~[-1, 1], same k-NN graph structure.
"""

import math

import torch

from .similarity_graph import (
    BASE_NODE_DIM, NODE_DIM, EDGE_DIM, GLOBAL_DIM,
    NODE_NAMES, EDGE_NAMES, GLOBAL_NAMES,
)
from .techniques_v2 import N_TECHNIQUES
from .graph_features import (
    soft_rank,
    augment_node_features_lineage,
    compute_shared_intermediates,
    compute_base_node_features,
    compute_edge_features,
    build_spatial_edge_index,
    compute_global_features,
    make_degenerate_global,
)


def build_similarity_graph_gpu(
    x: torch.Tensor,
    fitness: torch.Tensor,
    step_num: int,
    max_steps: int,
    ndim: int,
    lb: float = -100.0,
    ub: float = 100.0,
    generation_ids: torch.Tensor = None,
    technique_success_rates: torch.Tensor = None,
    prev_best: float = None,
    k_neighbors: int = 8,
    eval_budget_frac: float = None,
    improvement_ema: float = None,
    n_techniques: int = N_TECHNIQUES,
    parent_info: dict = None,
    K: int = 4,
    max_K: int = 4,
    beta: float = 10.0,
):
    """Build k-NN similarity graph entirely on GPU (differentiable).

    All node and edge features are differentiable w.r.t. x and fitness.
    Topology (edge_index) is computed from detached coordinates for
    stable k-NN sparsity.

    Args:
        x:            (N, D) coordinates, float, on device (may require grad)
        fitness:      (N,) fitness values, float, on device (may require grad)
        step_num:     current generation number
        max_steps:    total generations
        ndim:         problem dimensionality D (for global feature)
        lb, ub:       coordinate bounds (scalar)
        generation_ids: (N,) int tensor
        technique_success_rates: (K,) tensor on device
        prev_best:    best fitness from previous generation (scalar)
        k_neighbors:  number of nearest neighbors
        eval_budget_frac: evaluation budget fraction
        improvement_ema:  EMA-smoothed improvement rate
        parent_info:  dict with lineage info (see augment_node_features_lineage)
        K:            number of operators
        beta:         sharpness for soft rank / sigmoid approximations

    Returns:
        nodes_t:     (N, 16)  float32 on device
        edges_t:     (2, E)   long    on device
        edge_attr_t: (E, 9)   float32 on device
        global_t:    (1, 16)  float32 on device
    """
    dev = x.device
    N = x.shape[0]
    gen_frac = step_num / max(max_steps, 1)

    # ---- Trivial / degenerate cases ----
    if N <= 1:
        base_features = torch.zeros(N, BASE_NODE_DIM, device=dev)
        if N == 1:
            base_features[0, 8] = gen_frac * 2 - 1
        node_features = augment_node_features_lineage(base_features, parent_info, K=K, max_K=max_K)
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev)
        edge_features = torch.zeros(0, EDGE_DIM, device=dev)
        global_t = make_degenerate_global(
            N, ndim, step_num, max_steps, dev,
            technique_success_rates, eval_budget_frac, n_techniques)
        return node_features, edge_index, edge_features, global_t

    # ---- Shared intermediates ----
    shared = compute_shared_intermediates(
        x, fitness, k_neighbors=k_neighbors, lb=lb, ub=ub, beta=beta)

    # ---- Edge index ----
    edge_index = build_spatial_edge_index(N, shared['knn_idx'])
    E = edge_index.shape[1]

    if E == 0:
        base_features = torch.zeros(N, BASE_NODE_DIM, device=dev)
        base_features[:, 8] = gen_frac * 2 - 1
        node_features = augment_node_features_lineage(base_features, parent_info, K=K, max_K=max_K)
        edge_features = torch.zeros(0, EDGE_DIM, device=dev)
        global_t = make_degenerate_global(
            N, ndim, step_num, max_steps, dev,
            technique_success_rates, eval_budget_frac, n_techniques)
        return node_features, edge_index, edge_features, global_t

    # ---- Node features (9 base + 7 lineage) ----
    base_features = compute_base_node_features(
        shared, step_num, max_steps, generation_ids)
    node_features = augment_node_features_lineage(base_features, parent_info, K=K, max_K=max_K)

    # ---- Edge features (9) ----
    edge_features = compute_edge_features(shared, edge_index, beta=beta)

    # ---- Global features (16) ----
    global_t = compute_global_features(
        shared, ndim, step_num, max_steps,
        technique_success_rates=technique_success_rates,
        prev_best=prev_best,
        eval_budget_frac=eval_budget_frac,
        improvement_ema=improvement_ema,
        n_techniques=n_techniques,
        beta=beta,
    )

    return node_features, edge_index, edge_features, global_t


# ======================================================================
# Batched graph construction for multi-population training
# ======================================================================

def build_batched_uniform_graphs_gpu(
    xs: torch.Tensor,
    fitnesses: torch.Tensor,
    step_nums: list,
    max_steps: int,
    ndim: int,
    prev_bests: torch.Tensor = None,
    k_neighbors: int = 8,
    K: int = 4,
    max_K: int = 4,
    beta: float = 10.0,
    lb: float = -100.0,
    ub: float = 100.0,
):
    """Build B similarity graphs in parallel when all have the same N.

    All heavy computation (cdist, soft_rank, kNN, features) is batched
    as (B, N, ...) tensors — one kernel per operation, not B kernels.

    Args:
        xs:        (B, N, D) coordinates
        fitnesses: (B, N) fitness values
        step_nums: list of B ints
        max_steps: int
        ndim:      int (same for all)
        prev_bests: (B,) or None

    Returns:
        all_nodes, all_edges, all_edge_attr, all_global, v_indices, e_indices
        (same format as build_batched_similarity_graphs_gpu)
    """
    B, N, D = xs.shape
    dev = xs.device
    span = ub - lb
    span_safe = span if abs(span) > 1e-12 else 1.0

    # ── Normalize coordinates ──
    coords_norm = (xs.float() - lb) / span_safe  # (B, N, D)

    # ── Z-scored log-fitness ──
    log_fit = torch.log(fitnesses.float().clamp(min=1e-30))  # (B, N)
    lf_mean = log_fit.mean(dim=1, keepdim=True)
    lf_std = log_fit.std(dim=1, keepdim=True).clamp(min=1e-4)
    fit = (log_fit - lf_mean) / lf_std  # (B, N)

    # ── Batched pairwise distances ──
    dist_matrix = torch.cdist(coords_norm, coords_norm)  # (B, N, N) — 1 kernel!
    dist_topo = torch.cdist(coords_norm.detach(), coords_norm.detach())
    # Mask diagonal
    diag_mask = torch.eye(N, device=dev, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
    dist_topo = dist_topo.masked_fill(diag_mask, float('inf'))

    # ── Batched kNN ──
    k = min(k_neighbors, N - 1)
    _, knn_idx = torch.topk(dist_topo, k, dim=2, largest=False)  # (B, N, k)

    # ── Batched soft_rank (fitness) ──
    fit_rank = soft_rank(fit, beta=beta) / max(N - 1, 1)  # (B, N)
    fit_mean = fit.mean(dim=1)  # (B,)
    fit_std = fit.std(dim=1).clamp(min=1e-8)  # (B,)
    fit_zscore = ((fit - fit_mean.unsqueeze(1)) / fit_std.unsqueeze(1)).clamp(-3, 3)

    # ── Batched best position (softmin) ──
    softmin_w = torch.softmax(-fit / fit_std.unsqueeze(1), dim=1)  # (B, N)
    best_x = (softmin_w.unsqueeze(2) * coords_norm).sum(dim=1)  # (B, D)
    d2b = coords_norm - best_x.unsqueeze(1)  # (B, N, D)
    dist_to_best = (d2b.pow(2).sum(dim=2) + 1e-8).sqrt()  # (B, N)
    dtb_rank = soft_rank(dist_to_best, beta=beta) / max(N - 1, 1)  # (B, N)

    # ── Batched local density ──
    row_exp = torch.arange(N, device=dev).unsqueeze(1).expand(N, k)  # (N, k)
    # Gather kNN distances: (B, N, k)
    knn_dists = dist_matrix[
        torch.arange(B, device=dev).reshape(B, 1, 1).expand(B, N, k),
        row_exp.unsqueeze(0).expand(B, N, k),
        knn_idx
    ]
    local_density = knn_dists.mean(dim=2)  # (B, N)
    max_density = local_density.max(dim=1, keepdim=True).values.clamp(min=1e-8)

    # ── Batched gradient consistency ──
    # coords_norm: (B, N, D), knn_idx: (B, N, k) → nn_coords: (B, N, k, D)
    b_idx = torch.arange(B, device=dev).reshape(B, 1, 1).expand(B, N, k)
    nn_coords = coords_norm[b_idx, knn_idx]  # (B, N, k, D)
    diffs = nn_coords - coords_norm.unsqueeze(2)  # (B, N, k, D)
    dists_nn = (diffs.pow(2).sum(dim=3, keepdim=True) + 1e-8).sqrt()
    directions = diffs / dists_nn
    nn_fit = fit.gather(1, knn_idx.reshape(B, -1)).reshape(B, N, k)  # (B, N, k)
    f_diffs = nn_fit - fit.unsqueeze(2)
    grads = f_diffs / dists_nn.squeeze(-1)
    weighted = grads.unsqueeze(-1) * directions
    weighted_sum = weighted.sum(dim=2)  # (B, N, D)
    total_abs_grad = grads.abs().sum(dim=2).clamp(min=1e-8)  # (B, N)
    ws_norm = (weighted_sum.pow(2).sum(dim=2) + 1e-8).sqrt()
    gradient_consistency = (ws_norm / total_abs_grad).clamp(0, 1)

    # ── Batched local convexity ──
    mean_nn_fit = nn_fit.mean(dim=2)  # (B, N)
    local_convexity = ((mean_nn_fit - fit) / fit_std.unsqueeze(1)).clamp(-3, 3) / 3.0

    # ── Batched NBC ratio ──
    nn_dist_min = dist_topo.min(dim=2).values  # (B, N) — min dist ignoring self
    fit_diff_ij = fit.unsqueeze(1) - fit.unsqueeze(2)  # (B, N, N)
    better_prob = torch.sigmoid(beta * fit_diff_ij)
    better_prob_masked = better_prob * (~diag_mask).float()
    has_better = better_prob_masked.sum(dim=2) > 0.5  # (B, N)
    max_dist = dist_matrix.detach().amax(dim=(1, 2), keepdim=True) + 1.0
    dist_safe = torch.where(diag_mask, max_dist, dist_matrix)
    masked_dist = dist_safe + (1.0 - better_prob_masked) * max_dist
    dist_temp = masked_dist.std(dim=2).clamp(min=1e-4).unsqueeze(2)
    nbn_softmin = (torch.softmax(-masked_dist / dist_temp, dim=2) * masked_dist).sum(dim=2)
    nbc_ratio = torch.where(
        has_better,
        nbn_softmin / (nn_dist_min + 1e-8),
        torch.ones_like(nn_dist_min)
    ).clamp(0, 5)

    # ── Node features (B, N, 9) ──
    gen_fracs = torch.tensor(
        [s / max(max_steps, 1) for s in step_nums],
        device=dev, dtype=torch.float32)  # (B,)
    tenure = gen_fracs.unsqueeze(1).expand(B, N)  # placeholder

    node_feats = torch.stack([
        fit_rank * 2 - 1,
        fit_zscore / 3.0,
        dtb_rank * 2 - 1,
        (1 - local_density / max_density).clamp(0, 1) * 2 - 1,
        gradient_consistency * 2 - 1,
        local_convexity,
        (nbc_ratio / 2.5).clamp(0, 2) - 1,
        tenure * 2 - 1,
        gen_fracs.unsqueeze(1).expand(B, N) * 2 - 1,
    ], dim=2)  # (B, N, 9)

    # Lineage: zeros (no parent info in ES-Single)
    lineage = torch.zeros(B, N, NODE_DIM - BASE_NODE_DIM, device=dev)
    all_node_feats = torch.cat([node_feats, lineage], dim=2)  # (B, N, 16)

    # ── Edge index + features (per-graph, can't fully avoid loop for variable E) ──
    # But kNN with same k gives same E per graph → vectorizable
    # Build edge_index for all B graphs at once
    # Each graph has the same kNN structure: bidirectional deduplicated
    # For uniform N,k: E is the same for all graphs

    # Build ONE template edge_index, then replicate with offsets
    template_ei = build_spatial_edge_index(N, knn_idx[0])  # (2, E)
    E = template_ei.shape[1]

    # Replicate for all B graphs with node offsets
    offsets = torch.arange(B, device=dev).unsqueeze(1) * N  # (B, 1)
    all_edges = (template_ei.unsqueeze(0) + offsets.unsqueeze(2)).reshape(2, B * E)

    # Edge features — batched
    src_templ = template_ei[0]  # (E,)
    dst_templ = template_ei[1]  # (E,)

    # Gather per-graph edge data using template indices
    edge_dists = dist_matrix[:, src_templ, dst_templ]  # (B, E)
    max_edge_dist = edge_dists.max(dim=1, keepdim=True).values.clamp(min=1e-8)

    e0 = (edge_dists / max_edge_dist).clamp(0, 1) * 2 - 1
    e1 = ((fit[:, src_templ] - fit[:, dst_templ]) / fit_std.unsqueeze(1)).clamp(-3, 3) / 3.0

    centroid = coords_norm.mean(dim=1, keepdim=True)  # (B, 1, D)
    c_src = coords_norm[:, src_templ] - centroid  # (B, E, D)
    c_dst = coords_norm[:, dst_templ] - centroid
    dot = (c_src * c_dst).sum(dim=2)
    norm_s = (c_src.pow(2).sum(dim=2) + 1e-8).sqrt()
    norm_d = (c_dst.pow(2).sum(dim=2) + 1e-8).sqrt()
    e2 = dot / (norm_s * norm_d)

    # Mutual kNN: build batched adjacency
    adj = torch.zeros(B, N, N, device=dev)
    row_exp_b = row_exp.unsqueeze(0).expand(B, N, k)
    adj.scatter_(2, knn_idx, 1.0)
    e3 = adj[:, src_templ, dst_templ] * adj[:, dst_templ, src_templ]  # (B, E)

    edge_delta = (coords_norm[:, src_templ] - coords_norm[:, dst_templ]).abs()
    delta_sum = edge_delta.sum(dim=2)
    delta_sq_sum = (edge_delta ** 2).sum(dim=2)
    eff_dims = (delta_sum ** 2) / (delta_sq_sum + 1e-8)
    e4 = (eff_dims / max(D, 1)).clamp(0, 1) * 2 - 1

    # Edge soft_rank — batched
    e5_rank = soft_rank(edge_dists, beta=beta)  # (B, E)
    e5 = e5_rank / max(E - 1, 1) * 2 - 1

    e6 = (fit_rank[:, src_templ] - fit_rank[:, dst_templ]).abs() * 2 - 1
    e7 = ((fit[:, src_templ] - fit[:, dst_templ]).abs()
           / (fit_std.unsqueeze(1) * edge_dists + 1e-8)).clamp(0, 3) / 1.5 - 1

    e8 = (adj[:, src_templ] * adj[:, dst_templ]).sum(dim=2) / max(k, 1) * 2 - 1

    all_edge_feats = torch.stack([e0, e1, e2, e3, e4, e5, e6, e7, e8], dim=2)  # (B, E, 9)
    all_edge_attr = all_edge_feats.reshape(B * E, EDGE_DIM)

    # ── Global features (B, 16) — simplified for ES ──
    progress = gen_fracs
    diversity = coords_norm.std(dim=1).mean(dim=1)  # (B,)
    imp_rate = torch.zeros(B, device=dev)  # placeholder
    # FDC
    best_fit = fit.min(dim=1).values  # (B,)
    fdc_num = ((fit - fit_mean.unsqueeze(1)) * dist_to_best).mean(dim=1)
    fdc_den = fit_std * dist_to_best.std(dim=1).clamp(min=1e-8)
    fdc = (fdc_num / fdc_den).clamp(-1, 1)

    gc_mean = gradient_consistency.mean(dim=1)
    conv_frac = (local_convexity > 0).float().mean(dim=1) * 2 - 1
    nbc_mean = nbc_ratio.mean(dim=1) / 2.5
    norm_dim = torch.full((B,), math.log10(ndim / 100.0), device=dev)

    # Technique success rates: zeros (not available in ES)
    tech_rates = torch.zeros(B, N_TECHNIQUES, device=dev)

    global_feats = torch.cat([
        progress.unsqueeze(1) * 2 - 1,
        diversity.unsqueeze(1) * 2 - 1,
        imp_rate.unsqueeze(1),
        fdc.unsqueeze(1),
        gc_mean.unsqueeze(1) * 2 - 1,
        conv_frac.unsqueeze(1),
        nbc_mean.unsqueeze(1) * 2 - 1,
        norm_dim.unsqueeze(1),
        tech_rates,
    ], dim=1)  # (B, 8 + N_TECHNIQUES)

    # Pad to GLOBAL_DIM if needed
    if global_feats.shape[1] < GLOBAL_DIM:
        global_feats = torch.nn.functional.pad(
            global_feats, (0, GLOBAL_DIM - global_feats.shape[1]))
    global_feats = global_feats[:, :GLOBAL_DIM]  # (B, 16)

    # ── Assemble into PyG batch format ──
    all_nodes = all_node_feats.reshape(B * N, NODE_DIM)
    v_indices = torch.arange(B, device=dev).unsqueeze(1).expand(B, N).reshape(-1)
    e_indices = torch.arange(B, device=dev).unsqueeze(1).expand(B, E).reshape(-1)

    return all_nodes, all_edges, all_edge_attr, global_feats, v_indices, e_indices


def build_batched_similarity_graphs_gpu(
    xs: list,
    fitnesses: list,
    step_nums: list,
    max_steps: int,
    ndims: list,
    parent_infos: list = None,
    **kwargs,
):
    """Build B similarity graphs and merge into a single PyG-style batch.

    Returns:
        all_nodes, all_edges, all_edge_attr, all_global, v_indices, e_indices
    """
    B = len(xs)
    dev = xs[0].device

    nodes_list = []
    edges_list = []
    edge_attr_list = []
    global_list = []
    v_idx_list = []
    e_idx_list = []
    node_offset = 0

    if parent_infos is None:
        parent_infos = [None] * B

    per_graph_kw = {}
    shared_kw = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and len(v) == B:
            per_graph_kw[k] = v
        else:
            shared_kw[k] = v

    for b in range(B):
        kw_b = {k: v[b] for k, v in per_graph_kw.items()}
        kw_b.update(shared_kw)
        n_t, ei_t, ea_t, g_t = build_similarity_graph_gpu(
            xs[b], fitnesses[b],
            step_num=step_nums[b], max_steps=max_steps, ndim=ndims[b],
            parent_info=parent_infos[b],
            **kw_b,
        )
        N_b = n_t.shape[0]
        E_b = ei_t.shape[1]

        nodes_list.append(n_t)
        edges_list.append(ei_t + node_offset)
        edge_attr_list.append(ea_t)
        global_list.append(g_t.squeeze(0))
        v_idx_list.append(torch.full((N_b,), b, dtype=torch.long, device=dev))
        e_idx_list.append(torch.full((E_b,), b, dtype=torch.long, device=dev))

        node_offset += N_b

    all_nodes = torch.cat(nodes_list, dim=0)
    all_edges = torch.cat(edges_list, dim=1) if edges_list else torch.zeros(2, 0, dtype=torch.long, device=dev)
    all_edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else torch.zeros(0, EDGE_DIM, device=dev)
    all_global = torch.stack(global_list, dim=0)
    v_indices = torch.cat(v_idx_list)
    e_indices = torch.cat(e_idx_list) if e_idx_list else torch.zeros(0, dtype=torch.long, device=dev)

    return all_nodes, all_edges, all_edge_attr, all_global, v_indices, e_indices


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Differentiable similarity graph -- gradient flow test")
    print("=" * 60)

    torch.manual_seed(42)
    N, D = 100, 10
    dev = "cpu"

    x = torch.randn(N, D, device=dev, requires_grad=True)
    fitness_raw = torch.randn(N, device=dev, requires_grad=True)
    fitness = fitness_raw * 1000 + 5000
    fitness.retain_grad()

    nodes_t, edges_t, edge_attr_t, global_t = build_similarity_graph_gpu(
        x, fitness, step_num=7, max_steps=50, ndim=D,
    )

    print(f"\nShapes:")
    print(f"  nodes:     {nodes_t.shape}")
    print(f"  edges:     {edges_t.shape}")
    print(f"  edge_attr: {edge_attr_t.shape}")
    print(f"  global:    {global_t.shape}")

    loss_nodes = nodes_t.sum()
    loss_nodes.backward(retain_graph=True)

    x_grad_nodes = x.grad.clone() if x.grad is not None else None
    f_grad_nodes = fitness.grad.clone() if fitness.grad is not None else None
    x.grad = None
    fitness.grad = None

    print(f"\n--- Gradient flow through node_features.sum() ---")
    print(f"  x.grad norm:       {x_grad_nodes.norm().item():.6f}" if x_grad_nodes is not None else "  x.grad: NONE")
    print(f"  fitness.grad norm: {f_grad_nodes.norm().item():.6f}" if f_grad_nodes is not None else "  fitness.grad: NONE")

    loss_edges = edge_attr_t.sum()
    loss_edges.backward(retain_graph=True)
    x_grad_edges = x.grad.clone() if x.grad is not None else None
    f_grad_edges = fitness.grad.clone() if fitness.grad is not None else None
    x.grad = None
    fitness.grad = None

    loss_global = global_t.sum()
    loss_global.backward()
    x_grad_global = x.grad.clone() if x.grad is not None else None
    f_grad_global = fitness.grad.clone() if fitness.grad is not None else None

    all_pass = True
    for name, g in [("x (nodes)", x_grad_nodes), ("fitness (nodes)", f_grad_nodes),
                     ("x (edges)", x_grad_edges), ("fitness (edges)", f_grad_edges),
                     ("x (global)", x_grad_global), ("fitness (global)", f_grad_global)]:
        ok = g is not None and g.norm().item() > 0
        if not ok:
            all_pass = False
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    print(f"\n{'PASS' if all_pass else 'FAIL'}: All gradient checks "
          f"{'passed' if all_pass else 'FAILED'}.")

    print(f"\nNode feature ranges:")
    for i, name in enumerate(NODE_NAMES):
        print(f"  {name:25s}: [{nodes_t[:, i].min().item():.3f}, "
              f"{nodes_t[:, i].max().item():.3f}]")
    print(f"\nEdge feature ranges:")
    for i, name in enumerate(EDGE_NAMES):
        print(f"  {name:25s}: [{edge_attr_t[:, i].min().item():.3f}, "
              f"{edge_attr_t[:, i].max().item():.3f}]")
    print(f"\nGlobal features:")
    for i, name in enumerate(GLOBAL_NAMES):
        print(f"  {name:25s}: {global_t[0, i].item():.3f}")
