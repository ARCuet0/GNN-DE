"""
temporal_graph.py — W=2 temporal parent-child graph builder (GPU-resident).

Constructs a graph with TWO generations: current (indices 0..N-1) and parent
(indices N..2N-1), connected by spatial k-NN edges within each generation
and directed temporal edges (child -> parent) between generations.

Graph structure (~200 nodes, ~1700 edges for N=100, k=8):
  - 200 nodes: 100 current + 100 parent
  - ~1600 spatial edges: k-NN within each generation (bidirectional, deduplicated)
  - ~100 temporal edges: child i -> parent_map[i] + N
  - Total: ~1700 edges

Node features (16): 9 base + 7 zero-pad (lineage features not used for
  temporal graph since parent-child info is in the temporal edges).

Edge features (10): same 9 spatial features + 1 is_temporal flag.

Global features (16): computed from current generation only.

All tensors stay on the same device — zero CPU transfers.
Differentiable: features carry gradients; topology stays discrete.
"""

import torch
import torch.nn.functional as F

from .similarity_graph import (
    BASE_NODE_DIM, NODE_DIM, EDGE_DIM, GLOBAL_DIM,
    EDGE_NAMES,
)
from .techniques_v2 import N_TECHNIQUES
from .graph_features import (
    soft_rank,
    compute_shared_intermediates,
    compute_base_node_features,
    compute_edge_features,
    build_spatial_edge_index,
    compute_global_features,
)

# Temporal graph has EDGE_DIM + 1 (is_temporal flag)
TEMPORAL_EDGE_DIM = EDGE_DIM + 1
TEMPORAL_EDGE_NAMES = EDGE_NAMES + ['is_temporal']


def build_temporal_graph_gpu(
    x_curr: torch.Tensor,
    fitness_curr: torch.Tensor,
    x_prev: torch.Tensor,
    fitness_prev: torch.Tensor,
    parent_map: torch.Tensor,
    operator_probs: torch.Tensor = None,
    step_num: int = 1,
    max_steps: int = 50,
    ndim: int = 30,
    k: int = 8,
    technique_success_rates: torch.Tensor = None,
    prev_best: float = None,
    eval_budget_frac: float = None,
    improvement_ema: float = None,
    n_techniques: int = N_TECHNIQUES,
    beta: float = 10.0,
):
    """Build W=2 temporal parent-child graph entirely on GPU (differentiable).

    Args:
        x_curr:      (N, D) current population coordinates
        fitness_curr: (N,) current population fitness
        x_prev:      (N, D) parent population coordinates
        fitness_prev: (N,) parent population fitness
        parent_map:  (N,) long tensor, parent_map[i] = index in prev gen
        operator_probs: (N, K) soft routing probabilities (unused, reserved)
        step_num:    current generation number
        max_steps:   total generations
        ndim:        problem dimensionality D
        k:           k-NN neighbors for spatial edges
        technique_success_rates: (K,) tensor
        prev_best:   best fitness from previous generation
        eval_budget_frac: budget fraction override
        improvement_ema: EMA-smoothed improvement
        n_techniques: number of techniques
        beta:        sharpness for soft rank / sigmoid approximations

    Returns:
        nodes_t:     (2N, 16)    float32 on device
        edges_t:     (2, E_tot)  long    on device
        edge_attr_t: (E_tot, 10) float32 on device
        global_t:    (1, 16)     float32 on device
    """
    dev = x_curr.device
    N = x_curr.shape[0]
    D = x_curr.shape[1]

    # ================================================================
    # 1. Shared intermediates for both generations
    # ================================================================
    shared_curr = compute_shared_intermediates(
        x_curr, fitness_curr, k_neighbors=k, beta=beta)
    prev_step = max(step_num - 1, 0)
    shared_prev = compute_shared_intermediates(
        x_prev, fitness_prev, k_neighbors=k, beta=beta)

    # ================================================================
    # 2. Node features (9 base, padded to 16)
    # ================================================================
    nf_curr = compute_base_node_features(shared_curr, step_num, max_steps)
    nf_prev = compute_base_node_features(shared_prev, prev_step, max_steps)

    nf_curr_padded = F.pad(nf_curr, (0, NODE_DIM - BASE_NODE_DIM))
    nf_prev_padded = F.pad(nf_prev, (0, NODE_DIM - BASE_NODE_DIM))
    nodes_t = torch.cat([nf_curr_padded, nf_prev_padded], dim=0)

    # ================================================================
    # 3. Spatial edges within current generation
    # ================================================================
    k_eff = min(k, N - 1)
    ei_curr = build_spatial_edge_index(N, shared_curr['knn_idx'])
    ef_curr = compute_edge_features(shared_curr, ei_curr, beta=beta)

    # ================================================================
    # 4. Spatial edges within parent generation (offset by N)
    # ================================================================
    ei_prev = build_spatial_edge_index(N, shared_prev['knn_idx'])
    ef_prev = compute_edge_features(shared_prev, ei_prev, beta=beta)
    ei_prev = ei_prev + N  # offset indices

    # ================================================================
    # 5. Temporal edges: child i -> parent_map[i] + N
    # ================================================================
    child_idx = torch.arange(N, device=dev, dtype=torch.long)
    parent_idx = parent_map.long() + N
    temporal_ei = torch.stack([child_idx, parent_idx], dim=0)

    # Temporal edge features (9 + 1 is_temporal)
    cn_curr = shared_curr['coords_norm']
    cn_prev = shared_prev['coords_norm']
    fit_curr_z = shared_curr['fit']
    fit_prev_z = shared_prev['fit']
    fr_curr = shared_curr['fit_rank']
    fr_prev = shared_prev['fit_rank']
    cent_curr = shared_curr['centroid']
    cent_prev = shared_prev['centroid']

    cn_child = cn_curr[child_idx]
    cn_parent = cn_prev[parent_map]

    disp = cn_child - cn_parent
    disp_norm = (disp.pow(2).sum(dim=1) + 1e-8).sqrt()
    max_disp = disp_norm.max().clamp(min=1e-8)

    te0 = (disp_norm / max_disp).clamp(0, 1) * 2 - 1

    combined_fit_std = torch.cat([fit_curr_z, fit_prev_z]).std().clamp(min=1e-8)
    te1 = ((fit_curr_z[child_idx] - fit_prev_z[parent_map]) / combined_fit_std
           ).clamp(-3, 3) / 3.0

    cent_dir = cent_curr - cent_prev
    cent_dir_norm = (cent_dir.pow(2).sum() + 1e-8).sqrt()
    disp_norm_safe = disp_norm.clamp(min=1e-8)
    te2 = ((disp * cent_dir.unsqueeze(0)).sum(dim=1) /
           (disp_norm_safe * cent_dir_norm)).clamp(-1, 1)

    te3 = torch.ones(N, device=dev)

    disp_abs = disp.abs()
    disp_abs_sum = disp_abs.sum(dim=1)
    disp_abs_sq_sum = (disp_abs ** 2).sum(dim=1)
    t_eff_dims = (disp_abs_sum ** 2) / (disp_abs_sq_sum + 1e-8)
    te4 = (t_eff_dims / max(D, 1)).clamp(0, 1) * 2 - 1

    t_dist_ranks_soft = soft_rank(disp_norm, beta=beta)
    te5 = t_dist_ranks_soft / max(N - 1, 1) * 2 - 1

    te6 = (fr_curr[child_idx] - fr_prev[parent_map]).abs() * 2 - 1

    te7 = ((fit_curr_z[child_idx] - fit_prev_z[parent_map]).abs() /
           (combined_fit_std * disp_norm_safe + 1e-8)).clamp(0, 3) / 1.5 - 1

    improvement = (fit_prev_z[parent_map] - fit_curr_z[child_idx])
    improvement_scaled = (improvement / (fit_prev_z[parent_map].abs() + 1e-8)
                          ).clamp(-1, 1)
    te8 = improvement_scaled

    temporal_ef = torch.stack([te0, te1, te2, te3, te4, te5, te6, te7, te8],
                              dim=1)

    # ================================================================
    # 6. Concatenate all edges, add is_temporal flag (10th feature)
    # ================================================================
    spatial_flag_curr = torch.zeros(ef_curr.shape[0], 1, device=dev)
    spatial_flag_prev = torch.zeros(ef_prev.shape[0], 1, device=dev)
    ef_curr_10 = torch.cat([ef_curr, spatial_flag_curr], dim=1)
    ef_prev_10 = torch.cat([ef_prev, spatial_flag_prev], dim=1)

    temporal_flag = torch.ones(N, 1, device=dev)
    temporal_ef_10 = torch.cat([temporal_ef, temporal_flag], dim=1)

    edges_t = torch.cat([ei_curr, ei_prev, temporal_ei], dim=1)
    edge_attr_t = torch.cat([ef_curr_10, ef_prev_10, temporal_ef_10], dim=0)

    # ================================================================
    # 7. Global features from current generation only
    # ================================================================
    global_t = compute_global_features(
        shared_curr, ndim, step_num, max_steps,
        technique_success_rates=technique_success_rates,
        prev_best=prev_best,
        eval_budget_frac=eval_budget_frac,
        improvement_ema=improvement_ema,
        n_techniques=n_techniques,
        beta=beta,
    )

    return nodes_t, edges_t, edge_attr_t, global_t


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    torch.manual_seed(42)

    N = 100
    D = 30
    K = 4
    k_nn = 8
    dev = "cpu"

    print("=" * 60)
    print("Temporal graph smoke test (differentiable)")
    print("=" * 60)

    print("\n[Test 1] Shape and edge count")
    x_curr = torch.randn(N, D, device=dev, dtype=torch.float64) * 50
    x_prev = torch.randn(N, D, device=dev, dtype=torch.float64) * 50
    fitness_curr = torch.randn(N, device=dev, dtype=torch.float64).abs() * 1000 + 100
    fitness_prev = torch.randn(N, device=dev, dtype=torch.float64).abs() * 1000 + 200
    parent_map = torch.arange(N, device=dev, dtype=torch.long)
    operator_probs = torch.softmax(torch.randn(N, K, device=dev), dim=-1)

    nodes_t, edges_t, edge_attr_t, global_t = build_temporal_graph_gpu(
        x_curr, fitness_curr, x_prev, fitness_prev,
        parent_map, operator_probs,
        step_num=5, max_steps=50, ndim=D, k=k_nn,
    )

    assert nodes_t.shape == (2 * N, NODE_DIM)
    assert edge_attr_t.shape[1] == TEMPORAL_EDGE_DIM
    assert global_t.shape == (1, GLOBAL_DIM)

    E_total = edges_t.shape[1]
    print(f"  Nodes: {nodes_t.shape[0]}, Edges: {E_total}, "
          f"Node dim: {nodes_t.shape[1]}, Edge dim: {edge_attr_t.shape[1]}")
    assert E_total >= N
    print("  PASS")

    print("\n[Test 2] Temporal edge connectivity")
    is_temporal = edge_attr_t[:, -1]
    temporal_mask = is_temporal > 0.5
    n_temporal = temporal_mask.sum().item()
    assert n_temporal == N
    temporal_edges = edges_t[:, temporal_mask]
    for i in range(N):
        match = (temporal_edges[0] == i) & (temporal_edges[1] == parent_map[i].item() + N)
        assert match.any()
    print(f"  All {N} temporal edges verified. PASS")

    print("\n[Test 3] Gradient flow")
    x_c = torch.randn(N, D, device=dev, requires_grad=True)
    x_p = torch.randn(N, D, device=dev, requires_grad=True)
    f_c = (x_c ** 2).sum(dim=1)
    f_p = (x_p ** 2).sum(dim=1)
    pm = torch.arange(N, device=dev, dtype=torch.long)

    nodes, edges, eattr, glob = build_temporal_graph_gpu(
        x_c, f_c, x_p, f_p, pm,
        step_num=3, max_steps=50, ndim=D, k=k_nn,
    )
    loss = nodes.sum() + eattr.sum() + glob.sum()
    loss.backward()
    assert x_c.grad is not None and x_c.grad.abs().sum() > 0
    assert x_p.grad is not None and x_p.grad.abs().sum() > 0
    print(f"  x_curr grad norm: {x_c.grad.norm():.4f}")
    print(f"  x_prev grad norm: {x_p.grad.norm():.4f}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
