"""
graph_builder_single.py — Single-population and list-batched graph builders.

Contains:
  - build_similarity_graph_gpu (single population, k-NN sparse, PyG format)
  - build_batched_similarity_graphs_gpu (list of populations → PyG batch)
"""

import torch

from .similarity_graph import (
    BASE_NODE_DIM, LINEAGE_DIM, NODE_DIM, EDGE_DIM, GLOBAL_DIM,
    NODE_NAMES, EDGE_NAMES, GLOBAL_NAMES,
)
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
    prev_best: float = None,
    k_neighbors: int = 8,
    eval_budget_frac: float = None,
    improvement_ema: float = None,
    parent_info: dict = None,
    stagnation_counter: int = 0,
    prev_x: torch.Tensor = None,
    prev_fitness: torch.Tensor = None,
    beta: float = 10.0,
    delta_fitness: float = 0.0,
    contraction_rate: float = 0.0,
):
    """Build k-NN similarity graph entirely on GPU (differentiable).

    Args:
        x:            (N, D) coordinates on device
        fitness:      (N,) fitness values on device
        step_num:     current generation number
        max_steps:    total generations
        ndim:         problem dimensionality D
        lb, ub:       coordinate bounds (scalar)
        generation_ids: (N,) int tensor
        prev_best:    best fitness from previous generation (scalar)
        k_neighbors:  number of nearest neighbors
        eval_budget_frac: evaluation budget fraction
        improvement_ema:  EMA-smoothed improvement rate
        parent_info:  dict with lineage info (see augment_node_features_lineage)
        stagnation_counter: generations since population best improved
        prev_x:       (N, D) previous generation coordinates
        prev_fitness:  (N,) previous generation fitness
        beta:         sharpness for soft rank / sigmoid approximations

    Returns:
        nodes_t:     (N, 9)   float32 on device
        edges_t:     (2, E)   long    on device
        edge_attr_t: (E, 4)   float32 on device
        global_t:    (1, 13)  float32 on device
    """
    dev = x.device
    N = x.shape[0]
    gen_frac = step_num / max(max_steps, 1)

    # ---- Trivial / degenerate cases ----
    if N <= 1:
        base_features = torch.zeros(N, BASE_NODE_DIM, device=dev)
        node_features = augment_node_features_lineage(base_features, parent_info)
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=dev)
        edge_features = torch.zeros(0, EDGE_DIM, device=dev)
        global_t = make_degenerate_global(
            N, ndim, step_num, max_steps, dev,
            eval_budget_frac=eval_budget_frac,
            stagnation_counter=stagnation_counter)
        return node_features, edge_index, edge_features, global_t

    # ---- Shared intermediates ----
    shared = compute_shared_intermediates(
        x, fitness, k_neighbors=k_neighbors, lb=lb, ub=ub, beta=beta)

    # ---- Edge index ----
    edge_index = build_spatial_edge_index(N, shared['knn_idx'])
    E = edge_index.shape[1]

    if E == 0:
        base_features = torch.zeros(N, BASE_NODE_DIM, device=dev)
        node_features = augment_node_features_lineage(base_features, parent_info)
        edge_features = torch.zeros(0, EDGE_DIM, device=dev)
        global_t = make_degenerate_global(
            N, ndim, step_num, max_steps, dev,
            eval_budget_frac=eval_budget_frac,
            stagnation_counter=stagnation_counter)
        return node_features, edge_index, edge_features, global_t

    # ---- Node features (6 base + 3 lineage) ----
    base_features = compute_base_node_features(
        shared, step_num, max_steps, generation_ids)
    node_features = augment_node_features_lineage(base_features, parent_info)

    # ---- Edge features (4) ----
    edge_features = compute_edge_features(shared, edge_index, beta=beta)

    # ---- Global features (13) ----
    global_t = compute_global_features(
        shared, ndim, step_num, max_steps,
        prev_best=prev_best,
        eval_budget_frac=eval_budget_frac,
        improvement_ema=improvement_ema,
        stagnation_counter=stagnation_counter,
        prev_x=prev_x,
        prev_fitness=prev_fitness,
        parent_info=parent_info,
        beta=beta,
        delta_fitness=delta_fitness,
        contraction_rate=contraction_rate,
    )

    return node_features, edge_index, edge_features, global_t


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
        _result = build_similarity_graph_gpu(
            xs[b], fitnesses[b],
            step_num=step_nums[b], max_steps=max_steps, ndim=ndims[b],
            parent_info=parent_infos[b],
            **kw_b,
        )
        n_t, ei_t, ea_t, g_t = _result[0], _result[1], _result[2], _result[3]
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
