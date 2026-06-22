"""
similarity_graph.py — Build similarity-based population graph for MOS+GNN.

v3: Cleaned feature set with temporal + cloud-landscape interaction features.

Graph structure:
  - Nodes: current population (gs elite individuals)
  - Edges: k-nearest neighbors in coordinate space (bidirectional)
  - Node features (9 = 6 base + 3 lineage):
        fit_rank, dist_to_best_rank, local_density_inv,
        gradient_consistency, elite_tenure, gen_frac,
        displacement_mag, improvement_ratio, parent_fitness_rank
  - Edge features (4): distance_percentile, fitness_rank_diff,
                        cosine_similarity, mutual_knn
  - Global features (11):
        gen_frac, diversity, improvement_rate, population_fdc,
        mean_grad_consistency, convexity_fraction, mean_nbc_ratio,
        stagnation_counter, front_vs_tail, direction_consensus,
        density_quality_corr

Output format matches PNA encoder expectations.
"""
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

# Feature dimension constants (importable)
BASE_NODE_DIM = 5   # fit_rank, dist_to_best_rank, local_density_inv, gradient_consistency, dist_to_centroid_rank
LINEAGE_DIM = 3     # displacement_mag, improvement_ratio, parent_fitness_rank
NODE_DIM = BASE_NODE_DIM + LINEAGE_DIM   # 9
EDGE_DIM = 4        # distance_percentile, fitness_rank_diff, cosine_similarity, mutual_knn
GLOBAL_DIM = 16     # 13 base + fitness_kurtosis + fitness_ruggedness + rank_distance_spearman

NODE_NAMES = [
    # Base (5):
    'fit_rank', 'dist_to_best_rank', 'local_density_inv',
    'gradient_consistency', 'dist_to_centroid_rank',
    # Lineage (3):
    'displacement_rank', 'improvement_rank', 'parent_fitness_rank',
]
EDGE_NAMES = [
    'distance_percentile', 'fitness_rank_diff',
    'cosine_similarity', 'mutual_knn',
]
GLOBAL_NAMES = [
    'eval_budget_frac', 'diversity', 'mean_local_convexity', 'population_fdc',
    'mean_grad_consistency', 'convexity_fraction', 'mean_nbc_ratio',
    'stagnation_counter', 'front_vs_tail', 'direction_consensus',
    'density_quality_corr', 'delta_fitness', 'contraction_rate',
    'fitness_kurtosis', 'fitness_ruggedness', 'rank_distance_spearman',
]


def build_similarity_graph(elite, bounds, step_num, max_steps,
                           generation_ids=None,
                           prev_best=None,
                           k_neighbors=8,
                           eval_budget_frac=None,
                           improvement_ema=None,
                           stagnation_counter=0):
    """
    Build a k-NN similarity graph from the current population.

    Args:
        elite: (gs, D+4) population array [ID, FITNESS, GEN, N_DIM, coords...]
        bounds: (D, 2) lower/upper bounds
        step_num: current generation number
        max_steps: total generations
        generation_ids: (gs,) which generation each individual was created in
        prev_best: best fitness from previous generation
        k_neighbors: number of nearest neighbors for graph edges
        eval_budget_frac: evaluation budget fraction (overrides step_frac)
        improvement_ema: EMA-smoothed improvement rate
        stagnation_counter: generations since population best improved

    Returns:
        nodes_t:     (N, 9)  torch.float32 — node features, ~[-1, 1]
        edges_t:     (2, E)  torch.long    — edge index (bidirectional)
        edge_attr_t: (E, 4)  torch.float32 — edge features, ~[-1, 1]
        global_t:    (1, 11) torch.float32 — global features
        node_ids_t:  (N,)    torch.int32   — original node IDs
    """
    # Column indices
    ID, FITNESS, GEN = 0, 1, 2
    COORDS = np.s_[4:]

    gs = len(elite)
    coords = elite[:, COORDS].astype(np.float64)
    fitness = elite[:, FITNESS].astype(np.float64)
    node_ids = elite[:, ID].astype(np.int32)
    gen_col = elite[:, GEN]
    D = coords.shape[1]

    bounds_span = bounds[:, 1] - bounds[:, 0]
    bounds_span_safe = np.where(bounds_span < 1e-12, 1.0, bounds_span)

    # ---- Handle trivial graph (gs <= 1) ----
    if gs <= 1:
        gf = step_num / max(max_steps, 1)
        node_features = np.zeros((gs, NODE_DIM), dtype=np.float32)
        if gs == 1:
            node_features[0, 5] = gf * 2 - 1  # gen_frac at index 5
            node_features[0, 6] = -1.0          # displacement_mag default
            node_features[0, 8] = -1.0          # parent_fitness_rank default
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_features = np.zeros((0, EDGE_DIM), dtype=np.float32)
        stag_norm = float(np.tanh(stagnation_counter / 20.0) * 2 - 1)
        global_cols = [
            np.clip(eval_budget_frac, 0, 1) * 2 - 1
                if eval_budget_frac is not None
                else gf * 2 - 1,                          # 0: gen_frac
            0.0, 0.0, 0.0, -1.0, -1.0, -1.0,            # 1-6: diversity..mean_nbc
            stag_norm,                                     # 7: stagnation_counter
            0.0, 0.0, 0.0,                                # 8-10: front_vs_tail, direction_consensus, density_quality_corr
        ]
        global_features = np.array(global_cols, dtype=np.float32)
        nodes_t = torch.from_numpy(node_features)
        edges_t = torch.from_numpy(edge_index).to(torch.long)
        edge_attr_t = torch.from_numpy(edge_features)
        global_t = torch.from_numpy(global_features).unsqueeze(0)
        node_ids_t = torch.from_numpy(node_ids).to(torch.int32)
        return nodes_t, edges_t, edge_attr_t, global_t, node_ids_t

    # ---- Normalize coordinates to [0,1] ----
    coords_norm = (coords - bounds[:, 0]) / bounds_span_safe

    # ---- Pairwise distances ----
    dist_matrix = squareform(pdist(coords_norm, metric="euclidean"))
    np.fill_diagonal(dist_matrix, np.inf)

    # ---- k-NN edges ----
    k = min(k_neighbors, gs - 1)
    src_list, dst_list = [], []
    knn_sets = []

    for i in range(gs):
        nn_idx = np.argpartition(dist_matrix[i], k)[:k]
        knn_sets.append(set(nn_idx.tolist()))
        for j in nn_idx:
            src_list.append(i)
            dst_list.append(j)
            src_list.append(j)
            dst_list.append(i)

    # Deduplicate edges
    edge_set = set()
    src_dedup, dst_dedup = [], []
    for s, d in zip(src_list, dst_list):
        if (s, d) not in edge_set:
            edge_set.add((s, d))
            src_dedup.append(s)
            dst_dedup.append(d)

    src_arr = np.array(src_dedup, dtype=np.int64)
    dst_arr = np.array(dst_dedup, dtype=np.int64)
    E = len(src_arr)

    # ================================================================
    # SHARED INTERMEDIATES
    # ================================================================
    fit_sorted_idx = np.argsort(fitness)
    fit_rank = np.zeros(gs)
    fit_rank[fit_sorted_idx] = np.arange(gs) / max(gs - 1, 1)  # [0, 1]
    fit_mean = np.mean(fitness)
    fit_std = np.std(fitness) + 1e-8
    fit_zscore = np.clip((fitness - fit_mean) / fit_std, -3, 3)

    centroid = np.mean(coords_norm, axis=0)
    best_idx = np.argmin(fitness)
    dist_to_best = np.linalg.norm(coords_norm - coords_norm[best_idx], axis=1)

    # Local density (average k-NN distance)
    max_finite = (np.max(dist_matrix[~np.isinf(dist_matrix)]) + 1
                  if np.any(~np.isinf(dist_matrix)) else 1.0)
    dist_matrix_safe = np.where(np.isinf(dist_matrix), max_finite, dist_matrix)
    knn_dists = np.partition(dist_matrix_safe, k, axis=1)[:, :k]
    local_density = np.mean(knn_dists, axis=1)
    max_density = np.max(local_density) + 1e-8

    gen_frac = step_num / max(max_steps, 1)

    # ---- Gradient consistency (per node) ----
    gradient_consistency = np.zeros(gs, dtype=np.float64)
    for i in range(gs):
        neighbors = list(knn_sets[i])
        if not neighbors:
            continue
        nn = np.array(neighbors)
        diffs = coords_norm[nn] - coords_norm[i]
        dists_nn = np.linalg.norm(diffs, axis=1) + 1e-8
        directions = diffs / dists_nn[:, None]
        f_diffs = fitness[nn] - fitness[i]
        grads = f_diffs / dists_nn
        weighted = grads[:, None] * directions
        weighted_sum = np.sum(weighted, axis=0)
        total_abs_grad = np.sum(np.abs(grads)) + 1e-8
        gradient_consistency[i] = np.clip(
            np.linalg.norm(weighted_sum) / total_abs_grad, 0, 1
        )

    # ---- Local convexity (per node) ----
    local_convexity = np.zeros(gs, dtype=np.float64)
    for i in range(gs):
        neighbors = list(knn_sets[i])
        if not neighbors:
            continue
        mean_neighbor_fit = np.mean(fitness[np.array(neighbors)])
        local_convexity[i] = np.clip(
            (mean_neighbor_fit - fitness[i]) / fit_std, -3, 3
        ) / 3.0

    # ---- NBC ratio (per node) ----
    nn_dist = np.min(dist_matrix_safe, axis=1)
    fitness_better_mask = fitness[None, :] < fitness[:, None]
    nbn_matrix = np.where(fitness_better_mask, dist_matrix_safe, np.inf)
    nbn_dist = np.min(nbn_matrix, axis=1)
    # Best individual has no better neighbor → nbn_dist=inf → cap at max
    nbn_dist = np.where(np.isinf(nbn_dist), nn_dist, nbn_dist)
    nbc_ratio = np.clip(nbn_dist / (nn_dist + 1e-8), 0, 5)

    # ================================================================
    # NODE FEATURES (9)
    # ================================================================

    # dist_to_best_rank: rank among population (D-invariant)
    dtb_rank = np.zeros(gs)
    dtb_sorted = np.argsort(dist_to_best)
    dtb_rank[dtb_sorted] = np.arange(gs) / max(gs - 1, 1)

    # Elite tenure: how long since individual was created
    if generation_ids is not None:
        tenure = np.clip((step_num - generation_ids) / max(max_steps, 1), 0, 1)
    else:
        tenure = np.clip((step_num - gen_col) / max(max_steps, 1), 0, 1)

    base_features = np.column_stack([
        fit_rank * 2 - 1,                                      # 0: fit_rank
        dtb_rank * 2 - 1,                                      # 1: dist_to_best_rank
        np.clip(1 - local_density / max_density, 0, 1) * 2 - 1,  # 2: local_density_inv
        gradient_consistency * 2 - 1,                           # 3: gradient_consistency
        tenure * 2 - 1,                                         # 4: elite_tenure
        np.full(gs, gen_frac * 2 - 1),                          # 5: gen_frac
    ]).astype(np.float32)
    # Lineage defaults (CPU version has no parent info):
    # displacement=-1 (tanh(0)*2-1), improvement=0, parent_rank=-1 (0*2-1)
    lineage_defaults = np.zeros((gs, LINEAGE_DIM), dtype=np.float32)
    lineage_defaults[:, 0] = -1.0    # displacement_mag = tanh(0)*2-1
    lineage_defaults[:, 1] = 0.0     # improvement_ratio = 0
    lineage_defaults[:, 2] = -1.0    # parent_rank = 0*2-1
    node_features = np.concatenate([base_features, lineage_defaults], axis=1)

    # ================================================================
    # EDGE FEATURES (4)
    # ================================================================
    edge_dists = dist_matrix[src_arr, dst_arr]

    edge_cols = []

    # 0: distance percentile (rank among all edges)
    dist_ranks = np.argsort(np.argsort(edge_dists)).astype(np.float32)
    edge_cols.append(dist_ranks / max(E - 1, 1) * 2 - 1)

    # 1: fitness rank difference
    edge_cols.append(np.abs(fit_rank[src_arr] - fit_rank[dst_arr]) * 2 - 1)

    # 2: cosine similarity (centered coordinates)
    c_src = coords_norm[src_arr] - centroid
    c_dst = coords_norm[dst_arr] - centroid
    dot = np.sum(c_src * c_dst, axis=1)
    norm_src = np.linalg.norm(c_src, axis=1) + 1e-8
    norm_dst = np.linalg.norm(c_dst, axis=1) + 1e-8
    edge_cols.append(dot / (norm_src * norm_dst))

    # 3: mutual k-NN
    mutual_knn = np.zeros(E, dtype=np.float32)
    for idx in range(E):
        s, d = src_arr[idx], dst_arr[idx]
        if d in knn_sets[s] and s in knn_sets[d]:
            mutual_knn[idx] = 1.0
    edge_cols.append(mutual_knn)

    edge_features = np.column_stack(edge_cols).astype(np.float32)

    # ================================================================
    # GLOBAL FEATURES (11)
    # ================================================================
    pop_diversity = float(np.std(coords_norm))

    if improvement_ema is not None:
        improvement_rate = np.clip(float(improvement_ema), -1, 1)
    elif prev_best is not None:
        improvement_rate = float(
            np.clip((prev_best - np.min(fitness)) / (abs(prev_best) + 1e-8), -1, 1)
        )
    else:
        improvement_rate = 0.0

    # Population FDC (fitness-distance correlation)
    if np.std(dist_to_best) > 1e-8 and np.std(fitness) > 1e-8:
        fdc = np.corrcoef(dist_to_best, fitness)[0, 1]
        fdc = fdc if not np.isnan(fdc) else 0.0
    else:
        fdc = 0.0

    # Global duals of per-node landscape features
    mean_grad_consistency = float(np.mean(gradient_consistency))
    convexity_fraction = float(np.mean(local_convexity > 0))
    mean_nbc = float(np.clip(np.mean(nbc_ratio), 0, 5))

    # Stagnation counter (normalized)
    stag_norm = float(np.tanh(stagnation_counter / 20.0) * 2 - 1)

    # Density-quality correlation (Spearman approx via rank correlation)
    density_rank = np.zeros(gs)
    density_rank[np.argsort(local_density)] = np.arange(gs) / max(gs - 1, 1)
    if np.std(density_rank) > 1e-8 and np.std(fit_rank) > 1e-8:
        dq_corr = float(np.corrcoef(density_rank, fit_rank)[0, 1])
        dq_corr = dq_corr if not np.isnan(dq_corr) else 0.0
    else:
        dq_corr = 0.0

    global_cols = [
        np.clip(eval_budget_frac, 0, 1) * 2 - 1
            if eval_budget_frac is not None
            else gen_frac * 2 - 1,                        # 0: gen_frac
        np.clip(pop_diversity, 0, 1) * 2 - 1,             # 1: diversity
        improvement_rate,                                   # 2: improvement [-1,1]
        fdc,                                                # 3: population FDC [-1,1]
        mean_grad_consistency * 2 - 1,                     # 4: mean grad consistency
        convexity_fraction * 2 - 1,                        # 5: convexity fraction
        np.clip(mean_nbc / 2.5, 0, 2) - 1,                # 6: mean NBC ratio
        stag_norm,                                          # 7: stagnation_counter
        0.0,                                                # 8: front_vs_tail (CPU: no parent info)
        0.0,                                                # 9: direction_consensus (CPU: no prev_x)
        dq_corr,                                            # 10: density_quality_corr
    ]

    global_features = np.array(global_cols, dtype=np.float32)

    # ================================================================
    # Convert to torch tensors
    # ================================================================
    nodes_t = torch.from_numpy(node_features)
    edges_t = torch.from_numpy(np.stack([src_arr, dst_arr])).to(torch.long)
    edge_attr_t = torch.from_numpy(edge_features)
    global_t = torch.from_numpy(global_features).unsqueeze(0)  # (1, G)
    node_ids_t = torch.from_numpy(node_ids).to(torch.int32)

    return nodes_t, edges_t, edge_attr_t, global_t, node_ids_t


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    D = 10
    gs = 50
    bounds = np.column_stack([np.full(D, -100), np.full(D, 100)])

    rng = np.random.default_rng(42)
    elite = np.zeros((gs, D + 4), dtype=np.float32)
    elite[:, 0] = np.arange(1, gs + 1)  # IDs
    elite[:, 1] = rng.uniform(100, 10000, gs)  # fitness
    elite[:, 2] = rng.choice([5, 6, 7], gs)  # generation
    elite[:, 3] = D  # n_dim
    elite[:, 4:] = rng.uniform(-100, 100, (gs, D))  # coords

    nodes, edges, edge_attr, glob, ids = build_similarity_graph(
        elite, bounds, step_num=7, max_steps=50,
        prev_best=200.0, stagnation_counter=5,
    )

    print(f"Nodes:     {nodes.shape}  (expected: [{gs}, {NODE_DIM}])")
    print(f"Edges:     {edges.shape}")
    print(f"Edge attr: {edge_attr.shape}  (expected: [E, {EDGE_DIM}])")
    print(f"Global:    {glob.shape}  (expected: [1, {GLOBAL_DIM}])")
    print(f"Node IDs:  {ids.shape}")

    assert nodes.shape[1] == NODE_DIM, f"node_dim: {nodes.shape[1]} vs {NODE_DIM}"
    assert edge_attr.shape[1] == EDGE_DIM, f"edge_dim: {edge_attr.shape[1]} vs {EDGE_DIM}"
    assert glob.shape[1] == GLOBAL_DIM, f"global_dim: {glob.shape[1]} vs {GLOBAL_DIM}"

    print(f"\nNode feature ranges:")
    for i, name in enumerate(NODE_NAMES):
        print(f"  {name:25s}: [{nodes[:, i].min():.3f}, {nodes[:, i].max():.3f}]")
    print(f"\nEdge feature ranges:")
    for i, name in enumerate(EDGE_NAMES):
        print(f"  {name:25s}: [{edge_attr[:, i].min():.3f}, {edge_attr[:, i].max():.3f}]")
    print(f"\nGlobal features:")
    for i, name in enumerate(GLOBAL_NAMES):
        print(f"  {name:25s}: {glob[0, i]:.3f}")

    print("\nAll assertions passed!")
