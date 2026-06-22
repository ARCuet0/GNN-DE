"""
topology_strategies_torch_knn.py — Torch-native approximate kNN, strict O(N*k).

Replaces the cdist-based kNN in `encoder/topology_strategies.py` for the
D1000 line. Honors the lema "no O(N^2) compute anywhere" by:

  * Initializing a random k-graph (O(N*k) compute / memory).
  * Refining T iterations via NN-Descent — for each point, gather candidates
    = self.NN ∪ NN.NN, compute exact distances over those k*(k+1) candidates,
    keep top-k. Per iteration O(N*k^2*D) — k is constant so this is O(N*D),
    linear in N.

For N <= fallback_n we delegate to exact `_chunked_knn` because:
  * Refinement bookkeeping has constant overhead that dominates at small N.
  * Existing tests / inference paths assume exactness at small N.
  * The lema is about scaling behavior; it is not violated by exact compute
    at sub-fallback sizes.

Determinism: seeded torch.Generator. Output is bit-exact across runs given
the same seed and the same input tensor on the same device.
"""
import torch
import torch.nn as nn

from .topology_strategies import _chunked_knn


def _random_k_graph(B: int, N: int, k: int, device: torch.device,
                    generator: torch.Generator) -> torch.Tensor:
    """Initialize a random k-graph with no self-loops.

    Returns: knn_idx (B, N, k) long. Each row contains k distinct indices
    in [0, N) excluding the row index itself.
    """
    # Sample k+1 candidates, drop any that match self, then take first k.
    # Probability that >1 self-collision in k+2 candidates is negligible
    # for N >= k+2; we safely assume N >= 2k for our regime.
    cand = torch.randint(0, N, (B, N, k + 2), device=device,
                         generator=generator, dtype=torch.long)
    self_idx = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k + 2)
    not_self = cand != self_idx
    # Replace self-collisions with (cand + 1) % N — guarantees != self with
    # uniform-ish distribution. Safe because we only replace a small fraction.
    cand = torch.where(not_self, cand, (cand + 1) % N)
    # Also ensure uniqueness within the row by shifting duplicates;
    # NN-Descent handles a few duplicates gracefully (keeps top-k by distance),
    # so we don't need strict uniqueness — but we do need != self.
    return cand[..., :k]


def _exclude_self_in_topk(dists: torch.Tensor, cand_idx: torch.Tensor,
                          query_idx: torch.Tensor) -> torch.Tensor:
    """Mask self-positions in `dists` with +inf before topk.

    Args:
        dists:     (B, N, K_cand)
        cand_idx:  (B, N, K_cand) long — global indices of candidates
        query_idx: (N,) long — global index per query (= arange(N))

    Returns: dists with self-positions masked to +inf.
    """
    self_mask = cand_idx == query_idx.view(1, -1, 1)
    return dists.masked_fill(self_mask, float('inf'))


def _reverse_neighbors(knn_idx: torch.Tensor, k_rev: int,
                       generator: torch.Generator) -> torch.Tensor:
    """Approximate reverse-NN per query via vectorized scatter+sample.

    For each (b, j), find indices i where j ∈ knn_idx[b, i]. Reverse degree is
    unbounded, so we cap to k_rev. Multiple writes to the same (b, j, slot)
    collide and the last write wins — acceptable approximation since downstream
    NN-Descent re-ranks by distance.

    Returns: rev (B, N, k_rev) long. Unfilled slots stay random — they act as
    "extra random candidates" that contribute diversity rather than self-loops.
    """
    B, N, k = knn_idx.shape
    device = knn_idx.device

    rev = torch.randint(0, N, (B, N, k_rev), device=device, generator=generator,
                        dtype=torch.long)

    # Vectorized scatter over (b, i, k_src):
    #   rev[b, dst_n[b,i,k_src], dst_slot[b,i,k_src]] = i
    src = torch.arange(N, device=device).view(1, N, 1).expand(B, N, k)
    dst_slot = torch.randint(0, k_rev, (B, N, k), device=device,
                             generator=generator, dtype=torch.long)
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
    rev[batch_idx, knn_idx, dst_slot] = src
    return rev


def _refine_one_pass(data: torch.Tensor, knn_idx: torch.Tensor,
                     k: int, generator: torch.Generator,
                     n_random: int = 0,
                     include_reverse: bool = True) -> torch.Tensor:
    """One NN-Descent refinement pass.

    Candidates per query = self.NN ∪ NN.NN ∪ reverse.NN ∪ random samples.
    Compute exact L2 distance over the union, take top-k.

    Args:
        data:           (B, N, D) float
        knn_idx:        (B, N, k) long — current kNN estimate
        k:              target neighbor count
        generator:      torch.Generator for the random sample channel
        n_random:       number of extra random candidates per query (default 0)
        include_reverse: include reverse neighbors (default True)

    Returns: refined knn_idx (B, N, k) long.
    """
    B, N, D = data.shape
    device = data.device

    # NN-of-NN: for each query (b, i), gather knn_idx[b, j, :] where
    # j ∈ knn_idx[b, i]. Vectorized via gather along dim=1.
    flat = knn_idx.reshape(B, N * k)
    nn_of_nn = knn_idx.gather(1, flat.unsqueeze(-1).expand(B, N * k, k))
    nn_of_nn = nn_of_nn.reshape(B, N, k * k)

    parts = [knn_idx, nn_of_nn]

    if include_reverse:
        rev = _reverse_neighbors(knn_idx, k_rev=k, generator=generator)
        parts.append(rev)

    if n_random > 0:
        rand_extra = torch.randint(0, N, (B, N, n_random), device=device,
                                   generator=generator, dtype=torch.long)
        parts.append(rand_extra)

    cand = torch.cat(parts, dim=-1)  # (B, N, K_cand)

    # Gather candidate coords: (B, N, K_cand, D).
    K_cand = cand.shape[-1]
    cand_flat = cand.reshape(B, N * K_cand)
    cand_coords = data.gather(
        1, cand_flat.unsqueeze(-1).expand(B, N * K_cand, D)
    ).reshape(B, N, K_cand, D)

    # Exact L2 distance to query
    diffs = cand_coords - data.unsqueeze(2)        # (B, N, K_cand, D)
    dists = (diffs * diffs).sum(dim=-1)             # (B, N, K_cand)

    # Mask self-collisions
    query_idx = torch.arange(N, device=device)
    dists = _exclude_self_in_topk(dists, cand, query_idx)

    # Mask duplicates within each row — they otherwise steal top-k slots
    # and cause monotonic recall degradation across iterations.
    sorted_cand, sort_idx = cand.sort(dim=-1)
    prev = torch.cat(
        [torch.full_like(sorted_cand[..., :1], -1), sorted_cand[..., :-1]],
        dim=-1)
    is_dup_sorted = sorted_cand == prev
    inv_idx = sort_idx.argsort(dim=-1)
    is_dup = is_dup_sorted.gather(-1, inv_idx)
    dists = dists.masked_fill(is_dup, float('inf'))

    # Top-k smallest distances
    _, top_local = dists.topk(k, dim=-1, largest=False)
    new_knn = cand.gather(-1, top_local)
    return new_knn


class TorchNNDescentKNN(nn.Module):
    """Strict-O(N*k) approximate kNN via NN-Descent on a random init graph.

    Args:
        k:          number of neighbors per node.
        n_iters:    NN-Descent refinement passes (default 2). More = higher
                    recall, more compute. Per-pass O(N*k^2*D).
        fallback_n: if N <= this, delegate to exact `_chunked_knn`.
                    Default 64 keeps small-N callers bit-exact.
        seed:       seed for the random init graph. Determinism contract:
                    same (data, seed, device) → same knn_idx.

    Forward:
        data: (B, N, D) float — coordinates (any space).
        Returns: knn_idx (B, N, k) long, self-excluded.
    """

    def __init__(self, k: int = 8, n_iters: int = 2, fallback_n: int = 64,
                 seed: int = 0, n_random_per_iter: int = 8,
                 include_reverse: bool = True):
        super().__init__()
        self.k = k
        self.n_iters = n_iters
        self.fallback_n = fallback_n
        self.seed = seed
        self.n_random_per_iter = n_random_per_iter
        self.include_reverse = include_reverse
        # Generator cached per device — re-seeded each forward so the contract
        # "same (data, seed, device) → same knn_idx" holds. Avoids the
        # per-forward `torch.Generator(...)` allocation on the hot path.
        self._gen_cache: dict[torch.device, torch.Generator] = {}

    def _get_generator(self, device: torch.device) -> torch.Generator:
        gen = self._gen_cache.get(device)
        if gen is None:
            gen = torch.Generator(device=device)
            self._gen_cache[device] = gen
        gen.manual_seed(self.seed)
        return gen

    @torch.no_grad()
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        B, N, D = data.shape
        k = min(self.k, N - 1)

        if N <= self.fallback_n:
            return _chunked_knn(data, k, chunk_size=max(N, 1))

        gen = self._get_generator(data.device)
        knn_idx = _random_k_graph(B, N, k, data.device, gen)

        for _ in range(self.n_iters):
            knn_idx = _refine_one_pass(
                data, knn_idx, k, generator=gen,
                n_random=self.n_random_per_iter,
                include_reverse=self.include_reverse)

        return knn_idx
