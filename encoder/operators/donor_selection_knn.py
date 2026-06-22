"""
DonorSelectionKNN — kNN-restricted GATv2 attention for donor selection.

Replaces the all-to-all (B, N, N, R, d) intermediate of DonorSelectionGATv2
with a sparse (B, N, k_donor, R, d) intermediate, where for each parent the
candidate set = kNN(parent) ∪ pbest_pool. Strict O(N*k_donor*R*d) compute
and memory.

Inherits projection params (lin_l, lin_r, att, alpha_fit, fit_sign) from
DonorSelectionGATv2 — same state_dict keys, so legacy checkpoints load
directly via warmstart (test_donor_knn_warmstart).

Returns:
    logits (B, N, k_donor, R) — raw attention scores, self-positions masked.
    cand_idx (B, N, k_donor) long — global donor indices per local slot.

Downstream: BatchedDiffAttDE.compute_params (de_heads.py) consumes these via
the donor_cand_idx kwarg path; gumbel_softmax samples on the local axis and
the realized index is resolved to a global index via cand_idx.
"""
import math

import torch
import torch.nn.functional as F

from .donor_selection import DonorSelectionGATv2


class DonorSelectionKNN(DonorSelectionGATv2):
    """kNN-restricted GATv2 donor selection — O(N*k_donor) memory."""

    def __init__(self, hidden_dim: int = 128, attn_dim: int = 16,
                 n_roles: int = 3, p_pool_frac: float = 0.1):
        super().__init__(hidden_dim=hidden_dim, attn_dim=attn_dim,
                         n_roles=n_roles)
        self.p_pool_frac = p_pool_frac

    def _build_cand_idx(self, knn_idx: torch.Tensor,
                        fit_signed: torch.Tensor) -> torch.Tensor:
        """Build per-parent candidate index = kNN(parent) ∪ pbest_pool.

        knn_idx: (B, N, k_graph) long — graph kNN per parent
        fit_signed: (B, N) — node_feat[..., 0] = fit_rank*2-1 in [-1, 1]
        Returns cand_idx (B, N, k_donor) long, k_donor = k_graph + p_pool.
        Layout: [pbest_pool indices, kNN indices]. Duplicates may appear when
        a parent's kNN entry is also in pbest_pool — duplicate slots are
        masked to -1e9 in `forward` so gumbel-softmax doesn't double-count
        them (which would silently bias sampling toward the overlap).
        """
        B, N, k_graph = knn_idx.shape
        p_pool = max(2, math.ceil(N * self.p_pool_frac))
        _, pool_idx = fit_signed.topk(p_pool, dim=-1, largest=False)
        pool_expanded = pool_idx.unsqueeze(1).expand(B, N, p_pool)
        return torch.cat([pool_expanded, knn_idx], dim=-1)

    def forward_asym(self, *args, **kwargs):
        """Disabled on the kNN head — forward_asym is the inherited all-to-all
        O(N²) path. Calling it on a kNN instance would silently defeat the
        D1000 lema. Use forward(h, node_feat, knn_idx, fit_signed) instead."""
        raise NotImplementedError(
            "DonorSelectionKNN.forward_asym is disabled (would invoke the "
            "inherited O(N^2) all-to-all path). Use forward(...) with "
            "knn_idx + fit_signed for the kNN-restricted path.")

    @staticmethod
    def _duplicate_mask(cand_idx: torch.Tensor) -> torch.Tensor:
        """Return (B, N, K) bool: True at slot j iff cand_idx[..., j] also
        appears at some j' < j (i.e. j is a NON-FIRST occurrence). Used to
        mask duplicate candidate slots in `scores` so gumbel-softmax doesn't
        give over-represented donors a higher selection probability.
        """
        sorted_cand, sort_idx = cand_idx.sort(dim=-1)
        prev = torch.cat(
            [torch.full_like(sorted_cand[..., :1], -1), sorted_cand[..., :-1]],
            dim=-1)
        is_dup_sorted = sorted_cand == prev
        inv_idx = sort_idx.argsort(dim=-1)
        return is_dup_sorted.gather(-1, inv_idx)

    def forward(self, h: torch.Tensor, node_feat: torch.Tensor,
                knn_idx: torch.Tensor,
                fit_signed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """kNN-restricted donor attention.

        Args:
            h:          (B, N, hidden_dim) backbone embedding.
            node_feat:  (B, N, >=1) — idx 0 must be fit_rank*2-1 ∈ [-1, 1].
            knn_idx:    (B, N, k_graph) long — graph kNN per parent
                        (already self-excluded).
            fit_signed: (B, N) — convenience alias for node_feat[..., 0]
                        (identical numerically). Passed separately so callers
                        can use a different ranking signal for the pool if
                        desired without touching node_feat.

        Returns:
            logits   (B, N, k_donor, n_roles)
            cand_idx (B, N, k_donor) long
        """
        B, N, H = h.shape
        R, d = self.n_roles, self.attn_dim
        device = h.device

        cand_idx = self._build_cand_idx(knn_idx, fit_signed)  # (B, N, K)
        K = cand_idx.shape[-1]

        # Project query (parent) and candidates side.
        x_l = self.lin_l(h).view(B, N, R, d)                    # parent
        x_r_full = self.lin_r(h).view(B, N, R, d)               # all candidates

        # Gather candidate projections via cand_idx — flatten then gather along
        # dim=1 to avoid the (B, N, N, R*d) intermediate that all-to-all
        # attention would materialize.
        x_r_flat = x_r_full.reshape(B, N, R * d)                # (B, N, R*d)
        cand_flat = cand_idx.reshape(B, N * K)                  # (B, N*K)
        x_r_gathered = x_r_flat.gather(
            1, cand_flat.unsqueeze(-1).expand(B, N * K, R * d)
        ).reshape(B, N, K, R, d)

        # GATv2 message: LeakyReLU(W_l h_i + W_r h_j) projected onto att.
        # x_l: (B, N, R, d), x_r_gathered: (B, N, K, R, d).
        msg = F.leaky_relu(x_l.unsqueeze(2) + x_r_gathered, 0.2)  # (B, N, K, R, d)
        scores = (msg * self.att.unsqueeze(2)).sum(dim=-1)        # (B, N, K, R)

        # Fitness bias keyed by candidate (j) only.
        # node_feat[..., 0] for each candidate via gather along N.
        nf0 = node_feat[..., 0]                                   # (B, N)
        nf0_gathered = nf0.gather(
            1, cand_flat).reshape(B, N, K)                        # (B, N, K)
        fit_bias = -nf0_gathered                                  # higher = better
        alpha = self.alpha_fit.clamp(min=0.0)                     # (R,)
        bias_coef = (self.fit_sign * alpha).view(1, 1, 1, R)
        bias = bias_coef * fit_bias.unsqueeze(-1)                 # (B, N, K, R)

        scores = scores + bias

        # Mask self-positions AND duplicate slots with -1e9 so gumbel-softmax
        # neither picks self nor double-counts a candidate that appears in
        # both pbest_pool and the parent's kNN.
        query_idx = torch.arange(N, device=device).view(1, N, 1)
        invalid = (cand_idx == query_idx) | self._duplicate_mask(cand_idx)
        scores = scores.masked_fill(invalid.unsqueeze(-1), -1e9)

        return scores, cand_idx
