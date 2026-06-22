"""
DonorSelectionGATv2 — all-to-all GATv2 attention module for donor selection.

Produces logits (B, N, N, n_roles) for per-individual selection of
pbest / r1 / r2 (in DE terms). Lives at the top of SparseGATv2Backbone,
AFTER message-passing + edge/global readouts.

Design decisions:
- GATv2 attention family to match the backbone's message-passing layers:
      score_ij = att^T · LeakyReLU(W_l h_i + W_r h_j)
  NOT scaled dot-product (Q·K), to keep a single attention class in the stack.
- Signed fitness bias per role (fit_sign fixed buffer = [+1, +1, -1]).
- alpha_fit learnable, init = 5.0 so at step 0 the module behaves as a
  classical "topk by fitness" / "botk by fitness" selector. softplus-like
  positivity (clamp(min=0)) prevents sign flipping of fit_sign.
- att initialized to match SparseGATv2Layer convention
  (std = 2 / sqrt(attn_dim), line 65 of sparse_gatv2_layer.py).
- Consumes fit_bias = -node_feat[..., 0]. NOTE the documented sign inversion
  (finding_donor_rank_sign_inversion_2026_06_12): node_feat ch0 is +1 = best
  (soft_rank gives the min-fitness agent the MAX rank), so -ch0 with
  fit_sign=[+1,+1,-1] biases pbest/r1 toward the WORST candidate and r2 toward
  the best — inverted vs the intended DE prior. The deployed checkpoint was
  trained with this convention and its learned embedding attention overcame
  the prior (net pro-fitness selection, measured). DO NOT flip the sign
  without retraining: it changes deployed-model behavior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DonorSelectionGATv2(nn.Module):
    """All-to-all GATv2 attention, n_roles independent heads (pbest, r1, r2).

    NO value aggregation — we only surface scores as selection logits.
    NO softmax, NO sampling — the operator decides that with gumbel_softmax.
    """

    def __init__(self, hidden_dim: int = 128, attn_dim: int = 16,
                 n_roles: int = 3, query_chunk_size: int | None = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim
        self.n_roles = n_roles

        # Per-role GATv2 projections (left/right for source/target).
        # Output dim = attn_dim * n_roles; reshape to per-role in forward.
        self.lin_l = nn.Linear(hidden_dim, attn_dim * n_roles)
        self.lin_r = nn.Linear(hidden_dim, attn_dim * n_roles)

        # att: attention direction per role. Init matches SparseGATv2Layer:
        #      std = 2 / sqrt(attn_dim) per element → peaked attention at start.
        self.att = nn.Parameter(
            torch.randn(1, 1, n_roles, attn_dim) * (2.0 / attn_dim ** 0.5))

        # Fitness bias scaling (per-role, learnable, init high for warm-start).
        # Effective alpha is clamp(min=0) so signs cannot flip the role semantics.
        self.alpha_fit = nn.Parameter(torch.tensor([5.0] * n_roles))

        # Role signs: pbest/r1 prefer BEST (positive bias); r2 prefers WORST.
        # Buffer (not learnable) — sign belongs to the role, not to training.
        if n_roles == 3:
            sign = torch.tensor([1.0, 1.0, -1.0])
        else:
            # For n_roles != 3, default all-positive and let alpha_fit learn;
            # the supported canonical case is 3 (DE pbest/r1/r2).
            sign = torch.ones(n_roles)
        self.register_buffer('fit_sign', sign)

        # Cache for the diagonal mask — built lazily, keyed by (N, device).
        # Avoids allocating torch.eye(N) every forward on the BPTT hot path.
        self._eye_cache: torch.Tensor | None = None

        # Chunk size for forward_asym (memory cap). None = monolithic
        # (BPTT-safe; default). Set via ctor or assigned per-instance for
        # inference at large N to avoid the O(N²·R·d) intermediate OOM.
        self.query_chunk_size: int | None = query_chunk_size

    def forward(self, h: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        """Symmetric (legacy) call — query and candidate are the same population.

        Args:
            h:         (B, N, hidden_dim) — backbone node embeddings
            node_feat: (B, N, >=1) — index 0 must be fit_rank*2-1 ∈ [-1, 1],
                       where +1 = best agent (min fitness for minimization;
                       soft_rank gives the minimum the maximum rank).

        Returns:
            logits (B, N, N, n_roles) — raw attention scores, diagonal set
            to −1e9 to prohibit self-selection. NO softmax applied here.

        For graph-native archive callers (E9_archive_K50), use forward_asym
        with a wider candidate pool plus cand_mask.
        """
        return self.forward_asym(h, h, node_feat, node_feat, cand_mask=None)

    def forward_asym(self, h_query: torch.Tensor, h_cand: torch.Tensor,
                     node_feat_query: torch.Tensor,
                     node_feat_cand: torch.Tensor,
                     cand_mask: torch.Tensor = None,
                     query_chunk_size: int = None) -> torch.Tensor:
        """Asymmetric attention: queries (active parents) × candidates (pop ∪ archive).

        Args:
            h_query:         (B, N_q, H) — active parents (query side).
            h_cand:          (B, N_c, H) — candidate pool, N_c = N_q + K_archive.
                             First N_q rows MUST correspond to the same active
                             parents as h_query (so the diagonal self-mask is
                             well-defined). Subsequent rows are archive nodes.
            node_feat_query: (B, N_q, >=1) — query node features (idx 0 = fit_rank*2-1).
            node_feat_cand:  (B, N_c, >=1) — candidate features (idx 0 used for fit_bias).
            cand_mask:       Optional (B, N_c) bool. False slots are forbidden
                             as donors on ALL channels (D9 warmup; per-batch
                             archive_mask propagates through this).
            query_chunk_size: Optional int. If set, splits N_q into chunks of
                             this size to bound peak memory of the (B, N_q, N_c,
                             R, d) intermediate tensor at O(C * N_c * R * d).
                             Bit-exact equivalent to monolithic path.
                             None or >= N_q  →  monolithic.

        Returns:
            logits (B, N_q, N_c, n_roles).

            Masking applied (in order):
              1. Diagonal of the active region [:N_q] (no self-selection).
              2. Archive slots [N_q:] on channel 0 (pbest cannot be from
                 archive — D4).
              3. cand_mask=False slots across ALL channels (D9 invalid slots).

        When N_q == N_c and cand_mask is None, the result is bit-exact to the
        legacy forward(h, node_feat).
        """
        B, N_q, _ = h_query.shape
        N_c = h_cand.shape[1]
        R, d = self.n_roles, self.attn_dim
        device = h_query.device

        # Fall back to per-instance default if no kwarg passed.
        if query_chunk_size is None:
            query_chunk_size = self.query_chunk_size

        # Project query (left) and candidate (right) sides — once.
        x_l_full = self.lin_l(h_query).view(B, N_q, R, d)
        x_r = self.lin_r(h_cand).view(B, N_c, R, d)

        # Fitness bias is keyed by candidate (j), not query. KNOWN sign
        # inversion vs the intended DE prior (see module docstring and
        # finding_donor_rank_sign_inversion_2026_06_12): kept as trained —
        # the deployed checkpoint learned to overcome it. Do not flip
        # without retraining.
        fit_bias = -node_feat_cand[..., 0]                   # (B, N_c)
        alpha = self.alpha_fit.clamp(min=0.0)                # (R,)
        bias_coef = (self.fit_sign * alpha).view(1, 1, 1, R)
        bias = bias_coef * fit_bias.unsqueeze(1).unsqueeze(-1)  # (B, 1, N_c, R)

        # Shared masks (computed once, applied per chunk).
        # Eye cache for the symmetric monolithic path — bit-exact preserved.
        if query_chunk_size is None or query_chunk_size >= N_q:
            eye = self._eye_cache
            if (eye is None or eye.shape[-1] != N_q or eye.device != device):
                eye = torch.eye(N_q, dtype=torch.bool, device=device)
                self._eye_cache = eye

        pbest_mask = None
        if N_c > N_q:
            pbest_mask = torch.zeros(1, 1, N_c, R, dtype=torch.bool,
                                     device=device)
            pbest_mask[..., N_q:, 0] = True

        invalid = None
        if cand_mask is not None:
            invalid = (~cand_mask).unsqueeze(1).unsqueeze(-1)  # (B, 1, N_c, 1)

        # Chunk schedule.
        if query_chunk_size is None or query_chunk_size >= N_q:
            chunks = [(0, N_q)]
        else:
            chunks = [(s, min(s + query_chunk_size, N_q))
                      for s in range(0, N_q, query_chunk_size)]

        # Single-chunk fast path == monolithic — keep its exact ops to
        # preserve bit-exactness with prior calls (and avoid unnecessary cat).
        if len(chunks) == 1:
            msg = F.leaky_relu(
                x_l_full.unsqueeze(2) + x_r.unsqueeze(1), 0.2)
            scores = (msg * self.att).sum(dim=-1)
            scores = scores + bias
            # ── 1. Self-mask on active diagonal [:N_q].
            if N_c == N_q:
                diag_mask = eye
            else:
                diag_mask = torch.zeros(N_q, N_c, dtype=torch.bool,
                                        device=device)
                diag_mask[:, :N_q] = eye
            scores = scores.masked_fill(diag_mask.unsqueeze(0).unsqueeze(-1),
                                        -1e9)
            if pbest_mask is not None:
                scores = scores.masked_fill(pbest_mask, -1e9)
            if invalid is not None:
                scores = scores.masked_fill(invalid, -1e9)
            return scores

        # Multi-chunk path — chunked over N_q.
        out_chunks = []
        for start, end in chunks:
            chunk_n = end - start
            x_l_c = x_l_full[:, start:end]                                # (B, C, R, d)
            msg = F.leaky_relu(x_l_c.unsqueeze(2) + x_r.unsqueeze(1), 0.2)
            scores_c = (msg * self.att).sum(dim=-1)                       # (B, C, N_c, R)
            scores_c = scores_c + bias

            # Diag mask for rows [start:end]: True at column = (start + i_local).
            # Always within [0, N_q), so within [0, N_c) since N_c >= N_q.
            diag_chunk = torch.zeros(chunk_n, N_c, dtype=torch.bool,
                                     device=device)
            i_local = torch.arange(chunk_n, device=device)
            diag_chunk[i_local, start + i_local] = True
            scores_c = scores_c.masked_fill(
                diag_chunk.unsqueeze(0).unsqueeze(-1), -1e9)

            if pbest_mask is not None:
                scores_c = scores_c.masked_fill(pbest_mask, -1e9)
            if invalid is not None:
                scores_c = scores_c.masked_fill(invalid, -1e9)

            out_chunks.append(scores_c)

        return torch.cat(out_chunks, dim=1)
