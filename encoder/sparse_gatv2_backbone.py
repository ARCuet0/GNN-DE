"""
sparse_gatv2_backbone.py — Sparse GATv2 backbone with gather-based attention.

O(N·k) memory instead of O(N²). Uses knn_idx (B,N,k) for neighbor indexing.
All operations use gather — no scatter, fully vmap-compatible.
"""
import enum
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .sparse_gatv2_layer import SparseGATv2Layer

log = logging.getLogger(__name__)


@dataclass
class SparseTopologyCache:
    """Sparse graph topology for vmap-compatible forward pass.

    All neighbor info stored as (B, N, k) indices — no dense (B, N, N) tensors.
    """
    knn_idx: torch.Tensor        # (B, N, k) long — neighbor indices
    edge_feat: torch.Tensor      # (B, N, k, edge_dim) — sparse edge features
    B: int
    N: int
    k: int
    node_feat: Optional[torch.Tensor] = None    # (B, N, node_dim)
    global_feat: Optional[torch.Tensor] = None  # (B, global_dim)
    alive: Optional[torch.Tensor] = None        # (B, N) bool — active individuals


class TopologyMode(enum.Enum):
    COORDINATE_KNN = "coordinate_knn"
    EMBEDDING_KNN = "embedding_knn"
    LEARNED_SCORER = "learned_scorer"
    # D1000 line: torch-native NN-Descent kNN, strict O(N*k) per generation.
    TORCH_KNN = "torch_knn"


class SparseGATv2Backbone(nn.Module):
    """Multi-layer sparse GATv2 backbone.

    Same parameter shapes as DenseGATv2Backbone — weights are interchangeable.
    Uses SparseGATv2Layer with gather-based attention for O(N·k) memory.

    Output 4-tuple:
        h:          (B, N, gatv2_hidden)
        e:          (B, N, k, gatv2_hidden)  — sparse edge features
        h_per_head: (B, N, n_heads, head_dim)
        h_global:   (B, global_out_dim)
    """

    def __init__(self, node_in=8, edge_in=4, global_in=16,
                 gatv2_hidden=128, gatv2_layers=2, n_heads=8,
                 global_out_dim=128, dropout=0.1,
                 enable_donor_selection=True,
                 donor_attn_dim=16, donor_n_roles=3,
                 # D1000 line — kNN-restricted donor.
                 donor_kind: str = 'all2all',
                 donor_pbest_frac: float = 0.1,
                 donor_chunk_size: int | None = None):
        super().__init__()
        assert gatv2_hidden % n_heads == 0
        assert donor_kind in ('all2all', 'knn'), donor_kind

        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.donor_kind = donor_kind

        # Input projections (same as dense)
        self.node_norm = nn.LayerNorm(node_in)
        self.node_proj = nn.Linear(node_in, gatv2_hidden)
        self.edge_proj = nn.Linear(edge_in, gatv2_hidden)
        self.global_norm = nn.LayerNorm(global_in)
        self.global_proj = nn.Linear(global_in, gatv2_hidden)

        # Sparse GATv2 layers
        self.layers = nn.ModuleList([
            SparseGATv2Layer(
                hidden_dim=gatv2_hidden,
                edge_dim=gatv2_hidden,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(gatv2_layers)
        ])

        # Edge-to-node readout
        self.edge_readout = nn.Linear(gatv2_hidden, gatv2_hidden)

        # Global readout
        self.global_readout = nn.Sequential(
            nn.Linear(gatv2_hidden * 2, gatv2_hidden),
            nn.ReLU(),
            nn.Linear(gatv2_hidden, global_out_dim),
        )

        # Donor selection head — TWO families share state_dict shapes for
        # warmstart compatibility:
        #   donor_kind='all2all' (legacy):
        #       DonorSelectionGATv2  →  logits (B, N, N, R), O(N^2) memory.
        #   donor_kind='knn' (D1000 line):
        #       DonorSelectionKNN    →  logits (B, N, k_donor, R) +
        #                               cand_idx (B, N, k_donor), strict O(N*k_donor).
        # Both consume node_feat[..., 0] (fit_rank*2-1) as the fitness bias.
        if enable_donor_selection:
            if donor_kind == 'knn':
                from .operators.donor_selection_knn import DonorSelectionKNN
                self.donor_selector = DonorSelectionKNN(
                    hidden_dim=gatv2_hidden,
                    attn_dim=donor_attn_dim,
                    n_roles=donor_n_roles,
                    p_pool_frac=donor_pbest_frac,
                )
            else:
                from .operators.donor_selection import DonorSelectionGATv2
                self.donor_selector = DonorSelectionGATv2(
                    hidden_dim=gatv2_hidden,
                    attn_dim=donor_attn_dim,
                    n_roles=donor_n_roles,
                    query_chunk_size=donor_chunk_size,
                )
        else:
            self.donor_selector = None

        total = sum(p.numel() for p in self.parameters())
        log.info("SparseGATv2Backbone: %d params (node_in=%d, hidden=%d, "
                 "n_heads=%d, layers=%d, global_out=%d, donor_selector=%s)",
                 total, node_in, gatv2_hidden, n_heads, gatv2_layers,
                 global_out_dim, enable_donor_selection)

    def encode(self, node_feat, global_feat, cache: SparseTopologyCache,
               n_active: Optional[int] = None,
               donor_mask: Optional[torch.Tensor] = None,
               **_ignored):
        # **_ignored absorbs kwargs that the outer TemporalSparseGATv2Backbone
        # injects (coords/fitness, used by B2 — TemporalSetAttentionEdge —
        # but ignored here). Keeps direct-instantiation call sites
        # (e.g. test_sparse_integration) working when opt_variant forwards
        # the live (coords, fitness) into the encode call.
        """Run sparse GATv2 backbone + donor selection head.

        Args:
            node_feat:   (B, N, node_in). Index 0 MUST be fit_rank*2-1 ∈ [-1,1]
                         (produced by `build_sparse_graphs_gpu`). This index is
                         consumed by `donor_selector` as the fitness bias.
            global_feat: (B, global_in)
            cache:       SparseTopologyCache with knn_idx and edge_feat
            n_active:    Optional[int]. If provided, the global readout
                         (mean+max pool that feeds h_global) is restricted to
                         the first `n_active` rows of `h`. Used by the
                         graph-native archive (E9_archive_K50) to prevent
                         archive nodes from contaminating population-state
                         globals (D6 in archive_design.md). When None,
                         behavior is unchanged from the no-archive baseline.
            donor_mask:  Optional (B, N_total) bool. Forwarded to the
                         donor_selector as `cand_mask` so that invalid
                         (warmup-False) archive slots are forbidden as donors
                         on all 3 channels (D9). Only consumed when
                         n_active is also provided AND n_active < N_total
                         (i.e. there are archive candidates).

        Returns:
            BackboneOutput NamedTuple with fields:
                h, e, h_per_head, h_global, donor_logits (may be None),
                h_pooled (None — filled by TemporalSparseGATv2Backbone wrapper).
        """
        from .backbone_output import BackboneOutput

        B, N = node_feat.shape[:2]

        # Input projections
        h = self.node_proj(self.node_norm(node_feat))   # (B, N, H)
        e = self.edge_proj(cache.edge_feat)              # (B, N, k, H)
        h_g = self.global_proj(self.global_norm(global_feat))  # (B, H)

        # Inject global context
        h = h + h_g.unsqueeze(1)

        # Sequential sparse GATv2 layers
        for i, layer in enumerate(self.layers):
            h, e = layer(h, cache.knn_idx, e)
            # Re-inject global between layers (except last)
            if i < len(self.layers) - 1:
                h = h + h_g.unsqueeze(1)

        # Edge-to-node readout: mean over k neighbors
        e_agg = e.mean(dim=2)                  # (B, N, H)
        h = h + self.edge_readout(e_agg)

        # Per-head view
        h_per_head = h.view(B, N, self.n_heads, self.head_dim)

        # Global readout — optionally restricted to first n_active rows so
        # that archive nodes (when present) do not contaminate population
        # state. No-op when n_active is None or equals N.
        if n_active is not None and n_active < N:
            h_for_globals = h[:, :n_active]
        else:
            h_for_globals = h
        h_mean = h_for_globals.mean(dim=1)                  # (B, H)
        h_max = h_for_globals.max(dim=1).values             # (B, H)
        h_global = self.global_readout(torch.cat([h_mean, h_max], dim=-1))

        # Donor selection logits (None if disabled via ctor).
        # Two paths:
        #   donor_kind='all2all' (legacy): DonorSelectionGATv2 returns
        #       (B, N_q, N_c, R) logits; cand_idx is None (de_heads gathers
        #       from full pool_coords identity-mapped).
        #   donor_kind='knn' (D1000): DonorSelectionKNN returns
        #       (logits, cand_idx) over a sparse axis k_donor.
        donor_logits = None
        donor_cand_idx = None
        if self.donor_selector is not None:
            if self.donor_kind == 'knn':
                # knn-restricted donor needs knn_idx + fit_signed
                # (= node_feat[..., 0]). Active parents only when n_active<N.
                if n_active is not None and n_active < N:
                    h_q = h[:, :n_active]
                    nf_q = node_feat[:, :n_active]
                    knn_q = cache.knn_idx[:, :n_active]
                else:
                    h_q = h
                    nf_q = node_feat
                    knn_q = cache.knn_idx
                fit_signed_q = nf_q[..., 0]
                donor_logits, donor_cand_idx = self.donor_selector(
                    h_q, nf_q, knn_q, fit_signed_q)
            else:
                if n_active is not None and n_active < N:
                    donor_logits = self.donor_selector.forward_asym(
                        h[:, :n_active], h,
                        node_feat[:, :n_active], node_feat,
                        cand_mask=donor_mask)
                else:
                    donor_logits = self.donor_selector(h, node_feat)

        return BackboneOutput(
            h=h, e=e, h_per_head=h_per_head, h_global=h_global,
            donor_logits=donor_logits, h_pooled=None,
            donor_cand_idx=donor_cand_idx,
        )

    def forward(self, node_feat, global_feat, cache, **kwargs):
        """Alias for encode() — required by torch.func.functional_call."""
        return self.encode(node_feat, global_feat, cache,
                           n_active=kwargs.get('n_active'),
                           donor_mask=kwargs.get('donor_mask'))
