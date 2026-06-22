"""
set_attention_edge_backbone.py — B2 arm of the 4-variant topology/edges
ablation (plan 2026-05-29).

Identical to `set_attention_backbone.py` (B1: all-to-all self-attn, no edges)
EXCEPT that each self-attention block receives a dense edge-feature bias
projected from the 3-d all-to-all edge_attr (built by
`build_dense_edge_attr_gpu`). This isolates the contribution of the explicit
pairwise inductive bias on top of all-to-all topology, complementing B1
(which has neither) and A (which has both k-NN topology and edges).

The shared submodules — temporal encoder, induced-point pooler, node/global
input projections, global readout, donor head — are kept identical to the
sparse arm so the inputs are common across A / B1 / B2.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

from .backbone_compat import BackboneCompatMixin
from .backbone_output import BackboneOutput
from .npa_layers import InducedPointPooler, TemporalDimPooler
from .sparse_gatv2_backbone import SparseTopologyCache, TopologyMode
from .temporal_attention import TemporalAttentionEncoder, _SelfAttnBlock
from .graph_builder_sparse import build_dense_edge_attr_gpu

log = logging.getLogger(__name__)

DENSE_EDGE_DIM = 3   # e0_dense, e1_dense, e2_dense (reciprocity is dropped)


class SetAttentionEdgeBackbone(nn.Module):
    """All-to-all self-attention + dense edge-feature bias.

    Mirrors `SetAttentionBackbone` shape-for-shape on the shared submodules
    so warmstart partial loads behave consistently across the ablation arms.
    Each self-attn block i carries an extra `edge_bias_projs[i] : Linear(3, 1)`
    that projects the dense edge_attr into a `(B, 1, N, N)` additive logit
    bias.
    """

    def __init__(self, node_in=8, edge_in=4, global_in=16,
                 gatv2_hidden=128, gatv2_layers=2, n_heads=8,
                 global_out_dim=128, dropout=0.1,
                 enable_donor_selection=True,
                 donor_attn_dim=16, donor_n_roles=3,
                 donor_chunk_size: int | None = None,
                 lb: float = -100.0, ub: float = 100.0):
        super().__init__()
        assert gatv2_hidden % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.donor_kind = 'all2all'
        self.lb = lb
        self.ub = ub

        # Shared shapes with B1 / A — node/global IO is the same model.
        self.node_norm = nn.LayerNorm(node_in)
        self.node_proj = nn.Linear(node_in, gatv2_hidden)
        self.global_norm = nn.LayerNorm(global_in)
        self.global_proj = nn.Linear(global_in, gatv2_hidden)

        # Set-attention inner stack (same shapes as B1).
        self.layers = nn.ModuleList([
            _SelfAttnBlock(gatv2_hidden, n_heads, gatv2_hidden * 4, dropout)
            for _ in range(gatv2_layers)
        ])

        # Shared readout shape with B1 / A.
        self.global_readout = nn.Sequential(
            nn.Linear(gatv2_hidden * 2, gatv2_hidden),
            nn.ReLU(),
            nn.Linear(gatv2_hidden, global_out_dim),
        )

        if enable_donor_selection:
            from .operators.donor_selection import DonorSelectionGATv2
            self.donor_selector = DonorSelectionGATv2(
                hidden_dim=gatv2_hidden,
                attn_dim=donor_attn_dim,
                n_roles=donor_n_roles,
                query_chunk_size=donor_chunk_size,
            )
        else:
            self.donor_selector = None

        # Per-layer edge-bias projections: (3,) → (1,) scalar bias per pair.
        # bias=False so a zeroed weight gives exactly zero bias for the
        # B1-parity sanity test. Constructed LAST in __init__ so all shared
        # modules above (layers, global_readout, donor_selector) get
        # RNG-aligned init with `SetAttentionBackbone` (B1) under a shared
        # torch.manual_seed — preserves the zero-bias-parity contract.
        self.edge_bias_projs = nn.ModuleList([
            nn.Linear(DENSE_EDGE_DIM, 1, bias=False)
            for _ in range(gatv2_layers)
        ])

        total = sum(p.numel() for p in self.parameters())
        log.info("SetAttentionEdgeBackbone: %d params (hidden=%d, heads=%d, "
                 "layers=%d, edges=DENSE_3D, donor_selector=%s)",
                 total, gatv2_hidden, n_heads, gatv2_layers,
                 enable_donor_selection)

    def encode(self, node_feat, global_feat, cache: SparseTopologyCache,
               n_active: Optional[int] = None,
               donor_mask: Optional[torch.Tensor] = None,
               coords: Optional[torch.Tensor] = None,
               fitness: Optional[torch.Tensor] = None):
        """Run set-attention-with-edge-bias + donor selection.

        Args:
            node_feat:   (B, N, node_in). Index 0 must be fit_rank*2-1 for the
                         donor head's fitness bias term.
            global_feat: (B, global_in).
            cache:       SparseTopologyCache. cache.knn_idx / cache.edge_feat
                         are accepted for signature parity but unused.
            n_active:    Optional[int] for archive-aware global readout.
            donor_mask:  Optional bool mask for archive candidates.
            coords:      (B, N, D) raw coords — REQUIRED for the dense edge
                         attr builder. If None, the call falls back to a
                         zero bias (degenerate identity with B1).
            fitness:     (B, N) fitness values — REQUIRED for dense edge attr.

        Returns:
            BackboneOutput with `e=(B,N,0,H)` sentinel (same convention as B1)
            and `donor_cand_idx=None`.
        """
        B, N = node_feat.shape[:2]

        # Build the dense edge bias once and reuse across layers via a
        # per-layer Linear(3,1) projection.
        if coords is not None and fitness is not None:
            edge_attr = build_dense_edge_attr_gpu(
                coords, fitness, lb=self.lb, ub=self.ub)  # (B, N, N, 3)
        else:
            edge_attr = node_feat.new_zeros(B, N, N, DENSE_EDGE_DIM)

        h = self.node_proj(self.node_norm(node_feat))            # (B, N, H)
        h_g = self.global_proj(self.global_norm(global_feat))     # (B, H)
        h = h + h_g.unsqueeze(1)

        for i, layer in enumerate(self.layers):
            # (B, N, N, 3) -> (B, N, N, 1) -> (B, 1, N, N) for head broadcast.
            edge_bias = self.edge_bias_projs[i](edge_attr).squeeze(-1)
            edge_bias = edge_bias.unsqueeze(1)
            h = layer(h, edge_bias=edge_bias)
            if i < len(self.layers) - 1:
                h = h + h_g.unsqueeze(1)

        h_per_head = h.view(B, N, self.n_heads, self.head_dim)

        if n_active is not None and n_active < N:
            h_for_globals = h[:, :n_active]
        else:
            h_for_globals = h
        h_mean = h_for_globals.mean(dim=1)
        h_max = h_for_globals.max(dim=1).values
        h_global = self.global_readout(torch.cat([h_mean, h_max], dim=-1))

        donor_logits = None
        if self.donor_selector is not None:
            if n_active is not None and n_active < N:
                donor_logits = self.donor_selector.forward_asym(
                    h[:, :n_active], h,
                    node_feat[:, :n_active], node_feat,
                    cand_mask=donor_mask)
            else:
                donor_logits = self.donor_selector(h, node_feat)

        e_sentinel = h.new_zeros(B, N, 0, self.gatv2_hidden)
        return BackboneOutput(
            h=h, e=e_sentinel, h_per_head=h_per_head, h_global=h_global,
            donor_logits=donor_logits, h_pooled=None,
            donor_cand_idx=None,
        )

    def forward(self, node_feat, global_feat, cache, **kwargs):
        return self.encode(node_feat, global_feat, cache,
                           n_active=kwargs.get('n_active'),
                           donor_mask=kwargs.get('donor_mask'),
                           coords=kwargs.get('coords'),
                           fitness=kwargs.get('fitness'))


class TemporalSetAttentionEdgeBackbone(BackboneCompatMixin, nn.Module):
    """Outer wrapper mirroring `TemporalSetAttentionBackbone`.

    The temporal encoder + induced-point pooler are unchanged across all 4
    ablation arms — those are inputs to the inductive-bias test, not subjects
    of it. The only diff vs B1 is the inner backbone (`SetAttentionEdgeBackbone`).
    """

    def __init__(self, d_rnn=64, d_temporal=64, gru_window=16,
                 node_in=8, edge_in=4, global_in=16,
                 gatv2_hidden=128, gatv2_layers=2, n_heads=8,
                 global_out_dim=128, dropout=0.1,
                 temporal_encoder='attention',
                 temporal_layers=2,
                 topology_mode=TopologyMode.COORDINATE_KNN,
                 k_neighbors=8,
                 pooler_type='induced',
                 device='cpu',
                 donor_kind: str = 'all2all',
                 donor_pbest_frac: float = 0.1,
                 donor_chunk_size: int | None = None,
                 lb: float = -100.0, ub: float = 100.0,
                 **_ignored):
        super().__init__()
        assert donor_kind == 'all2all', (
            f"TemporalSetAttentionEdgeBackbone supports only "
            f"donor_kind='all2all'; got {donor_kind!r}.")
        self.d_temporal = d_temporal
        self.gru_window = gru_window
        self.device = device
        self.topology_mode = topology_mode
        self.k_neighbors = k_neighbors
        self.use_checkpoint = False
        self.lb = lb
        self.ub = ub

        if temporal_encoder == 'attention':
            n_attn_heads = max(1, d_rnn // 16)
            while d_rnn % n_attn_heads != 0:
                n_attn_heads -= 1
            self.temporal = TemporalAttentionEncoder(
                d_model=d_rnn, n_layers=temporal_layers, n_heads=n_attn_heads,
                dropout=dropout, coord_range=100.0)
        else:
            from .npa_layers import TemporalGRUEncoder
            self.temporal = TemporalGRUEncoder(
                d_model=d_rnn, d_rnn=d_rnn)

        if pooler_type == 'induced':
            self.pooler = InducedPointPooler(d_rnn=d_rnn, d_out=d_temporal)
        else:
            self.pooler = TemporalDimPooler(d_rnn=d_rnn, d_out=d_temporal)

        self.backbone = SetAttentionEdgeBackbone(
            node_in=node_in + d_temporal,
            edge_in=edge_in, global_in=global_in,
            gatv2_hidden=gatv2_hidden, gatv2_layers=gatv2_layers,
            n_heads=n_heads, global_out_dim=global_out_dim,
            dropout=dropout,
            donor_chunk_size=donor_chunk_size,
            lb=lb, ub=ub,
        )

        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.pna_out = global_out_dim

        total = sum(p.numel() for p in self.parameters())
        temp_p = sum(p.numel() for p in self.temporal.parameters())
        pool_p = sum(p.numel() for p in self.pooler.parameters())
        log.info("TemporalSetAttentionEdgeBackbone: %d params (%d temporal, "
                 "%d pooler, %d set_attention_edge)",
                 total, temp_p, pool_p, total - temp_p - pool_p)

    def encode(self, node_feat, global_feat, cache,
               coords_hist=None, fitness_hist=None, n_valid=None,
               coords=None, fitness=None,
               n_active: Optional[int] = None,
               donor_mask: Optional[torch.Tensor] = None, **_ignored):
        """Drop-in for `TemporalSparseGATv2Backbone.encode`.

        Needs `coords` and `fitness` to build the dense edge bias. The opt
        loop already forwards both (`GenerationStep` passes the live
        population state). For a coords-less call we fall back to a zero
        bias (degenerate parity with B1).
        """
        B = node_feat.shape[0]
        N = node_feat.shape[1]

        # Temporal encoding — verbatim from sparse/set-attn arms so the
        # input bias is identical across the 4 ablation variants.
        if coords_hist is not None and n_valid is not None:
            nv = n_valid if isinstance(n_valid, int) else n_valid.item()
            if coords_hist.dim() == 4:
                B_t, W_t, N_t, D_t = coords_hist.shape
                fh = fitness_hist.float()
                fh_safe = fh.clamp(min=1e-30)
                f_min = fh_safe.amin(dim=(1, 2), keepdim=True)
                f_range = (fh_safe.amax(dim=(1, 2), keepdim=True) - f_min).clamp(min=1e-30)
                fh_norm = (fh_safe - f_min) / f_range
                was_training_t = self.temporal.training
                was_training_p = self.pooler.training
                self.temporal.eval()
                self.pooler.eval()
                try:
                    with torch.amp.autocast('cuda', enabled=False):
                        ch_flat = coords_hist.float().permute(1, 0, 2, 3).reshape(W_t, B_t * N_t, D_t)
                        fh_flat = fh_norm.permute(1, 0, 2).reshape(W_t, B_t * N_t)
                        h_temporal = self.temporal(ch_flat, fh_flat, nv, N_t * B_t, D_t)
                        h_flat = self.pooler(h_temporal)
                        h_pooled = h_flat.view(B_t, N_t, -1)
                finally:
                    self.temporal.train(was_training_t)
                    self.pooler.train(was_training_p)
            else:
                h_temporal = self.temporal(
                    coords_hist.float(), fitness_hist.float(), nv)
                h_pooled = self.pooler(h_temporal)
                h_pooled = h_pooled.unsqueeze(0).expand(B, -1, -1)
        else:
            h_pooled = node_feat.new_zeros(B, N, self.d_temporal)

        node_feat_aug = torch.cat([node_feat, h_pooled], dim=-1)

        # If fitness is missing, synthesize from node_feat[..., 0] (fit_rank
        # bias ∈ [-1, 1]) so that the dense edge bias is still meaningful.
        if coords is not None and fitness is None:
            # node_feat[..., 0] is fit_rank*2-1, monotonic in true rank;
            # the dense builder only consumes ranks, so this is sufficient.
            fitness = (node_feat[..., 0] + 1.0) * 0.5 + 1e-6

        out = self.backbone.encode(node_feat_aug, global_feat, cache,
                                   n_active=n_active,
                                   donor_mask=donor_mask,
                                   coords=coords, fitness=fitness)
        return out._replace(h_pooled=h_pooled)

    def forward(self, node_feat, global_feat, cache, **kwargs):
        return self.encode(node_feat, global_feat, cache, **kwargs)
