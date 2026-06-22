"""
set_attention_backbone.py — Set-attention ablation arm of the deployed backbone.

Drops the GATv2 message-passing entirely. Reuses the temporal encoder, the
induced-point pooler, the input projections, the global readout, and the
all-to-all DonorSelectionGATv2 head from the sparse backbone. The only
component that changes is the inner stack: three Pre-LN multi-head
self-attention blocks operate over the N (individual) axis with no edge
features and no positional embedding (the population is a set).

Used as the falsification arm for the graph inductive-bias claim. See
`analysis/preregistration_set_attn_2026_05_21.md`.
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

log = logging.getLogger(__name__)


class SetAttentionBackbone(nn.Module):
    """Edge-free message-passing replacement for `SparseGATv2Backbone`.

    Same parameter shapes for `node_norm`, `node_proj`, `global_norm`,
    `global_proj`, `global_readout`, and `donor_selector` so warmstart
    checkpoints saved against the sparse arm can load these submodules
    via `strict=False` partial load.

    Returns the same `BackboneOutput` NamedTuple. The `e` field is a
    `(B, N, 0, gatv2_hidden)` zero-axis sentinel — set-attention has no
    edges, but downstream code unpacks the field.
    """

    def __init__(self, node_in=8, edge_in=4, global_in=16,
                 gatv2_hidden=128, gatv2_layers=2, n_heads=8,
                 global_out_dim=128, dropout=0.1,
                 enable_donor_selection=True,
                 donor_attn_dim=16, donor_n_roles=3,
                 donor_chunk_size: int | None = None):
        super().__init__()
        assert gatv2_hidden % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        # Donor-kind tracking parallel to SparseGATv2Backbone; set-attention
        # only supports all-to-all (the knn-restricted donor head would
        # consume topology, which is precisely what we're ablating away).
        self.donor_kind = 'all2all'

        # Input projections — shapes match SparseGATv2Backbone for warmstart.
        # `edge_in` is accepted (signature parity) but unused.
        self.node_norm = nn.LayerNorm(node_in)
        self.node_proj = nn.Linear(node_in, gatv2_hidden)
        self.global_norm = nn.LayerNorm(global_in)
        self.global_proj = nn.Linear(global_in, gatv2_hidden)

        # Set-attention layers — replace SparseGATv2Layer stack. Tokens are
        # individuals. No positional embedding (permutation-invariant). No
        # edges. FFN width 4× matches the temporal encoder convention.
        self.layers = nn.ModuleList([
            _SelfAttnBlock(gatv2_hidden, n_heads, gatv2_hidden * 4, dropout)
            for _ in range(gatv2_layers)
        ])

        # Global readout — same shape as SparseGATv2Backbone for warmstart.
        self.global_readout = nn.Sequential(
            nn.Linear(gatv2_hidden * 2, gatv2_hidden),
            nn.ReLU(),
            nn.Linear(gatv2_hidden, global_out_dim),
        )

        # All-to-all donor selection — reused unchanged from sparse arm.
        # DonorSelectionGATv2 consumes only h and node_feat[..., 0]
        # (fit_rank*2-1); no edges, no topology.
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

        total = sum(p.numel() for p in self.parameters())
        log.info("SetAttentionBackbone: %d params (node_in=%d, hidden=%d, "
                 "n_heads=%d, layers=%d, global_out=%d, donor_selector=%s, "
                 "edges=DISABLED)",
                 total, node_in, gatv2_hidden, n_heads, gatv2_layers,
                 global_out_dim, enable_donor_selection)

    def encode(self, node_feat, global_feat, cache: SparseTopologyCache,
               n_active: Optional[int] = None,
               donor_mask: Optional[torch.Tensor] = None):
        """Run set-attention backbone + donor selection head.

        Args:
            node_feat:   (B, N, node_in). Index 0 MUST be fit_rank*2-1
                         (consumed by donor_selector as fitness bias).
            global_feat: (B, global_in).
            cache:       SparseTopologyCache. Only `cache.knn_idx` and
                         `cache.edge_feat` are present for interface parity;
                         neither is consumed by the set-attention path.
            n_active:    Optional[int]. Restricts global readout + donor
                         queries to the first `n_active` rows of `h`, matching
                         the archive-aware semantics of SparseGATv2Backbone.
            donor_mask:  Optional (B, N_total) bool. Forwarded to donor_selector
                         as `cand_mask` to forbid invalid archive candidates.

        Returns:
            BackboneOutput with `e` as a (B, N, 0, gatv2_hidden) zero-axis
            sentinel and `donor_cand_idx=None` (all2all only).
        """
        B, N = node_feat.shape[:2]

        # Input projections (identical to SparseGATv2Backbone)
        h = self.node_proj(self.node_norm(node_feat))   # (B, N, H)
        h_g = self.global_proj(self.global_norm(global_feat))  # (B, H)

        # Inject global context into every token
        h = h + h_g.unsqueeze(1)

        # Set-attention layers. _SelfAttnBlock expects (B, L, D); we feed
        # (B, N, H) — N is the set-token axis.
        for i, layer in enumerate(self.layers):
            h = layer(h)
            # Re-inject global between layers (except last), mirroring
            # SparseGATv2Backbone.encode loop semantics.
            if i < len(self.layers) - 1:
                h = h + h_g.unsqueeze(1)

        # Per-head view (same convention as sparse arm)
        h_per_head = h.view(B, N, self.n_heads, self.head_dim)

        # Global readout — optionally restricted to the first n_active rows.
        if n_active is not None and n_active < N:
            h_for_globals = h[:, :n_active]
        else:
            h_for_globals = h
        h_mean = h_for_globals.mean(dim=1)
        h_max = h_for_globals.max(dim=1).values
        h_global = self.global_readout(torch.cat([h_mean, h_max], dim=-1))

        # Donor selection (all-to-all, no edges).
        donor_logits = None
        if self.donor_selector is not None:
            if n_active is not None and n_active < N:
                donor_logits = self.donor_selector.forward_asym(
                    h[:, :n_active], h,
                    node_feat[:, :n_active], node_feat,
                    cand_mask=donor_mask)
            else:
                donor_logits = self.donor_selector(h, node_feat)

        # Edge sentinel: set-attention has no edges. (B, N, 0, H) carries
        # zero data but satisfies positional unpacking + shape assertions.
        e_sentinel = h.new_zeros(B, N, 0, self.gatv2_hidden)

        return BackboneOutput(
            h=h, e=e_sentinel, h_per_head=h_per_head, h_global=h_global,
            donor_logits=donor_logits, h_pooled=None,
            donor_cand_idx=None,
        )

    def forward(self, node_feat, global_feat, cache, **kwargs):
        return self.encode(node_feat, global_feat, cache,
                           n_active=kwargs.get('n_active'),
                           donor_mask=kwargs.get('donor_mask'))


class TemporalSetAttentionBackbone(BackboneCompatMixin, nn.Module):
    """Temporal attention + induced-point pooler + set-attention backbone.

    Outer wrapper mirroring `TemporalSparseGATv2Backbone`. Same constructor
    signature so model_factory can dispatch between the two backbones with
    no other code changes.

    The temporal encoder + induced-point pooler are unchanged — those are
    inputs to the inductive-bias test, not subjects of it.
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
                 # Donor-kind / pbest_frac / chunk_size accepted for signature
                 # parity with the sparse backbone. donor_kind != 'all2all'
                 # is rejected — the knn-restricted donor head would consume
                 # topology, defeating the ablation.
                 donor_kind: str = 'all2all',
                 donor_pbest_frac: float = 0.1,
                 donor_chunk_size: int | None = None,
                 **_ignored):
        super().__init__()
        assert donor_kind == 'all2all', (
            f"TemporalSetAttentionBackbone supports only donor_kind='all2all'; "
            f"got {donor_kind!r}. The knn-restricted donor head requires "
            f"topology, which is precisely what this backbone ablates.")
        self.d_temporal = d_temporal
        self.gru_window = gru_window
        self.device = device
        # topology_mode and k_neighbors are accepted (signature parity) but
        # unused — set-attention is graph-free.
        self.topology_mode = topology_mode
        self.k_neighbors = k_neighbors
        self.use_checkpoint = False

        # Temporal encoder (identical to sparse arm)
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

        # Pooler (identical to sparse arm)
        if pooler_type == 'induced':
            self.pooler = InducedPointPooler(d_rnn=d_rnn, d_out=d_temporal)
        else:
            self.pooler = TemporalDimPooler(d_rnn=d_rnn, d_out=d_temporal)

        # Inner set-attention backbone
        self.backbone = SetAttentionBackbone(
            node_in=node_in + d_temporal,
            edge_in=edge_in, global_in=global_in,
            gatv2_hidden=gatv2_hidden, gatv2_layers=gatv2_layers,
            n_heads=n_heads, global_out_dim=global_out_dim,
            dropout=dropout,
            donor_chunk_size=donor_chunk_size,
        )

        # Expose attributes for variant compatibility
        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.pna_out = global_out_dim

        total = sum(p.numel() for p in self.parameters())
        temp_p = sum(p.numel() for p in self.temporal.parameters())
        pool_p = sum(p.numel() for p in self.pooler.parameters())
        log.info("TemporalSetAttentionBackbone: %d params (%d temporal, %d pooler, "
                 "%d set_attention)",
                 total, temp_p, pool_p, total - temp_p - pool_p)

    def encode(self, node_feat, global_feat, cache,
               coords_hist=None, fitness_hist=None, n_valid=None,
               coords=None, n_active: Optional[int] = None,
               donor_mask: Optional[torch.Tensor] = None, **_ignored):
        """Drop-in for `TemporalSparseGATv2Backbone.encode`.

        Set-attention path: ignores cache.knn_idx and cache.edge_feat. The
        temporal encoder + pooler produce h_pooled exactly as the sparse arm,
        and h_pooled is concatenated onto node_feat to feed the inner
        backbone. Same `_replace(h_pooled=...)` composition pattern.
        """
        B = node_feat.shape[0]
        N = node_feat.shape[1]

        # Temporal encoding (logic copied verbatim from
        # sparse_temporal_backbone.encode to keep the two arms bit-equal on
        # the temporal/pooler stage)
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

        # Set-attention is topology-free: ignore cache.knn_idx / cache.edge_feat
        # entirely. We still pass `cache` through for signature parity.
        out = self.backbone.encode(node_feat_aug, global_feat, cache,
                                   n_active=n_active,
                                   donor_mask=donor_mask)
        return out._replace(h_pooled=h_pooled)

    def forward(self, node_feat, global_feat, cache, **kwargs):
        return self.encode(node_feat, global_feat, cache, **kwargs)
