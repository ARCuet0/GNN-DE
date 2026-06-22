"""
identity_backbone.py — C arm of the 4-variant topology/edges ablation
(plan 2026-05-29).

The relational floor: replaces the 3 message-passing layers (GATv2 in A;
self-attention in B1/B2) with a per-node `Linear(node_in → H)`. The
`temporal` encoder + `InducedPointPooler` + `DonorSelectionGATv2` head are
retained — the temporal pre-encoder is the shared signal extractor across
all 4 variants, and the donor head is the policy reader (not the population
encoder). Without it the DE step cannot select donors and the architecture
stops being TersQ.

Defining property: for the inner h-output, h[i] depends only on
node_feat[i, :] + global_feat — NOT on node_feat[j] for j ≠ i. The donor
head still mixes via all-to-all attention; that mixing is downstream, in
the policy reader, not in the population encoder.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

from .backbone_compat import BackboneCompatMixin
from .backbone_output import BackboneOutput
from .npa_layers import InducedPointPooler, TemporalDimPooler
from .sparse_gatv2_backbone import SparseTopologyCache, TopologyMode
from .temporal_attention import TemporalAttentionEncoder

log = logging.getLogger(__name__)


class IdentityBackbone(nn.Module):
    """Per-node MLP — no relational layers. Donor head retained for the
    DE step.
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
        self.donor_kind = 'all2all'

        # Shared shapes — same names as A/B1/B2 for warmstart parity if ever
        # needed; in the cold-start ablation they just match the structure.
        self.node_norm = nn.LayerNorm(node_in)
        self.node_proj = nn.Linear(node_in, gatv2_hidden)
        self.global_norm = nn.LayerNorm(global_in)
        self.global_proj = nn.Linear(global_in, gatv2_hidden)

        # No relational layers. `layers` is an empty ModuleList — kept for
        # symmetry with the other backbones' attribute names.
        self.layers = nn.ModuleList()

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

        total = sum(p.numel() for p in self.parameters())
        log.info("IdentityBackbone: %d params (node_in=%d, hidden=%d, "
                 "n_heads=%d, donor_selector=%s, relational_layers=NONE)",
                 total, node_in, gatv2_hidden, n_heads,
                 enable_donor_selection)

    def encode(self, node_feat, global_feat, cache: SparseTopologyCache,
               n_active: Optional[int] = None,
               donor_mask: Optional[torch.Tensor] = None):
        """h[i] = node_proj(node_norm(node_feat[i])) + global_proj(...).

        No mixing across individuals at the h stage. The donor head mixes
        downstream as the policy reader.
        """
        B, N = node_feat.shape[:2]

        h = self.node_proj(self.node_norm(node_feat))             # (B, N, H)
        h_g = self.global_proj(self.global_norm(global_feat))     # (B, H)
        h = h + h_g.unsqueeze(1)
        # Intentionally no per-N mixing here.

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
                           donor_mask=kwargs.get('donor_mask'))


class TemporalIdentityBackbone(BackboneCompatMixin, nn.Module):
    """Outer wrapper mirroring the other temporal backbones.

    Temporal encoder + InducedPointPooler are unchanged (shared input
    signal); the inner stack is the non-relational `IdentityBackbone`.
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
                 **_ignored):
        super().__init__()
        assert donor_kind == 'all2all', (
            f"TemporalIdentityBackbone supports only donor_kind='all2all'; "
            f"got {donor_kind!r}.")
        self.d_temporal = d_temporal
        self.gru_window = gru_window
        self.device = device
        self.topology_mode = topology_mode
        self.k_neighbors = k_neighbors
        self.use_checkpoint = False

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

        # gatv2_layers is accepted for signature parity but unused (the
        # identity inner stack has zero relational layers by construction).
        self.backbone = IdentityBackbone(
            node_in=node_in + d_temporal,
            edge_in=edge_in, global_in=global_in,
            gatv2_hidden=gatv2_hidden, gatv2_layers=gatv2_layers,
            n_heads=n_heads, global_out_dim=global_out_dim,
            dropout=dropout,
            donor_chunk_size=donor_chunk_size,
        )

        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.pna_out = global_out_dim

        total = sum(p.numel() for p in self.parameters())
        temp_p = sum(p.numel() for p in self.temporal.parameters())
        pool_p = sum(p.numel() for p in self.pooler.parameters())
        log.info("TemporalIdentityBackbone: %d params (%d temporal, %d pooler, "
                 "%d identity_inner)",
                 total, temp_p, pool_p, total - temp_p - pool_p)

    def encode(self, node_feat, global_feat, cache,
               coords_hist=None, fitness_hist=None, n_valid=None,
               coords=None, n_active: Optional[int] = None,
               donor_mask: Optional[torch.Tensor] = None, **_ignored):
        B = node_feat.shape[0]
        N = node_feat.shape[1]

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

        out = self.backbone.encode(node_feat_aug, global_feat, cache,
                                   n_active=n_active,
                                   donor_mask=donor_mask)
        return out._replace(h_pooled=h_pooled)

    def forward(self, node_feat, global_feat, cache, **kwargs):
        return self.encode(node_feat, global_feat, cache, **kwargs)
