"""
temporal_backbone.py — PNA + GATv2 backbone augmented with GRU temporal features.

Composes TemporalGRUEncoder + TemporalDimPooler + PNAGATv2Backbone.
The GRU encodes per-(individual, dimension) temporal dynamics from a ring
buffer of recent generations, pools across D to get per-individual temporal
embeddings, and concatenates them with the standard graph node features
before feeding into PNA layer 0.

Drop-in replacement for PNAGATv2Backbone. Returns the same 4-tuple from
encode(): (h, e, h_per_head, h_global).
"""
import logging

import torch
import torch.nn as nn

from .backbone import PNAGATv2Backbone
from .npa_layers import TemporalGRUEncoder, TemporalDimPooler
from .similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM

log = logging.getLogger(__name__)


class TemporalPNAGATv2Backbone(nn.Module):
    """PNA + GATv2 with GRU temporal node features injected at PNA input.

    Architecture:
        coords_hist (W, N, D) + fitness_hist (W, N)
            → TemporalGRUEncoder → (N, D, d_rnn)
            → TemporalDimPooler  → (N, d_temporal)
            → cat([node_feat, h_temporal]) → (N, NODE_DIM + d_temporal)
            → PNAGATv2Backbone(node_in=NODE_DIM + d_temporal)
            → (h, e, h_per_head, h_global)

    When no temporal history is available (gen 0), zeros are used for the
    temporal features — the PNA node_proj Linear maps them to near-zero
    contributions via LayerNorm.
    """

    def __init__(self, d_rnn=32, d_temporal=32, gru_window=8,
                 node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
                 pna_hidden=64, pna_out=32, pna_layers=4,
                 gatv2_hidden=64, gatv2_layers=2, n_heads=4,
                 dropout=0.1, device='cpu',
                 # Silently ignore PNA checkpoint — training from scratch
                 pna_checkpoint=None, **_ignored):
        super().__init__()
        self.d_temporal = d_temporal
        self.device = device

        # --- Temporal encoder ---
        self.gru = TemporalGRUEncoder(d_model=d_rnn, d_rnn=d_rnn)
        self.pooler = TemporalDimPooler(d_rnn=d_rnn, d_out=d_temporal)

        # --- Inner PNA+GATv2 backbone with expanded node_in ---
        self._inner = PNAGATv2Backbone(
            node_in=node_in + d_temporal,
            edge_in=edge_in, global_in=global_in,
            pna_hidden=pna_hidden, pna_out=pna_out, pna_layers=pna_layers,
            gatv2_hidden=gatv2_hidden, gatv2_layers=gatv2_layers,
            n_heads=n_heads, dropout=dropout, device=device,
        )

        # Never freeze PNA — training from scratch
        self._inner.unfreeze_pna()

        # Expose attributes expected by BudgetMOSRouter
        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.pna_out = pna_out
        self._pna_frozen = False

        if pna_checkpoint is not None:
            log.warning("TemporalPNAGATv2Backbone trains from scratch; "
                        "ignoring pna_checkpoint=%s", pna_checkpoint)

        total = sum(p.numel() for p in self.parameters())
        gru_p = sum(p.numel() for p in self.gru.parameters())
        pool_p = sum(p.numel() for p in self.pooler.parameters())
        log.info("TemporalPNAGATv2Backbone: %d params (%d GRU, %d pooler, "
                 "%d inner backbone, n_heads=%d, d_temporal=%d)",
                 total, gru_p, pool_p, total - gru_p - pool_p,
                 n_heads, d_temporal)

    # ------------------------------------------------------------------
    # Properties for BudgetMOSRouter compatibility
    # ------------------------------------------------------------------

    @property
    def pna(self):
        return self._inner.pna

    @property
    def gatv2_layers(self):
        return self._inner.gatv2_layers

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, node_feat, edge_index, edge_attr, global_feat,
               v_indices=None, e_indices=None,
               coords_hist=None, fitness_hist=None, n_valid=None,
               **_ignored):
        """Run GRU temporal encoding + PNA + GATv2.

        Args:
            node_feat, edge_index, edge_attr, global_feat, v_indices, e_indices:
                Standard graph inputs (same as PNAGATv2Backbone).
            coords_hist:   (W, N, D) ring buffer coordinates (oldest first)
            fitness_hist:  (W, N) ring buffer fitness values
            n_valid:       0-dim long tensor — valid timesteps in buffer

        Returns:
            h, e, h_per_head, h_global — same as PNAGATv2Backbone.
        """
        N = node_feat.shape[0]

        if (coords_hist is not None
                and n_valid is not None
                and n_valid.item() > 0):
            h_temporal = self.gru(coords_hist, fitness_hist, n_valid)
            h_pooled = self.pooler(h_temporal)          # (N, d_temporal)
        else:
            h_pooled = node_feat.new_zeros(N, self.d_temporal)

        node_feat_aug = torch.cat([node_feat, h_pooled], dim=-1)

        return self._inner.encode(
            node_feat_aug, edge_index, edge_attr, global_feat,
            v_indices, e_indices)

    # ------------------------------------------------------------------
    # PNA utilities (delegate to inner backbone)
    # ------------------------------------------------------------------

    def freeze_pna(self):
        self._pna_frozen = True
        self._inner.freeze_pna()

    def unfreeze_pna(self):
        self._pna_frozen = False
        self._inner.unfreeze_pna()

    def override_degree_histogram(self, deg_hist=None):
        self._inner.override_degree_histogram(deg_hist)

    def load_pna_checkpoint(self, checkpoint_path):
        log.warning("TemporalPNAGATv2Backbone trains from scratch; "
                    "ignoring checkpoint %s", checkpoint_path)

    def get_param_groups(self, lr_gru=3e-4, **kwargs):
        """Return optimizer param groups with differential learning rates."""
        temporal_params = (list(self.gru.parameters()) +
                           list(self.pooler.parameters()))
        groups = [{'params': temporal_params, 'lr': lr_gru, 'name': 'temporal'}]
        groups.extend(self._inner.get_param_groups(**kwargs))
        return groups

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result
