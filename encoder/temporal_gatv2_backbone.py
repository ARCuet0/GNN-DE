"""
temporal_gatv2_backbone.py — GATv2-only backbone with GRU temporal features.

Composes TemporalGRUEncoder + TemporalDimPooler + GATv2OnlyBackbone.
Same pattern as temporal_backbone.py but without PNA.

Drop-in replacement: returns (h, e, h_per_head, h_global).
~73K params total.
"""
import logging

import torch
import torch.nn as nn

from .gatv2_backbone import GATv2OnlyBackbone
from .npa_layers import TemporalGRUEncoder, TemporalDimPooler
from .temporal_attention import TemporalAttentionEncoder
from .similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM

log = logging.getLogger(__name__)


class TemporalGATv2Backbone(nn.Module):
    """GATv2-only backbone with GRU temporal node features.

    Architecture:
        coords_hist (W, N, D) + fitness_hist (W, N)
            → TemporalGRUEncoder → (N, D, d_rnn)
            → TemporalDimPooler  → (N, d_temporal)
            → cat([node_feat, h_temporal]) → (N, NODE_DIM + d_temporal)
            → GATv2OnlyBackbone(node_in=NODE_DIM + d_temporal)
            → (h, e, h_per_head, h_global)
    """

    def __init__(self, d_rnn=32, d_temporal=32, gru_window=8,
                 node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
                 gatv2_hidden=64, gatv2_layers=2, n_heads=4,
                 global_out_dim=32, dropout=0.1, device='cpu',
                 use_readout_tokens=False, n_readout=4,
                 gru_fp16=False,
                 gru_checkpoint=False,
                 temporal_encoder='gru',
                 # Silently ignored
                 pna_hidden=None, pna_out=None, pna_layers=None,
                 pna_checkpoint=None, **_ignored):
        super().__init__()
        self.d_temporal = d_temporal
        self.device = device
        self.gru_fp16 = gru_fp16

        # --- Temporal encoder ---
        if temporal_encoder == 'attention':
            n_attn_heads = max(1, d_rnn // 16)
            while d_rnn % n_attn_heads != 0:
                n_attn_heads -= 1
            self.gru = TemporalAttentionEncoder(
                d_model=d_rnn, n_layers=2, n_heads=n_attn_heads,
                dropout=dropout, coord_range=100.0)
            # ^ CEC2017 deployed regime. Override post-construction for BBOB.
        else:
            self.gru = TemporalGRUEncoder(d_model=d_rnn, d_rnn=d_rnn,
                                           gradient_checkpointing=gru_checkpoint)
        self.pooler = TemporalDimPooler(d_rnn=d_rnn, d_out=d_temporal)

        # --- Inner GATv2-only backbone with expanded node_in ---
        self._inner = GATv2OnlyBackbone(
            node_in=node_in + d_temporal,
            edge_in=edge_in, global_in=global_in,
            gatv2_hidden=gatv2_hidden, gatv2_layers=gatv2_layers,
            n_heads=n_heads, global_out_dim=global_out_dim,
            dropout=dropout, device=device,
            use_readout_tokens=use_readout_tokens, n_readout=n_readout,
        )

        # Expose attributes for BudgetMOSRouter compatibility
        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.pna_out = self._inner.pna_out
        self._pna_frozen = False

        total = sum(p.numel() for p in self.parameters())
        gru_p = sum(p.numel() for p in self.gru.parameters())
        pool_p = sum(p.numel() for p in self.pooler.parameters())
        log.info("TemporalGATv2Backbone: %d params (%d GRU, %d pooler, "
                 "%d inner GATv2, n_heads=%d, d_temporal=%d)",
                 total, gru_p, pool_p, total - gru_p - pool_p,
                 n_heads, d_temporal)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pna(self):
        return None

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
        N = node_feat.shape[0]

        if (coords_hist is not None
                and n_valid is not None
                and n_valid.item() > 0):
            if self.gru_fp16:
                with torch.autocast('cuda', dtype=torch.float16):
                    h_temporal = self.gru(coords_hist, fitness_hist, n_valid)
                h_temporal = h_temporal.float()
            else:
                h_temporal = self.gru(coords_hist, fitness_hist, n_valid)
            h_pooled = self.pooler(h_temporal)
        else:
            h_pooled = node_feat.new_zeros(N, self.d_temporal)

        node_feat_aug = torch.cat([node_feat, h_pooled], dim=-1)

        return self._inner.encode(
            node_feat_aug, edge_index, edge_attr, global_feat,
            v_indices, e_indices)

    # ------------------------------------------------------------------
    # Compatibility interface
    # ------------------------------------------------------------------

    def freeze_pna(self):
        self._pna_frozen = True

    def unfreeze_pna(self):
        self._pna_frozen = False

    def override_degree_histogram(self, deg_hist=None):
        pass

    def load_pna_checkpoint(self, checkpoint_path):
        log.warning("TemporalGATv2Backbone has no PNA; ignoring %s",
                    checkpoint_path)

    def get_param_groups(self, lr_gru=3e-4, lr_proj=3e-4, lr_gatv2=3e-4,
                         **_ignored):
        """Return optimizer param groups."""
        temporal_params = (list(self.gru.parameters()) +
                           list(self.pooler.parameters()))
        groups = [{'params': temporal_params, 'lr': lr_gru, 'name': 'temporal'}]
        groups.extend(self._inner.get_param_groups(
            lr_proj=lr_proj, lr_gatv2=lr_gatv2))
        return groups

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result
