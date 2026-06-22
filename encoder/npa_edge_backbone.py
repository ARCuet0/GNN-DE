"""
npa_edge_backbone.py — NPABackbone variant with edge feature injection.

Ablation variant: after attention pool (D→N) and before feature injection,
aggregates edge features from the k-NN graph via scatter_mean and fuses
them into the per-individual embedding.

Same interface as NPABackbone: encode() → (h, e_agg, h_per_head, h_global).
Unlike base NPA, returns e_agg (N, hidden_dim) instead of None.
"""

import logging

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from .npa_backbone import NPABackbone
from .similarity_graph import EDGE_DIM

log = logging.getLogger(__name__)


class NPAEdgeBackbone(NPABackbone):
    """NPABackbone + edge feature injection after pool."""

    def __init__(self, edge_in=EDGE_DIM, **kwargs):
        super().__init__(**kwargs)
        hidden = self.hidden_dim

        # Edge aggregation: project edge features, scatter_mean to nodes,
        # then fuse with the per-individual embedding.
        self.edge_proj = nn.Linear(edge_in, hidden)
        self.edge_fuse = nn.Linear(hidden * 2, hidden)
        self.edge_norm = nn.LayerNorm(hidden)

        # Re-log param count
        total = sum(p.numel() for p in self.parameters())
        edge_p = (sum(p.numel() for p in self.edge_proj.parameters())
                  + sum(p.numel() for p in self.edge_fuse.parameters())
                  + sum(p.numel() for p in self.edge_norm.parameters()))
        log.info("NPAEdgeBackbone: %d params (+%d edge injection)", total, edge_p)

    def encode(self, node_feat, edge_index, edge_attr, global_feat,
               v_indices=None, e_indices=None,
               coords_hist=None, fitness_hist=None,
               valid_mask=None, n_valid=None,
               coords_current=None, fitness_current=None,
               f_init=None, fes_ratio=None, **_ignored):
        """Same as NPABackbone.encode but injects edge features post-pool."""
        if coords_hist is not None:
            h, h_global = self._encode_with_edges(
                coords_hist, fitness_hist, valid_mask, n_valid,
                coords_current, fitness_current, f_init, v_indices,
                edge_index, edge_attr, fes_ratio)
        else:
            h, _, h_per_head, h_global = self._encode_fallback(
                coords_current, fitness_current, v_indices)
            return h, None, h_per_head, h_global

        h_per_head = h.view(h.shape[0], self.n_heads, self.head_dim)

        # Edge readout: scatter_mean edge embeddings to destination nodes
        e_emb = self.edge_proj(edge_attr)  # (E, hidden_dim)
        e_agg = scatter_mean(e_emb, edge_index[1], dim=0,
                             dim_size=h.shape[0])  # (N, hidden_dim)

        return h, e_agg, h_per_head, h_global

    def _encode_with_edges(self, coords_hist, fitness_hist,
                           valid_mask, n_valid,
                           coords_current, fitness_current,
                           f_init, v_indices,
                           edge_index, edge_attr, fes_ratio):
        """Three-stage attention + edge injection."""
        n_pop = None
        if v_indices is not None:
            B = v_indices[-1].item() + 1
            if B > 1:
                n_pop = coords_current.shape[0] // B

        h_grid, h_global_raw = self.grid_attn(
            coords_hist, fitness_hist, n_valid,
            fitness_current, fes_ratio, n_pop=n_pop)

        h_global = self.global_proj(h_global_raw)

        # Pool D→N
        N_total = h_grid.shape[0]
        q = self.pool_query.unsqueeze(0).expand(N_total, -1, -1)
        h = self.pool_attn(q, h_grid).squeeze(1)
        h = self.pool_norm(self.pool_proj(h))   # (N, hidden_dim)

        # Edge injection: aggregate neighbor edge features
        e_emb = torch.relu(self.edge_proj(edge_attr))  # (E, hidden_dim)
        e_ctx = scatter_mean(e_emb, edge_index[1], dim=0,
                             dim_size=N_total)           # (N, hidden_dim)
        h = self.edge_norm(self.edge_fuse(
            torch.cat([h, e_ctx], dim=-1)))              # (N, hidden_dim)

        # Standard feature injection
        h = self.feature_injector(
            h, coords_current, fitness_current, v_indices)

        return h, h_global
