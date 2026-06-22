"""
dense_gatv2_backbone.py — Dense GATv2 backbone with TopologyCache.

No scatter, no .item(), no nonzero() — fully vmap-compatible.
Drop-in replacement for GATv2OnlyBackbone when used with TopologyCache.

Returns same 4-tuple: (h, e_flat, h_per_head, h_global).
"""
import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from .backbone_compat import BackboneCompatMixin
from .dense_gatv2_layer import DenseGATv2Layer
from .similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM

log = logging.getLogger(__name__)


# ── TopologyCache ─────────────────────────────────────────────────

@dataclass
class TopologyCache:
    """Pre-computed graph topology for vmap-compatible forward pass.

    All tensors are dense (B, N, N) — no sparse ops needed.
    Computed once per generation, reused across M ES perturbations.
    """
    adj: torch.Tensor       # (P, N, N) bool — adjacency mask
    edge_feat: torch.Tensor # (P, N, N, edge_dim) — dense edge features
    B: int                  # number of graphs (P = M*B or just B)
    N: int                  # nodes per graph (same for all)
    node_feat: torch.Tensor = None   # (P, N, node_dim) — optional, from dense builder
    global_feat: torch.Tensor = None # (P, global_dim) — optional, from dense builder


def precompute_topology(
    edge_indices: List[torch.Tensor],
    edge_attrs: List[torch.Tensor],
    B: int,
    N: int,
) -> TopologyCache:
    """Convert B sparse graphs to dense TopologyCache.

    Args:
        edge_indices: list of B tensors, each (2, E_b) — sparse edge indices
        edge_attrs:   list of B tensors, each (E_b, edge_dim) — sparse edge features
        B: number of graphs
        N: number of nodes per graph (must be same for all)

    Returns:
        TopologyCache with dense (B, N, N) adjacency and (B, N, N, edge_dim) features.
    """
    assert len(edge_indices) == B
    assert len(edge_attrs) == B

    edge_dim = edge_attrs[0].shape[1]
    device = edge_attrs[0].device

    adj = torch.zeros(B, N, N, dtype=torch.bool, device=device)
    edge_feat = torch.zeros(B, N, N, edge_dim, device=device)

    for b in range(B):
        ei = edge_indices[b]
        ea = edge_attrs[b]
        src, dst = ei[0], ei[1]
        adj[b, src, dst] = True
        edge_feat[b, src, dst] = ea

    return TopologyCache(adj=adj, edge_feat=edge_feat, B=B, N=N)


# ── DenseGATv2Backbone ────────────────────────────────────────────

class DenseGATv2Backbone(BackboneCompatMixin, nn.Module):
    """Dense GATv2-only encoder with global feature injection.

    Architecture:
        node_feat (B, N, node_in) → LayerNorm + node_proj → h (B, N, hidden)
        edge_feat from cache (B, N, N, edge_in) → edge_proj → e (B, N, N, hidden)
        global_feat (B, global_in) → global_proj → h_g (B, hidden) [re-injected]

        DenseGATv2 layers → global readout (mean + max + MLP) → h_global

    Returns 4-tuple from encode():
        h:          (B, N, gatv2_hidden)
        e:          (B, N, N, edge_dim_internal)  — dense edge features
        h_per_head: (B, N, n_heads, head_dim)
        h_global:   (B, global_out_dim)
    """

    def __init__(self, node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
                 gatv2_hidden=64, gatv2_layers=2, n_heads=4,
                 global_out_dim=32, dropout=0.1, device='cpu',
                 # Compatibility kwargs (silently ignored)
                 pna_hidden=None, pna_out=None, pna_layers=None,
                 pna_checkpoint=None, use_readout_tokens=None,
                 n_readout=None, **_ignored):
        super().__init__()
        assert gatv2_hidden % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.pna_out = global_out_dim
        self.device = device

        # Input projections
        self.node_norm = nn.LayerNorm(node_in)
        self.node_proj = nn.Linear(node_in, gatv2_hidden)
        self.edge_proj = nn.Linear(edge_in, gatv2_hidden)
        self.global_norm = nn.LayerNorm(global_in)
        self.global_proj = nn.Linear(global_in, gatv2_hidden)

        # Dense GATv2 layers
        self.layers = nn.ModuleList([
            DenseGATv2Layer(
                hidden_dim=gatv2_hidden,
                edge_dim=gatv2_hidden,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(gatv2_layers)
        ])

        # Edge-to-node readout (dense: mean over neighbors)
        self.edge_readout = nn.Linear(gatv2_hidden, gatv2_hidden)

        # Global readout: mean + max -> MLP
        self.global_readout = nn.Sequential(
            nn.Linear(gatv2_hidden * 2, gatv2_hidden),
            nn.ReLU(),
            nn.Linear(gatv2_hidden, global_out_dim),
        )

        total = sum(p.numel() for p in self.parameters())
        log.info("DenseGATv2Backbone: %d params (node_in=%d, hidden=%d, "
                 "n_heads=%d, layers=%d, global_out=%d)",
                 total, node_in, gatv2_hidden, n_heads, gatv2_layers,
                 global_out_dim)

    def encode(self, node_feat, global_feat, cache: TopologyCache, **_ignored):
        """
        Args:
            node_feat:   (B, N, node_in)
            global_feat: (B, global_in)
            cache:       TopologyCache with adj (B,N,N) and edge_feat (B,N,N,edge_in)

        Returns:
            h:          (B, N, gatv2_hidden)
            e:          (B, N, N, gatv2_hidden)  — internal edge dim
            h_per_head: (B, N, n_heads, head_dim)
            h_global:   (B, global_out_dim)
        """
        B, N = cache.B, cache.N

        # Project inputs
        h = self.node_proj(self.node_norm(node_feat))       # (B, N, H)
        e = self.edge_proj(cache.edge_feat)                  # (B, N, N, H)
        h_g = self.global_proj(self.global_norm(global_feat))  # (B, H)

        # Global context injection: broadcast h_g to all nodes
        h = h + h_g.unsqueeze(1)  # (B, N, H)

        # GATv2 layers with global re-injection between layers
        for i, layer in enumerate(self.layers):
            h, e = layer(h, cache.adj, e)
            if i < len(self.layers) - 1:
                h = h + h_g.unsqueeze(1)

        # Edge-to-node readout: mean of neighbor edge features
        # adj: (B, N, N) -> count of neighbors per node
        adj_float = cache.adj.float()  # (B, N, N)
        n_neighbors = adj_float.sum(dim=2, keepdim=True).clamp(min=1)  # (B, N, 1)

        # Mean edge aggregation: sum(adj * e, dim=2) / n_neighbors
        # e: (B, N, N, H), adj_float: (B, N, N) -> (B, N, N, 1)
        e_masked = e * adj_float.unsqueeze(-1)         # zero out non-edges
        e_agg = e_masked.sum(dim=2) / n_neighbors      # (B, N, H)
        h = h + self.edge_readout(e_agg)

        # Per-head view
        h_per_head = h.view(B, N, self.n_heads, self.head_dim)

        # Global readout: mean + max over nodes
        h_mean = h.mean(dim=1)                          # (B, H)
        h_max = h.max(dim=1).values                     # (B, H)
        h_global = self.global_readout(torch.cat([h_mean, h_max], dim=-1))

        return h, e, h_per_head, h_global

    def forward(self, node_feat, global_feat, cache: TopologyCache):
        """Alias for encode() — required by torch.func.functional_call."""
        return self.encode(node_feat, global_feat, cache)

    def get_param_groups(self, lr_proj=3e-4, lr_gatv2=3e-4, **_ignored):
        proj_params = (list(self.node_norm.parameters()) +
                       list(self.node_proj.parameters()) +
                       list(self.edge_proj.parameters()) +
                       list(self.global_norm.parameters()) +
                       list(self.global_proj.parameters()) +
                       list(self.global_readout.parameters()))
        gatv2_params = (list(p for layer in self.layers
                             for p in layer.parameters()) +
                        list(self.edge_readout.parameters()))
        return [
            {'params': proj_params, 'lr': lr_proj, 'name': 'projections'},
            {'params': gatv2_params, 'lr': lr_gatv2, 'name': 'gatv2'},
        ]

    # to() inherited from BackboneCompatMixin
