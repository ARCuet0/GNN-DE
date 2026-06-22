"""
gatv2_layer.py — GATv2 message-passing layer with concat=True.

Shared across all systems that use GATv2 (NEURAL_META_K4, GNN_MOS_Classic,
NEURAL_ELA_MOS, HyperOPT). Extracted from DifferentiableOperators.py.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATv2ConcatLayer(nn.Module):
    """GATv2 message-passing layer with concat=True (preserves per-head identity)."""

    def __init__(self, hidden_dim, edge_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        head_dim = hidden_dim // n_heads

        self.gat = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=n_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            add_self_loops=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, edge_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        # Pre-norm: normalize input before GAT, keep clean residual highway
        h = self.gat(self.norm(x), edge_index, edge_attr=edge_attr)
        h = self.dropout(h)
        h = h + x  # residual

        src, dst = edge_index
        edge_input = torch.cat([h[src], h[dst], self.edge_norm(edge_attr)], dim=-1)
        edge_attr = self.edge_mlp(edge_input)

        return h, edge_attr
