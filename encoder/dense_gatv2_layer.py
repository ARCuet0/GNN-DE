"""
dense_gatv2_layer.py — Dense GATv2 layer using matmul-based attention.

No scatter, no .item(), no nonzero() — fully vmap-compatible.
Drop-in replacement for GATv2ConcatLayer when topology is dense (B, N, N).

GATv2 formula (Brody et al. 2022):
    score_ij = att^T LeakyReLU( W_l x_i + W_r x_j + W_e e_ij )
    alpha_ij = softmax_j(score_ij)   (over neighbors j of i)
    h_i = sum_j alpha_ij * W_r x_j   + residual

Shapes: all (B, N, ...) — batched over B graphs of same size N.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGATv2Layer(nn.Module):
    """Dense GATv2 message-passing layer with concat=True.

    Args:
        hidden_dim: node hidden dimension (must be divisible by n_heads)
        edge_dim: edge feature dimension
        n_heads: number of attention heads
        dropout: attention dropout
    """

    def __init__(self, hidden_dim, edge_dim, n_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # GATv2 linear transforms (matching PyG GATv2Conv param names)
        self.lin_l = nn.Linear(hidden_dim, hidden_dim)   # source (left)
        self.lin_r = nn.Linear(hidden_dim, hidden_dim)   # target (right)
        self.lin_edge = nn.Linear(edge_dim, hidden_dim)  # edge features
        # Init att vectors with norm ~2.0 per head (matching SSL pretrained)
        # to enable peaked attention from the start (avoids gradient starvation)
        self.att = nn.Parameter(torch.randn(1, 1, n_heads, self.head_dim) * (2.0 / self.head_dim ** 0.5))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Edge update MLP (matches GATv2ConcatLayer)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, edge_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x, adj, edge_feat):
        """
        Args:
            x:         (B, N, hidden_dim) node features
            adj:       (B, N, N) bool — adjacency mask (True where edge exists)
            edge_feat: (B, N, N, edge_dim) edge features (zero where no edge)

        Returns:
            h:         (B, N, hidden_dim) updated node features
            edge_out:  (B, N, N, edge_dim) updated edge features
        """
        B, N, H = x.shape

        # Pre-norm residual
        x_normed = self.norm(x)

        # Linear transforms: (B, N, H) -> (B, N, n_heads, head_dim)
        x_l = self.lin_l(x_normed).view(B, N, self.n_heads, self.head_dim)
        x_r = self.lin_r(x_normed).view(B, N, self.n_heads, self.head_dim)

        # Edge transform: (B, N, N, edge_dim) -> (B, N, N, n_heads, head_dim)
        e_lin = self.lin_edge(edge_feat).view(B, N, N, self.n_heads, self.head_dim)

        # GATv2 attention scores: att^T LeakyReLU(x_l[i] + x_r[j] + e_ij)
        # x_l[i]: (B, N, 1, heads, head_dim) broadcast over j
        # x_r[j]: (B, 1, N, heads, head_dim) broadcast over i
        msg = x_l.unsqueeze(2) + x_r.unsqueeze(1) + e_lin  # (B, N, N, heads, hd)
        msg = F.leaky_relu(msg, 0.2)

        # Score: (B, N, N, n_heads)
        scores = (msg * self.att).sum(dim=-1)

        # Masked softmax: set non-edges to -inf before softmax
        # adj: (B, N, N) -> (B, N, N, 1)
        mask = ~adj.unsqueeze(-1)  # True where NO edge
        scores = scores.masked_fill(mask, float('-inf'))

        # Softmax over source dimension (dim=2, over j for each i)
        alpha = F.softmax(scores, dim=2)  # (B, N, N, n_heads)

        # Handle isolated nodes: softmax(-inf, -inf, ...) = NaN -> replace with 0
        alpha = alpha.nan_to_num(0.0)

        # Apply dropout to attention weights
        alpha = self.dropout(alpha)

        # Aggregate: h_i = sum_j alpha_ij * x_r[j]
        # alpha: (B, N_dst, N_src, n_heads), x_r: (B, N_src, n_heads, head_dim)
        # Use einsum to avoid materializing (B, N, N, heads, head_dim) tensor
        h = torch.einsum('bijk,bjkd->bikd', alpha, x_r)  # (B, N, n_heads, head_dim)

        # Concat heads: (B, N, hidden_dim)
        h = h.reshape(B, N, self.hidden_dim) + self.bias

        # Residual connection
        h = h + x

        # Edge update: MLP(h[src] || h[dst] || edge_feat_normed)
        # h[i]: (B, N, 1, H) broadcast; h[j]: (B, 1, N, H) broadcast
        edge_input = torch.cat([
            h.unsqueeze(2).expand(B, N, N, H),   # h[i]
            h.unsqueeze(1).expand(B, N, N, H),   # h[j]
            self.edge_norm(edge_feat),
        ], dim=-1)  # (B, N, N, 2H + edge_dim)
        edge_out = self.edge_mlp(edge_input)  # (B, N, N, edge_dim)

        return h, edge_out
