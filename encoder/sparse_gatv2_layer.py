"""
sparse_gatv2_layer.py — Sparse GATv2 layer using gather-based attention.

O(N·k) memory instead of O(N²). Same parameter shapes as DenseGATv2Layer —
weights are directly interchangeable.

GATv2 formula (Brody et al. 2022):
    score_ij = att^T LeakyReLU( W_l x_i + W_r x_j + W_e e_ij )
    alpha_ij = softmax_j(score_ij)   (over k neighbors j of i)
    h_i = sum_j alpha_ij * W_r x_j   + residual

All gather operations use static-shape knn_idx — fully vmap-compatible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gather_neighbors(x: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
    """Gather k neighbors of each node from x.

    Args:
        x: (B, N, *F) — node features (any trailing dims)
        knn_idx: (B, N, k) long — neighbor indices

    Returns:
        x_neighbors: (B, N, k, *F)
    """
    B, N, k = knn_idx.shape
    # Flatten trailing dims for gather, then reshape back
    trailing = x.shape[2:]
    F_flat = 1
    for d in trailing:
        F_flat *= d
    x_flat = x.reshape(B, N, F_flat)  # (B, N, F)

    # Expand knn_idx to (B, N*k, F) for gather along dim=1
    idx_flat = knn_idx.reshape(B, N * k)  # (B, N*k)
    idx_exp = idx_flat.unsqueeze(-1).expand(-1, -1, F_flat)  # (B, N*k, F)

    gathered = torch.gather(x_flat, 1, idx_exp)  # (B, N*k, F)
    return gathered.reshape(B, N, k, *trailing)


class SparseGATv2Layer(nn.Module):
    """Sparse GATv2 message-passing layer with gather-based attention.

    Same parameters as DenseGATv2Layer — weights are interchangeable.
    Forward operates on knn_idx (B,N,k) instead of adj (B,N,N).

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

        # Same parameter names as DenseGATv2Layer for weight compatibility
        self.lin_l = nn.Linear(hidden_dim, hidden_dim)   # source (left)
        self.lin_r = nn.Linear(hidden_dim, hidden_dim)   # target (right)
        self.lin_edge = nn.Linear(edge_dim, hidden_dim)  # edge features
        self.att = nn.Parameter(
            torch.randn(1, 1, n_heads, self.head_dim) * (2.0 / self.head_dim ** 0.5))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Edge update MLP (same structure as dense)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, edge_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x, knn_idx, edge_feat):
        """Sparse GATv2 forward with gather-based attention.

        Args:
            x:         (B, N, hidden_dim) node features
            knn_idx:   (B, N, k) long — neighbor indices
            edge_feat: (B, N, k, edge_dim) sparse edge features

        Returns:
            h:         (B, N, hidden_dim) updated node features
            edge_out:  (B, N, k, edge_dim) updated edge features
        """
        B, N, H = x.shape
        k = knn_idx.shape[2]
        heads = self.n_heads
        hd = self.head_dim

        # Pre-norm residual
        x_normed = self.norm(x)

        # Linear transforms: (B, N, H) → (B, N, heads, hd)
        x_l = self.lin_l(x_normed).view(B, N, heads, hd)
        x_r = self.lin_r(x_normed).view(B, N, heads, hd)

        # Edge transform: (B, N, k, edge_dim) → (B, N, k, heads, hd)
        e_lin = self.lin_edge(edge_feat).view(B, N, k, heads, hd)

        # Gather k neighbors of x_r: (B, N, k, heads, hd)
        x_r_k = gather_neighbors(x_r, knn_idx)

        # GATv2 scores: att^T LeakyReLU(x_l[i] + x_r[j] + e_ij)
        # x_l: (B, N, 1, heads, hd) broadcast over k
        msg = x_l.unsqueeze(2) + x_r_k + e_lin  # (B, N, k, heads, hd)
        msg = F.leaky_relu(msg, 0.2)

        # Score: (B, N, k, n_heads)
        scores = (msg * self.att).sum(dim=-1)

        # Softmax over k neighbors (dim=2)
        alpha = F.softmax(scores, dim=2)  # (B, N, k, heads)
        alpha = self.dropout(alpha)

        # Aggregate: h_i = sum_j alpha_ij * x_r[j]
        h = (alpha.unsqueeze(-1) * x_r_k).sum(dim=2)  # (B, N, heads, hd)

        # Concat heads + bias + residual
        h = h.reshape(B, N, H) + self.bias + x

        # Edge update: MLP(h[i] || h[j] || edge_feat_normed)
        h_i = h.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, H)
        h_j = gather_neighbors(h, knn_idx)           # (B, N, k, H)

        edge_input = torch.cat([
            h_i, h_j, self.edge_norm(edge_feat),
        ], dim=-1)  # (B, N, k, 2H + edge_dim)
        edge_out = self.edge_mlp(edge_input)  # (B, N, k, edge_dim)

        return h, edge_out
