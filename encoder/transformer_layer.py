"""
transformer_layer.py — Dense self-attention layer with sparse edge bias.

Drop-in alternative to GATv2ConcatLayer. Uses full N×N attention instead of
sparse message-passing on a kNN graph, with optional additive bias from
kNN edge features at (src, dst) positions.

No PyG dependency — uses only torch.nn and torch.einsum.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseEdgeBiasedAttentionLayer(nn.Module):
    """Dense self-attention with sparse edge bias and sparse edge update.

    Node update: standard multi-head self-attention over all N nodes.
    Edge bias:   kNN edge features projected to per-head scalars, added at
                 (src, dst) positions in the N×N attention logit matrix.
    Edge update: identical to GATv2ConcatLayer — cat([h[src], h[dst], e_ij])
                 through an MLP, only for existing (sparse) edges.
    """

    def __init__(self, hidden_dim, edge_dim, n_heads=4, dropout=0.1,
                 use_edge_bias=True):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.use_edge_bias = use_edge_bias

        # Pre-norm (same pattern as GATv2ConcatLayer)
        self.norm = nn.LayerNorm(hidden_dim)

        # QKV in one projection
        self.W_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Sparse edge bias
        if use_edge_bias:
            self.edge_bias_proj = nn.Linear(edge_dim, n_heads)

        # FFN block (transformers need this for feature mixing)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        # Edge update (identical to GATv2ConcatLayer)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, edge_dim),
            nn.LeakyReLU(),
        )

    def forward(self, x, edge_index, edge_attr, batch_mask=None):
        """
        Args:
            x:          (N, hidden_dim) node features
            edge_index: (2, E) sparse kNN connectivity
            edge_attr:  (E, edge_dim) sparse edge features
            batch_mask: (N, N) bool — True where attention is allowed
                        (None for single graph: all-to-all)

        Returns:
            h:         (N, hidden_dim) updated node features
            edge_attr: (E, edge_dim) updated edge features (sparse)
        """
        N = x.shape[0]

        # --- Self-attention ---
        x_normed = self.norm(x)

        # QKV: (N, 3*hidden) → 3 × (N, n_heads, head_dim)
        qkv = self.W_qkv(x_normed).view(N, 3, self.n_heads, self.head_dim)
        Q, K, V = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Dense attention logits: (N, N, n_heads)
        scale = self.head_dim ** -0.5
        logits = torch.einsum('ihd,jhd->ijh', Q, K) * scale

        # Sparse edge bias at kNN positions
        if self.use_edge_bias and edge_attr is not None:
            bias = self.edge_bias_proj(edge_attr)  # (E, n_heads)
            src, dst = edge_index
            logits[src, dst] = logits[src, dst] + bias

        # Batch mask: prevent cross-graph attention
        if batch_mask is not None:
            logits = logits.masked_fill(
                ~batch_mask.unsqueeze(-1), float('-inf')
            )

        attn = F.softmax(logits, dim=1)  # softmax over keys

        # Store attention weights for diagnostics (detached, no grad cost)
        self._last_attn = attn.detach()

        attn = self.attn_dropout(attn)

        # Value aggregation: (N, n_heads, head_dim)
        h = torch.einsum('ijh,jhd->ihd', attn, V)
        h = h.reshape(N, self.hidden_dim)
        h = self.out_proj(h)
        h = h + x  # residual

        # --- FFN ---
        h = h + self.ffn(self.ffn_norm(h))

        # --- Sparse edge update (identical to GATv2ConcatLayer) ---
        src, dst = edge_index
        edge_input = torch.cat(
            [h[src], h[dst], self.edge_norm(edge_attr)], dim=-1
        )
        edge_attr = self.edge_mlp(edge_input)

        return h, edge_attr
