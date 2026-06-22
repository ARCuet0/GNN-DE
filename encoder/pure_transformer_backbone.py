"""
pure_transformer_backbone.py — Pure attention backbone without PNA.

Dense self-attention over all nodes with three-scale interaction:
  - Node↔Node: multi-head attention (Q·K^T)
  - Edge bias: sparse edge features bias attention logits
  - Global conditioning: learned temperature + residual gate

Each layer updates nodes, edges, AND global bidirectionally.
No PNA, no k-NN message passing — purely attention-based.

Drop-in replacement for PNAGATv2Backbone / TransformerBackbone.
"""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM
from .temporal_graph import TEMPORAL_EDGE_DIM

log = logging.getLogger(__name__)


# ======================================================================
# Single transformer layer: node↔edge↔global bidirectional
# ======================================================================

class PureTransformerLayer(nn.Module):
    """One layer of three-scale transformer.

    Forward: (h, e, z) → (h', e', z')
      h: (N, H)  node embeddings
      e: (E, H)  edge embeddings
      z: (B, H)  global embeddings
    """

    def __init__(self, hidden_dim=64, n_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        # --- Attention ---
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.W_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.edge_bias_proj = nn.Linear(hidden_dim, n_heads)
        self.temp_proj = nn.Linear(hidden_dim, n_heads)

        # --- Edge value modulation ---
        self.edge_val_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # --- Global gate + output ---
        self.gate_proj = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

        # --- Edge update ---
        self.norm_e = nn.LayerNorm(hidden_dim)
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # --- Global update ---
        self.norm_z = nn.LayerNorm(hidden_dim)
        self.global_update = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, h, e, z, edge_index, v_indices, batch_mask):
        """
        Args:
            h: (N, H) node embeddings
            e: (E, H) edge embeddings
            z: (B, H) global embeddings
            edge_index: (2, E) sparse connectivity
            v_indices: (N,) node-to-graph mapping (None = single graph)
            batch_mask: (N, N) bool — True where attention is allowed
        Returns:
            h_new, e_new, z_new
        """
        N, H = h.shape
        n_heads = self.n_heads
        head_dim = self.head_dim
        src, dst = edge_index

        # ── 1. Multi-head attention with edge bias + global temperature ──
        h_norm = self.norm_h(h)
        qkv = self.W_qkv(h_norm).reshape(N, 3, n_heads, head_dim)
        Q, K, V = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (N, n_heads, head_dim)

        # Dense attention logits: (N, N, n_heads)
        attn_logits = torch.einsum('ihd,jhd->ijh', Q, K) * self.scale

        # Sparse edge bias
        edge_bias = self.edge_bias_proj(e)  # (E, n_heads)
        attn_logits[src, dst] = attn_logits[src, dst] + edge_bias

        # Global-conditioned temperature: (B, n_heads)
        temp = F.softplus(self.temp_proj(z)) + 0.1  # floor to prevent div-by-zero

        # Broadcast temperature to nodes
        if v_indices is not None:
            temp_per_node = temp[v_indices]  # (N, n_heads)
        else:
            temp_per_node = temp.expand(N, -1)  # single graph

        # Divide by learned temperature: (N, N, n_heads) / (N, 1, n_heads)
        attn_logits = attn_logits / temp_per_node.unsqueeze(1)

        # Batch mask: prevent cross-graph attention
        if batch_mask is not None:
            attn_logits = attn_logits.masked_fill(
                ~batch_mask.unsqueeze(-1), float('-inf'))

        weights = torch.softmax(attn_logits, dim=1)  # softmax over keys (j)
        weights = self.drop(weights)

        # ── 2. Value aggregation with edge modulation ──
        # Base message: dense attention over V
        msg_base = torch.einsum('ijh,jhd->ihd', weights, V)  # (N, n_heads, head_dim)

        # Edge value modulation: sparse contribution
        edge_val = self.edge_val_mlp(e).reshape(-1, n_heads, head_dim)  # (E, nh, hd)
        # Weight by attention at edge positions
        edge_weights = weights[src, dst]  # (E, n_heads)
        edge_msg = edge_val * edge_weights.unsqueeze(-1)  # (E, nh, hd)
        # Scatter-add to destination nodes
        msg_edge = torch.zeros_like(msg_base)
        msg_edge.scatter_add_(
            0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(edge_msg), edge_msg)

        node_msg = (msg_base + msg_edge).reshape(N, H)

        # ── 3. Global-conditioned residual gate ──
        if v_indices is not None:
            z_broadcast = z[v_indices]  # (N, H)
        else:
            z_broadcast = z.expand(N, -1)

        gate_input = torch.cat([h, node_msg, z_broadcast], dim=-1)  # (N, 3H)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # (N, H)
        h_new = gate * self.out_mlp(node_msg) + (1 - gate) * h

        # ── 4. Edge update (bidirectional from updated nodes) ──
        e_norm = self.norm_e(e)
        e_input = torch.cat([h_new[src], h_new[dst], e_norm], dim=-1)  # (E, 3H)
        e_new = e + self.edge_update(e_input)

        # ── 5. Global update (pooled from updated nodes) ──
        z_norm = self.norm_z(z)
        if v_indices is not None:
            B = z.shape[0]
            # Per-graph pooling
            h_mean = torch.zeros(B, H, device=h.device, dtype=h.dtype)
            h_mean.scatter_reduce_(
                0, v_indices.unsqueeze(-1).expand_as(h_new),
                h_new, reduce='mean', include_self=False)
            h_max = torch.full((B, H), -1e9, device=h.device, dtype=h.dtype)
            h_max.scatter_reduce_(
                0, v_indices.unsqueeze(-1).expand_as(h_new),
                h_new, reduce='amax', include_self=False)
            # std via mean of squared deviations
            h_sq = torch.zeros(B, H, device=h.device, dtype=h.dtype)
            h_sq.scatter_reduce_(
                0, v_indices.unsqueeze(-1).expand_as(h_new),
                (h_new - h_mean[v_indices]) ** 2,
                reduce='mean', include_self=False)
            h_std = h_sq.sqrt().clamp(min=1e-6)
        else:
            h_mean = h_new.mean(dim=0, keepdim=True)
            h_max = h_new.max(dim=0, keepdim=True).values
            h_std = h_new.std(dim=0, keepdim=True).clamp(min=1e-6)

        z_input = torch.cat([h_mean, h_max, h_std, z_norm], dim=-1)  # (B, 4H)
        z_new = z + self.global_update(z_input)

        return h_new, e_new, z_new


# ======================================================================
# Full backbone: encoding + L layers + output projection
# ======================================================================

class PureTransformerBackbone(nn.Module):
    """Pure attention backbone — no PNA, no k-NN message passing.

    Drop-in replacement for PNAGATv2Backbone / TransformerBackbone.
    Returns the same 4-tuple: (h, e, h_per_head, h_global).
    """

    def __init__(
        self,
        node_in=NODE_DIM,
        edge_in=EDGE_DIM,
        global_in=GLOBAL_DIM,
        hidden_dim=64,
        global_out_dim=32,
        n_layers=4,
        n_heads=4,
        dropout=0.1,
        device='cpu',
        # Compatibility kwargs (silently mapped or ignored)
        pna_hidden=None, pna_out=None, pna_layers=None,
        pna_checkpoint=None,
        attn_hidden=None, attn_layers=None,
        gatv2_hidden=None, gatv2_layers=None,
        use_edge_bias=None,
        **_ignored,
    ):
        super().__init__()

        # Map compatibility aliases
        hidden_dim = attn_hidden or gatv2_hidden or hidden_dim
        n_layers = attn_layers or gatv2_layers or n_layers
        global_out_dim = pna_out or global_out_dim

        self.hidden_dim = hidden_dim
        self.global_out_dim = global_out_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.device = device

        # Sentinel — no PNA
        self.pna = None
        self._pna_frozen = False

        # ── Initial encoders ──
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Temporal edge compat: project TEMPORAL_EDGE_DIM → EDGE_DIM
        self.temporal_edge_proj = nn.Linear(TEMPORAL_EDGE_DIM, edge_in)

        # ── Transformer layers ──
        self.layers = nn.ModuleList([
            PureTransformerLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # ── Output projection: z (hidden_dim) → h_global (global_out_dim) ──
        self.global_out_proj = nn.Linear(hidden_dim, global_out_dim)

        # Compatibility aliases
        self.node_bridge = self.node_encoder
        self.edge_bridge = self.edge_encoder
        self.global_bridge = self.global_encoder

        if pna_checkpoint is not None:
            log.warning("PureTransformerBackbone has no PNA; "
                        "ignoring checkpoint %s", pna_checkpoint)

        total = sum(p.numel() for p in self.parameters())
        log.info("PureTransformerBackbone: %d params (%d layers, "
                 "hidden=%d, heads=%d)", total, n_layers, hidden_dim, n_heads)

    # ------------------------------------------------------------------
    # Encode — main interface
    # ------------------------------------------------------------------

    def encode(self, node_feat, edge_index, edge_attr, global_feat,
               v_indices=None, e_indices=None):
        """Encode batched graph.

        Args:
            node_feat:   (N, node_in)   node features
            edge_index:  (2, E)         sparse connectivity
            edge_attr:   (E, edge_in)   edge features
            global_feat: (B, global_in) global features
            v_indices:   (N,) node-to-graph mapping (None = single graph)
            e_indices:   (E,) edge-to-graph mapping (unused here)

        Returns:
            h:          (N, hidden_dim)
            e:          (E, hidden_dim)
            h_per_head: (N, n_heads, head_dim)
            h_global:   (B, global_out_dim)
        """
        # Temporal edge compatibility
        if edge_attr.shape[1] == TEMPORAL_EDGE_DIM:
            edge_attr = self.temporal_edge_proj(edge_attr)

        # Initial encoding
        h = self.node_encoder(node_feat)       # (N, H)
        e = self.edge_encoder(edge_attr)       # (E, H)
        z = self.global_encoder(global_feat)   # (B, H)

        # Build batch mask (prevents cross-graph attention)
        batch_mask = self._build_batch_mask(v_indices, h.device)

        # L transformer layers
        for layer in self.layers:
            h, e, z = layer(h, e, z, edge_index, v_indices, batch_mask)

        # Output
        h_global = self.global_out_proj(z)     # (B, global_out_dim)
        h_per_head = h.view(-1, self.n_heads, self.head_dim)

        return h, e, h_per_head, h_global

    # ------------------------------------------------------------------
    # Batch mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_batch_mask(v_indices, device):
        """Build (N, N) boolean mask: True where nodes are in the same graph."""
        if v_indices is None:
            return None
        return v_indices.unsqueeze(0) == v_indices.unsqueeze(1)  # (N, N)

    # ------------------------------------------------------------------
    # Compatibility interface (drop-in for PNAGATv2Backbone)
    # ------------------------------------------------------------------

    @property
    def gatv2_layers(self):
        """Compatibility alias for downstream code accessing backbone.gatv2_layers."""
        return self.layers

    def set_phase(self, phase):
        """No-op: all params always trainable (no PNA to freeze/unfreeze)."""
        pass

    def freeze_pna(self):
        """No-op: no PNA."""
        pass

    def unfreeze_pna(self):
        """No-op: no PNA."""
        pass

    def override_degree_histogram(self, deg_hist=None):
        """No-op: no PNA."""
        pass

    def load_pna_checkpoint(self, checkpoint_path):
        """No-op: no PNA."""
        log.warning("PureTransformerBackbone has no PNA; "
                    "ignoring checkpoint %s", checkpoint_path)

    def get_param_groups(self, lr=3e-4, **kwargs):
        """Return optimizer param groups with differential learning rates."""
        lr_enc = kwargs.get('lr_bridges', kwargs.get('lr_pna', lr))
        lr_lay = kwargs.get('lr_attn', kwargs.get('lr_gatv2', lr))

        enc_params = (list(self.node_encoder.parameters()) +
                      list(self.edge_encoder.parameters()) +
                      list(self.global_encoder.parameters()) +
                      list(self.temporal_edge_proj.parameters()))
        layer_params = (list(self.layers.parameters()) +
                        list(self.global_out_proj.parameters()))

        return [
            {'params': enc_params, 'lr': lr_enc, 'name': 'encoders'},
            {'params': layer_params, 'lr': lr_lay, 'name': 'transformer'},
        ]

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result
