"""
transformer_backbone.py — Shared PNA + Transformer encoder backbone.

Drop-in alternative to PNAGATv2Backbone. Replaces GATv2 sparse message-passing
with dense self-attention + sparse edge bias while reusing the same PNA
perceptual encoder and bridge layers.

Returns the same 4-tuple from encode():
    h:          (N, attn_hidden)        per-node output (all heads concatenated)
    e:          (E, attn_hidden)        edge embeddings
    h_per_head: (N, n_heads, head_dim)  raw attention head outputs (not mixed)
    h_global:   (B, pna_out_dim)        PNA graph-level readout
"""
import contextlib
import logging

import torch
import torch.nn as nn
from torch_scatter import scatter_mean

from .pna_model import PNAFeatureExtractor
from .transformer_layer import SparseEdgeBiasedAttentionLayer
from .similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM
from .temporal_graph import TEMPORAL_EDGE_DIM

log = logging.getLogger(__name__)


class TransformerBackbone(nn.Module):
    """Shared encoder: PNA (perception) + bridges + dense self-attention (action).

    Identical to PNAGATv2Backbone except GATv2ConcatLayer is replaced by
    SparseEdgeBiasedAttentionLayer. The encode() interface is identical so all
    downstream consumers can use either backbone interchangeably.
    """

    def __init__(self, node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
                 pna_hidden=64, pna_out=32, pna_layers=4,
                 attn_hidden=64, attn_layers=2, n_heads=4,
                 dropout=0.1, use_edge_bias=True,
                 pna_checkpoint=None, device='cpu'):
        super().__init__()
        assert attn_hidden % n_heads == 0, \
            f"attn_hidden={attn_hidden} must be divisible by n_heads={n_heads}"

        self.n_heads = n_heads
        self.head_dim = attn_hidden // n_heads
        self.attn_hidden = attn_hidden
        self.pna_out = pna_out
        self.device = device
        self._pna_frozen = True

        # ---- PNA encoder (perception) ----
        self.pna = PNAFeatureExtractor(
            node_in=node_in,
            edge_in=edge_in,
            global_in=global_in,
            hidden_dim=pna_hidden,
            out_dim=pna_out,
            num_layers=pna_layers,
            device=device,
        )
        self.override_degree_histogram()

        if pna_checkpoint is not None:
            self.load_pna_checkpoint(pna_checkpoint)

        # Freeze PNA initially
        for param in self.pna.parameters():
            param.requires_grad = False

        # ---- Bridges: PNA outputs → attention inputs ----
        pna_cat_dim = pna_out * 2  # cat([h_nodes, h_node_edge])
        self.node_bridge = nn.Linear(pna_cat_dim, attn_hidden)
        self.edge_bridge = nn.Linear(pna_out, attn_hidden)
        self.global_bridge = nn.Linear(pna_out, attn_hidden)

        # ---- Transformer attention layers ----
        self.attn_layers = nn.ModuleList([
            SparseEdgeBiasedAttentionLayer(
                hidden_dim=attn_hidden,
                edge_dim=attn_hidden,
                n_heads=n_heads,
                dropout=dropout,
                use_edge_bias=use_edge_bias,
            )
            for _ in range(attn_layers)
        ])

        # ---- Edge-to-node readout ----
        self.edge_readout = nn.Linear(attn_hidden, attn_hidden)

        # ---- Temporal edge projection (backward compat) ----
        self.temporal_edge_proj = nn.Linear(TEMPORAL_EDGE_DIM, edge_in)
        with torch.no_grad():
            self.temporal_edge_proj.weight.zero_()
            min_dim = min(edge_in, TEMPORAL_EDGE_DIM)
            self.temporal_edge_proj.weight[:min_dim, :min_dim] = torch.eye(min_dim)
            self.temporal_edge_proj.bias.zero_()

        # Log param counts
        total = sum(p.numel() for p in self.parameters())
        pna_p = sum(p.numel() for p in self.pna.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        log.info("TransformerBackbone: %d total (%d PNA, %d trainable, n_heads=%d)",
                 total, pna_p, trainable, n_heads)

    # ------------------------------------------------------------------
    # Batch mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_batch_mask(v_indices):
        """Build (N, N) boolean mask preventing cross-graph attention.

        Returns None for single-graph input (all-to-all attention).
        """
        if v_indices is None:
            return None
        # mask[i,j] = True iff node i and node j belong to the same graph
        return v_indices.unsqueeze(0) == v_indices.unsqueeze(1)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, node_feat, edge_index, edge_attr, global_feat,
               v_indices=None, e_indices=None, **_ignored):
        """Run PNA + bridges + dense self-attention with h_global re-injection.

        Args:
            node_feat:   (N, node_in) node features
            edge_index:  (2, E) connectivity
            edge_attr:   (E, edge_in) edge features
            global_feat: (B, global_in) global features
            v_indices:   (N,) node-to-graph mapping (None for single graph)
            e_indices:   (E,) edge-to-graph mapping (None for single graph)

        Returns:
            h:          (N, attn_hidden) per-node output (concat of all heads)
            e:          (E, attn_hidden) edge embeddings
            h_per_head: (N, n_heads, head_dim) raw attention head outputs
            h_global:   (B, pna_out) PNA graph-level readout
        """
        # Handle temporal edges
        if edge_attr.shape[1] == TEMPORAL_EDGE_DIM:
            edge_attr = self.temporal_edge_proj(edge_attr)

        # PNA (frozen or trainable)
        ctx = torch.no_grad() if self._pna_frozen else contextlib.nullcontext()
        with ctx:
            h_nodes, h_edges, h_global, h_node_edge = self.pna(
                node_feat, edge_index, edge_attr, global_feat,
                v_indices, e_indices,
            )

        # Bridges
        h_in = torch.cat([h_nodes, h_node_edge], dim=-1)
        h = self.node_bridge(h_in)
        e = self.edge_bridge(h_edges)

        # Global context injection
        h_g = self.global_bridge(h_global)
        if v_indices is not None:
            h = h + h_g[v_indices]
        else:
            h = h + h_g[0]

        # Batch mask for dense attention
        batch_mask = self._build_batch_mask(v_indices)

        # Transformer layers with h_global re-injection between layers
        for i, layer in enumerate(self.attn_layers):
            h, e = layer(h, edge_index, e, batch_mask=batch_mask)
            if i < len(self.attn_layers) - 1:
                # Re-inject global context to prevent dilution
                if v_indices is not None:
                    h = h + h_g[v_indices]
                else:
                    h = h + h_g[0]

        # Edge readout: aggregate edge embeddings back to nodes
        e_agg = scatter_mean(e, edge_index[1], dim=0, dim_size=h.shape[0])
        h = h + self.edge_readout(e_agg)

        # Per-head view
        h_per_head = h.view(-1, self.n_heads, self.head_dim)

        return h, e, h_per_head, h_global

    def encode_with_pna(self, node_feat, edge_index, edge_attr, global_feat,
                        v_indices=None, e_indices=None):
        """Like encode(), but also returns pre-bridge PNA node features.

        Returns:
            h, e, h_per_head, h_global — same as encode()
            h_pna: (N, pna_out*2) — cat([h_nodes, h_node_edge]) before bridge
        """
        if edge_attr.shape[1] == TEMPORAL_EDGE_DIM:
            edge_attr = self.temporal_edge_proj(edge_attr)

        ctx = torch.no_grad() if self._pna_frozen else contextlib.nullcontext()
        with ctx:
            h_nodes, h_edges, h_global, h_node_edge = self.pna(
                node_feat, edge_index, edge_attr, global_feat,
                v_indices, e_indices,
            )

        h_pna = torch.cat([h_nodes, h_node_edge], dim=-1)

        h = self.node_bridge(h_pna)
        e = self.edge_bridge(h_edges)

        h_g = self.global_bridge(h_global)
        if v_indices is not None:
            h = h + h_g[v_indices]
        else:
            h = h + h_g[0]

        batch_mask = self._build_batch_mask(v_indices)

        for i, layer in enumerate(self.attn_layers):
            h, e = layer(h, edge_index, e, batch_mask=batch_mask)
            if i < len(self.attn_layers) - 1:
                if v_indices is not None:
                    h = h + h_g[v_indices]
                else:
                    h = h + h_g[0]

        e_agg = scatter_mean(e, edge_index[1], dim=0, dim_size=h.shape[0])
        h = h + self.edge_readout(e_agg)
        h_per_head = h.view(-1, self.n_heads, self.head_dim)

        return h, e, h_per_head, h_global, h_pna

    def encode_layerwise(self, node_feat, edge_index, edge_attr, global_feat,
                         v_indices=None, e_indices=None):
        """Single forward pass returning intermediates at every encoder level.

        Returns dict with 8 representations (all on input device):
            h_nodes:       (N, pna_out)       — PNA raw node embeddings
            h_node_edge:   (N, pna_out)       — PNA per-node edge context
            h_pna:         (N, pna_out*2)     — cat([h_nodes, h_node_edge])
            h_post_bridge: (N, attn_hidden)   — after bridge + global inject
            h_attn0:       (N, attn_hidden)   — after attention layer 0 + re-inject
            h_transformer: (N, attn_hidden)   — final (last layer + edge readout)
            h_global:      (B, pna_out)       — PNA graph-level readout
            h_per_head:    (N, n_heads, head_dim) — raw attention heads
        """
        if edge_attr.shape[1] == TEMPORAL_EDGE_DIM:
            edge_attr = self.temporal_edge_proj(edge_attr)

        ctx = torch.no_grad() if self._pna_frozen else contextlib.nullcontext()
        with ctx:
            h_nodes, h_edges, h_global, h_node_edge = self.pna(
                node_feat, edge_index, edge_attr, global_feat,
                v_indices, e_indices,
            )

        h_pna = torch.cat([h_nodes, h_node_edge], dim=-1)

        h = self.node_bridge(h_pna)
        e = self.edge_bridge(h_edges)

        h_g = self.global_bridge(h_global)
        if v_indices is not None:
            h = h + h_g[v_indices]
        else:
            h = h + h_g[0]
        h_post_bridge = h.clone()

        batch_mask = self._build_batch_mask(v_indices)

        # Attention layer 0
        h, e = self.attn_layers[0](h, edge_index, e, batch_mask=batch_mask)
        if len(self.attn_layers) > 1:
            if v_indices is not None:
                h = h + h_g[v_indices]
            else:
                h = h + h_g[0]
        h_attn0 = h.clone()

        # Remaining layers
        for layer in self.attn_layers[1:]:
            h, e = layer(h, edge_index, e, batch_mask=batch_mask)

        e_agg = scatter_mean(e, edge_index[1], dim=0, dim_size=h.shape[0])
        h_final = h + self.edge_readout(e_agg)
        h_per_head = h_final.view(-1, self.n_heads, self.head_dim)

        return {
            'h_nodes': h_nodes,
            'h_node_edge': h_node_edge,
            'h_pna': h_pna,
            'h_post_bridge': h_post_bridge,
            'h_attn0': h_attn0,
            'h_transformer': h_final,
            'h_global': h_global,
            'h_per_head': h_per_head,
        }

    # ------------------------------------------------------------------
    # PNA utilities (identical to PNAGATv2Backbone)
    # ------------------------------------------------------------------

    def override_degree_histogram(self, deg_hist=None):
        """Override PNA degree histogram for k-NN graphs (k=8, bidirectional)."""
        if deg_hist is None:
            deg_hist = torch.zeros(20, dtype=torch.long)
            deg_hist[6] = 100
            deg_hist[8] = 500
            deg_hist[10] = 2000
            deg_hist[12] = 3000
            deg_hist[14] = 2000
            deg_hist[16] = 500

        self.pna.deg_histogram.copy_(deg_hist)

        deg_float = deg_hist.float()
        N_total = deg_float.sum()
        if N_total == 0:
            log.warning("override_degree_histogram: all-zero histogram, skipping")
            return
        bin_degree = torch.arange(deg_hist.numel(), dtype=torch.float)
        avg_deg_lin = (bin_degree * deg_float).sum() / N_total
        avg_deg_log = ((bin_degree + 1).log() * deg_float).sum() / N_total
        for layer in self.pna.layers:
            aggr = layer.pna_conv.aggr_module
            aggr.avg_deg_lin.fill_(avg_deg_lin.item())
            aggr.avg_deg_log.fill_(avg_deg_log.item())

    def load_pna_checkpoint(self, checkpoint_path):
        """Load pretrained PNA encoder weights from SSL checkpoint.

        Handles multiple checkpoint formats:
          - {'encoder_state_dict': ...}
          - {'model_state_dict': {'encoder.xxx': ...}}
          - {'model_state_dict': {'pna.xxx': ...}}
          - Raw state dict
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        if 'encoder_state_dict' in ckpt:
            state = ckpt['encoder_state_dict']
        elif 'model_state_dict' in ckpt:
            inner = ckpt['model_state_dict']
            state = {}
            for key, val in inner.items():
                if key.startswith('encoder.'):
                    state[key.replace('encoder.', '')] = val
                elif key.startswith('pna.'):
                    state[key.replace('pna.', '')] = val
            if not state:
                state = inner
        else:
            state = {}
            for key, val in ckpt.items():
                if key.startswith('encoder.'):
                    state[key.replace('encoder.', '')] = val
            if not state:
                state = ckpt

        # Handle node_proj width expansion (NODE_DIM 9 → 16)
        node_proj_key = 'node_proj.0.weight'
        if (node_proj_key in state
                and state[node_proj_key].shape[1] !=
                self.pna.node_proj[0].weight.shape[1]):
            old_w = state.pop(node_proj_key)
            old_in = old_w.shape[1]
            new_w = self.pna.node_proj[0].weight.data.clone()
            new_w[:, :old_in] = old_w
            self.pna.node_proj[0].weight.data.copy_(new_w)
            log.info("node_proj weight expanded: %d -> %d", old_in,
                     new_w.shape[1])

        result = self.pna.load_state_dict(state, strict=False)
        log.info("Loaded PNA from %s (%d keys, missing=%s, unexpected=%s)",
                 checkpoint_path, len(state),
                 result.missing_keys, result.unexpected_keys)
        self.override_degree_histogram()

    def freeze_pna(self):
        """Freeze PNA parameters (no gradients)."""
        self._pna_frozen = True
        for param in self.pna.parameters():
            param.requires_grad = False

    def unfreeze_pna(self):
        """Unfreeze PNA parameters (gradients enabled)."""
        self._pna_frozen = False
        for param in self.pna.parameters():
            param.requires_grad = True

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result

    def get_param_groups(self, lr_pna=3e-5, lr_bridges=3e-4, lr_attn=3e-4):
        """Return optimizer param groups with differential learning rates."""
        groups = []
        # PNA (only if unfrozen)
        pna_params = [p for p in self.pna.parameters() if p.requires_grad]
        if pna_params:
            groups.append({'params': pna_params, 'lr': lr_pna, 'name': 'pna'})
        # Bridges
        bridge_params = (list(self.node_bridge.parameters()) +
                         list(self.edge_bridge.parameters()) +
                         list(self.global_bridge.parameters()) +
                         list(self.temporal_edge_proj.parameters()))
        groups.append({'params': bridge_params, 'lr': lr_bridges, 'name': 'bridges'})
        # Transformer attention + edge_readout
        attn_params = (list(p for l in self.attn_layers for p in l.parameters()) +
                       list(self.edge_readout.parameters()))
        groups.append({'params': attn_params, 'lr': lr_attn, 'name': 'transformer'})
        return groups
