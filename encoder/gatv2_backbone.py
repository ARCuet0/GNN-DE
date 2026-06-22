"""
gatv2_backbone.py — GATv2-only encoder backbone (no PNA).

Replaces PNAGATv2Backbone by eliminating PNA message-passing entirely.
Raw node/edge features are projected directly into GATv2 layers. Global
features (13-dim handcrafted landscape stats) are projected and re-injected
between GATv2 layers. Post-GATv2 global readout uses scatter_mean + scatter_max.

When use_readout_tokens=True, n_readout learnable virtual nodes participate
in GATv2 attention (fully connected to all real nodes per graph). After the
last GATv2 layer, these tokens are extracted and mean-pooled to produce
h_global (B, gatv2_hidden) — no extra layers, no projection.

Drop-in replacement: returns the same 4-tuple (h, e, h_per_head, h_global).
~65K params vs ~590K for PNA+GATv2.
"""
import logging

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from .gatv2_layer import GATv2ConcatLayer
from .similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM

log = logging.getLogger(__name__)


class GATv2OnlyBackbone(nn.Module):
    """GATv2-only encoder with global feature injection.

    Architecture:
        node_feat → LayerNorm + node_proj → h (N, hidden)
        edge_attr → edge_proj → e (E, hidden)
        global_feat → global_proj → h_g (B, hidden)  [re-injected between layers]

        GATv2 layers with h_g re-injection → edge readout →
        GlobalPoolReadout (scatter_mean + scatter_max + MLP) → h_global

    When use_readout_tokens=True:
        n_readout virtual nodes are appended per graph, fully connected
        to all real nodes. They participate in GATv2 attention directly.
        h_global = mean(readout_tokens) after final layer. Output dim = gatv2_hidden.

    Returns 4-tuple from encode():
        h:          (N, gatv2_hidden)
        e:          (E, gatv2_hidden)
        h_per_head: (N, n_heads, head_dim)
        h_global:   (B, global_out_dim)
    """

    def __init__(self, node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
                 gatv2_hidden=64, gatv2_layers=2, n_heads=4,
                 global_out_dim=32, dropout=0.1, device='cpu',
                 use_readout_tokens=False, n_readout=4,
                 # Compatibility kwargs (silently ignored)
                 pna_hidden=None, pna_out=None, pna_layers=None,
                 pna_checkpoint=None, **_ignored):
        super().__init__()
        assert gatv2_hidden % n_heads == 0, \
            f"gatv2_hidden={gatv2_hidden} must be divisible by n_heads={n_heads}"

        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.device = device
        self._pna_frozen = False
        self.use_readout_tokens = use_readout_tokens

        if use_readout_tokens:
            self.n_readout = n_readout
            self.readout_tokens = nn.Parameter(
                torch.randn(n_readout, gatv2_hidden) * 0.02)
            self.pna_out = gatv2_hidden  # h_global is 64-dim
        else:
            self.pna_out = global_out_dim

        # ---- Input projections ----
        self.node_norm = nn.LayerNorm(node_in)
        self.node_proj = nn.Linear(node_in, gatv2_hidden)
        self.edge_proj = nn.Linear(edge_in, gatv2_hidden)
        self.global_proj = nn.Linear(global_in, gatv2_hidden)

        # ---- GATv2 layers ----
        self.gatv2_layers = nn.ModuleList([
            GATv2ConcatLayer(
                hidden_dim=gatv2_hidden,
                edge_dim=gatv2_hidden,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(gatv2_layers)
        ])

        # ---- Edge-to-node readout ----
        self.edge_readout = nn.Linear(gatv2_hidden, gatv2_hidden)

        # ---- Post-GATv2 global readout (only without readout tokens) ----
        if not use_readout_tokens:
            self.global_readout = nn.Sequential(
                nn.Linear(gatv2_hidden * 2, gatv2_hidden),
                nn.ReLU(),
                nn.Linear(gatv2_hidden, global_out_dim),
            )

        total = sum(p.numel() for p in self.parameters())
        log.info("GATv2OnlyBackbone: %d params (node_in=%d, hidden=%d, "
                 "n_heads=%d, layers=%d, global_out=%d, readout_tokens=%s)",
                 total, node_in, gatv2_hidden, n_heads, gatv2_layers,
                 self.pna_out, use_readout_tokens)

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, node_feat, edge_index, edge_attr, global_feat,
               v_indices=None, e_indices=None, **_ignored):
        N_real = node_feat.shape[0]

        # Project inputs
        h = self.node_proj(self.node_norm(node_feat))
        e = self.edge_proj(edge_attr)
        h_g = self.global_proj(global_feat)

        # Global context injection
        if v_indices is not None:
            h = h + h_g[v_indices]
            B = global_feat.shape[0]
        else:
            h = h + h_g[0]
            B = 1

        # --- Append readout tokens as virtual nodes ---
        if self.use_readout_tokens:
            n_r = self.n_readout
            # Expand readout tokens per graph: (B * n_r, H)
            rt = self.readout_tokens.unsqueeze(0).expand(B, -1, -1)
            rt = rt.reshape(B * n_r, self.gatv2_hidden)

            # v_indices for virtual nodes
            rt_v = torch.arange(B, device=h.device).repeat_interleave(n_r)

            # Build fully-connected edges: real ↔ virtual per graph
            # For each graph g: real nodes [start_g..end_g] ↔ virtual [N_real + g*n_r .. N_real + (g+1)*n_r]
            fc_src, fc_dst = [], []
            if v_indices is not None:
                for g in range(B):
                    real_idx = (v_indices == g).nonzero(as_tuple=True)[0]
                    virt_idx = torch.arange(n_r, device=h.device) + N_real + g * n_r
                    # real → virtual
                    grid = torch.meshgrid(real_idx, virt_idx, indexing='ij')
                    fc_src.append(grid[0].reshape(-1))
                    fc_dst.append(grid[1].reshape(-1))
                    # virtual → real
                    fc_src.append(grid[1].reshape(-1))
                    fc_dst.append(grid[0].reshape(-1))
            else:
                real_idx = torch.arange(N_real, device=h.device)
                virt_idx = torch.arange(n_r, device=h.device) + N_real
                grid = torch.meshgrid(real_idx, virt_idx, indexing='ij')
                fc_src.append(grid[0].reshape(-1))
                fc_dst.append(grid[1].reshape(-1))
                fc_src.append(grid[1].reshape(-1))
                fc_dst.append(grid[0].reshape(-1))

            fc_edges = torch.stack([torch.cat(fc_src), torch.cat(fc_dst)])  # (2, E_fc)
            E_fc = fc_edges.shape[1]

            # Edge features for virtual edges: zeros (learned via attention)
            fc_edge_attr = e.new_zeros(E_fc, self.gatv2_hidden)

            # Concatenate
            h = torch.cat([h, rt], dim=0)
            edge_index = torch.cat([edge_index, fc_edges], dim=1)
            e = torch.cat([e, fc_edge_attr], dim=0)
            v_indices_aug = torch.cat([v_indices, rt_v]) if v_indices is not None else None

        # GATv2 with global re-injection between layers (real nodes only)
        for i, layer in enumerate(self.gatv2_layers):
            h, e = layer(h, edge_index, e)
            if i < len(self.gatv2_layers) - 1:
                if v_indices is not None:
                    h[:N_real] = h[:N_real] + h_g[v_indices]
                else:
                    h[:N_real] = h[:N_real] + h_g[0]

        # --- Separate real nodes and readout tokens ---
        if self.use_readout_tokens:
            h_real = h[:N_real]
            h_rt = h[N_real:]  # (B * n_r, H)
            e_real = e[:edge_attr.shape[0]]  # original edges only

            # Edge readout (real nodes only)
            e_agg = scatter(e_real, edge_index[:, :edge_attr.shape[0]][1],
                           dim=0, dim_size=N_real, reduce='mean')
            h_real = h_real + self.edge_readout(e_agg)

            h_per_head = h_real.view(-1, self.n_heads, self.head_dim)

            # h_global from readout tokens: mean per graph
            h_rt = h_rt.view(B, n_r, self.gatv2_hidden)
            h_global = h_rt.mean(dim=1)  # (B, gatv2_hidden=64)

            return h_real, e_real, h_per_head, h_global

        # Edge readout
        e_agg = scatter(e, edge_index[1], dim=0, dim_size=h.shape[0], reduce='mean')
        h = h + self.edge_readout(e_agg)

        # Per-head view
        h_per_head = h.view(-1, self.n_heads, self.head_dim)

        # Global readout: scatter_mean + scatter_max
        if v_indices is not None:
            g_mean = scatter(h, v_indices, dim=0, dim_size=B, reduce='mean')
            g_max = scatter(h, v_indices, dim=0, dim_size=B, reduce='max')
        else:
            g_mean = h.mean(dim=0, keepdim=True)
            g_max = h.max(dim=0, keepdim=True).values
        h_global = self.global_readout(torch.cat([g_mean, g_max], dim=-1))

        return h, e, h_per_head, h_global

    # ------------------------------------------------------------------
    # Compatibility interface (drop-in for PNAGATv2Backbone)
    # ------------------------------------------------------------------

    def freeze_pna(self):
        self._pna_frozen = True

    def unfreeze_pna(self):
        self._pna_frozen = False

    def override_degree_histogram(self, deg_hist=None):
        pass

    def load_pna_checkpoint(self, checkpoint_path):
        log.warning("GATv2OnlyBackbone has no PNA; ignoring checkpoint %s",
                    checkpoint_path)

    @property
    def pna(self):
        return None

    def get_param_groups(self, lr_proj=3e-4, lr_gatv2=3e-4, **_ignored):
        """Return optimizer param groups."""
        global_params = (
            [self.readout_tokens]
        ) if self.use_readout_tokens else list(self.global_readout.parameters())
        proj_params = (list(self.node_norm.parameters()) +
                       list(self.node_proj.parameters()) +
                       list(self.edge_proj.parameters()) +
                       list(self.global_proj.parameters()) +
                       global_params)
        gatv2_params = (list(p for l in self.gatv2_layers
                             for p in l.parameters()) +
                        list(self.edge_readout.parameters()))
        return [
            {'params': proj_params, 'lr': lr_proj, 'name': 'projections'},
            {'params': gatv2_params, 'lr': lr_gatv2, 'name': 'gatv2'},
        ]

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result
