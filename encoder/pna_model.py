"""
pna_model.py - PNA-based GNN feature extractor with GraphNorm + FiLM conditioning

Drop-in replacement for EncoderCoreDecoder:
  - Same forward signature: forward(x, edge_index, edge_attr, u, v_indices, e_indices)
  - Same return format: (node_emb, edge_emb, global_emb, node_edge_emb)

Uses:
  - PNAConv (multiple aggregators + scalers) for message passing
  - GraphNorm for per-graph normalization within a batch
  - FiLM (Feature-wise Linear Modulation) conditioning from global features u
    at every PNA layer (both node and edge representations)

FiLM conditioning: global features u are projected to (gamma, beta) pairs per layer.
Applied post-residual+dropout for nodes and post-edge-residual for edges.
Uses (1 + gamma) * x + beta so untrained FiLM acts as identity (gamma=0, beta=0).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Sequential, LeakyReLU

from torch_scatter import scatter_mean, scatter_max
from torch_geometric.nn import PNAConv, GraphNorm


# ======================================================================
# PNA Edge Layer: PNAConv + GraphNorm + FiLM + edge update + residuals
# ======================================================================
class PNAEdgeLayer(nn.Module):
    """Single PNA message-passing layer with edge updates and FiLM conditioning."""

    def __init__(self, hidden_dim, edge_dim, aggregators, scalers,
                 deg_histogram, towers=4, dropout=0.1, global_dim=0,
                 film_clamp=3.0):
        super().__init__()
        self.film_clamp = film_clamp

        self.pna_conv = PNAConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg_histogram,
            edge_dim=edge_dim,
            towers=towers,
            pre_layers=1,
            post_layers=1,
        )
        self.graph_norm = GraphNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Edge update MLP: src_node || dst_node || edge_attr -> edge_attr
        self.edge_mlp = Sequential(
            Linear(2 * hidden_dim + edge_dim, edge_dim),
            LeakyReLU(),
            LayerNorm(edge_dim),
        )

        # FiLM conditioning from global features (if global_dim > 0)
        self.has_film = global_dim > 0
        if self.has_film:
            # Node FiLM: u -> (gamma, beta) for node representations
            self.node_film = Sequential(
                Linear(global_dim, hidden_dim),
                LeakyReLU(),
                Linear(hidden_dim, hidden_dim * 2),  # gamma + beta
            )
            # Edge FiLM: u -> (gamma_e, beta_e) for edge representations
            self.edge_film = Sequential(
                Linear(global_dim, hidden_dim),
                LeakyReLU(),
                Linear(hidden_dim, edge_dim * 2),  # gamma_e + beta_e
            )
            # Zero-init last layer so gamma=0, beta=0 at init → true identity
            nn.init.zeros_(self.node_film[-1].weight)
            nn.init.zeros_(self.node_film[-1].bias)
            nn.init.zeros_(self.edge_film[-1].weight)
            nn.init.zeros_(self.edge_film[-1].bias)

    def forward(self, x, edge_index, edge_attr, batch=None, u_graph=None):
        """
        Args:
            x: [N, hidden_dim] node features
            edge_index: [2, E] connectivity
            edge_attr: [E, edge_dim] edge features
            batch: [N] graph membership for GraphNorm
            u_graph: [B, global_dim] global features for FiLM (optional)

        Returns:
            (x_new, edge_attr_new)
        """
        # Node update via PNA + GraphNorm + residual + dropout
        x_new = self.pna_conv(x, edge_index, edge_attr)
        x_new = self.graph_norm(x_new, batch)
        x_new = x_new + x  # residual
        x_new = self.dropout(x_new)

        # Node FiLM: apply AFTER residual + dropout
        if self.has_film and u_graph is not None and batch is not None:
            film_params = self.node_film(u_graph)           # [B, hidden*2]
            gamma, beta = film_params.chunk(2, dim=-1)      # each [B, hidden]
            gamma = gamma.clamp(-self.film_clamp, self.film_clamp)
            beta = beta.clamp(-self.film_clamp, self.film_clamp)
            # Expand from [B, hidden] to [N, hidden] using batch index
            gamma_n = gamma[batch]                           # [N, hidden]
            beta_n = beta[batch]                             # [N, hidden]
            x_new = x_new * (1 + gamma_n) + beta_n          # affine modulation

        # Edge update: concat src, dst, edge_attr -> MLP + residual
        row, col = edge_index[0], edge_index[1]
        e_in = torch.cat([x_new[row], x_new[col], edge_attr], dim=-1)
        edge_attr_new = self.edge_mlp(e_in) + edge_attr  # residual

        # Edge FiLM: apply AFTER edge residual
        if self.has_film and u_graph is not None and batch is not None:
            e_film = self.edge_film(u_graph)                 # [B, edge_dim*2]
            gamma_e, beta_e = e_film.chunk(2, dim=-1)        # each [B, edge_dim]
            gamma_e = gamma_e.clamp(-self.film_clamp, self.film_clamp)
            beta_e = beta_e.clamp(-self.film_clamp, self.film_clamp)
            # Expand to per-edge using source node's graph membership
            e_batch = batch[row]                             # [E]
            gamma_e = gamma_e[e_batch]                       # [E, edge_dim]
            beta_e = beta_e[e_batch]                         # [E, edge_dim]
            edge_attr_new = edge_attr_new * (1 + gamma_e) + beta_e

        return x_new, edge_attr_new


# ======================================================================
# PNA Feature Extractor: drop-in replacement for EncoderCoreDecoder
# ======================================================================
class PNAFeatureExtractor(nn.Module):
    """
    PNA-based GNN feature extractor with FiLM conditioning.

    Drop-in replacement for EncoderCoreDecoder with the same interface:
      forward(x, edge_index, edge_attr, u, v_indices=None, e_indices=None)
      -> (node_emb, edge_emb, global_emb, node_edge_emb)

    Global features u are injected into every PNA layer via FiLM conditioning,
    allowing message-passing to be aware of dimensionality, progress, diversity, etc.
    """

    def __init__(self, node_in=7, edge_in=10, global_in=2,
                 hidden_dim=64, out_dim=32, num_layers=4,
                 towers=4, dropout=0.1, device="cpu",
                 deg_histogram=None, rwse_dim=None, film_clamp=3.0):
        super().__init__()
        # rwse_dim: accepted for backward compatibility but unused (RWSE removed).
        # Deprecated — will be removed in a future version.
        self.hidden_dim = hidden_dim

        # Degree histogram for PNA scalers
        # Caller can pass a domain-specific histogram (e.g. k-NN graph);
        # default matches genealogical graph statistics from MOS.
        if deg_histogram is None:
            deg_histogram = torch.zeros(20, dtype=torch.long)
            deg_histogram[0] = 100   # elite nodes (no incoming edges)
            deg_histogram[2] = 5000  # children (2 parents each)
            deg_histogram[4] = 500   # some nodes with higher connectivity
        self.register_buffer("deg_histogram", deg_histogram)

        # Aggregators and scalers
        self.aggregators = ["mean", "max", "min", "std"]
        self.scalers = ["identity", "amplification", "attenuation"]

        # Input projections (no RWSE — removed, near-constant on k-NN graphs)
        self.node_proj = Sequential(
            Linear(node_in, hidden_dim),
            LeakyReLU(),
            LayerNorm(hidden_dim),
        )
        self.edge_proj = Sequential(
            Linear(edge_in, hidden_dim),
            LeakyReLU(),
            LayerNorm(hidden_dim),
        )

        # PNA layers with FiLM conditioning from global features
        self.layers = nn.ModuleList([
            PNAEdgeLayer(
                hidden_dim=hidden_dim,
                edge_dim=hidden_dim,
                aggregators=self.aggregators,
                scalers=self.scalers,
                deg_histogram=self.deg_histogram,
                towers=towers,
                dropout=dropout,
                global_dim=global_in,
                film_clamp=film_clamp,
            )
            for _ in range(num_layers)
        ])

        # Output projections
        self.node_out = Sequential(
            Linear(hidden_dim, out_dim),
            LeakyReLU(),
            LayerNorm(out_dim),
        )
        self.edge_out = Sequential(
            Linear(hidden_dim, out_dim),
            LeakyReLU(),
            LayerNorm(out_dim),
        )

        # Per-node edge context: aggregate incident edges per node (Fix C)
        self.node_edge_out = Sequential(
            Linear(hidden_dim, out_dim),
            LeakyReLU(),
            LayerNorm(out_dim),
        )

        # Global readout: scatter_mean + scatter_max + edge_mean + global_attr -> MLP
        # Fix A: added hidden_dim for edge pooling (was hidden_dim * 2 + global_in)
        self.global_mlp = Sequential(
            Linear(hidden_dim * 3 + global_in, hidden_dim),
            LeakyReLU(),
            Linear(hidden_dim, out_dim),
            LeakyReLU(),
            LayerNorm(out_dim),
        )

    def _sanitize(self, tensor):
        """Replace NaN/Inf with per-column worst (max finite) value.

        Clones before mutation to avoid in-place modification of input tensors,
        which would crash if the tensor requires_grad or is shared by the caller
        (bug prevention: Round 3 fix W-G3-3).
        """
        if not (torch.isnan(tensor).any() or torch.isinf(tensor).any()):
            return tensor
        # Clone to avoid in-place mutation on caller's tensor or autograd leaf
        tensor = tensor.clone()
        if tensor.dim() >= 2:
            for col in range(tensor.shape[1]):
                col_fin = torch.isfinite(tensor[:, col])
                if not col_fin.all():
                    fv = tensor[col_fin, col]
                    tensor[~col_fin, col] = fv.max() if fv.numel() > 0 else 0.0
        else:
            fin = torch.isfinite(tensor)
            if not fin.all():
                fv = tensor[fin]
                tensor[~fin] = fv.max() if fv.numel() > 0 else 0.0
        return tensor

    def forward(self, x, edge_index, edge_attr, u, v_indices=None, e_indices=None):
        """
        Forward pass with same interface as EncoderCoreDecoder.

        Args:
            x: [N, node_in] node features
            edge_index: [2, E] connectivity (long)
            edge_attr: [E, edge_in] edge features
            u: [B, global_in] global features (FiLM-conditioned into every layer)
            v_indices: [N] node-to-graph mapping (long), None for single graph
            e_indices: [E] edge-to-graph mapping (long), None for single graph

        Returns:
            (node_emb [N, out_dim], edge_emb [E, out_dim],
             global_emb [B, out_dim], node_edge_emb [N, out_dim])
        """
        # Default indices for single graph — handle each independently
        if v_indices is None:
            v_indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        if e_indices is None:
            e_indices = torch.zeros(edge_attr.shape[0], dtype=torch.long, device=x.device)

        # Sanitize inputs
        x = self._sanitize(x)
        edge_attr = self._sanitize(edge_attr)
        u = self._sanitize(u)

        # Project inputs to hidden dimension
        x = self.node_proj(x)           # [N, hidden_dim]
        edge_attr_h = self.edge_proj(edge_attr)  # [E, hidden_dim]

        # PNA message-passing layers with FiLM conditioning
        for layer in self.layers:
            x, edge_attr_h = layer(x, edge_index, edge_attr_h,
                                   batch=v_indices, u_graph=u)

        # Output projections
        node_emb = self.node_out(x)           # [N, out_dim]
        edge_emb = self.edge_out(edge_attr_h)  # [E, out_dim]

        # Per-node edge context: aggregate incident edges to destination nodes (Fix C)
        node_edge_ctx = scatter_mean(
            edge_attr_h, edge_index[1], dim=0, dim_size=x.shape[0])  # [N, hidden_dim]
        node_edge_emb = self.node_edge_out(node_edge_ctx)  # [N, out_dim]

        # Global readout: mean + max pooling over nodes + edge mean + global attr (Fix A)
        num_graphs = u.shape[0]  # B = number of graphs (from global features)
        g_mean = scatter_mean(x, v_indices, dim=0, dim_size=num_graphs)  # [B, hidden_dim]
        g_max = scatter_max(x, v_indices, dim=0, dim_size=num_graphs)[0] # [B, hidden_dim]
        g_max = g_max.clamp(min=-1e9)  # scatter_max fills empty positions with -inf
        e_mean = scatter_mean(edge_attr_h, e_indices, dim=0, dim_size=num_graphs)  # [B, hidden_dim]
        global_in = torch.cat([g_mean, g_max, e_mean, u], dim=-1)    # [B, hidden_dim*3 + global_in]
        global_emb = self.global_mlp(global_in)                        # [B, out_dim]

        # Sanitize outputs only at inference — during training, NaN/Inf should
        # propagate so gradient bugs surface instead of being silently masked.
        if not self.training:
            node_emb = self._sanitize(node_emb)
            edge_emb = self._sanitize(edge_emb)
            global_emb = self._sanitize(global_emb)
            node_edge_emb = self._sanitize(node_edge_emb)

        return node_emb, edge_emb, global_emb, node_edge_emb
