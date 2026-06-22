"""
npa_backbone.py — Neural Population Analyzer backbone.

Three-level fully neural architecture replacing handcrafted graph features:
  Level 1: GRU temporal per dimension (shared weights)
  Level 2: Transformer cross-dimension (per individual, FiLM-conditioned)
  Level 3: Transformer cross-individual (permutation equivariant, distance bias)

Drop-in replacement for PNAGATv2Backbone / PureTransformerBackbone.
Returns the same 4-tuple: (h, e, h_per_head, h_global).
"""

import logging
import math

import torch
import torch.nn as nn

from .npa_layers import (
    TemporalGRUEncoder,
    TwoStageAttention,
    ThreeStageAttention,
    _CrossAttnBlock,
    PopulationTransformer,
    IndividualFeatureInjector,
)

log = logging.getLogger(__name__)


class NPABackbone(nn.Module):
    """Neural Population Analyzer — fully neural population encoder.

    Replaces handcrafted graph features with a 3-level hierarchy that
    operates on raw coordinates and fitness values.
    """

    def __init__(
        self,
        # NPA-specific
        window: int = 8,
        d_model: int = 32,
        d_rnn: int = 32,
        d_ind: int = 64,
        hidden_dim: int = 64,
        global_out_dim: int = 32,
        n_heads: int = 4,
        level2_layers: int = 2,
        level3_layers: int = 2,
        max_D: int = 100,
        dropout: float = 0.1,
        device: str = 'cpu',
        # Compatibility kwargs (silently ignored)
        node_in=None, edge_in=None, global_in=None,
        pna_hidden=None, pna_out=None, pna_layers=None,
        pna_checkpoint=None, attn_hidden=None, attn_layers=None,
        gatv2_hidden=None, gatv2_layers=None, use_edge_bias=None,
        **_ignored,
    ):
        super().__init__()

        # Map compatibility aliases
        hidden_dim = attn_hidden or gatv2_hidden or hidden_dim
        global_out_dim = pna_out or global_out_dim

        self.window = window
        self.hidden_dim = hidden_dim
        self.global_out_dim = global_out_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.d_rnn = d_rnn
        self.d_ind = d_ind
        self.device = device

        # Sentinels for compatibility
        self.pna = None
        self._pna_frozen = False
        self._is_npa = True

        # Three-stage factored attention (replaces GRU + TwoStageAttention)
        self.grid_attn = ThreeStageAttention(
            n_feat=7, d_model=d_rnn, d_global_out=hidden_dim,
            n_layers=level2_layers, n_heads=n_heads,
            max_W=window, max_D=max_D, dropout=dropout)

        # Legacy attributes (kept for checkpoint compatibility probing)
        self.temporal_gru = None
        self.cross_dim = None

        # Between Level 2 and readout: per-individual feature injection (post-pool)
        self.feature_injector = IndividualFeatureInjector(hidden_dim)

        # Attention pool: learnable query attends to D tokens per individual
        self.pool_query = nn.Parameter(torch.randn(1, d_rnn) * 0.02)
        self.pool_attn = _CrossAttnBlock(d_rnn, n_heads, dropout)
        self.pool_proj = nn.Linear(d_rnn, hidden_dim)
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, global_out_dim)

        # Keep for backward compat (loading old checkpoints)
        self.pop_transformer = None

        # Fallback path (gen 0, no history): per-dim projection
        self.fallback_dim_proj = nn.Linear(2, d_rnn)
        self.fallback_ind_proj = nn.Linear(d_rnn, d_ind)

        if pna_checkpoint is not None:
            log.warning("NPABackbone has no PNA; ignoring checkpoint %s",
                        pna_checkpoint)

        self._init_weights(level2_layers, level3_layers)

        total = sum(p.numel() for p in self.parameters())
        log.info("NPABackbone: %d params (window=%d, d_rnn=%d, d_ind=%d, "
                 "hidden=%d, heads=%d, L2=%d, L3=%d)",
                 total, window, d_rnn, d_ind, hidden_dim, n_heads,
                 level2_layers, level3_layers)

    # ------------------------------------------------------------------
    def _init_weights(self, n_cross_layers, n_pop_layers):
        """GPT-2 style scaled init for ThreeStageAttention."""

        # ── ThreeStageAttention: input projection ──
        ga = self.grid_attn
        nn.init.xavier_uniform_(ga.input_proj.weight)
        nn.init.zeros_(ga.input_proj.bias)

        # ── Attention layers: scaled residual init ──
        cross_scale = 1.0 / math.sqrt(3 * n_cross_layers)  # 3 stages per layer
        # Self-attention blocks (dim + ind layers)
        for layer in list(ga.dim_layers) + list(ga.ind_layers):
            nn.init.xavier_uniform_(layer.W_qkv.weight)
            nn.init.zeros_(layer.W_qkv.bias)
            nn.init.xavier_uniform_(layer.out_proj.weight)
            layer.out_proj.weight.data.mul_(cross_scale)
            nn.init.zeros_(layer.out_proj.bias)
            nn.init.xavier_uniform_(layer.ffn[0].weight)
            nn.init.zeros_(layer.ffn[0].bias)
            nn.init.xavier_uniform_(layer.ffn[3].weight)
            layer.ffn[3].weight.data.mul_(cross_scale)
            nn.init.zeros_(layer.ffn[3].bias)
        # Cross-attention blocks
        for layer in ga.cross_layers:
            nn.init.xavier_uniform_(layer.W_q.weight)
            nn.init.zeros_(layer.W_q.bias)
            nn.init.xavier_uniform_(layer.W_kv.weight)
            nn.init.zeros_(layer.W_kv.bias)
            nn.init.xavier_uniform_(layer.out_proj.weight)
            layer.out_proj.weight.data.mul_(cross_scale)
            nn.init.zeros_(layer.out_proj.bias)
            nn.init.xavier_uniform_(layer.ffn[0].weight)
            nn.init.zeros_(layer.ffn[0].bias)
            nn.init.xavier_uniform_(layer.ffn[3].weight)
            layer.ffn[3].weight.data.mul_(cross_scale)
            nn.init.zeros_(layer.ffn[3].bias)

        # ── Readout projection ──
        nn.init.xavier_uniform_(ga.readout_proj.weight)
        nn.init.zeros_(ga.readout_proj.bias)

        # ── Pool attention + projection: Xavier ──
        pa = self.pool_attn
        nn.init.xavier_uniform_(pa.W_q.weight)
        nn.init.zeros_(pa.W_q.bias)
        nn.init.xavier_uniform_(pa.W_kv.weight)
        nn.init.zeros_(pa.W_kv.bias)
        nn.init.xavier_uniform_(pa.out_proj.weight)
        nn.init.zeros_(pa.out_proj.bias)
        nn.init.xavier_uniform_(pa.ffn[0].weight)
        nn.init.zeros_(pa.ffn[0].bias)
        nn.init.xavier_uniform_(pa.ffn[3].weight)
        nn.init.zeros_(pa.ffn[3].bias)
        nn.init.xavier_uniform_(self.pool_proj.weight)
        nn.init.zeros_(self.pool_proj.bias)

        # ── Feature injector: Xavier for fuse ──
        nn.init.xavier_uniform_(self.feature_injector.fuse.weight)
        nn.init.zeros_(self.feature_injector.fuse.bias)

    # ------------------------------------------------------------------
    # Encode — main interface
    # ------------------------------------------------------------------

    def encode(self, node_feat, edge_index, edge_attr, global_feat,
               v_indices=None, e_indices=None,
               # NPA-specific kwargs (passed via **encode_kwargs)
               coords_hist=None, fitness_hist=None,
               valid_mask=None, n_valid=None,
               coords_current=None, fitness_current=None,
               f_init=None, fes_ratio=None, **_ignored):
        """Encode population state.

        When coords_hist is provided, uses the 3-level neural path.
        Otherwise falls back to a simple per-dim projection (gen 0).

        Returns:
            h:          (N, hidden_dim)      per-individual embedding
            e:          None                 (no sparse edges in NPA)
            h_per_head: (N, n_heads, head_dim)
            h_global:   (B, global_out_dim)
        """
        if coords_hist is not None:
            return self._encode_neural(
                coords_hist, fitness_hist,
                valid_mask, n_valid,
                coords_current, fitness_current,
                f_init, v_indices, fes_ratio=fes_ratio)
        else:
            return self._encode_fallback(
                coords_current, fitness_current, v_indices)

    def _pool_and_project(self, h_grid, coords_current, fitness_current,
                          v_indices):
        """Pool D→N, project d_rnn→hidden_dim, global readout, inject features.

        Args:
            h_grid: (N, D, d_rnn)
        Returns:
            h, None, h_per_head, h_global
        """
        # Pool D tokens per individual
        h = h_grid.mean(dim=1)                                    # (N, d_rnn)
        h = self.pool_norm(self.pool_proj(h))                     # (N, hidden_dim)

        # Global readout
        if v_indices is not None:
            B = v_indices[-1].item() + 1
            h_global = torch.zeros(B, self.hidden_dim,
                                   device=h.device, dtype=h.dtype)
            h_global.scatter_reduce_(
                0, v_indices.unsqueeze(-1).expand_as(h),
                h, reduce='mean', include_self=False)
        else:
            h_global = h.mean(dim=0, keepdim=True)
        h_global = self.global_proj(h_global)                     # (B, global_out_dim)

        # Per-individual features
        h = self.feature_injector(
            h, coords_current, fitness_current, v_indices)

        h_per_head = h.view(h.shape[0], self.n_heads, self.head_dim)
        return h, None, h_per_head, h_global

    def _encode_neural(self, coords_hist, fitness_hist,
                       valid_mask, n_valid,
                       coords_current, fitness_current,
                       f_init, v_indices, fes_ratio=None):
        """Three-stage factored attention → pool → project."""
        n_pop = None
        if v_indices is not None:
            B = v_indices[-1].item() + 1
            if B > 1:
                n_pop = coords_current.shape[0] // B

        h_grid, h_global_raw = self.grid_attn(
            coords_hist, fitness_hist, n_valid,
            fitness_current, fes_ratio, n_pop=n_pop)

        h_global = self.global_proj(h_global_raw)       # (B, global_out_dim)

        # Per-individual: attention pool over D, project, inject features
        N_total = h_grid.shape[0]
        q = self.pool_query.unsqueeze(0).expand(N_total, -1, -1)  # (N, 1, d_rnn)
        h = self.pool_attn(q, h_grid).squeeze(1)       # (N, d_rnn)
        h = self.pool_norm(self.pool_proj(h))           # (N, hidden_dim)
        h = self.feature_injector(
            h, coords_current, fitness_current, v_indices)
        h_per_head = h.view(h.shape[0], self.n_heads, self.head_dim)
        return h, None, h_per_head, h_global

    def _encode_fallback(self, coords_current, fitness_current, v_indices):
        """Fallback for generation 0: single-timestep dummy buffer → grid_attn."""
        N, D = coords_current.shape
        # Construct (1, N, D) coords_hist and (1, N) fitness_hist
        coords_hist = coords_current.unsqueeze(0)
        fitness_hist = fitness_current.unsqueeze(0)
        n_valid = torch.tensor(1, device=coords_current.device, dtype=torch.long)
        return self._encode_neural(
            coords_hist, fitness_hist, None, n_valid,
            coords_current, fitness_current, None, v_indices)

    # ------------------------------------------------------------------
    # Convenience: skip dummy graph tensors
    # ------------------------------------------------------------------

    def encode_npa(self, coords_hist, fitness_hist, valid_mask, n_valid,
                   coords_current, fitness_current, f_init, v_indices=None):
        """Direct NPA encode without dummy graph tensors."""
        N = coords_current.shape[0]
        d = coords_current.device
        return self.encode(
            torch.zeros(N, 1, device=d),
            torch.zeros(2, 0, device=d, dtype=torch.long),
            torch.zeros(0, 1, device=d),
            torch.zeros(1, 1, device=d),
            v_indices=v_indices,
            coords_hist=coords_hist,
            fitness_hist=fitness_hist,
            valid_mask=valid_mask,
            n_valid=n_valid,
            coords_current=coords_current,
            fitness_current=fitness_current,
            f_init=f_init,
        )

    # ------------------------------------------------------------------
    # Compatibility interface (drop-in for PNAGATv2Backbone)
    # ------------------------------------------------------------------

    @property
    def gatv2_layers(self):
        """Compatibility alias (empty — no PopTransformer)."""
        return nn.ModuleList()

    def set_phase(self, phase):
        """No-op: all params always trainable."""
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
        log.warning("NPABackbone has no PNA; ignoring checkpoint %s",
                    checkpoint_path)

    def get_param_groups(self, lr=3e-4, **kwargs):
        """Return optimizer param groups split by level."""
        lr_gru = kwargs.get('lr_pna', lr)       # reuse lr_pna for GRU
        lr_dim = kwargs.get('lr_bridges', lr)    # reuse lr_bridges for cross-dim
        lr_pop = kwargs.get('lr_gatv2', lr)      # reuse lr_gatv2 for pop-transformer

        return [
            {'params': list(self.temporal_gru.parameters()),
             'lr': lr_gru, 'name': 'npa_gru'},
            {'params': (list(self.cross_dim.parameters()) +
                        list(self.fallback_dim_proj.parameters()) +
                        list(self.fallback_ind_proj.parameters())),
             'lr': lr_dim, 'name': 'npa_cross_dim'},
            {'params': (list(self.pool_proj.parameters()) +
                        list(self.pool_norm.parameters()) +
                        list(self.global_proj.parameters())),
             'lr': lr_pop, 'name': 'npa_readout'},
        ]

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result
