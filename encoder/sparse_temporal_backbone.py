"""
sparse_temporal_backbone.py — Temporal Attention + Sparse GATv2 backbone.

Same structure as dense_temporal_backbone.py but with O(N·k) sparse attention.
Supports 3 topology modes: coordinate kNN, embedding kNN, learned scorer.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .backbone_compat import BackboneCompatMixin
from .sparse_gatv2_backbone import SparseGATv2Backbone, SparseTopologyCache, TopologyMode
from .topology_strategies import build_topology
from .npa_layers import InducedPointPooler, TemporalDimPooler
from .temporal_attention import TemporalAttentionEncoder

log = logging.getLogger(__name__)


class TemporalSparseGATv2Backbone(BackboneCompatMixin, nn.Module):
    """Temporal Attention + Sparse GATv2 backbone.

    Three topology modes:
        COORDINATE_KNN:  kNN in coordinate space (default, like dense but sparse)
        EMBEDDING_KNN:   kNN in temporal embedding space (zero new params)
        LEARNED_SCORER:  q/k projections → attention-based topology

    Returns 4-tuple: (h, e, h_per_head, h_global).
    """

    def __init__(self, d_rnn=64, d_temporal=64, gru_window=16,
                 node_in=8, edge_in=4, global_in=16,
                 gatv2_hidden=128, gatv2_layers=2, n_heads=8,
                 global_out_dim=128, dropout=0.1,
                 temporal_encoder='attention',
                 temporal_layers=2,
                 topology_mode=TopologyMode.COORDINATE_KNN,
                 k_neighbors=8,
                 pooler_type='induced',
                 device='cpu',
                 # D1000 line: pass-through for inner SparseGATv2Backbone.
                 donor_kind: str = 'all2all',
                 donor_pbest_frac: float = 0.1,
                 donor_chunk_size: int | None = None,
                 # Compatibility kwargs
                 **_ignored):
        super().__init__()
        self.d_temporal = d_temporal
        self.gru_window = gru_window
        self.device = device
        self.topology_mode = topology_mode
        self.k_neighbors = k_neighbors
        self.use_checkpoint = False

        # Temporal encoder (identical to dense version)
        if temporal_encoder == 'attention':
            n_attn_heads = max(1, d_rnn // 16)
            while d_rnn % n_attn_heads != 0:
                n_attn_heads -= 1
            self.temporal = TemporalAttentionEncoder(
                d_model=d_rnn, n_layers=temporal_layers, n_heads=n_attn_heads,
                dropout=dropout, coord_range=100.0)
            # ^ CEC2017 deployed regime. BBOB / LSGO callers must override
            # backbone.temporal.coord_range post-construction (see
            # eval_bbob_smoke.py).
        else:
            from .npa_layers import TemporalGRUEncoder
            self.temporal = TemporalGRUEncoder(
                d_model=d_rnn, d_rnn=d_rnn)

        if pooler_type == 'induced':
            self.pooler = InducedPointPooler(d_rnn=d_rnn, d_out=d_temporal)
        else:
            self.pooler = TemporalDimPooler(d_rnn=d_rnn, d_out=d_temporal)

        # Inner sparse GATv2 backbone
        self.backbone = SparseGATv2Backbone(
            node_in=node_in + d_temporal,
            edge_in=edge_in, global_in=global_in,
            gatv2_hidden=gatv2_hidden, gatv2_layers=gatv2_layers,
            n_heads=n_heads, global_out_dim=global_out_dim,
            dropout=dropout,
            donor_kind=donor_kind,
            donor_pbest_frac=donor_pbest_frac,
            donor_chunk_size=donor_chunk_size,
        )

        # Topology strategy
        if topology_mode == TopologyMode.LEARNED_SCORER:
            d_in = gatv2_hidden  # uses projected node features
            self.topology = build_topology(topology_mode, k=k_neighbors,
                                           d_in=d_in, d_k=16)
        else:
            self.topology = build_topology(topology_mode, k=k_neighbors)

        # Expose attributes for variant compatibility
        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.pna_out = global_out_dim

        total = sum(p.numel() for p in self.parameters())
        temp_p = sum(p.numel() for p in self.temporal.parameters())
        pool_p = sum(p.numel() for p in self.pooler.parameters())
        topo_p = sum(p.numel() for p in self.topology.parameters())
        log.info("TemporalSparseGATv2Backbone: %d params (%d temporal, %d pooler, "
                 "%d sparse_gatv2, %d topology, mode=%s, k=%d)",
                 total, temp_p, pool_p, total - temp_p - pool_p - topo_p,
                 topo_p, topology_mode.value, k_neighbors)

    def encode(self, node_feat, global_feat, cache,
               coords_hist=None, fitness_hist=None, n_valid=None,
               coords=None, n_active: Optional[int] = None,
               donor_mask: Optional[torch.Tensor] = None, **_ignored):
        """
        Args:
            node_feat:    (B, N, node_in) float32
            global_feat:  (B, global_in) float32
            cache:        SparseTopologyCache (or will build from coords)
            coords_hist:  (W, N, D) — temporal window
            fitness_hist: (W, N) — temporal window
            n_valid:      int — number of valid timesteps
            coords:       (B, N, D) — raw coordinates (for COORDINATE_KNN
                          when cache doesn't have precomputed knn_idx)
            n_active:     Optional[int]. Forwarded to inner SparseGATv2Backbone
                          to restrict the global readout to h[:, :n_active]
                          (graph-native archive D6). None = unchanged baseline.

        Returns:
            h, e, h_per_head, h_global
        """
        B = node_feat.shape[0]
        N = node_feat.shape[1]

        # Temporal encoding
        if coords_hist is not None and n_valid is not None:
            nv = n_valid if isinstance(n_valid, int) else n_valid.item()
            if coords_hist.dim() == 4:
                # Batched: (B, W, N, D) — per-batch temporal encoding
                B_t, W_t, N_t, D_t = coords_hist.shape
                # Pre-normalize fitness per-batch to [0,1] before flatten,
                # preventing cross-batch leakage in the temporal encoder's
                # global min/max normalization.
                fh = fitness_hist.float()  # (B, W, N)
                # Finite guard only: per-batch min-max below is sign-agnostic, so
                # non-positive (BBOB) fitness must NOT be floored to 1e-30 (that
                # collapses it). See finding_bbob_fitness_blindness_clamp_2026_06_05.
                _fmax = torch.finfo(fh.dtype).max
                fh_safe = fh.clamp(min=-_fmax, max=_fmax)
                f_min = fh_safe.amin(dim=(1, 2), keepdim=True)  # (B, 1, 1)
                f_range = (fh_safe.amax(dim=(1, 2), keepdim=True) - f_min).clamp(min=1e-30)
                fh_norm = (fh_safe - f_min) / f_range  # (B, W, N) in [0, 1]
                # Batched temporal: flatten B batches into one forward pass.
                # Each batch has N×D sequences of length W — total B×N×D sequences.
                # No vmap needed: standard batched transformer + gradient flows.
                # eval() disables dropout intentionally: batched path needs
                # deterministic output for BPTT gradient checkpoint recomputation.
                # Dropout regularization is NOT active in this path by design.
                was_training_t = self.temporal.training
                was_training_p = self.pooler.training
                self.temporal.eval()
                self.pooler.eval()
                try:
                    with torch.amp.autocast('cuda', enabled=False):
                        # (B, W, N, D) → (W, B*N, D) for temporal encoder
                        ch_flat = coords_hist.float().permute(1, 0, 2, 3).reshape(W_t, B_t * N_t, D_t)
                        fh_flat = fh_norm.permute(1, 0, 2).reshape(W_t, B_t * N_t)
                        # temporal: (W, B*N, D) → (B*N, D, d_model)
                        h_temporal = self.temporal(ch_flat, fh_flat, nv, N_t * B_t, D_t)
                        # pooler: (B*N, D, d_model) → (B*N, d_out)
                        h_flat = self.pooler(h_temporal)
                        # Reshape back: (B*N, d_out) → (B, N, d_out)
                        h_pooled = h_flat.view(B_t, N_t, -1)
                finally:
                    self.temporal.train(was_training_t)
                    self.pooler.train(was_training_p)
            else:
                # Legacy: (W, N, D) → single batch, broadcast
                h_temporal = self.temporal(
                    coords_hist.float(), fitness_hist.float(), nv)
                h_pooled = self.pooler(h_temporal)
                h_pooled = h_pooled.unsqueeze(0).expand(B, -1, -1)
        else:
            h_pooled = node_feat.new_zeros(B, N, self.d_temporal)

        # Augmented node features
        node_feat_aug = torch.cat([node_feat, h_pooled], dim=-1)

        # If cache is already a SparseTopologyCache, use it directly
        if isinstance(cache, SparseTopologyCache):
            if self.use_checkpoint and torch.is_grad_enabled():
                def _backbone_fwd(nf, gf):
                    return self.backbone.encode(nf, gf, cache,
                                                 n_active=n_active,
                                                 donor_mask=donor_mask)
                out = grad_checkpoint(
                    _backbone_fwd, node_feat_aug, global_feat,
                    use_reentrant=False)
                return out._replace(h_pooled=h_pooled)
            out = self.backbone.encode(node_feat_aug, global_feat, cache,
                                        n_active=n_active,
                                        donor_mask=donor_mask)
            return out._replace(h_pooled=h_pooled)

        # Otherwise, build sparse cache using topology strategy
        if self.topology_mode == TopologyMode.COORDINATE_KNN:
            assert coords is not None, "COORDINATE_KNN requires coords argument"
            knn_idx = self.topology(coords.float())
        elif self.topology_mode == TopologyMode.EMBEDDING_KNN:
            knn_idx = self.topology(h_pooled.detach())
        elif self.topology_mode == TopologyMode.LEARNED_SCORER:
            h_proj = self.backbone.node_proj(self.backbone.node_norm(node_feat_aug))
            knn_idx = self.topology(h_proj.detach())

        # Build sparse edge features
        from .similarity_graph_gpu import build_sparse_graphs_gpu
        sparse_cache = build_sparse_graphs_gpu(
            coords.float() if coords is not None else node_feat.new_zeros(B, N, 1),
            node_feat.new_ones(B, N),  # placeholder fitness
            step_num=0, max_steps=1, ndim=1,
            k_neighbors=self.k_neighbors,
            knn_idx=knn_idx,
        )
        # Override with provided features (including edge_feat to avoid
        # placeholder fitness producing zero fitness_diff/rank_diff)
        sparse_cache.node_feat = cache.node_feat if hasattr(cache, 'node_feat') and cache.node_feat is not None else sparse_cache.node_feat
        sparse_cache.global_feat = cache.global_feat if hasattr(cache, 'global_feat') and cache.global_feat is not None else sparse_cache.global_feat
        sparse_cache.edge_feat = cache.edge_feat if hasattr(cache, 'edge_feat') and cache.edge_feat is not None else sparse_cache.edge_feat

        out = self.backbone.encode(node_feat_aug, global_feat, sparse_cache,
                                    n_active=n_active,
                                    donor_mask=donor_mask)
        return out._replace(h_pooled=h_pooled)

    def forward(self, node_feat, global_feat, cache, **kwargs):
        """Alias for encode() — required by torch.func.functional_call."""
        return self.encode(node_feat, global_feat, cache, **kwargs)
