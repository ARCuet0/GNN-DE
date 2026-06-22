"""
dense_temporal_backbone.py — Temporal Attention + Dense GATv2 backbone.

The canonical attn64_W16_bf16 backbone:
  1. TemporalAttentionEncoder: coords_hist (W, N, D) → h_temporal (N, D, d_rnn)
  2. TemporalDimPooler: (N, D, d_rnn) → (N, d_temporal)
  3. cat([node_feat, h_temporal]) → (N, node_in + d_temporal)
  4. DenseGATv2Backbone: → h (B, N, H), h_per_head (B, N, n_heads, hd), h_global

Fully vmap-compatible. No scatter, no .item(), no nonzero().
Drop-in replacement for TemporalGATv2Backbone.
"""
import logging
from typing import Optional

import torch
import torch.nn as nn

from .backbone_compat import BackboneCompatMixin
from .dense_gatv2_backbone import DenseGATv2Backbone, TopologyCache
from .npa_layers import TemporalDimPooler
from .temporal_attention import TemporalAttentionEncoder

log = logging.getLogger(__name__)


class TemporalDenseGATv2Backbone(BackboneCompatMixin, nn.Module):
    """Temporal Attention + Dense GATv2 — the attn64_W16_bf16 backbone.

    Returns 4-tuple: (h, e, h_per_head, h_global).
    """

    def __init__(self, d_rnn=64, d_temporal=64, gru_window=16,
                 node_in=8, edge_in=4, global_in=13,
                 gatv2_hidden=128, gatv2_layers=2, n_heads=8,
                 global_out_dim=128, dropout=0.1,
                 temporal_encoder='attention',
                 device='cpu',
                 # Compatibility kwargs (silently ignored)
                 pna_hidden=None, pna_out=None, pna_layers=None,
                 pna_checkpoint=None, n_readout=None,
                 use_readout_tokens=None, gru_fp16=None,
                 gru_checkpoint=None, **_ignored):
        super().__init__()
        self.d_temporal = d_temporal
        self.gru_window = gru_window
        self.device = device

        # Temporal encoder
        if temporal_encoder == 'attention':
            n_attn_heads = max(1, d_rnn // 16)
            while d_rnn % n_attn_heads != 0:
                n_attn_heads -= 1
            self.temporal = TemporalAttentionEncoder(
                d_model=d_rnn, n_layers=2, n_heads=n_attn_heads,
                dropout=dropout, coord_range=100.0)
            # ^ CEC2017 deployed regime. Override post-construction for BBOB.
        else:
            from .npa_layers import TemporalGRUEncoder
            self.temporal = TemporalGRUEncoder(
                d_model=d_rnn, d_rnn=d_rnn)

        self.pooler = TemporalDimPooler(d_rnn=d_rnn, d_out=d_temporal)

        # Inner dense GATv2 backbone with expanded node_in
        self.backbone = DenseGATv2Backbone(
            node_in=node_in + d_temporal,
            edge_in=edge_in, global_in=global_in,
            gatv2_hidden=gatv2_hidden, gatv2_layers=gatv2_layers,
            n_heads=n_heads, global_out_dim=global_out_dim,
            dropout=dropout,
        )

        # Expose attributes for variant compatibility
        self.n_heads = n_heads
        self.head_dim = gatv2_hidden // n_heads
        self.gatv2_hidden = gatv2_hidden
        self.pna_out = global_out_dim

        total = sum(p.numel() for p in self.parameters())
        temp_p = sum(p.numel() for p in self.temporal.parameters())
        pool_p = sum(p.numel() for p in self.pooler.parameters())
        log.info("TemporalDenseGATv2Backbone: %d params (%d temporal, %d pooler, "
                 "%d dense_gatv2, n_heads=%d, d_temporal=%d)",
                 total, temp_p, pool_p, total - temp_p - pool_p,
                 n_heads, d_temporal)

    def encode(self, node_feat, global_feat, cache: TopologyCache,
               coords_hist=None, fitness_hist=None, n_valid=None,
               **_ignored):
        """
        Args:
            node_feat:    (B, N, node_in) float32
            global_feat:  (B, global_in) float32
            cache:        TopologyCache
            coords_hist:  (W, N, D) float64 — temporal window (optional)
            fitness_hist: (W, N) float32 — temporal window (optional)
            n_valid:      int or tensor — number of valid timesteps

        Returns:
            h, e, h_per_head, h_global — same as DenseGATv2Backbone.encode()
        """
        B, N = cache.B, cache.N

        if coords_hist is not None and n_valid is not None:
            # Pass int to avoid .item() under vmap
            nv = n_valid if isinstance(n_valid, int) else n_valid.item()
            # Cast to float32 — coords_hist may be float64 (CEC2017 precision)
            h_temporal = self.temporal(
                coords_hist.float(), fitness_hist.float(), nv)  # (N, D, d_rnn)
            h_pooled = self.pooler(h_temporal)  # (N, d_temporal)
            # Expand to (B, N, d_temporal)
            h_pooled = h_pooled.unsqueeze(0).expand(B, -1, -1)
        else:
            h_pooled = node_feat.new_zeros(B, N, self.d_temporal)

        # Concatenate temporal features with node features
        node_feat_aug = torch.cat([node_feat, h_pooled], dim=-1)  # (B, N, node_in + d_temporal)

        return self.backbone.encode(node_feat_aug, global_feat, cache)

    def forward(self, node_feat, global_feat, cache, **kwargs):
        """Alias for encode() — required by torch.func.functional_call."""
        return self.encode(node_feat, global_feat, cache, **kwargs)

    def load_legacy_checkpoint(self, ckpt_path):
        """Load weights from ssl_nextstep_attn64_W16_fp16.pth (legacy format).

        Legacy key mapping:
            gru.*              → self.temporal.*
            pooler.*           → self.pooler.*
            _inner.gatv2_layers.{i}.gat.{param} → self.backbone.layers.{i}.{param}
            _inner.{other}     → self.backbone.{other}
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        legacy_sd = ckpt.get('backbone_state_dict', ckpt)

        new_sd = {}
        unmapped = []

        for key, val in legacy_sd.items():
            if key.startswith('gru.'):
                new_key = 'temporal.' + key[len('gru.'):]
            elif key.startswith('pooler.'):
                new_key = key  # same
            elif key.startswith('_inner.gatv2_layers.'):
                # _inner.gatv2_layers.{i}.gat.{param} → backbone.layers.{i}.{param}
                # _inner.gatv2_layers.{i}.{other} → backbone.layers.{i}.{other}
                rest = key[len('_inner.gatv2_layers.'):]
                parts = rest.split('.', 2)  # ['0', 'gat', 'lin_l.weight'] or ['0', 'norm', 'weight']
                layer_idx = parts[0]
                sub = parts[1]
                param_rest = parts[2] if len(parts) > 2 else ''

                if sub == 'gat':
                    # _inner.gatv2_layers.0.gat.lin_l.weight → backbone.layers.0.lin_l.weight
                    new_key = f'backbone.layers.{layer_idx}.{param_rest}'
                else:
                    # _inner.gatv2_layers.0.norm.weight → backbone.layers.0.norm.weight
                    new_key = f'backbone.layers.{layer_idx}.{sub}' + (f'.{param_rest}' if param_rest else '')
            elif key.startswith('_inner.readout_tokens'):
                # Skip readout tokens (DenseGATv2Backbone uses mean+max readout, not tokens)
                unmapped.append(key)
                continue
            elif key.startswith('_inner.'):
                # _inner.node_proj.* → backbone.node_proj.*
                new_key = 'backbone.' + key[len('_inner.'):]
            else:
                unmapped.append(key)
                continue

            # Fix att shape: legacy (1, n_heads, hd) → dense (1, 1, n_heads, hd)
            if new_key.endswith('.att') and val.dim() == 3:
                val = val.unsqueeze(0)

            new_sd[new_key] = val

        missing, unexpected = self.load_state_dict(new_sd, strict=False)

        if unmapped:
            log.info("Legacy checkpoint: %d unmapped keys (readout tokens, etc.): %s",
                     len(unmapped), unmapped[:5])
        if missing:
            log.warning("Missing keys after legacy load: %s", missing[:10])
        if unexpected:
            log.warning("Unexpected keys after legacy load: %s", unexpected[:10])

        log.info("Loaded legacy checkpoint: %d keys mapped, %d missing, %d unexpected",
                 len(new_sd), len(missing), len(unexpected))

    def load_ssl_checkpoint(self, ckpt_path):
        """Load SSL checkpoint — auto-detects legacy vs native format."""
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('backbone_state_dict', ckpt)
        # Detect legacy format by checking key prefixes
        sample_keys = list(sd.keys())[:5]
        is_legacy = any(k.startswith(('gru.', '_inner.')) for k in sample_keys)
        if is_legacy:
            log.info("Detected legacy format, using key remapping")
            return self.load_legacy_checkpoint(ckpt_path)
        # Native format — direct load
        missing, unexpected = self.load_state_dict(sd, strict=False)
        if missing:
            log.warning("load_ssl_checkpoint: missing keys: %s", missing[:10])
        if unexpected:
            log.warning("load_ssl_checkpoint: unexpected keys: %s", unexpected[:10])
        log.info("Loaded SSL checkpoint: %s (%d keys)", ckpt_path, len(sd))

    def get_param_groups(self, lr_temporal=3e-4, lr_proj=3e-4, lr_gatv2=3e-4,
                         **_ignored):
        temporal_params = (list(self.temporal.parameters()) +
                           list(self.pooler.parameters()))
        groups = [{'params': temporal_params, 'lr': lr_temporal, 'name': 'temporal'}]
        groups.extend(self.backbone.get_param_groups(lr_proj=lr_proj, lr_gatv2=lr_gatv2))
        return groups

    # to() inherited from BackboneCompatMixin
