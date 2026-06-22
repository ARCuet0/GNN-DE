from .backbone import PNAGATv2Backbone
from .transformer_backbone import TransformerBackbone
from .pure_transformer_backbone import PureTransformerBackbone


def create_backbone(backbone_type='pna_gatv2', **kwargs):
    """Factory for encoder backbones.

    Args:
        backbone_type: 'pna_gatv2', 'transformer', 'pure_transformer',
                       or 'transformer_ssl'
        **kwargs: forwarded to the backbone constructor

    Returns:
        Backbone instance with encode() → (h, e, h_per_head, h_global)
    """
    if backbone_type == 'pna_gatv2':
        # Map attn_hidden/attn_layers to gatv2_hidden/gatv2_layers if present
        kw = dict(kwargs)
        if 'attn_hidden' in kw:
            kw.setdefault('gatv2_hidden', kw.pop('attn_hidden'))
        if 'attn_layers' in kw:
            kw.setdefault('gatv2_layers', kw.pop('attn_layers'))
        kw.pop('use_edge_bias', None)  # not applicable to GATv2
        return PNAGATv2Backbone(**kw)
    elif backbone_type == 'transformer':
        # Map gatv2_hidden/gatv2_layers to attn_hidden/attn_layers if present
        kw = dict(kwargs)
        if 'gatv2_hidden' in kw:
            kw.setdefault('attn_hidden', kw.pop('gatv2_hidden'))
        if 'gatv2_layers' in kw:
            kw.setdefault('attn_layers', kw.pop('gatv2_layers'))
        return TransformerBackbone(**kw)
    elif backbone_type == 'pure_transformer':
        kw = dict(kwargs)
        if 'gatv2_hidden' in kw:
            kw.setdefault('hidden_dim', kw.pop('gatv2_hidden'))
        if 'gatv2_layers' in kw:
            kw.setdefault('n_layers', kw.pop('gatv2_layers'))
        if 'attn_hidden' in kw:
            kw.setdefault('hidden_dim', kw.pop('attn_hidden'))
        if 'attn_layers' in kw:
            kw.setdefault('n_layers', kw.pop('attn_layers'))
        kw.pop('use_edge_bias', None)
        return PureTransformerBackbone(**kw)
    elif backbone_type == 'transformer_ssl':
        from TRANSFORMER_SSL.backbone import TransformerSSLBackbone
        kw = dict(kwargs)
        # Map L2O aliases to TransformerSSLBackbone params
        # Ignore gatv2_hidden — SSL checkpoint dictates dims
        kw.pop('gatv2_hidden', None)
        kw.pop('attn_hidden', None)
        hidden_dim = kw.pop('hidden_dim', 64)
        # Infer input dims from checkpoint or current constants
        from encoder.similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM
        node_in, edge_in, global_in = NODE_DIM, EDGE_DIM, GLOBAL_DIM
        ckpt_path = kw.get('pna_checkpoint')
        if ckpt_path:
            import torch as _t
            _c = _t.load(ckpt_path, map_location='cpu', weights_only=False)
            _sd = _c.get('encoder_state_dict', _c)
            for key_prefix in ('node_encoder_mlp.0', 'node_encoder.0'):
                if f'{key_prefix}.weight' in _sd:
                    node_in = _sd[f'{key_prefix}.weight'].shape[1]
                    hidden_dim = _sd[f'{key_prefix}.weight'].shape[0]
                    break
            for key_prefix in ('edge_encoder_mlp.0', 'edge_encoder.0'):
                if f'{key_prefix}.weight' in _sd:
                    edge_in = _sd[f'{key_prefix}.weight'].shape[1]
                    break
            for key_prefix in ('global_encoder_mlp.0', 'global_encoder.0'):
                if f'{key_prefix}.weight' in _sd:
                    global_in = _sd[f'{key_prefix}.weight'].shape[1]
                    break
            # Infer gate_rank from checkpoint
            if 'layers.0.gate_proj.0.weight' in _sd:
                _gate_rank_ckpt = _sd['layers.0.gate_proj.0.weight'].shape[0]
            else:
                _gate_rank_ckpt = None
            del _c
        else:
            _gate_rank_ckpt = None
        n_layers = kw.pop('gatv2_layers', None) or kw.pop('attn_layers', None) or kw.pop('n_layers', 4)
        n_heads = kw.pop('n_heads', 4)
        gate_rank = kw.pop('gate_rank', 0) or _gate_rank_ckpt or 16
        dropout = kw.pop('dropout', 0.1)
        ckpt_path = kw.pop('pna_checkpoint', None)
        device = kw.pop('device', 'cpu')
        # Discard unused PNA/GATv2 kwargs
        for drop in ('pna_out', 'pna_layers', 'pna_hidden', 'n_heads_pna',
                      'use_edge_bias', 'global_out_dim'):
            kw.pop(drop, None)
        backbone = TransformerSSLBackbone(
            node_in=node_in, edge_in=edge_in, global_in=global_in,
            hidden_dim=hidden_dim, n_layers=n_layers,
            n_heads=n_heads, dropout=dropout, gate_rank=gate_rank,
        )
        if ckpt_path:
            import torch as _torch
            ckpt = _torch.load(ckpt_path, map_location=device, weights_only=False)
            sd = ckpt.get('encoder_state_dict', ckpt)
            result = backbone.load_state_dict(sd, strict=False)
            if result.missing_keys:
                import logging
                logging.getLogger(__name__).warning(
                    "TransformerSSL missing keys: %s", result.missing_keys)
        # Expose attributes expected by BudgetMOSRouter
        backbone.pna = None
        backbone._pna_frozen = False
        return backbone
    elif backbone_type == 'temporal_pna_gatv2':
        from .temporal_backbone import TemporalPNAGATv2Backbone
        kw = dict(kwargs)
        if 'attn_hidden' in kw:
            kw.setdefault('gatv2_hidden', kw.pop('attn_hidden'))
        if 'attn_layers' in kw:
            kw.setdefault('gatv2_layers', kw.pop('attn_layers'))
        kw.pop('use_edge_bias', None)
        return TemporalPNAGATv2Backbone(**kw)
    elif backbone_type == 'gatv2_only':
        from .gatv2_backbone import GATv2OnlyBackbone
        kw = dict(kwargs)
        if 'attn_hidden' in kw:
            kw.setdefault('gatv2_hidden', kw.pop('attn_hidden'))
        if 'attn_layers' in kw:
            kw.setdefault('gatv2_layers', kw.pop('attn_layers'))
        kw.pop('use_edge_bias', None)
        return GATv2OnlyBackbone(**kw)
    elif backbone_type == 'temporal_gatv2':
        from .temporal_gatv2_backbone import TemporalGATv2Backbone
        kw = dict(kwargs)
        if 'attn_hidden' in kw:
            kw.setdefault('gatv2_hidden', kw.pop('attn_hidden'))
        if 'attn_layers' in kw:
            kw.setdefault('gatv2_layers', kw.pop('attn_layers'))
        kw.pop('use_edge_bias', None)
        return TemporalGATv2Backbone(**kw)
    elif backbone_type == 'temporal_dense_gatv2':
        from .dense_temporal_backbone import TemporalDenseGATv2Backbone
        kw = dict(kwargs)
        if 'attn_hidden' in kw:
            kw.setdefault('gatv2_hidden', kw.pop('attn_hidden'))
        if 'attn_layers' in kw:
            kw.setdefault('gatv2_layers', kw.pop('attn_layers'))
        kw.pop('use_edge_bias', None)
        return TemporalDenseGATv2Backbone(**kw)
    elif backbone_type == 'dense_gatv2':
        from .dense_gatv2_backbone import DenseGATv2Backbone
        kw = dict(kwargs)
        if 'attn_hidden' in kw:
            kw.setdefault('gatv2_hidden', kw.pop('attn_hidden'))
        if 'attn_layers' in kw:
            kw.setdefault('gatv2_layers', kw.pop('attn_layers'))
        kw.pop('use_edge_bias', None)
        return DenseGATv2Backbone(**kw)
    elif backbone_type == 'npa':
        from .npa_backbone import NPABackbone
        kw = dict(kwargs)
        if 'gatv2_hidden' in kw:
            kw.setdefault('hidden_dim', kw.pop('gatv2_hidden'))
        if 'attn_hidden' in kw:
            kw.setdefault('hidden_dim', kw.pop('attn_hidden'))
        for drop in ('pna_out', 'pna_layers', 'pna_hidden', 'n_heads_pna',
                      'use_edge_bias', 'gatv2_layers', 'attn_layers'):
            kw.pop(drop, None)
        return NPABackbone(**kw)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type!r}. "
                         f"Expected 'pna_gatv2', 'temporal_pna_gatv2', "
                         f"'gatv2_only', 'temporal_gatv2', 'temporal_dense_gatv2', 'dense_gatv2', "
                         f"'transformer', 'pure_transformer', "
                         f"'transformer_ssl', or 'npa'.")
