"""Model construction for L2O training."""
import logging
import os

import torch

log = logging.getLogger(__name__)


def build_backbone(args, device):
    from encoder.sparse_temporal_backbone import TemporalSparseGATv2Backbone
    from encoder.sparse_gatv2_backbone import TopologyMode

    topo_map = {
        'embedding_knn': TopologyMode.EMBEDDING_KNN,
        'coordinate_knn': TopologyMode.COORDINATE_KNN,
        'learned_scorer': TopologyMode.LEARNED_SCORER,
        'torch_knn': TopologyMode.TORCH_KNN,
    }

    raw_chunk = int(getattr(args, 'donor_chunk_size', 0))
    chunk_size = raw_chunk if raw_chunk > 0 else None  # 0 or negative → None

    # Dispatch on --backbone-type. set_attention is the 2026-05-21 graph-vs-set
    # ablation; set_attention_edge (B2) and identity (C) are the new arms of
    # the 2026-05-29 RLDE-AFL-style topology/edges ladder. Default keeps the
    # deployed sparse GATv2 behavior unchanged.
    backbone_type = getattr(args, 'backbone_type', 'sparse_gatv2')
    if backbone_type == 'set_attention':
        from encoder.set_attention_backbone import TemporalSetAttentionBackbone
        return TemporalSetAttentionBackbone(
            d_rnn=args.d_rnn, d_temporal=args.d_rnn,
            gru_window=args.gru_window,
            node_in=8, edge_in=4, global_in=16,
            gatv2_hidden=args.gatv2_hidden,
            gatv2_layers=args.gatv2_layers,
            n_heads=args.n_heads,
            global_out_dim=args.gatv2_hidden,
            dropout=args.dropout,
            temporal_encoder='attention',
            temporal_layers=args.temporal_layers,
            topology_mode=topo_map[args.topology],
            k_neighbors=args.k_neighbors,
            pooler_type=args.pooler,
            n_induced=args.n_induced,
            device=device,
            donor_kind=getattr(args, 'donor_kind', 'all2all'),
            donor_pbest_frac=getattr(args, 'donor_pbest_frac', 0.1),
            donor_chunk_size=chunk_size,
        ).to(device)

    if backbone_type == 'set_attention_edge':
        from encoder.set_attention_edge_backbone import (
            TemporalSetAttentionEdgeBackbone)
        return TemporalSetAttentionEdgeBackbone(
            d_rnn=args.d_rnn, d_temporal=args.d_rnn,
            gru_window=args.gru_window,
            node_in=8, edge_in=4, global_in=16,
            gatv2_hidden=args.gatv2_hidden,
            gatv2_layers=args.gatv2_layers,
            n_heads=args.n_heads,
            global_out_dim=args.gatv2_hidden,
            dropout=args.dropout,
            temporal_encoder='attention',
            temporal_layers=args.temporal_layers,
            topology_mode=topo_map[args.topology],
            k_neighbors=args.k_neighbors,
            pooler_type=args.pooler,
            device=device,
            donor_kind=getattr(args, 'donor_kind', 'all2all'),
            donor_pbest_frac=getattr(args, 'donor_pbest_frac', 0.1),
            donor_chunk_size=chunk_size,
            lb=args.lb, ub=args.ub,
        ).to(device)

    if backbone_type == 'identity':
        from encoder.identity_backbone import TemporalIdentityBackbone
        return TemporalIdentityBackbone(
            d_rnn=args.d_rnn, d_temporal=args.d_rnn,
            gru_window=args.gru_window,
            node_in=8, edge_in=4, global_in=16,
            gatv2_hidden=args.gatv2_hidden,
            gatv2_layers=args.gatv2_layers,
            n_heads=args.n_heads,
            global_out_dim=args.gatv2_hidden,
            dropout=args.dropout,
            temporal_encoder='attention',
            temporal_layers=args.temporal_layers,
            topology_mode=topo_map[args.topology],
            k_neighbors=args.k_neighbors,
            pooler_type=args.pooler,
            device=device,
            donor_kind=getattr(args, 'donor_kind', 'all2all'),
            donor_pbest_frac=getattr(args, 'donor_pbest_frac', 0.1),
            donor_chunk_size=chunk_size,
        ).to(device)

    return TemporalSparseGATv2Backbone(
        d_rnn=args.d_rnn, d_temporal=args.d_rnn,
        gru_window=args.gru_window,
        node_in=8, edge_in=4, global_in=16,
        gatv2_hidden=args.gatv2_hidden,
        gatv2_layers=args.gatv2_layers,
        n_heads=args.n_heads,
        global_out_dim=args.gatv2_hidden,
        dropout=args.dropout,
        temporal_encoder='attention',
        temporal_layers=args.temporal_layers,
        topology_mode=topo_map[args.topology],
        k_neighbors=args.k_neighbors,
        pooler_type=args.pooler,
        n_induced=args.n_induced,
        device=device,
        donor_kind=getattr(args, 'donor_kind', 'all2all'),
        donor_pbest_frac=getattr(args, 'donor_pbest_frac', 0.1),
        donor_chunk_size=chunk_size,
    ).to(device)


def build_variant(args, device):
    from encoder.variants.neural_k4 import (
        NeuralK4Variant,
        BATCHED_OPERATOR_CLASSES,
        BATCHED_OPERATOR_CLASSES_K5,
        BATCHED_OPERATOR_CLASSES_DIRECT,
        BATCHED_OPERATOR_CLASSES_K5_ATT,
        BATCHED_OPERATOR_CLASSES_NEURAL,
        BATCHED_OPERATOR_CLASSES_NEURAL_ATT,
        BATCHED_OPERATOR_CLASSES_GATED,
    )

    from encoder.variants.operator_sets import BATCHED_OPERATOR_CLASSES_K2, BATCHED_OPERATOR_CLASSES_K1

    ops_map = {
        'gated': BATCHED_OPERATOR_CLASSES_GATED,
        'classic': BATCHED_OPERATOR_CLASSES,
        'direct': BATCHED_OPERATOR_CLASSES_DIRECT,
        'neural': BATCHED_OPERATOR_CLASSES_NEURAL,
        'neural_att': BATCHED_OPERATOR_CLASSES_NEURAL_ATT,
        'k5': BATCHED_OPERATOR_CLASSES_K5,
        'k5_att': BATCHED_OPERATOR_CLASSES_K5_ATT,
        'k2': BATCHED_OPERATOR_CLASSES_K2,
        'k1': BATCHED_OPERATOR_CLASSES_K1,
    }

    op_classes = ops_map[args.operators]
    K = len(op_classes)

    # head_dim = gatv2_hidden when the stateless BatchedDiffAttDE is present
    # (it reads the full backbone embedding, no per-op proj). Legacy presets
    # with _make_proj-based heads keep head_dim=16.
    from encoder.operators.de_heads import BatchedDiffAttDE
    _stateless_de = any(issubclass(cls, BatchedDiffAttDE) for cls in op_classes)
    head_dim = args.gatv2_hidden if _stateless_de else 16

    variant = NeuralK4Variant(
        K=K, head_dim=head_dim, gatv2_hidden=args.gatv2_hidden,
        operator_classes=op_classes,
        pool_dim=args.pool_dim,
        gate_node_feat_dim=getattr(args, 'gate_node_feat', 0),
        gate_type=getattr(args, 'gate_type', 'adaptive'),
        fcr_mode=getattr(args, 'fcr_mode', 'beta'),
    ).to(device)

    # Legacy tau clamp does NOT change parameter shapes, so it's safe to flip
    # at construction (ckpt resume reads the same `tau` parameter regardless).
    if getattr(args, 'legacy_tau_clamp', False):
        for h in getattr(variant, 'heads', []):
            if hasattr(h, 'tau_mode'):
                h.tau_mode = 'clamp'
    # learn_sigma DOES change `mlp[-1]` output shape (2→3); the expansion
    # must happen AFTER ckpt resume so the trained 2-row weights survive.
    # train_distributed.apply_post_resume_head_fixes() handles that.
    return variant


def apply_post_resume_head_fixes(variant, args):
    """Expand AdaptiveFCRCauchy from 2-output to 3-output after ckpt resume.

    Must run AFTER `variant.load_state_dict(...)` so the trained mlp[-1]
    rows for μ_F and μ_CR are preserved. The σ_F row gets fresh init.

    No-op if `--fcr-learn-sigma` is off.
    """
    if not getattr(args, 'fcr_learn_sigma', False):
        return
    from encoder.operators.adaptive_fcr_cauchy import AdaptiveFCRCauchy
    for h in getattr(variant, 'heads', []):
        adaptive_fcr = getattr(h, 'adaptive_fcr', None)
        if isinstance(adaptive_fcr, AdaptiveFCRCauchy):
            adaptive_fcr.expand_to_learn_sigma()


def build_model(args, device):
    from encoder.opt_variant import GenerationStep

    backbone = build_backbone(args, device)
    assert backbone.gatv2_hidden == args.gatv2_hidden, (
        f"backbone.gatv2_hidden={backbone.gatv2_hidden} != "
        f"args.gatv2_hidden={args.gatv2_hidden}")
    variant = build_variant(args, device)
    _beta_str = os.environ.get('OVERRIDE_SOFT_MIN_BETA')
    if _beta_str is not None:
        try:
            _beta = float(_beta_str)
        except ValueError:
            raise ValueError(
                f"OVERRIDE_SOFT_MIN_BETA='{_beta_str}' is not a valid float")
        log.warning("OVERRIDE_SOFT_MIN_BETA=%.4f (env var override)", _beta)
    else:
        _beta = 20.0
    surrogate = None
    build_graph_fn = None
    if getattr(args, 'gate_type', '') == 'surrogate':
        from encoder.variants.pairwise_surrogate import PairwiseSurrogate
        from encoder.similarity_graph_gpu import build_sparse_graphs_gpu
        surrogate = PairwiseSurrogate(
            backbone_dim=args.gatv2_hidden).to(device)
        build_graph_fn = build_sparse_graphs_gpu

    augment = getattr(args, 'surrogate_augment', 'rebuild')
    if augment == 'delta' and surrogate is None:
        raise ValueError(
            "--surrogate-augment delta requires --gate-type surrogate "
            "(the delta-graph path only fires inside _run_surrogate). "
            f"Got --gate-type={getattr(args, 'gate_type', None)!r}.")

    gen_step = GenerationStep(
        backbone, variant, eval_fn=None, soft_min_beta=_beta,
        surrogate=surrogate, build_graph_fn=build_graph_fn,
        archive_capacity=getattr(args, 'archive_capacity', 0),
        surrogate_augment_strategy=augment,
        lb=args.lb, ub=args.ub).to(device)
    return backbone, variant, gen_step
