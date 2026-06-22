"""
analyze_ssl_nextstep.py — Checkpoint analysis for SSL next-step prediction.

Unified analysis across 4 backbones: pna_gatv2, temporal_gatv2, npa, npa_edges.

Modes:
  --weights-only    Weight health + head analysis (no data needed, fast)
  (default)         Full analysis: weights + embeddings + R² breakdowns

Usage:
    # Single checkpoint (weights only):
    python -m encoder.analyze_ssl_nextstep \
        --checkpoint checkpoints/ssl_nextstep/ssl_nextstep_pna_gatv2.pth \
        --backbone pna_gatv2 --weights-only

    # Single checkpoint (full with data):
    python -m encoder.analyze_ssl_nextstep \
        --checkpoint checkpoints/ssl_nextstep/ssl_nextstep_pna_gatv2.pth \
        --backbone pna_gatv2 --data-dir DATASETS/NPA_GPU --device cuda

    # Compare all checkpoints in directory:
    python -m encoder.analyze_ssl_nextstep \
        --checkpoint-dir checkpoints/ssl_nextstep/ \
        --data-dir DATASETS/NPA_GPU --device cuda
"""

import argparse
import glob
import json
import logging
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# CEC2017 categories (function ID → category)
CEC_CATEGORIES = {}
for fid in range(1, 4):
    CEC_CATEGORIES[fid] = 'unimodal'
for fid in range(4, 11):
    CEC_CATEGORIES[fid] = 'multimodal'
for fid in range(11, 21):
    CEC_CATEGORIES[fid] = 'hybrid'
for fid in range(21, 30):
    CEC_CATEGORIES[fid] = 'composition'

GEN_BUCKETS = {'early': (0, 15), 'mid': (16, 33), 'late': (34, 49)}

from .similarity_graph import NODE_NAMES, NODE_DIM, EDGE_DIM, GLOBAL_DIM
from .ssl_nextstep_dataset import PREDICT_FEATURES

FEATURE_NAMES = [NODE_NAMES[i] for i in PREDICT_FEATURES]
GLOBAL_NAMES = [
    'eval_budget_frac', 'diversity', 'improvement_rate', 'population_fdc',
    'mean_grad_consistency', 'convexity_fraction', 'mean_nbc_ratio',
    'stagnation_counter', 'front_vs_tail', 'direction_consensus',
    'density_quality_corr', 'delta_fitness', 'contraction_rate',
]


# ======================================================================
# Utilities
# ======================================================================

def effective_dim(X, threshold=0.95):
    """SVD-based effective dimensionality."""
    if X.ndim == 1 or min(X.shape) < 2:
        return 1, np.array([1.0])
    X_c = X - X.mean(axis=0)
    try:
        _, S, _ = np.linalg.svd(X_c, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1, np.array([1.0])
    var = S ** 2
    total = var.sum()
    if total < 1e-30:
        return 0, np.zeros(min(X.shape))
    explained = np.cumsum(var) / total
    return int(np.searchsorted(explained, threshold)) + 1, explained


def compute_r2(pred, target):
    """Per-feature R². Inputs: (N, F) tensors."""
    ss_res = ((pred - target) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return (1 - ss_res / (ss_tot + 1e-8)).cpu().numpy()


def _classify_module(key, backbone_name):
    """Classify a state_dict key into a module group."""
    if 'pna.' in key:
        if 'film' in key.lower():
            return 'pna.film'
        if 'node_proj' in key:
            return 'pna.input_proj'
        return 'pna.layers'
    if 'bridge' in key:
        return 'bridges'
    if 'gatv2' in key:
        return 'gatv2'
    if 'edge_readout' in key:
        return 'edge_readout'
    if 'gru' in key or 'temporal' in key:
        return 'gru_temporal'
    if 'pooler' in key or 'pool' in key:
        return 'pooler'
    if 'grid_attn' in key:
        if 'dim_layers' in key:
            return 'attn.stage1_dim'
        if 'ind_layers' in key:
            return 'attn.stage2_ind'
        if 'cross_layers' in key:
            return 'attn.stage3_cross'
        return 'attn.other'
    if 'feature_injector' in key:
        return 'feature_injector'
    if 'edge_proj' in key or 'edge_fuse' in key or 'edge_norm' in key:
        return 'edge_injection'
    return 'other'


# ======================================================================
# 1. Weight Health Analysis
# ======================================================================

def analyze_weights(state_dict, backbone_name):
    """Analyze weight health: norms, drift, dead units, FiLM, effective rank."""
    report = {'backbone': backbone_name, 'modules': {}, 'pathologies': []}

    # Group weights by module
    from collections import defaultdict
    modules = defaultdict(list)
    for k, v in state_dict.items():
        if v.dim() < 2:
            continue
        group = _classify_module(k, backbone_name)
        fan_in = v.shape[1]
        fan_out = v.shape[0]
        xavier_std = math.sqrt(2.0 / (fan_in + fan_out))
        actual_std = v.std().item()
        ratio = actual_std / xavier_std if xavier_std > 0 else 0

        modules[group].append({
            'key': k,
            'shape': list(v.shape),
            'numel': v.numel(),
            'norm': v.norm().item(),
            'std': actual_std,
            'abs_max': v.abs().max().item(),
            'dead_pct': (v.abs() < 1e-6).float().mean().item() * 100,
            'xavier_ratio': ratio,
        })

    # Aggregate per module
    for group, weights in sorted(modules.items()):
        total_params = sum(w['numel'] for w in weights)
        avg_ratio = np.mean([w['xavier_ratio'] for w in weights])
        max_dead = max(w['dead_pct'] for w in weights)
        avg_dead = np.mean([w['dead_pct'] for w in weights])

        report['modules'][group] = {
            'n_tensors': len(weights),
            'total_params': total_params,
            'avg_xavier_ratio': round(avg_ratio, 3),
            'max_dead_pct': round(max_dead, 1),
            'avg_dead_pct': round(avg_dead, 1),
            'weights': weights,
        }

        # Pathology detection
        if avg_ratio < 0.1:
            report['pathologies'].append(
                f"{group}: DEAD (xavier_ratio={avg_ratio:.3f})")
        elif avg_ratio < 0.6 and total_params > 1000:
            report['pathologies'].append(
                f"{group}: UNDERTRAINED (xavier_ratio={avg_ratio:.3f}, "
                f"{total_params:,} params)")
        if max_dead > 50:
            report['pathologies'].append(
                f"{group}: HIGH DEAD UNITS ({max_dead:.0f}%)")

    # FiLM analysis (pna_gatv2 only)
    film_keys = [k for k in state_dict if 'film' in k.lower() and 'weight' in k]
    if film_keys:
        film_active = 0
        film_dead = 0
        for k in film_keys:
            if state_dict[k].abs().max().item() < 1e-6:
                film_dead += 1
            else:
                film_active += 1
        report['film'] = {
            'active': film_active,
            'dead': film_dead,
            'status': 'ALL DEAD' if film_active == 0 else
                      f'{film_active} active, {film_dead} dead',
        }
        if film_dead > 0:
            report['pathologies'].append(
                f"FiLM: {film_dead}/{film_active + film_dead} layers dead")

    # Effective rank on largest weight matrices
    report['effective_ranks'] = {}
    large_weights = [(k, v) for k, v in state_dict.items()
                     if v.dim() == 2 and v.numel() > 1000]
    large_weights.sort(key=lambda x: -x[1].numel())
    for k, v in large_weights[:10]:
        eff, _ = effective_dim(v.detach().cpu().numpy())
        max_rank = min(v.shape)
        report['effective_ranks'][k] = {
            'effective': eff, 'max': max_rank,
            'ratio': round(eff / max_rank, 3)}

    return report


# ======================================================================
# 2. Head Analysis
# ======================================================================

def analyze_heads(pretrainer_state, backbone_name):
    """Analyze prediction head weights."""
    report = {'backbone': backbone_name, 'heads': {}}

    # Node head output layer
    node_w = pretrainer_state.get('node_head.2.weight')
    node_b = pretrainer_state.get('node_head.2.bias')
    if node_w is not None:
        n_feat = node_w.shape[0]
        report['heads']['node'] = []
        for i in range(n_feat):
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f'feat_{i}'
            report['heads']['node'].append({
                'feature': name,
                'weight_norm': round(node_w[i].norm().item(), 4),
                'bias': round(node_b[i].item(), 4) if node_b is not None else None,
                'weight_std': round(node_w[i].std().item(), 4),
            })

    # Global head output layer
    global_w = pretrainer_state.get('global_head.2.weight')
    global_b = pretrainer_state.get('global_head.2.bias')
    if global_w is not None:
        n_global = global_w.shape[0]
        report['heads']['global'] = []
        for i in range(n_global):
            name = GLOBAL_NAMES[i] if i < len(GLOBAL_NAMES) else f'global_{i}'
            wnorm = global_w[i].norm().item()
            report['heads']['global'].append({
                'feature': name,
                'weight_norm': round(wnorm, 4),
                'bias': round(global_b[i].item(), 4) if global_b is not None else None,
                'status': 'DEAD' if wnorm < 0.01 else
                          'WEAK' if wnorm < 0.2 else 'OK',
            })

    # Edge head
    edge_w = pretrainer_state.get('edge_head.net.2.weight')
    if edge_w is not None:
        report['heads']['edge'] = {
            'output_shape': list(edge_w.shape),
            'norm': round(edge_w.norm().item(), 4),
        }
    else:
        report['heads']['edge'] = {'status': 'N/A (no edges)'}

    return report


# ======================================================================
# 3. Embedding Quality (requires forward pass)
# ======================================================================

@torch.no_grad()
def analyze_embeddings(pretrainer, val_dataset, device, max_batches=50):
    """Analyze embedding quality: effective dim, B/W ratio, dead dims."""
    from .ssl_nextstep_dataset import collate_nextstep
    from torch.utils.data import DataLoader

    pretrainer.eval()
    all_h, all_h_global = [], []
    all_fids = []

    loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_nextstep, num_workers=0)

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch_dev = {k: v.to(device) for k, v in batch.items()}

        # Get encode kwargs for temporal backbones
        encode_kw = dict(v_indices=batch_dev['v_indices'],
                         e_indices=batch_dev['e_indices'])
        for key in ('coords_hist', 'fitness_hist', 'n_valid',
                     'coordinates', 'fitness'):
            if key in batch_dev:
                if key == 'coordinates':
                    encode_kw['coords_current'] = batch_dev[key]
                elif key == 'fitness' and 'fitness_hist' in batch_dev:
                    encode_kw['fitness_current'] = batch_dev[key]
                else:
                    encode_kw[key] = batch_dev[key]

        h, e, _, h_global = pretrainer.backbone.encode(
            batch_dev['node_feat'], batch_dev['edge_index'],
            batch_dev['edge_attr'], batch_dev['global_feat'],
            **encode_kw)

        all_h.append(h.cpu().numpy())
        all_h_global.append(h_global.cpu().numpy())

    h_cat = np.concatenate(all_h)           # (N_total, embed_dim)
    hg_cat = np.concatenate(all_h_global)   # (B_total, global_dim)

    report = {}

    # Node embeddings
    eff, explained = effective_dim(h_cat)
    dead_dims = (h_cat.std(axis=0) < 1e-4).sum()
    report['h_nodes'] = {
        'effective_dim': eff,
        'max_dim': h_cat.shape[1],
        'ratio': round(eff / h_cat.shape[1], 3),
        'dead_dims': int(dead_dims),
        'mean': round(h_cat.mean(), 4),
        'std': round(h_cat.std(), 4),
        'abs_max': round(np.abs(h_cat).max(), 4),
    }

    # Global embeddings
    eff_g, _ = effective_dim(hg_cat)
    dead_g = (hg_cat.std(axis=0) < 1e-4).sum()
    report['h_global'] = {
        'effective_dim': eff_g,
        'max_dim': hg_cat.shape[1],
        'ratio': round(eff_g / hg_cat.shape[1], 3),
        'dead_dims': int(dead_g),
        'mean': round(hg_cat.mean(), 4),
        'std': round(hg_cat.std(), 4),
    }

    return report


# ======================================================================
# 3b. Activation Effective Rank per Layer (hook-based)
# ======================================================================

@torch.no_grad()
def analyze_activation_ranks(pretrainer, val_dataset, device, max_batches=30):
    """Effective rank of activations at each named layer."""
    from .ssl_nextstep_dataset import collate_nextstep
    from torch.utils.data import DataLoader

    pretrainer.eval()
    activations = {}
    hooks = []

    def _make_hook(name):
        def hook_fn(module, inp, out):
            t = out if isinstance(out, torch.Tensor) else out[0]
            if t is not None and t.dim() >= 2:
                # Flatten spatial dims, keep last dim
                flat = t.reshape(-1, t.shape[-1]).detach().cpu().numpy()
                if name not in activations:
                    activations[name] = []
                activations[name].append(flat)
        return hook_fn

    # Register hooks on key modules
    backbone = pretrainer.backbone
    hook_targets = {}

    # PNAGATv2-specific
    if hasattr(backbone, 'pna') and backbone.pna is not None:
        hook_targets['pna.node_proj'] = backbone.pna.node_proj
        for i, layer in enumerate(backbone.pna.layers):
            hook_targets[f'pna.layer_{i}'] = layer
        if hasattr(backbone, 'node_bridge'):
            hook_targets['node_bridge'] = backbone.node_bridge
        for i, layer in enumerate(backbone.gatv2_layers):
            hook_targets[f'gatv2.layer_{i}'] = layer

    # NPA-specific
    if hasattr(backbone, 'grid_attn'):
        ga = backbone.grid_attn
        hook_targets['attn.input_proj'] = ga.input_proj
        for i, layer in enumerate(ga.dim_layers):
            hook_targets[f'attn.dim_{i}'] = layer
        for i, layer in enumerate(ga.ind_layers):
            hook_targets[f'attn.ind_{i}'] = layer
        for i, layer in enumerate(ga.cross_layers):
            hook_targets[f'attn.cross_{i}'] = layer
        if hasattr(backbone, 'pool_proj'):
            hook_targets['pool_proj'] = backbone.pool_proj
        if hasattr(backbone, 'feature_injector'):
            hook_targets['feature_injector'] = backbone.feature_injector

    # TemporalGATv2-specific
    if hasattr(backbone, 'gru'):
        hook_targets['gru'] = backbone.gru
    if hasattr(backbone, 'pooler'):
        hook_targets['pooler'] = backbone.pooler

    # Edge injection (npa_edges)
    if hasattr(backbone, 'edge_fuse'):
        hook_targets['edge_fuse'] = backbone.edge_fuse

    for name, module in hook_targets.items():
        hooks.append(module.register_forward_hook(_make_hook(name)))

    # Run forward passes
    loader = DataLoader(val_dataset, batch_size=16, shuffle=False,
                        collate_fn=collate_nextstep, num_workers=0)
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        encode_kw = dict(v_indices=batch_dev['v_indices'],
                         e_indices=batch_dev['e_indices'])
        for key in ('coords_hist', 'fitness_hist', 'n_valid',
                     'coordinates', 'fitness'):
            if key in batch_dev:
                if key == 'coordinates':
                    encode_kw['coords_current'] = batch_dev[key]
                elif key == 'fitness' and 'fitness_hist' in batch_dev:
                    encode_kw['fitness_current'] = batch_dev[key]
                else:
                    encode_kw[key] = batch_dev[key]
        pretrainer.backbone.encode(
            batch_dev['node_feat'], batch_dev['edge_index'],
            batch_dev['edge_attr'], batch_dev['global_feat'], **encode_kw)

    for h in hooks:
        h.remove()

    # Compute effective dim per layer
    report = {}
    for name in sorted(activations.keys()):
        cat = np.concatenate(activations[name])
        eff, explained = effective_dim(cat)
        max_dim = cat.shape[1]
        dead = int((cat.std(axis=0) < 1e-4).sum())
        report[name] = {
            'effective_dim': eff,
            'max_dim': max_dim,
            'ratio': round(eff / max(max_dim, 1), 3),
            'dead_dims': dead,
            'std': round(float(cat.std()), 4),
        }

    return report


# ======================================================================
# 3c. Feature Attribution (input ablation)
# ======================================================================

@torch.no_grad()
def analyze_feature_attribution(pretrainer, val_dataset, device,
                                max_batches=20):
    """Measure R² drop when zeroing each input feature."""
    from .ssl_nextstep_dataset import collate_nextstep
    from torch.utils.data import DataLoader

    pretrainer.eval()
    loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_nextstep, num_workers=0)

    # Collect batches
    batches = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batches.append({k: v.to(device) for k, v in batch.items()})

    def _run_batches(batches_list):
        all_pred = []
        for batch in batches_list:
            _, _, pred_n, _ = pretrainer(batch)
            all_pred.append(pred_n.cpu())
        return torch.cat(all_pred)

    # Collect targets
    all_target = torch.cat([b['target_node_feat'][:, PREDICT_FEATURES].cpu()
                            for b in batches])

    # Baseline R² (no ablation)
    pred_base = _run_batches(batches)
    r2_base = compute_r2(pred_base, all_target).mean()

    report = {'baseline_r2': round(float(r2_base), 4), 'node_features': {},
              'edge_ablation': None, 'global_features': {}}

    # Node feature ablation
    n_node_feat = batches[0]['node_feat'].shape[1]
    for f_idx in range(n_node_feat):
        ablated = []
        for batch in batches:
            b = {k: v.clone() if k == 'node_feat' else v
                 for k, v in batch.items()}
            b['node_feat'][:, f_idx] = 0.0
            ablated.append(b)
        pred_abl = _run_batches(ablated)
        r2_abl = compute_r2(pred_abl, all_target).mean()
        name = NODE_NAMES[f_idx] if f_idx < len(NODE_NAMES) else f'feat_{f_idx}'
        drop = float(r2_base - r2_abl)
        report['node_features'][name] = {
            'r2_without': round(float(r2_abl), 4),
            'r2_drop': round(drop, 4),
            'importance': round(abs(drop), 4),
        }

    # Edge ablation (zero all edge_attr)
    ablated = []
    for batch in batches:
        b = {k: v.clone() if k == 'edge_attr' else v
             for k, v in batch.items()}
        b['edge_attr'] = torch.zeros_like(b['edge_attr'])
        ablated.append(b)
    pred_abl = _run_batches(ablated)
    r2_abl = compute_r2(pred_abl, all_target).mean()
    report['edge_ablation'] = {
        'r2_without': round(float(r2_abl), 4),
        'r2_drop': round(float(r2_base - r2_abl), 4),
    }

    # Global feature ablation
    n_global = batches[0]['global_feat'].shape[1]
    for g_idx in range(n_global):
        ablated = []
        for batch in batches:
            b = {k: v.clone() if k == 'global_feat' else v
                 for k, v in batch.items()}
            b['global_feat'][:, g_idx] = 0.0
            ablated.append(b)
        pred_abl = _run_batches(ablated)
        r2_abl = compute_r2(pred_abl, all_target).mean()
        name = GLOBAL_NAMES[g_idx] if g_idx < len(GLOBAL_NAMES) else f'g_{g_idx}'
        drop = float(r2_base - r2_abl)
        report['global_features'][name] = {
            'r2_without': round(float(r2_abl), 4),
            'r2_drop': round(drop, 4),
            'importance': round(abs(drop), 4),
        }

    return report


# ======================================================================
# 3d. Between/Within Variance Ratio
# ======================================================================

@torch.no_grad()
def analyze_bw_ratio(pretrainer, val_dataset, device, max_batches=50):
    """Between/Within variance ratio of embeddings grouped by category, D, gen."""
    from .ssl_nextstep_dataset import collate_nextstep
    from torch.utils.data import DataLoader
    from collections import defaultdict

    pretrainer.eval()
    groups = defaultdict(list)  # key → list of h vectors

    loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        collate_fn=collate_nextstep, num_workers=0)

    graph_idx = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch_dev = {k: v.to(device) for k, v in batch.items()}

        encode_kw = dict(v_indices=batch_dev['v_indices'],
                         e_indices=batch_dev['e_indices'])
        for key in ('coords_hist', 'fitness_hist', 'n_valid',
                     'coordinates', 'fitness'):
            if key in batch_dev:
                if key == 'coordinates':
                    encode_kw['coords_current'] = batch_dev[key]
                elif key == 'fitness' and 'fitness_hist' in batch_dev:
                    encode_kw['fitness_current'] = batch_dev[key]
                else:
                    encode_kw[key] = batch_dev[key]

        h, _, _, h_global = pretrainer.backbone.encode(
            batch_dev['node_feat'], batch_dev['edge_index'],
            batch_dev['edge_attr'], batch_dev['global_feat'], **encode_kw)

        # h_global per graph
        B = h_global.shape[0]
        for b in range(B):
            if graph_idx >= len(val_dataset):
                break
            fpath, off_t, *_ = val_dataset._index[graph_idx]
            s_t = val_dataset._load_file(fpath)[off_t]
            fid = s_t.get('fid', 0)
            ndim = s_t.get('ndim', 10)
            gen = s_t.get('gen', 0)

            cat = CEC_CATEGORIES.get(fid, 'unknown')
            gen_bucket = ('early' if gen <= 15 else
                         'mid' if gen <= 33 else 'late')

            hg = h_global[b].cpu().numpy()
            groups[f'cat:{cat}'].append(hg)
            groups[f'D:{ndim}'].append(hg)
            groups[f'gen:{gen_bucket}'].append(hg)
            groups[f'D:{ndim}_gen:{gen_bucket}'].append(hg)
            graph_idx += 1

    # Compute B/W ratio per grouping type
    def _bw_ratio(group_dict, prefix):
        relevant = {k: np.stack(v) for k, v in group_dict.items()
                    if k.startswith(prefix) and len(v) >= 5}
        if len(relevant) < 2:
            return None
        all_cat = np.concatenate(list(relevant.values()))
        grand_mean = all_cat.mean(axis=0)
        # Between: variance of group means around grand mean
        group_means = np.stack([v.mean(axis=0) for v in relevant.values()])
        B_var = ((group_means - grand_mean) ** 2).mean()
        # Within: average within-group variance
        W_var = np.mean([v.var(axis=0).mean() for v in relevant.values()])
        ratio = B_var / max(W_var, 1e-10)
        return {
            'ratio': round(float(ratio), 4),
            'B_var': round(float(B_var), 6),
            'W_var': round(float(W_var), 6),
            'n_groups': len(relevant),
            'status': ('COLLAPSED' if ratio < 0.01 else
                       'WEAK' if ratio < 0.1 else
                       'MODERATE' if ratio < 1.0 else 'GOOD'),
        }

    report = {}
    for prefix, label in [('cat:', 'by_category'), ('D:', 'by_dimension'),
                           ('gen:', 'by_generation')]:
        result = _bw_ratio(groups, prefix)
        if result:
            report[label] = result

    return report


# ======================================================================
# 4. R² Breakdowns (requires forward pass + metadata)
# ======================================================================

@torch.no_grad()
def analyze_r2(pretrainer, data_dir, device, backbone_name,
               max_pairs=10000):
    """R² breakdowns: per-feature, per-gen, per-category, per-strategy."""
    from .ssl_nextstep_dataset import NextStepPairDataset, collate_nextstep
    from .ssl_nextstep_pretrain import TEMPORAL_BACKBONES
    from torch.utils.data import DataLoader

    temporal = backbone_name in TEMPORAL_BACKBONES
    val_data = NextStepPairDataset(data_dir, split='val', temporal=temporal)

    if len(val_data) == 0:
        return {'error': 'No val data found'}

    pretrainer.eval()

    # Collect metadata from raw snapshots
    all_pred, all_target, all_current = [], [], []
    all_pred_g, all_target_g, all_current_g = [], [], []
    all_gen, all_fid, all_strategy, all_ndim = [], [], [], []

    n_use = min(len(val_data), max_pairs)
    for idx in range(n_use):
        fpath, off_t, *_ = val_data._index[idx]
        file_data = val_data._load_file(fpath)
        s_t = file_data[off_t]
        all_gen.append(s_t.get('gen', 0))
        all_fid.append(s_t.get('fid', 0))
        all_strategy.append(s_t.get('strategy', 'unknown'))
        all_ndim.append(s_t.get('ndim', 10))

    # Now do batched forward passes
    if temporal:
        from .ssl_pretrain_k2 import DimGroupBatchSampler
        sampler = DimGroupBatchSampler(
            val_data._ndims[:max_pairs], batch_size=32,
            drop_last=False, shuffle=False)
        loader = DataLoader(val_data, batch_sampler=sampler,
                            collate_fn=collate_nextstep, num_workers=0)
    else:
        from torch.utils.data import Subset
        subset = Subset(val_data, range(min(len(val_data), max_pairs)))
        loader = DataLoader(subset, batch_size=32, shuffle=False,
                            collate_fn=collate_nextstep, num_workers=0)

    node_offset = 0
    for batch in loader:
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        _, _, pred_n, pred_g = pretrainer(batch_dev)

        target_n = batch_dev['target_node_feat'][:, PREDICT_FEATURES]
        current_n = batch_dev['node_feat'][:, PREDICT_FEATURES]

        all_pred.append(pred_n.cpu())
        all_target.append(target_n.cpu())
        all_current.append(current_n.cpu())
        all_pred_g.append(pred_g.cpu())
        all_target_g.append(batch_dev['target_global'].cpu())
        all_current_g.append(batch_dev['global_feat'].cpu())

    pred_all = torch.cat(all_pred)
    target_all = torch.cat(all_target)
    current_all = torch.cat(all_current)
    pred_g_all = torch.cat(all_pred_g)
    target_g_all = torch.cat(all_target_g)
    current_g_all = torch.cat(all_current_g)

    report = {}

    # 4a. Overall per-feature R²
    r2_model = compute_r2(pred_all, target_all)
    r2_persist = compute_r2(current_all, target_all)
    report['per_feature'] = {}
    for i, name in enumerate(FEATURE_NAMES):
        report['per_feature'][name] = {
            'model': round(float(r2_model[i]), 4),
            'persistence': round(float(r2_persist[i]), 4),
            'delta': round(float(r2_model[i] - r2_persist[i]), 4),
        }
    report['mean_r2'] = round(float(r2_model.mean()), 4)
    report['mean_persist'] = round(float(r2_persist.mean()), 4)

    # 4b. Global R² per feature
    r2_g = compute_r2(pred_g_all, target_g_all)
    r2_gp = compute_r2(current_g_all, target_g_all)
    report['global_per_feature'] = {}
    for i, name in enumerate(GLOBAL_NAMES[:r2_g.shape[0]]):
        report['global_per_feature'][name] = {
            'model': round(float(r2_g[i]), 4),
            'persistence': round(float(r2_gp[i]), 4),
        }

    # 4c. Per-generation bucket (expand gen per node: N=100 per graph)
    N_per_graph = 100
    n_graphs = len(all_gen)
    gen_expanded = []
    for g in all_gen[:n_graphs]:
        gen_expanded.extend([g] * N_per_graph)
    gen_arr = np.array(gen_expanded[:pred_all.shape[0]])

    report['per_generation'] = {}
    for bucket_name, (lo, hi) in GEN_BUCKETS.items():
        mask = (gen_arr >= lo) & (gen_arr <= hi)
        if mask.sum() < 100:
            continue
        r2_m = compute_r2(pred_all[mask], target_all[mask])
        r2_p = compute_r2(current_all[mask], target_all[mask])
        report['per_generation'][bucket_name] = {
            'n_nodes': int(mask.sum()),
            'model_mean': round(float(r2_m.mean()), 4),
            'persist_mean': round(float(r2_p.mean()), 4),
            'delta': round(float(r2_m.mean() - r2_p.mean()), 4),
        }

    # 4d. Per-category
    fid_expanded = []
    for f in all_fid[:n_graphs]:
        fid_expanded.extend([f] * N_per_graph)
    fid_arr = np.array(fid_expanded[:pred_all.shape[0]])

    report['per_category'] = {}
    for cat in ['unimodal', 'multimodal', 'hybrid', 'composition']:
        cat_fids = {fid for fid, c in CEC_CATEGORIES.items() if c == cat}
        mask = np.isin(fid_arr, list(cat_fids))
        if mask.sum() < 100:
            continue
        r2_m = compute_r2(pred_all[mask], target_all[mask])
        r2_p = compute_r2(current_all[mask], target_all[mask])
        report['per_category'][cat] = {
            'n_nodes': int(mask.sum()),
            'model_mean': round(float(r2_m.mean()), 4),
            'persist_mean': round(float(r2_p.mean()), 4),
            'delta': round(float(r2_m.mean() - r2_p.mean()), 4),
        }

    # 4e. Per-strategy
    strat_expanded = []
    for s in all_strategy[:n_graphs]:
        strat_expanded.extend([s] * N_per_graph)
    strat_arr = np.array(strat_expanded[:pred_all.shape[0]])

    report['per_strategy'] = {}
    for strat in ['shade_only', 'mos_best', 'mos_top3', 'mos_top10', 'oracle']:
        mask = strat_arr == strat
        if mask.sum() < 100:
            continue
        r2_m = compute_r2(pred_all[mask], target_all[mask])
        r2_p = compute_r2(current_all[mask], target_all[mask])
        report['per_strategy'][strat] = {
            'n_nodes': int(mask.sum()),
            'model_mean': round(float(r2_m.mean()), 4),
            'persist_mean': round(float(r2_p.mean()), 4),
            'delta': round(float(r2_m.mean() - r2_p.mean()), 4),
        }

    # 4f. Per-dimensionality
    ndim_expanded = []
    for d in all_ndim[:n_graphs]:
        ndim_expanded.extend([d] * N_per_graph)
    ndim_arr = np.array(ndim_expanded[:pred_all.shape[0]])

    report['per_dimension'] = {}
    for D in sorted(set(all_ndim)):
        mask = ndim_arr == D
        if mask.sum() < 100:
            continue
        r2_m = compute_r2(pred_all[mask], target_all[mask])
        r2_p = compute_r2(current_all[mask], target_all[mask])
        report['per_dimension'][f'D={D}'] = {
            'n_nodes': int(mask.sum()),
            'model_mean': round(float(r2_m.mean()), 4),
            'persist_mean': round(float(r2_p.mean()), 4),
            'delta': round(float(r2_m.mean() - r2_p.mean()), 4),
            'per_feature': {
                FEATURE_NAMES[i]: round(float(r2_m[i]), 4)
                for i in range(len(FEATURE_NAMES))
            },
        }

    # 4g. Cross: dimension × generation
    report['dim_x_gen'] = {}
    for D in sorted(set(all_ndim)):
        for bucket_name, (lo, hi) in GEN_BUCKETS.items():
            mask = (ndim_arr == D) & (gen_arr >= lo) & (gen_arr <= hi)
            if mask.sum() < 50:
                continue
            r2_m = compute_r2(pred_all[mask], target_all[mask])
            r2_p = compute_r2(current_all[mask], target_all[mask])
            key = f'D={D}_{bucket_name}'
            report['dim_x_gen'][key] = {
                'n_nodes': int(mask.sum()),
                'model_mean': round(float(r2_m.mean()), 4),
                'persist_mean': round(float(r2_p.mean()), 4),
                'delta': round(float(r2_m.mean() - r2_p.mean()), 4),
            }

    return report


# ======================================================================
# 5. Cross-Backbone Comparison
# ======================================================================

def compare_backbones(reports):
    """Build comparison table from multiple backbone reports."""
    rows = []
    for r in reports:
        name = r['config']['backbone']
        total_params = sum(
            m['total_params'] for m in r['weights']['modules'].values())
        row = {
            'backbone': name,
            'params': total_params,
            'epoch': r['config'].get('epoch', '?'),
            'val_loss': r['config'].get('val_loss', '?'),
            'r2_mean': r.get('r2', {}).get('mean_r2', '?'),
            'r2_persist': r.get('r2', {}).get('mean_persist', '?'),
            'pathologies': len(r['weights'].get('pathologies', [])),
        }
        # Per-feature R²
        if 'r2' in r and 'per_feature' in r['r2']:
            for feat, vals in r['r2']['per_feature'].items():
                row[f'r2_{feat}'] = vals['model']
        rows.append(row)
    return rows


# ======================================================================
# Printing
# ======================================================================

def print_weight_report(report):
    print(f"\n{'='*70}")
    print(f"  WEIGHT HEALTH: {report['backbone']}")
    print(f"{'='*70}")

    for group, info in sorted(report['modules'].items()):
        status = 'OK'
        if info['avg_xavier_ratio'] < 0.1:
            status = 'DEAD'
        elif info['avg_xavier_ratio'] < 0.6:
            status = 'UNDERTRAINED'
        print(f"  {group:25s}  params={info['total_params']:>8,}  "
              f"xavier_ratio={info['avg_xavier_ratio']:.3f}  "
              f"dead={info['avg_dead_pct']:.1f}%  [{status}]")

    if 'film' in report:
        print(f"\n  FiLM: {report['film']['status']}")

    if report['pathologies']:
        print(f"\n  PATHOLOGIES:")
        for p in report['pathologies']:
            print(f"    - {p}")
    else:
        print(f"\n  No pathologies detected.")


def print_head_report(report):
    print(f"\n{'='*70}")
    print(f"  HEAD ANALYSIS: {report['backbone']}")
    print(f"{'='*70}")

    if 'node' in report['heads']:
        print(f"\n  Node head (next-step prediction):")
        for f in report['heads']['node']:
            print(f"    {f['feature']:25s}  w_norm={f['weight_norm']:.4f}  "
                  f"bias={f['bias']:+.4f}")

    if 'global' in report['heads']:
        print(f"\n  Global head:")
        for f in report['heads']['global']:
            print(f"    {f['feature']:25s}  w_norm={f['weight_norm']:.4f}  "
                  f"[{f['status']}]")

    if 'edge' in report['heads']:
        e = report['heads']['edge']
        if 'norm' in e:
            print(f"\n  Edge head: norm={e['norm']:.4f}")
        else:
            print(f"\n  Edge head: {e.get('status', 'N/A')}")


def print_embedding_report(report):
    print(f"\n{'='*70}")
    print(f"  EMBEDDING QUALITY")
    print(f"{'='*70}")

    h = report['h_nodes']
    print(f"  h_nodes:  eff_dim={h['effective_dim']}/{h['max_dim']} "
          f"({h['ratio']:.3f})  dead={h['dead_dims']}  "
          f"mean={h['mean']:.4f}  std={h['std']:.4f}")

    g = report['h_global']
    print(f"  h_global: eff_dim={g['effective_dim']}/{g['max_dim']} "
          f"({g['ratio']:.3f})  dead={g['dead_dims']}  "
          f"mean={g['mean']:.4f}  std={g['std']:.4f}")


def print_activation_ranks(report):
    print(f"\n{'='*70}")
    print(f"  ACTIVATION EFFECTIVE RANK (per layer)")
    print(f"{'='*70}")
    for name, info in report.items():
        status = ('COLLAPSED' if info['ratio'] < 0.1 else
                  'LOW' if info['ratio'] < 0.3 else '')
        print(f"  {name:30s}  eff={info['effective_dim']:>3d}/{info['max_dim']:<3d} "
              f"({info['ratio']:.3f})  dead={info['dead_dims']}  "
              f"std={info['std']:.4f}  {status}")


def print_attribution(report):
    print(f"\n{'='*70}")
    print(f"  FEATURE ATTRIBUTION (R² drop when zeroed)")
    print(f"{'='*70}")
    print(f"  Baseline R²: {report['baseline_r2']:.4f}")

    print(f"\n  Node features (sorted by importance):")
    sorted_nf = sorted(report['node_features'].items(),
                        key=lambda x: -x[1]['importance'])
    for name, vals in sorted_nf:
        bar = '#' * int(vals['importance'] * 200)
        print(f"    {name:25s}  drop={vals['r2_drop']:+.4f}  {bar}")

    if report['edge_ablation']:
        e = report['edge_ablation']
        print(f"\n  Edge features (all): drop={e['r2_drop']:+.4f}")

    print(f"\n  Global features (sorted by importance):")
    sorted_gf = sorted(report['global_features'].items(),
                        key=lambda x: -x[1]['importance'])
    for name, vals in sorted_gf:
        bar = '#' * int(vals['importance'] * 200)
        print(f"    {name:25s}  drop={vals['r2_drop']:+.4f}  {bar}")


def print_bw_ratio(report):
    print(f"\n{'='*70}")
    print(f"  BETWEEN/WITHIN VARIANCE RATIO (embedding discrimination)")
    print(f"{'='*70}")
    for label, info in report.items():
        print(f"  {label:20s}  B/W={info['ratio']:.4f}  "
              f"({info['n_groups']} groups)  [{info['status']}]")


def print_r2_report(report):
    print(f"\n{'='*70}")
    print(f"  R² BREAKDOWNS")
    print(f"{'='*70}")

    if 'per_feature' in report:
        print(f"\n  Per-feature (model / persistence / delta):")
        for name, vals in report['per_feature'].items():
            print(f"    {name:25s}  {vals['model']:+.4f}  "
                  f"{vals['persistence']:+.4f}  {vals['delta']:+.4f}")
        print(f"    {'MEAN':25s}  {report['mean_r2']:+.4f}  "
              f"{report['mean_persist']:+.4f}  "
              f"{report['mean_r2'] - report['mean_persist']:+.4f}")

    for section, label in [
        ('per_generation', 'Per generation bucket'),
        ('per_category', 'Per CEC2017 category'),
        ('per_strategy', 'Per strategy'),
        ('per_dimension', 'Per dimensionality'),
        ('dim_x_gen', 'Per dimension x generation'),
    ]:
        if section in report and report[section]:
            print(f"\n  {label}:")
            for key, vals in report[section].items():
                print(f"    {key:20s}  model={vals['model_mean']:+.4f}  "
                      f"persist={vals['persist_mean']:+.4f}  "
                      f"Δ={vals['delta']:+.4f}  (n={vals['n_nodes']:,})")

    if 'global_per_feature' in report:
        print(f"\n  Global R² per feature:")
        for name, vals in report['global_per_feature'].items():
            status = 'DEAD' if vals['model'] < -10 else ''
            print(f"    {name:25s}  model={vals['model']:+.4f}  "
                  f"persist={vals['persistence']:+.4f}  {status}")


def print_comparison(rows):
    print(f"\n{'='*70}")
    print(f"  CROSS-BACKBONE COMPARISON")
    print(f"{'='*70}")

    print(f"\n  {'Backbone':20s} {'Params':>8s} {'Epoch':>6s} "
          f"{'R² model':>9s} {'R² persist':>11s} {'Delta':>7s} {'Issues':>7s}")
    print(f"  {'-'*68}")
    for r in rows:
        r2 = r['r2_mean'] if isinstance(r['r2_mean'], float) else 0
        rp = r['r2_persist'] if isinstance(r['r2_persist'], float) else 0
        print(f"  {r['backbone']:20s} {r['params']:>8,} {str(r['epoch']):>6s} "
              f"{r2:>9.4f} {rp:>11.4f} {r2-rp:>+7.4f} {r['pathologies']:>7d}")


# ======================================================================
# Main
# ======================================================================

def analyze_single(ckpt_path, backbone_name, data_dir, device,
                   weights_only=False):
    """Full analysis of a single checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    config = ckpt.get('config', {})
    config['epoch'] = ckpt.get('epoch', '?')
    config['val_loss'] = ckpt.get('val_loss', '?')
    config['val_r2_mean'] = ckpt.get('val_r2_mean', '?')
    config['backbone'] = backbone_name

    result = {'config': config}

    # 1. Weights
    state = ckpt.get('backbone_state_dict', {})
    result['weights'] = analyze_weights(state, backbone_name)
    print_weight_report(result['weights'])

    # 2. Heads
    pt_state = ckpt.get('pretrainer_state_dict', {})
    result['heads'] = analyze_heads(pt_state, backbone_name)
    print_head_report(result['heads'])

    if weights_only:
        return result

    if not data_dir:
        log.warning("No --data-dir, skipping embedding/R² analysis")
        return result

    # Reconstruct model for forward passes
    from .ssl_nextstep_pretrain import _create_backbone, NextStepPretrainer, TEMPORAL_BACKBONES
    backbone = _create_backbone(backbone_name, device)
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    has_edges = backbone_name not in ('npa',)
    pretrainer = NextStepPretrainer(
        backbone, embed_dim=64, global_dim=32,
        edge_dim=EDGE_DIM, has_edges=has_edges, device=device)
    # Load head weights
    pretrainer.load_state_dict(ckpt['pretrainer_state_dict'])

    temporal = backbone_name in TEMPORAL_BACKBONES

    # 3. Embeddings
    from .ssl_nextstep_dataset import NextStepPairDataset
    val_data = NextStepPairDataset(data_dir, split='val', temporal=temporal)
    if len(val_data) > 0:
        result['embeddings'] = analyze_embeddings(
            pretrainer, val_data, device)
        print_embedding_report(result['embeddings'])

    # 3b. Activation effective rank per layer
    if len(val_data) > 0:
        result['activation_ranks'] = analyze_activation_ranks(
            pretrainer, val_data, device)
        print_activation_ranks(result['activation_ranks'])

    # 3c. Feature attribution
    if len(val_data) > 0:
        result['attribution'] = analyze_feature_attribution(
            pretrainer, val_data, device)
        print_attribution(result['attribution'])

    # 3d. Between/Within variance ratio
    if len(val_data) > 0:
        result['bw_ratio'] = analyze_bw_ratio(
            pretrainer, val_data, device)
        print_bw_ratio(result['bw_ratio'])

    # 4. R² breakdowns
    result['r2'] = analyze_r2(pretrainer, data_dir, device, backbone_name)
    print_r2_report(result['r2'])

    return result


def main():
    parser = argparse.ArgumentParser(
        description="SSL next-step checkpoint analysis")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory with ssl_nextstep_*.pth files")
    parser.add_argument("--backbone", type=str, default=None,
                        help="Backbone name (auto-detected from filename if omitted)")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--weights-only", action='store_true')
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Save JSON reports here (default: same as checkpoint)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    # Discover checkpoints
    if args.checkpoint:
        ckpts = [(args.checkpoint, args.backbone)]
    elif args.checkpoint_dir:
        ckpts = []
        for f in sorted(glob.glob(
                os.path.join(args.checkpoint_dir, 'ssl_nextstep_*.pth'))):
            # Extract backbone name from filename
            match = re.search(r'ssl_nextstep_(.+)\.pth', os.path.basename(f))
            if match:
                ckpts.append((f, match.group(1)))
        if not ckpts:
            log.error("No ssl_nextstep_*.pth found in %s", args.checkpoint_dir)
            return
    else:
        parser.error("Provide --checkpoint or --checkpoint-dir")

    all_reports = []
    for ckpt_path, backbone_name in ckpts:
        if backbone_name is None:
            match = re.search(r'ssl_nextstep_(.+)\.pth',
                              os.path.basename(ckpt_path))
            backbone_name = match.group(1) if match else 'unknown'

        log.info("Analyzing: %s [%s]", ckpt_path, backbone_name)
        report = analyze_single(
            ckpt_path, backbone_name, args.data_dir,
            args.device, args.weights_only)
        all_reports.append(report)

        # Save individual JSON
        out_dir = args.output_dir or os.path.dirname(ckpt_path)
        if out_dir:
            json_path = os.path.join(out_dir, f'analysis_{backbone_name}.json')
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            log.info("Saved: %s", json_path)

    # Comparison
    if len(all_reports) > 1:
        comp = compare_backbones(all_reports)
        print_comparison(comp)

        out_dir = args.output_dir or os.path.dirname(ckpts[0][0])
        if out_dir:
            comp_path = os.path.join(out_dir, 'comparison.json')
            with open(comp_path, 'w') as f:
                json.dump(comp, f, indent=2, default=str)
            log.info("Saved: %s", comp_path)


if __name__ == '__main__':
    main()
