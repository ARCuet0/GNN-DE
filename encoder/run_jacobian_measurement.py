"""
run_jacobian_measurement.py — Measure Jacobian spectral radius ρ on trained checkpoint.

Determines how many BPTT generations the system actually needs.

Usage:
    python -m encoder.run_jacobian_measurement \
        --checkpoint checkpoints/k4_cec_full/best_val.pth \
        --device cuda --dims 10
"""
import json
import logging
import math
import time
from pathlib import Path

import torch

from .measure_jacobian import estimate_spectral_radius_from_h, min_safe_bptt

log = logging.getLogger(__name__)


def measure_rho_at_gen(backbone, variant, fn, f_optimal, *,
                        D, N, target_gen, n_gens, device, graph_builder):
    """Run trajectory to target_gen, then measure ρ = ‖∂h_{t+1}/∂h_t‖."""
    from .opt_variant import GenerationStep
    from .train_hybrid import _ring_indices

    B = 1
    gen_step = GenerationStep(backbone, variant, eval_fn=fn)

    coords = torch.rand(B, N, D, dtype=torch.float64, device=device) * 200 - 100
    fitness = fn(coords.squeeze(0)).unsqueeze(0)

    W = getattr(backbone, 'gru_window', 16)
    coords_ring = torch.zeros(W, N, D, dtype=torch.float32, device=device)
    fitness_ring = torch.zeros(W, N, dtype=torch.float32, device=device)

    # Run to target_gen (no grad)
    with torch.no_grad():
        for gen in range(target_gen + 2):
            ri = gen % W
            coords_ring[ri] = coords.squeeze(0).float()
            fitness_ring[ri] = fitness.squeeze(0).float()
            n_valid = min(gen + 1, W)
            idx = _ring_indices(gen, W)

            cache = graph_builder(
                coords.float(), fitness.float(),
                step_num=gen, max_steps=n_gens, ndim=D, k_neighbors=8)

            temporal_kw = dict(
                coords_hist=coords_ring[idx],
                fitness_hist=fitness_ring[idx],
                n_valid=n_valid)

            if gen == target_gen:
                # Save state for Jacobian measurement
                saved_coords = coords.clone()
                saved_fitness = fitness.clone()
                saved_cache = cache
                saved_temporal = dict(temporal_kw)

            if gen == target_gen + 1:
                saved_cache_tp1 = cache
                saved_temporal_tp1 = dict(temporal_kw)

            result = gen_step.run(
                coords=coords, fitness=fitness, cache=cache,
                f_optimal=f_optimal, M=1,
                node_feat=cache.node_feat, global_feat=cache.global_feat,
                **temporal_kw)
            coords = result['new_coords']
            fitness = result['new_fitness']

    # Now measure Jacobian with grad enabled
    backbone.eval()

    # h_t: forward at target_gen with grad
    if coords.is_cuda:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            h_t_raw, _, _, _ = backbone.encode(
                saved_cache.node_feat, saved_cache.global_feat, saved_cache,
                **saved_temporal)
        h_t = h_t_raw.float()
    else:
        h_t, _, _, _ = backbone.encode(
            saved_cache.node_feat, saved_cache.global_feat, saved_cache,
            **saved_temporal)

    h_t_detached = h_t.detach().requires_grad_(True)

    # h_{t+1}: simulate dependency on h_t
    # The real Jacobian is through the population update (greedy selection)
    # which is non-differentiable. Instead, measure the backbone-internal
    # Jacobian: how much does h change between consecutive generations
    # given the same backbone weights but different population states?
    if coords.is_cuda:
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            h_tp1_raw, _, _, _ = backbone.encode(
                saved_cache_tp1.node_feat, saved_cache_tp1.global_feat,
                saved_cache_tp1, **saved_temporal_tp1)
        h_tp1 = h_tp1_raw.float()
    else:
        h_tp1, _, _, _ = backbone.encode(
            saved_cache_tp1.node_feat, saved_cache_tp1.global_feat,
            saved_cache_tp1, **saved_temporal_tp1)

    # Measure input sensitivity: how much does h change between gen t and t+1?
    # This is ||h_{t+1} - h_t|| / ||h_t|| — a proxy for the effective ρ
    h_diff = (h_tp1 - h_t).norm().item()
    h_norm = h_t.norm().item()
    rho_proxy = h_diff / max(h_norm, 1e-8)

    return rho_proxy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Measure Jacobian spectral radius ρ')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dims', type=int, nargs='+', default=[10])
    parser.add_argument('--sparse', action='store_true')
    parser.add_argument('--gen-points', type=int, nargs='+',
                        default=[50, 200, 500, 1000, 1500, 1900])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    # Load model
    device = args.device
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if args.sparse:
        from .sparse_temporal_backbone import TemporalSparseGATv2Backbone
        from .sparse_gatv2_backbone import TopologyMode
        from .similarity_graph_gpu import build_sparse_graphs_gpu
        backbone = TemporalSparseGATv2Backbone(
            d_rnn=64, d_temporal=64, gru_window=16,
            node_in=8, edge_in=4, global_in=13,
            gatv2_hidden=128, gatv2_layers=2, n_heads=8,
            global_out_dim=128, dropout=0.0,
            topology_mode=TopologyMode.COORDINATE_KNN, k_neighbors=8,
            device=device).to(device)
        graph_builder = build_sparse_graphs_gpu
    else:
        from .dense_temporal_backbone import TemporalDenseGATv2Backbone
        from .similarity_graph_gpu import build_dense_graphs_gpu
        backbone = TemporalDenseGATv2Backbone(
            d_rnn=64, d_temporal=64, gru_window=16,
            node_in=8, edge_in=4, global_in=13,
            gatv2_hidden=128, gatv2_layers=2, n_heads=8,
            global_out_dim=128, dropout=0.0,
            device=device).to(device)
        graph_builder = build_dense_graphs_gpu

    backbone.load_state_dict(ckpt['backbone_state_dict'])

    from .variants.neural_k4 import NeuralK4Variant
    variant = NeuralK4Variant(K=4, head_dim=16, gatv2_hidden=128).to(device)
    if 'variant_state_dict' in ckpt:
        variant.load_state_dict(ckpt['variant_state_dict'])

    from .cec2017_torch import CEC2017Torch, get_all_func_ids, FUNCTIONS

    print(f'\n{"FID":>4} {"Cat":>14} {"D":>3} | ', end='')
    for g in args.gen_points:
        print(f'  g={g:<5}', end='')
    print(' | min_bptt(5%) | min_bptt(1%)')
    print('-' * 100)

    for D in args.dims:
        N = 5 * D
        n_gens = (10000 * D) // N
        fids = get_all_func_ids(D)

        for fid in fids:
            torch.manual_seed(args.seed)
            fn = CEC2017Torch(fid, D, device)
            cat = FUNCTIONS[fid][1]

            rhos = []
            print(f'F{fid:02d}  {cat:>14} {D:3d} | ', end='', flush=True)

            for gen in args.gen_points:
                if gen >= n_gens - 2:
                    print(f'  {"--":>6}', end='')
                    continue
                try:
                    rho = measure_rho_at_gen(
                        backbone, variant, fn, fn.f_optimal,
                        D=D, N=N, target_gen=gen, n_gens=n_gens,
                        device=device, graph_builder=graph_builder)
                    rhos.append(rho)
                    print(f'  {rho:6.4f}', end='', flush=True)
                except Exception as e:
                    print(f'  {"ERR":>6}', end='')

            if rhos:
                mean_rho = sum(rhos) / len(rhos)
                L5 = min_safe_bptt(mean_rho, tolerance=0.05)
                L1 = min_safe_bptt(mean_rho, tolerance=0.01)
                print(f' | {L5:>12d} | {L1:>11d}')
            else:
                print(f' | {"--":>12} | {"--":>11}')
