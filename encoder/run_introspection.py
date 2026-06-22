"""
Run one optimization trajectory with full introspection.

Captures every intermediate tensor the model produces:
- GATv2 attention scores and weights per head per layer
- Routing logits, probabilities, and hard winners
- Per-operator parameters (DE: F, CR, pbest, diff; LS: step, sparsity; CMA: sigma, L)
- Per-operator deltas
- Population state at each generation
- Graph features (node, edge, global)

Saves to a .npz file for offline analysis.

Usage:
    python -m encoder.run_introspection \
        --checkpoint checkpoints/k4_sparse_embed_auc2/best_ema.pth \
        --fid 2 --D 10 --n-gens 200 \
        --output introspection_F02.npz
"""
import argparse
import logging
import torch
import numpy as np

log = logging.getLogger(__name__)


def run_introspection(backbone, variant, fn, f_optimal, *,
                      D, N, n_gens, device='cpu'):
    """Run one trajectory capturing all intermediate tensors."""
    from .opt_variant import GenerationStep
    from .similarity_graph_gpu import build_sparse_graphs_gpu
    from .trajectory import _ring_indices

    graph_builder = build_sparse_graphs_gpu
    gen_step = GenerationStep(backbone, variant, eval_fn=fn)

    B = 1
    coords = torch.rand(B, N, D, dtype=torch.float64, device=device) * 200 - 100
    fitness = fn(coords.reshape(-1, D)).reshape(B, N)

    W = getattr(backbone, 'gru_window', 16)
    coords_ring = torch.zeros(W, N, D, dtype=torch.float32, device=device)
    fitness_ring = torch.zeros(W, N, dtype=torch.float32, device=device)

    # Storage for all generations
    record = {
        'f_optimal': f_optimal,
        'D': D, 'N': N, 'n_gens': n_gens,
        # Per-gen arrays (appended each gen)
        'coords': [],          # (n_gens, N, D)
        'fitness': [],         # (n_gens, N)
        'best_fitness': [],    # (n_gens,)
        # Graph features
        'node_feat': [],       # (n_gens, N, 8)
        'edge_feat': [],       # (n_gens, N, k, 4)
        'global_feat': [],     # (n_gens, 13)
        'knn_idx': [],         # (n_gens, N, k)
        # Backbone
        'h_node': [],          # (n_gens, N, 128)
        'h_global': [],        # (n_gens, 128)
        'h_per_head': [],      # (n_gens, N, 8, 16)
        'attn_scores_L0': [],  # (n_gens, N, k, 8) layer 0
        'attn_weights_L0': [], # (n_gens, N, k, 8) layer 0
        'attn_scores_L1': [],  # (n_gens, N, k, 8) layer 1
        'attn_weights_L1': [], # (n_gens, N, k, 8) layer 1
        # Routing
        'logits': [],          # (n_gens, N, K)
        'routing_probs': [],   # (n_gens, N, K)
        'winner': [],          # (n_gens, N)
        # Offspring
        'delta': [],           # (n_gens, N, D)
        'offspring_fitness': [],  # (n_gens, N)
        'improved': [],        # (n_gens, N) bool
        # Counterfactual: fitness each operator WOULD produce
        'cf_fitness': [],      # (n_gens, N, K) — fitness of each operator's offspring
        'cf_improved': [],     # (n_gens, N, K) — bool: would each operator improve?
    }

    # Patch GATv2 layers to save attention scores as attributes
    attn_captures = {}
    sparse_bb = backbone.backbone if hasattr(backbone, 'backbone') else backbone
    _patch_attention_save(sparse_bb)

    with torch.no_grad():
        for gen in range(n_gens):
            # Record population state
            record['coords'].append(coords[0].cpu().numpy())
            record['fitness'].append(fitness[0].cpu().numpy())
            record['best_fitness'].append(fitness[0].min().item())

            # Temporal ring buffer
            ri = gen % W
            coords_ring[ri] = coords[0].detach().float()
            fitness_ring[ri] = fitness[0].detach().float()
            n_valid = min(gen + 1, W)
            idx = _ring_indices(gen, W)
            coords_hist = coords_ring[idx]
            fitness_hist = fitness_ring[idx]

            # Build graph
            cache = graph_builder(
                coords.float(), fitness.float(),
                step_num=gen, max_steps=n_gens, ndim=D, k_neighbors=8)

            record['node_feat'].append(cache.node_feat[0].cpu().numpy())
            record['edge_feat'].append(cache.edge_feat[0].cpu().numpy())
            record['global_feat'].append(cache.global_feat[0].cpu().numpy())
            record['knn_idx'].append(cache.knn_idx[0].cpu().numpy())

            temporal_kw = dict(coords_hist=coords_hist,
                               fitness_hist=fitness_hist, n_valid=n_valid)

            # ── Backbone forward (captures h_node, h_global) ──
            h, e, h_per_head, h_global = backbone.encode(
                cache.node_feat, cache.global_feat, cache, **temporal_kw)

            record['h_node'].append(h[0].cpu().numpy())
            record['h_global'].append(h_global[0].cpu().numpy())
            record['h_per_head'].append(h_per_head[0].cpu().numpy())

            # Attention scores from patched layers
            for li, layer in enumerate(sparse_bb.layers):
                if hasattr(layer, '_last_scores'):
                    record[f'attn_scores_L{li}'].append(layer._last_scores[0].cpu().numpy())
                    record[f'attn_weights_L{li}'].append(layer._last_alpha[0].cpu().numpy())

            # ── Variant step (routing + operator deltas) ──
            delta, extras = variant.step(
                h, h_per_head, h_global, coords, fitness,
                cache, D, M=1, gumbel_tau=1.0)

            record['logits'].append(extras.get('logits', torch.zeros(B, N, 4))[0].cpu().numpy())
            record['routing_probs'].append(extras.get('routing_probs', torch.zeros(B, N, 4))[0].cpu().numpy())
            winner = extras.get('winner', torch.zeros(1, B, N, dtype=torch.long))
            record['winner'].append(winner[0, 0].cpu().numpy())

            # ── Offspring eval + greedy selection ──
            offspring = (coords.unsqueeze(0) + delta).clamp(-100, 100)  # (M, B, N, D)
            off_flat = offspring.reshape(-1, D)
            off_fitness = fn(off_flat).reshape(1, B, N)

            # NoOp: keep parent fitness
            noop_mask = (winner == 3)
            if noop_mask.any():
                off_fitness = torch.where(noop_mask, fitness.unsqueeze(0), off_fitness)

            best_off = offspring[0]
            best_fit = off_fitness[0]
            improved = best_fit < fitness
            new_coords = torch.where(improved.unsqueeze(-1), best_off, coords)
            new_fitness = torch.where(improved, best_fit, fitness)

            record['delta'].append(delta[0, 0].cpu().numpy())
            record['offspring_fitness'].append(off_fitness[0, 0].cpu().numpy())
            record['improved'].append(improved[0].cpu().numpy())

            # ── Counterfactual: evaluate ALL operators' offspring ──
            deltas_k = extras.get('deltas_k')  # (M, B, N, K, D)
            if deltas_k is not None:
                K = deltas_k.shape[3]
                cf_fit = torch.zeros(N, K, dtype=torch.float64)
                cf_imp = torch.zeros(N, K, dtype=torch.bool)
                for k in range(K):
                    if k == 3:  # NoOp
                        cf_fit[:, k] = fitness[0]
                    else:
                        off_k = (coords + deltas_k[0, :, :, k, :]).clamp(-100, 100)  # (B, N, D)
                        cf_fit[:, k] = fn(off_k.reshape(-1, D)).reshape(N)
                    cf_imp[:, k] = cf_fit[:, k] < fitness[0]
                record['cf_fitness'].append(cf_fit.cpu().numpy())
                record['cf_improved'].append(cf_imp.cpu().numpy())

            # Update population
            coords = new_coords
            fitness = new_fitness

            # Early stop
            if (fitness.min() - f_optimal) < 0.1:
                log.info("Converged at gen %d, gap=%.4f", gen, fitness.min().item() - f_optimal)
                break

            if gen % 50 == 0:
                log.info("gen %d/%d | best=%.2f | gap=%.2f",
                         gen, n_gens, fitness.min().item(), fitness.min().item() - f_optimal)

    _unpatch_attention_save(sparse_bb)

    # Convert lists to arrays
    out = {}
    for k, v in record.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
            out[k] = np.stack(v)
        elif isinstance(v, list) and len(v) > 0:
            out[k] = np.array(v)
        else:
            out[k] = v

    return out


def _patch_attention_save(backbone):
    """Patch SparseGATv2Layers to save scores/alpha as attributes after forward."""
    from .sparse_gatv2_layer import gather_neighbors
    import torch.nn.functional as F

    for layer in backbone.layers:
        orig_forward = layer.forward

        def make_patched(layer, orig):
            def patched_forward(x, knn_idx, edge_feat, **kwargs):
                h_out, e_out = orig(x, knn_idx, edge_feat, **kwargs)

                # Recompute scores cheaply (same ops as forward, no grad)
                B, N, H = x.shape
                heads = layer.att.shape[2]
                hd = H // heads
                x_normed = layer.norm(x)
                x_l = layer.lin_l(x_normed).view(B, N, heads, hd)
                x_r = layer.lin_r(x_normed).view(B, N, heads, hd)
                e_lin = layer.lin_edge(edge_feat).view(B, N, knn_idx.shape[2], heads, hd)
                x_r_k = gather_neighbors(x_r, knn_idx)
                msg = F.leaky_relu(x_l.unsqueeze(2) + x_r_k + e_lin, 0.2)
                layer._last_scores = (msg * layer.att).sum(dim=-1)
                layer._last_alpha = F.softmax(layer._last_scores, dim=2)

                return h_out, e_out
            return patched_forward

        layer.forward = make_patched(layer, orig_forward)
        layer._orig_forward = orig_forward


def _unpatch_attention_save(backbone):
    """Restore original forward methods."""
    for layer in backbone.layers:
        if hasattr(layer, '_orig_forward'):
            layer.forward = layer._orig_forward
            del layer._orig_forward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run introspection on L2O model')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--fid', type=int, default=2, help='CEC2017 function ID')
    parser.add_argument('--D', type=int, default=10, help='Dimensionality')
    parser.add_argument('--n-gens', type=int, default=200, help='Number of generations')
    parser.add_argument('--output', default='introspection.npz', help='Output file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    torch.manual_seed(args.seed)
    device = 'cpu'

    # Load model
    from .sparse_temporal_backbone import TemporalSparseGATv2Backbone
    from .sparse_gatv2_backbone import TopologyMode
    from .variants.neural_k4 import NeuralK4Variant
    from .cec2017_torch import CEC2017Torch

    backbone = TemporalSparseGATv2Backbone(
        d_rnn=64, d_temporal=64, gru_window=16,
        node_in=8, edge_in=4, global_in=13,
        gatv2_hidden=128, gatv2_layers=2, n_heads=8,
        global_out_dim=128, dropout=0.0,
        topology_mode=TopologyMode.EMBEDDING_KNN,
        k_neighbors=8, device=device)

    variant = NeuralK4Variant(K=4, head_dim=16, gatv2_hidden=128)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    variant.load_state_dict(ckpt['variant_state_dict'])

    backbone.eval()
    variant.eval()

    fn = CEC2017Torch(args.fid, args.D, device)
    N = 5 * args.D  # pop_per_dim=5

    log.info("Running introspection: F%02d D=%d N=%d n_gens=%d",
             args.fid, args.D, N, args.n_gens)

    result = run_introspection(
        backbone, variant, fn, fn.f_optimal,
        D=args.D, N=N, n_gens=args.n_gens, device=device)

    # Save
    np.savez_compressed(args.output, **{k: v for k, v in result.items()
                                        if isinstance(v, np.ndarray)})

    n_gens_actual = len(result['best_fitness'])
    log.info("Saved %d generations to %s", n_gens_actual, args.output)
    log.info("Final gap: %.4f", result['best_fitness'][-1] - fn.f_optimal)

    # Quick summary
    winners = result['winner']  # (n_gens, N)
    op_names = ['DE/cpbest', 'MTS-LS1', 'CMA-ES', 'NoOp']
    for k in range(4):
        frac = (winners == k).mean()
        log.info("  %s: %.1f%%", op_names[k], frac * 100)
