"""
smac_sparse_topology.py — SMAC3 comparison of sparse topology strategies.

Optimizes topology choice + hyperparameters, all starting from the same
pretrained checkpoint. Objective: maximize validation gap closure.

Usage:
    python -m encoder.smac_sparse_topology \
        --checkpoint checkpoints/k4_cec_full/best_val.pth \
        --device cuda --n-trials 30 --train-steps 200
"""
import logging
import time
from pathlib import Path

import numpy as np
import torch

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)
from smac import HyperparameterOptimizationFacade, Scenario

log = logging.getLogger(__name__)


def train_and_evaluate(config: Configuration, seed: int = 0,
                       checkpoint: str = '', device: str = 'cuda',
                       train_steps: int = 200, budget_mult: int = 10000) -> float:
    """Train with given config and return negative validation gc.

    SMAC minimizes, so we return -val_gc.
    """
    from .sparse_temporal_backbone import TemporalSparseGATv2Backbone
    from .sparse_gatv2_backbone import TopologyMode
    from .similarity_graph_gpu import build_sparse_graphs_gpu
    from .variants.neural_k4 import NeuralK4Variant
    from .train_hybrid import train_hybrid

    torch.manual_seed(seed)

    topo_map = {
        'coordinate': TopologyMode.COORDINATE_KNN,
        'embedding': TopologyMode.EMBEDDING_KNN,
        'learned': TopologyMode.LEARNED_SCORER,
    }

    topology = config['topology']
    k = config['k_neighbors']
    lr = config['lr']

    # Build backbone
    backbone = TemporalSparseGATv2Backbone(
        d_rnn=64, d_temporal=64, gru_window=16,
        node_in=8, edge_in=4, global_in=13,
        gatv2_hidden=128, gatv2_layers=2, n_heads=8,
        global_out_dim=128, dropout=0.1,
        topology_mode=topo_map[topology],
        k_neighbors=k,
        device=device,
    ).to(device)

    variant = NeuralK4Variant(
        K=4, head_dim=16, gatv2_hidden=128,
    ).to(device)

    # Load pretrained weights (strict=False: LearnedScorer adds q_proj/k_proj)
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt['backbone_state_dict'], strict=False)
        if 'variant_state_dict' in ckpt:
            variant.load_state_dict(ckpt['variant_state_dict'])

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        history = train_hybrid(
            backbone, variant,
            n_steps=train_steps,
            budget_mult=budget_mult,
            pop_per_dim=5,
            dims=[10],
            M_es=2,
            sigma=0.01,
            bptt_window=80,
            lr=lr,
            lambda_es=0.1,
            max_grad_norm=1.0,
            graph_builder=build_sparse_graphs_gpu,
            device=device,
            save_dir=tmpdir,
            save_every=0,
            log_every=50,
            val_every=train_steps,  # validate only at the end
            patience=train_steps + 1,  # no early stopping
        )

    # Objective: mean (f_best - f*) across validation functions
    # Lower = better. SMAC minimizes, so return directly.
    final_step = history[-1] if history else {}
    val_results = final_step.get('val_results', [])

    if val_results:
        gaps = [r['final_best'] - r['f_optimal'] for r in val_results]
        mean_gap = float(np.mean(gaps))
    else:
        mean_gap = 1e10  # failure fallback

    log.info("SMAC trial: topology=%s k=%d lr=%.1e | mean_gap=%.4f (n_funcs=%d)",
             topology, k, lr, mean_gap, len(val_results))

    return mean_gap


def build_configspace() -> ConfigurationSpace:
    cs = ConfigurationSpace(seed=42)
    cs.add([
        Categorical('topology', ['coordinate', 'embedding', 'learned'],
                     default='coordinate'),
        Integer('k_neighbors', (4, 16), default=8),
        Float('lr', (1e-5, 1e-3), default=3e-4, log=True),
    ])
    return cs


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='SMAC3 comparison of sparse topology strategies')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n-trials', type=int, default=30)
    parser.add_argument('--train-steps', type=int, default=200)
    parser.add_argument('--budget-mult', type=int, default=10000)
    parser.add_argument('--output-dir', default='smac_output/sparse_topology')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    cs = build_configspace()

    scenario = Scenario(
        configspace=cs,
        deterministic=False,
        n_trials=args.n_trials,
        output_directory=Path(args.output_dir),
        seed=42,
    )

    # Wrap target function with fixed args
    def target(config, seed=0):
        return train_and_evaluate(
            config, seed=seed,
            checkpoint=args.checkpoint,
            device=args.device,
            train_steps=args.train_steps,
            budget_mult=args.budget_mult,
        )

    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=target,
    )

    incumbent = smac.optimize()

    log.info("=" * 60)
    log.info("SMAC RESULT — Best configuration:")
    log.info("  topology:    %s", incumbent['topology'])
    log.info("  k_neighbors: %d", incumbent['k_neighbors'])
    log.info("  lr:          %.1e", incumbent['lr'])
    log.info("=" * 60)

    # Print all evaluated configs ranked by performance
    rh = smac.runhistory
    configs_costs = []
    for run_key in rh:
        config = rh.get_config(run_key.config_id)
        cost = rh.get_cost(config)
        configs_costs.append((config, cost))

    configs_costs.sort(key=lambda x: x[1])
    log.info("\nAll configurations ranked by mean(f_best - f*):")
    for i, (cfg, cost) in enumerate(configs_costs):
        log.info("  %2d. gap=%.4f | topology=%-11s k=%-2d lr=%.1e",
                 i + 1, cost, cfg['topology'], cfg['k_neighbors'], cfg['lr'])
