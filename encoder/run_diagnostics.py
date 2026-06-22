"""
run_diagnostics.py — Collect per-individual, per-generation diagnostic data.

Loads a trained checkpoint, runs eval trajectories on CEC2017 with
diagnostics=True, and saves .npz files for Block 0 analysis.

Usage:
    python -m encoder.run_diagnostics \
        --checkpoint checkpoints/k4_cec_full/best_val.pth \
        --out-dir checkpoints/k4_cec_full/diagnostics \
        --dims 10 --device cuda
"""
import logging
import time
from pathlib import Path
from typing import List, Optional

import torch

log = logging.getLogger(__name__)


def collect_diagnostics(
    checkpoint: Path,
    out_dir: Path,
    dims: List[int],
    fids: Optional[List[int]] = None,
    budget_mult: int = 10000,
    pop_per_dim: int = 5,
    device: str = 'cpu',
    seed: int = 42,
    # Architecture params (default = canonical)
    gatv2_hidden: int = 128,
    n_heads: int = 8,
    global_out_dim: int = 128,
    head_dim: int = 16,
):
    """Run eval trajectories with diagnostic logging.

    Args:
        checkpoint: path to .pth checkpoint
        out_dir: directory for output .npz files
        dims: list of dimensionalities to evaluate
        fids: function IDs (None = all valid for each dim)
        budget_mult: budget = budget_mult * D evaluations
        pop_per_dim: N = pop_per_dim * D
        device: 'cpu' or 'cuda'
        seed: random seed for reproducibility
    """
    from .dense_temporal_backbone import TemporalDenseGATv2Backbone
    from .dense_gatv2_backbone import DenseGATv2Backbone
    from .variants.neural_k4 import NeuralK4Variant
    from .cec2017_torch import CEC2017Torch, get_all_func_ids
    from .train_hybrid import run_trajectory

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Detect backbone type from state_dict keys
    bb_keys = list(ckpt['backbone_state_dict'].keys())
    is_temporal = any(k.startswith('temporal.') for k in bb_keys)

    if is_temporal:
        backbone = TemporalDenseGATv2Backbone(
            d_rnn=64, d_temporal=64, gru_window=16,
            node_in=8, edge_in=4, global_in=16,
            gatv2_hidden=gatv2_hidden, gatv2_layers=2, n_heads=n_heads,
            global_out_dim=global_out_dim, dropout=0.0,
            device=device,
        ).to(device)
    else:
        backbone = DenseGATv2Backbone(
            node_in=8, edge_in=4, global_in=16,
            gatv2_hidden=gatv2_hidden, n_heads=n_heads,
            global_out_dim=global_out_dim,
            gatv2_layers=2, dropout=0.0,
        ).to(device)
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    backbone.eval()

    variant = NeuralK4Variant(
        K=4, head_dim=head_dim, gatv2_hidden=gatv2_hidden,
    ).to(device)
    variant.load_state_dict(ckpt['variant_state_dict'])
    variant.eval()

    total = 0
    for D in dims:
        N = pop_per_dim * D
        n_gens = (budget_mult * D) // N
        func_ids = fids if fids is not None else get_all_func_ids(D)

        for fid in func_ids:
            torch.manual_seed(seed)
            fn = CEC2017Torch(fid, D, device)

            t0 = time.perf_counter()
            with torch.no_grad():
                gc, _, stats = run_trajectory(
                    backbone, variant, fn, fn.f_optimal,
                    D=D, N=N, n_gens=n_gens,
                    bptt_window=n_gens, bptt_segment=-1,
                    diagnostics=True, device=device,
                )

            buf = stats['diag_buffer']
            out_path = out_dir / f'F{fid:02d}_D{D}.npz'
            buf.save(out_path, metadata={
                'fid': fid, 'D': D, 'N': N, 'n_gens': n_gens,
                'gc': gc, 'seed': seed,
                'checkpoint': str(checkpoint),
            })

            elapsed = time.perf_counter() - t0
            total += 1
            log.info("F%02d D%d | gc=%.4f | %d gens | %.1fs | → %s",
                     fid, D, gc, buf.gen_idx, elapsed, out_path.name)

    log.info("Done: %d diagnostic files saved to %s", total, out_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect Block 0 diagnostic data')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to trained .pth checkpoint')
    parser.add_argument('--out-dir', required=True,
                        help='Output directory for .npz files')
    parser.add_argument('--dims', type=int, nargs='+', default=[10])
    parser.add_argument('--fids', type=int, nargs='+', default=None,
                        help='Function IDs (default: all valid)')
    parser.add_argument('--budget-mult', type=int, default=10000)
    parser.add_argument('--pop-per-dim', type=int, default=5)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(name)s %(levelname)s %(message)s')

    collect_diagnostics(
        checkpoint=Path(args.checkpoint),
        out_dir=Path(args.out_dir),
        dims=args.dims,
        fids=args.fids,
        budget_mult=args.budget_mult,
        pop_per_dim=args.pop_per_dim,
        device=args.device,
        seed=args.seed,
    )
