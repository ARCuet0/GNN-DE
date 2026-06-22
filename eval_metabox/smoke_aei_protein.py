"""Smoke the AEI pipeline for GNN-DE on MetaBox protein-docking.

Runs GNN-DE and the Random_search reference on N protein instances x S seeds,
assembles the MetaBox `results` dict, and scores with the REAL Basic_Logger
AEI machinery (get_random_baseline + aei_cost + cal_aei). Validates the whole
comparison pipeline end to end before committing to the 51-seed protocol.

The AEI reported here is the "best objective value AEI" (cost arm) = what the
RLDE-AFL paper's Fig 4 uses. It is normalized against Random_search, so an AEI
for GNN-DE alone is impossible (the reference must be in the same run).

Usage:
    python eval_metabox/smoke_aei_protein.py --n-instances 8 --seeds 2
    python eval_metabox/smoke_aei_protein.py --n-instances 280 --seeds 3 \
        --out eval_metabox/results/aei_smoke_280x3.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TERSQ_ROOT = os.path.dirname(_HERE)
if _TERSQ_ROOT not in sys.path:
    sys.path.insert(0, _TERSQ_ROOT)

from types import SimpleNamespace

from eval_metabox._metabox_compat import (
    get_basic_logger, get_random_search, load_protein_testset)
from eval_metabox.gnnde_optimizer import GNN_DE


def make_cfg(maxFEs=500, n_logpoint=50, device='cpu'):
    return SimpleNamespace(
        maxFEs=maxFEs, n_logpoint=n_logpoint,
        log_interval=max(1, maxFEs // n_logpoint),
        device=device, full_meta_data=False,
        test_problem='protein',  # drives the cal_aei protein std x5 branch
    )


def _run_agent(opt, problem, seed):
    opt.seed(seed)
    res = opt.run_episode(problem)
    return np.asarray(res['cost'], dtype=np.float64), float(res['fes'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-instances', type=int, default=8,
                    help='number of protein instances (<=280; default 8 for a quick smoke)')
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--maxFEs', type=int, default=500)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = make_cfg(maxFEs=args.maxFEs, device=args.device)
    Random_search = get_random_search()

    instances = load_protein_testset('numpy', 'all')[:args.n_instances]
    print(f"[smoke] {len(instances)} instances x {args.seeds} seeds, maxFEs={args.maxFEs}",
          flush=True)

    gnnde = GNN_DE(cfg)
    rs = Random_search(cfg)

    AGENTS = ['GNN_DE', 'Random_search']
    cost = {}     # cost[pname][agent] = (seeds, record)
    fes = {}      # fes[pname][agent]  = (seeds,)
    T1 = {}
    T2 = {}

    t_start = time.perf_counter()
    for i, prob in enumerate(instances):
        pname = str(prob)
        cost[pname] = {}
        fes[pname] = {}
        T1[pname] = {}
        T2[pname] = {}
        for agent_name, opt in (('GNN_DE', gnnde), ('Random_search', rs)):
            curves = []
            fess = []
            for s in range(args.seeds):
                c, f = _run_agent(opt, prob, s)
                curves.append(c)
                fess.append(f)
            cost[pname][agent_name] = np.stack(curves)         # (seeds, record)
            fes[pname][agent_name] = np.asarray(fess)          # (seeds,)
            # Dummy-but-valid complexity timings (cost-arm AEI does not use them;
            # they only keep get_random_baseline finite). T2 > T1 > 0, T0 > 0.
            T1[pname][agent_name] = 1.0
            T2[pname][agent_name] = 2.0
        if (i + 1) % 20 == 0 or i + 1 == len(instances):
            el = time.perf_counter() - t_start
            print(f"[smoke] {i + 1}/{len(instances)} done ({el:.0f}s)", flush=True)

    results = {'T0': 1.0, 'T1': T1, 'T2': T2, 'fes': fes, 'cost': cost}

    # ---- REAL MetaBox AEI machinery -------------------------------------
    Basic_Logger = get_basic_logger()
    logger = Basic_Logger(cfg)
    baseline = logger.get_random_baseline(results, args.maxFEs)
    results_cost, aei_cost_mean, aei_cost_std = logger.aei_cost(results['cost'], baseline)

    # Per-instance head-to-head on final best (lower energy = better).
    wins = ties = losses = 0
    finals = {a: [] for a in AGENTS}
    for pname in cost:
        g = cost[pname]['GNN_DE'][:, -1].mean()
        r = cost[pname]['Random_search'][:, -1].mean()
        finals['GNN_DE'].append(g)
        finals['Random_search'].append(r)
        if g < r - 1e-9:
            wins += 1
        elif g > r + 1e-9:
            losses += 1
        else:
            ties += 1

    summary = {
        'n_instances': len(instances),
        'seeds': args.seeds,
        'maxFEs': args.maxFEs,
        'aei_cost_mean': {k: float(v) for k, v in aei_cost_mean.items()},
        'aei_cost_std': {k: float(v) for k, v in aei_cost_std.items()},
        'baseline_cost_avg': float(baseline['cost_avg']),
        'baseline_cost_std': float(baseline['cost_std']),
        'gnnde_vs_rs_final_best': {'wins': wins, 'ties': ties, 'losses': losses},
        'mean_final_best': {k: float(np.mean(v)) for k, v in finals.items()},
        'wall_seconds': time.perf_counter() - t_start,
    }

    print("\n==== AEI smoke summary ====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nNOTE: AEI here is normalized vs THIS run's Random_search, so it is "
          "comparable across the agents in this run, NOT directly to the paper's "
          "published bars (those need the same RS reference + the published "
          "baselines + the resolved bounds/FES/AEI-definition).", flush=True)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[smoke] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
