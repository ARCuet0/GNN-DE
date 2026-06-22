"""Magerit-side RAW runner for the protein-docking comparison.

Runs the requested agents on a SHARD of the 280 protein instances x S seeds and
dumps RAW per-seed results (final best + full cost curve + fes). It does NOT
import the MetaBox logger / compute AEI (the tersq.sif container lacks
pandas/matplotlib). AEI scoring + Wilcoxon happen LOCALLY via score_aei.py after
the shard JSONs are rsynced back. This keeps the heavy compute on the cluster and
the (millisecond) scoring + plotting off the container.

Granular storage (user pref): every seed's final cost, fes, and full curve are
saved, so AEI / FES-curve / Wilcoxon can be recomputed under any definition.

Usage (one SLURM array task):
    python eval_metabox/run_protein_raw.py --shard-id $SLURM_ARRAY_TASK_ID \
        --n-shards 28 --seeds 51 --maxFEs 500 \
        --agents GNN_DE Random_search MadDE NLSHADELBC \
        --out eval_metabox/results/protein_real_51s/shard_${SLURM_ARRAY_TASK_ID}.json
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
    _import, get_basic_optimizer, get_random_search, load_protein_testset)


def make_cfg(maxFEs, device):
    return SimpleNamespace(
        maxFEs=maxFEs, n_logpoint=50, log_interval=max(1, maxFEs // 50),
        device=device, full_meta_data=False, test_problem='protein')


def build_agent(name, cfg, ckpt):
    """Fresh agent instance (avoids any cross-episode state on the optimizer
    object; the GNN-DE model itself is cached module-level inside the wrapper)."""
    if name == 'GNN_DE':
        from eval_metabox.gnnde_optimizer import GNN_DE
        if ckpt:
            cfg = SimpleNamespace(**{**vars(cfg), 'gnnde_ckpt': ckpt})
        return GNN_DE(cfg)
    if name == 'Random_search':
        return get_random_search()(cfg)
    if name == 'MadDE':
        return _import('baseline.bbo.madde').MADDE(cfg)
    if name == 'NLSHADELBC':
        return _import('baseline.bbo.nlshadelbc').NLSHADELBC(cfg)
    raise ValueError(f'unknown agent {name!r}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard-id', type=int, required=True)
    ap.add_argument('--n-shards', type=int, required=True)
    ap.add_argument('--seeds', type=int, default=51)
    ap.add_argument('--maxFEs', type=int, default=500)
    ap.add_argument('--agents', nargs='+',
                    default=['GNN_DE', 'Random_search', 'MadDE', 'NLSHADELBC'])
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--ckpt', default=None, help='override deployed ckpt path')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    get_basic_optimizer()  # pre-load skeleton packages once
    cfg = make_cfg(args.maxFEs, args.device)

    all_instances = load_protein_testset('numpy', 'all')
    shards = np.array_split(np.arange(len(all_instances)), args.n_shards)
    my_idx = shards[args.shard_id]
    my_instances = [all_instances[i] for i in my_idx]
    print(f"[shard {args.shard_id}/{args.n_shards}] {len(my_instances)} instances "
          f"x {args.seeds} seeds, agents={args.agents}, maxFEs={args.maxFEs}", flush=True)

    data = {}
    t0 = time.perf_counter()
    for j, prob in enumerate(my_instances):
        pid = str(prob)
        data[pid] = {}
        for agent_name in args.agents:
            finals, fess, curves = [], [], []
            for s in range(args.seeds):
                opt = build_agent(agent_name, cfg, args.ckpt)
                opt.seed(s)
                res = opt.run_episode(prob)
                curve = [float(x) for x in res['cost']]
                finals.append(float(curve[-1]))
                fess.append(int(res['fes']))
                curves.append(curve)
            data[pid][agent_name] = {'finals': finals, 'fes': fess, 'curves': curves}
        el = time.perf_counter() - t0
        print(f"[shard {args.shard_id}] {j + 1}/{len(my_instances)} {pid} done ({el:.0f}s)",
              flush=True)

    out = {
        'meta': {
            'shard_id': args.shard_id, 'n_shards': args.n_shards,
            'instance_ids': [str(p) for p in my_instances],
            'seeds': args.seeds, 'maxFEs': args.maxFEs, 'agents': args.agents,
            'device': args.device, 'wall_seconds': time.perf_counter() - t0,
        },
        'data': data,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f"[shard {args.shard_id}] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
