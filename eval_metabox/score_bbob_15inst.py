"""15-instance BBOB W/L/T aggregator — per-instance net distribution.

Robustness check for the MetaBox BBOB difficult-D10 table. The published table is a
single landscape realization (instance_seed=3849). This re-runs the SAME validated
per-function Mann-Whitney U tally (GNN-DE vs each baseline, 51 vs 51 seeds, two-sided,
TIE when both medians < TAU or p >= ALPHA) on each instance independently, then
reports, per method, the DISTRIBUTION of net = W - L across the 15 instances
(min / median / max) and, per (method, function), in how many instances GNN-DE wins.

No scale-mixing: each instance is scored on its own seeds, so different landscape
difficulties never pool into one test. Answers "does the single-instance net hold
across COCO-style instance realizations, or is it a single-instance artifact?"

Usage:
    python score_bbob_15inst.py \
        --root eval_metabox/results/bbob_15inst \
        --legacy-dir eval_metabox/results/bbob_metabox \
        --out eval_metabox/results/bbob_15inst_WLT.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.stats import mannwhitneyu

TAU = 1e-4
ALPHA = 0.05
ORDER = ['DE', 'MADDE', 'NLSHADELBC', 'JDE21', 'Random_search',
         'DEDDQN', 'DEDQN', 'LDE', 'RLHPSDE', 'GLEET', 'RLDAS', 'RLDEAFL']


def _load_algo(d, algo):
    p = os.path.join(d, f'{algo}.json')
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    return {fn: np.asarray(v['finals'], float) for fn, v in j['data'].items()}


def _verdict(g, x):
    """Per-function verdict: 'W' (GNN-DE better), 'L' (worse), 'T' (tie)."""
    gm, xm = float(np.median(g)), float(np.median(x))
    if gm < TAU and xm < TAU:
        return 'T'
    try:
        p = float(mannwhitneyu(g, x, alternative='two-sided').pvalue)
    except Exception:
        p = 1.0
    if p < ALPHA and gm < xm:
        return 'W'
    if p < ALPHA and xm < gm:
        return 'L'
    return 'T'


def _instance_dirs(root, legacy_dir=None):
    """Ordered list of (label, dirpath): inst_* under root, then legacy as one more."""
    dirs = []
    for d in sorted(glob.glob(os.path.join(root, 'inst_*')),
                    key=lambda p: int(os.path.basename(p).split('_')[1])):
        dirs.append((os.path.basename(d), d))
    if legacy_dir and os.path.isdir(legacy_dir):
        dirs.append(('inst_3849', legacy_dir))
    return dirs


def aggregate(root, roster=None, legacy_dir=None):
    roster = roster or ORDER
    dirs = _instance_dirs(root, legacy_dir)
    per_algo = {}
    for algo in roster:
        nets, tallies, perfn = [], [], {}
        for label, d in dirs:
            gnn = _load_algo(d, 'GNN_DE')
            base = _load_algo(d, algo)
            if gnn is None or base is None:
                continue
            funcs = sorted(set(gnn) & set(base))
            w = t = l = 0
            for fn in funcs:
                v = _verdict(gnn[fn], base[fn])
                slot = perfn.setdefault(fn, {'W': 0, 'T': 0, 'L': 0, 'n_instances': 0})
                slot[v] += 1
                slot['n_instances'] += 1
                if v == 'W':
                    w += 1
                elif v == 'L':
                    l += 1
                else:
                    t += 1
            nets.append(w - l)
            tallies.append({'instance': label, 'W': w, 'T': t, 'L': l, 'net': w - l})
        if not nets:
            continue
        arr = np.asarray(nets, float)
        per_algo[algo] = {
            'n_instances': len(nets),
            'nets': nets,
            'net_min': int(min(nets)),
            'net_median': float(np.median(arr)),
            'net_max': int(max(nets)),
            'net_mean': float(np.mean(arr)),
            'per_instance': tallies,
            'per_function': perfn,
        }
    return {'reference': 'GNN_DE', 'test': 'mannwhitneyu_unpaired',
            'tau': TAU, 'alpha': ALPHA, 'aggregation': 'per_instance_net_distribution',
            'instances': [lbl for lbl, _ in dirs], 'per_algorithm': per_algo}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--legacy-dir', default=None,
                    help="single-instance dir (instance_seed=3849) to count as one more instance")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    res = aggregate(args.root, legacy_dir=args.legacy_dir)
    n_inst = len(res['instances'])
    print(f"GNN-DE vs roster on BBOB difficult-D10, {n_inst} instances "
          f"(per-instance MWU, net distribution)")
    print(f"{'algorithm':<14} {'net med':>8} {'net min':>8} {'net max':>8}  nets")
    print('-' * 78)
    for algo in ORDER:
        a = res['per_algorithm'].get(algo)
        if a is None:
            print(f"{algo:<14} {'(missing)':>8}")
            continue
        print(f"{algo:<14} {a['net_median']:>8.1f} {a['net_min']:>8d} "
              f"{a['net_max']:>8d}  {a['nets']}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(res, open(args.out, 'w'), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
