"""W/L/T of GNN-DE vs every paper algorithm on MetaBox BBOB difficult-D10.

Reads the per-algorithm JSONs produced by run_metabox_bbob.py (all on the SAME 16
held-out functions, instance_seed=3849, 20000 FES, 51 seeds) and computes, per
algorithm, the per-function GNN-DE-vs-baseline tally.

Test: two-sided Mann-Whitney U (UNPAIRED). The two sides are independent samples
(different population sizes and dynamics, per-index correlation ~0), so a paired
Wilcoxon is unjustified — same correction applied to the RLDE-AFL BBOB audit and
the CEC17 RLDE-AFL comparison. TIE when both medians < TAU (both effectively
solved) or p >= alpha. Lower gap = better.

Single fixed instance per function (instance_seed=3849): the 51 seeds vary
optimizer initialization on one landscape realization, NOT the COCO 15-instance
protocol. State this caveat with any result.

Usage:
    python score_bbob_wilcoxon.py --dir eval_metabox/results/bbob_metabox \
        --out eval_metabox/results/bbob_metabox_WLT.json
"""
import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import mannwhitneyu

TAU = 1e-4
ALPHA = 0.05

# Paper roster order (classic then learned), GNN-DE is the reference.
ORDER = ['DE', 'MADDE', 'NLSHADELBC', 'JDE21', 'Random_search',
         'DEDDQN', 'DEDQN', 'LDE', 'RLHPSDE', 'GLEET', 'RLDAS', 'RLDEAFL']


def load_algo(d, algo):
    p = os.path.join(d, f'{algo}.json')
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    return {fn: np.asarray(v['finals'], float) for fn, v in j['data'].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    gnn = load_algo(args.dir, 'GNN_DE')
    if gnn is None:
        sys.exit('GNN_DE.json missing')
    funcs = sorted(gnn.keys())
    print(f"GNN-DE vs paper algorithms on BBOB difficult-D10 "
          f"({len(funcs)} functions, 51 seeds, Mann-Whitney U, single instance/func)")
    print(f"{'algorithm':<14} {'W/T/L':>10} {'net':>5}   per-function (W=GNN-DE better)")
    print('-' * 80)

    summary = {}
    for algo in ORDER:
        b = load_algo(args.dir, algo)
        if b is None:
            print(f"{algo:<14} {'(missing)':>10}")
            continue
        w = t = l = 0
        wins, losses = [], []
        for fn in funcs:
            if fn not in b:
                continue
            g, x = gnn[fn], b[fn]
            gm, xm = float(np.median(g)), float(np.median(x))
            if gm < TAU and xm < TAU:
                t += 1; continue
            try:
                p = float(mannwhitneyu(g, x, alternative='two-sided').pvalue)
            except Exception:
                p = 1.0
            if p < ALPHA and gm < xm:
                w += 1; wins.append(fn)
            elif p < ALPHA and xm < gm:
                l += 1; losses.append(fn)
            else:
                t += 1
        net = w - l
        summary[algo] = {'W': w, 'T': t, 'L': l, 'net': net,
                         'gnnde_wins': wins, 'gnnde_losses': losses}
        print(f"{algo:<14} {f'{w}/{t}/{l}':>10} {net:>+5}   "
              f"losses@{losses if losses else '-'}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({'reference': 'GNN_DE', 'test': 'mannwhitneyu_unpaired',
                   'tau': TAU, 'alpha': ALPHA, 'n_functions': len(funcs),
                   'caveat': 'single fixed instance per function (instance_seed=3849)',
                   'per_algorithm': summary}, open(args.out, 'w'), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
