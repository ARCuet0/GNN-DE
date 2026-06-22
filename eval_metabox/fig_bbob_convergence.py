"""Convergence figure: GNN-DE vs the RLDE-AFL paper algorithms on MetaBox BBOB
difficult-D10 (16 held-out functions), from run_metabox_bbob.py outputs.

A 4x4 grid (one panel per function). Each panel overlays the per-FE median
best-so-far gap (normalized by each run's first recorded gap) for every algorithm,
with a shaded IQR band. GNN-DE and RLDE-AFL are drawn thicker. All runs share the
same instance_seed=3849 realization, 20000 FES, 51 seeds.

Caveat (print + caption): single fixed instance per function, not the COCO
15-instance protocol — this shows optimizer-init variance on one landscape.

Usage:
    python fig_bbob_convergence.py --dir eval_metabox/results/bbob_metabox \
        --out eval_metabox/results/fig_bbob_convergence.png
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

GAP_FLOOR = 1e-12
N_GRID = 200

# order: GNN-DE + RLDE-AFL highlighted; classic grey-ish; learned colored
ORDER = ['GNN_DE', 'RLDEAFL', 'DE', 'MADDE', 'NLSHADELBC', 'JDE21', 'Random_search',
         'DEDDQN', 'DEDQN', 'LDE', 'RLHPSDE', 'GLEET', 'RLDAS']
THICK = {'GNN_DE'}
COL = {
    'GNN_DE': '#1f4e9c', 'RLDEAFL': '#8B3BB2',
    'DE': '#9aa0a6', 'MADDE': '#6b6f73', 'NLSHADELBC': '#b0b4b8',
    'JDE21': '#4d5054', 'Random_search': '#cfd2d5',
    'DEDDQN': '#d1622b', 'DEDQN': '#e0962b', 'LDE': '#3b9b5c',
    'RLHPSDE': '#2bb0a3', 'GLEET': '#b23b3b', 'RLDAS': '#7a5cc2',
}
LABEL = {'GNN_DE': 'GNN-DE', 'RLDEAFL': 'RLDE-AFL', 'NLSHADELBC': 'NL-SHADE-LBC',
         'Random_search': 'Random', 'MADDE': 'MadDE'}


def load(d, algo):
    p = os.path.join(d, f'{algo}.json')
    if not os.path.isfile(p):
        return None
    return json.load(open(p))['data']


def seeds_to_band(curves, grid):
    """ABSOLUTE gap-to-optimum interpolated onto the FES grid (lower = closer to
    the optimum), so curves are directly comparable across methods within a panel."""
    mat = np.full((len(curves), len(grid)), np.nan)
    for i, gap in enumerate(curves):
        gap = np.asarray(gap, float)
        gap = np.minimum.accumulate(np.clip(gap, GAP_FLOOR, None))
        if gap[0] <= 0:
            continue
        # curves are at uniform FES checkpoints 0..maxFEs (len = n_logpoint+1)
        fes = np.linspace(grid[0], grid[-1], len(gap))
        idx = np.searchsorted(fes, grid, side='right') - 1
        valid = idx >= 0
        mat[i, valid] = gap[idx[valid]]
    with np.errstate(invalid='ignore'):
        return np.nanmedian(mat, 0), np.nanpercentile(mat, 25, 0), np.nanpercentile(mat, 75, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--maxFEs', type=int, default=20000)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    data = {a: load(args.dir, a) for a in ORDER}
    data = {a: d for a, d in data.items() if d is not None}
    present = [a for a in ORDER if a in data]
    if not present:
        raise SystemExit(f"no algorithm JSONs under {args.dir}")
    funcs = sorted(next(iter(data.values())).keys())
    grid = np.logspace(np.log10(50), np.log10(args.maxFEs), N_GRID)

    n = len(funcs)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.0 * nrow))
    axes = np.atleast_2d(axes)
    for k, fn in enumerate(funcs):
        ax = axes[k // ncol, k % ncol]
        for a in present:
            if fn not in data[a]:
                continue
            curves = data[a][fn]['curves']
            med, lo, hi = seeds_to_band(curves, grid)
            if np.isnan(med).all():
                continue
            lw = 2.1 if a in THICK else 1.0
            z = 5 if a in THICK else 2
            if a in THICK:
                ax.fill_between(grid, lo, hi, color=COL[a], alpha=0.12, lw=0, zorder=z - 1)
            ax.plot(grid, med, color=COL[a], lw=lw, zorder=z, label=LABEL.get(a, a))
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(grid[0], grid[-1])
        ax.set_title(fn, fontsize=8.5, pad=2)
        ax.tick_params(labelsize=7)
        ax.grid(True, which='major', ls=':', lw=0.4, alpha=0.5)
    for k in range(n, nrow * ncol):
        axes[k // ncol, k % ncol].axis('off')
    for r in range(nrow):
        axes[r, 0].set_ylabel('gap to optimum (median, IQR)', fontsize=8)
    for c in range(ncol):
        axes[nrow - 1, c].set_xlabel('FES', fontsize=8)

    handles = [Line2D([0], [0], color=COL[a], lw=2.2 if a in THICK else 1.4,
                      label=LABEL.get(a, a)) for a in present]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(handles), 7), fontsize=8.5, frameon=False)
    fig.suptitle('Convergence on BBOB difficult-D10 (16 held-out functions, 51 seeds, '
                 'single instance/func) — GNN-DE vs the MetaBox BBOB roster', fontsize=11, y=0.998)
    fig.tight_layout(rect=(0.01, 0.04, 1.0, 0.97))

    out = args.out or os.path.join(args.dir, 'fig_bbob_convergence.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    fig.savefig(out.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"wrote {out} ({len(present)} algos: {present})")


if __name__ == '__main__':
    main()
