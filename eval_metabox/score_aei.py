"""LOCAL scorer for the protein-docking comparison (runs off the cluster).

Loads the raw shard JSONs produced by run_protein_raw.py, assembles the MetaBox
results dict, and computes:
  * the cost-arm AEI (the RLDE-AFL Fig-4 metric) via the REAL MetaBox
    Basic_Logger.get_random_baseline + aei_cost + cal_aei (needs pandas/matplotlib,
    hence local-only), normalized vs Random_search, protein std x5;
  * per-instance W/T/L of GNN-DE vs each baseline on mean final best, plus a
    Wilcoxon signed-rank p-value across the 280 instance means.

Uses per-seed FINAL best (reshaped to (seeds, 1)) for the cost arm: aei_cost reads
only [:, -1], and MadDE/NL-SHADE-LBC return short (len-3) curves that cannot be
stacked alongside the len-51 ones. The full curves are preserved in the shards.

Usage:
    python eval_metabox/score_aei.py --shards-dir eval_metabox/results/protein_real_51s \
        --out eval_metabox/results/protein_real_51s_AEI.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TERSQ_ROOT = os.path.dirname(_HERE)
if _TERSQ_ROOT not in sys.path:
    sys.path.insert(0, _TERSQ_ROOT)

from types import SimpleNamespace

from eval_metabox._metabox_compat import get_basic_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards-dir', required=True)
    ap.add_argument('--extra-shards-dirs', nargs='*', default=[],
                    help='additional shard dirs (e.g. RLDE-AFL) to merge in by pid; '
                         'their agents are added to each instance alongside the primary set')
    ap.add_argument('--maxFEs', type=int, default=500)
    ap.add_argument('--reference', default='Random_search',
                    help='agent used as the AEI normalization baseline')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import re
    # Primary shards keyed by complex name (e.g. 1PPE_1); order across shards
    # (shard number ascending, insertion order within) = global array_split order.
    shard_files = sorted(
        glob.glob(os.path.join(args.shards_dir, 'shard_*.json')),
        key=lambda p: int(re.search(r'shard_(\d+)', p).group(1)))
    if not shard_files:
        sys.exit(f"no shard_*.json under {args.shards_dir}")

    # Global index -> instance-name map, so numeric-keyed extra shards (RLDE-AFL
    # uses 0..279) can be remapped onto the primary's name keys.
    idx_to_name = []
    for sf in shard_files:
        idx_to_name.extend(list(json.load(open(sf))['data'].keys()))

    # Merge shards.
    cost, fes, T1, T2 = {}, {}, {}, {}
    agents = None
    seeds = None
    for sf in shard_files:
        d = json.load(open(sf))
        if agents is None:
            agents = list(d['meta']['agents'])
            seeds = d['meta']['seeds']
        for pid, ag in d['data'].items():
            cost[pid], fes[pid], T1[pid], T2[pid] = {}, {}, {}, {}
            for a, v in ag.items():
                finals = np.asarray(v['finals'], dtype=np.float64)   # (seeds,)
                cost[pid][a] = finals.reshape(-1, 1)                  # (seeds, 1); [:, -1] = final
                fes[pid][a] = np.asarray(v['fes'], dtype=np.float64)  # (seeds,)
                T1[pid][a] = 1.0                                      # dummy, finite; cost arm ignores
                T2[pid][a] = 2.0

    # Merge extra shard dirs (e.g. RLDE-AFL) into the same per-pid dicts.
    for extra in args.extra_shards_dirs:
        efiles = sorted(glob.glob(os.path.join(extra, 'shard_*.json')))
        if not efiles:
            print(f"[score] WARNING: no shards under {extra}", flush=True)
            continue
        added = set()
        for sf in efiles:
            d = json.load(open(sf))
            for pid, ag in d['data'].items():
                # Remap numeric pid (global instance index) -> primary's name key.
                if pid not in cost and pid.isdigit() and int(pid) < len(idx_to_name):
                    pid = idx_to_name[int(pid)]
                if pid not in cost:
                    print(f"[score] WARNING: extra pid {pid} not in primary set, skipping",
                          flush=True)
                    continue
                for a, v in ag.items():
                    cost[pid][a] = np.asarray(v['finals'], dtype=np.float64).reshape(-1, 1)
                    fes[pid][a] = np.asarray(v['fes'], dtype=np.float64)
                    T1[pid][a] = 1.0
                    T2[pid][a] = 2.0
                    added.add(a)
        for a in added:
            if a not in agents:
                agents.append(a)
        print(f"[score] merged extra dir {extra}: agents {sorted(added)}", flush=True)
    n_inst = len(cost)
    print(f"[score] merged {len(shard_files)} shards -> {n_inst} instances, "
          f"agents={agents}, seeds={seeds}", flush=True)
    if n_inst != 280:
        print(f"[score] WARNING: expected 280 instances, got {n_inst} "
              f"(missing/failed shards?)", flush=True)

    results = {'T0': 1.0, 'T1': T1, 'T2': T2, 'fes': fes, 'cost': cost}

    # ---- REAL MetaBox cost-arm AEI ----
    cfg = SimpleNamespace(test_problem='protein')
    logger = get_basic_logger()(cfg)
    baseline = logger.get_random_baseline(results, args.maxFEs)
    _, aei_mean, aei_std = logger.aei_cost(results['cost'], baseline)

    # ---- per-instance W/T/L + Wilcoxon (GNN-DE vs each baseline) ----
    from scipy.stats import wilcoxon
    pids = sorted(cost.keys())
    mean_final = {a: np.array([cost[p][a][:, -1].mean() for p in pids]) for a in agents}
    head = {}
    ref_for_wtl = 'GNN_DE'
    for a in agents:
        if a == ref_for_wtl:
            continue
        g = mean_final[ref_for_wtl]
        b = mean_final[a]
        wins = int(np.sum(g < b - 1e-9))     # GNN-DE better (lower energy)
        losses = int(np.sum(g > b + 1e-9))
        ties = int(n_inst - wins - losses)
        try:
            p = float(wilcoxon(g, b).pvalue)
        except Exception as e:
            p = float('nan')
        head[a] = {'gnnde_wins': wins, 'ties': ties, 'gnnde_losses': losses,
                   'wilcoxon_p': p}

    # ---- RLDE-AFL head-to-head (if present): RLDE-AFL vs each other agent ----
    rlde_head = {}
    if 'RLDE_AFL' in agents:
        for a in agents:
            if a == 'RLDE_AFL':
                continue
            r = mean_final['RLDE_AFL']
            b = mean_final[a]
            wins = int(np.sum(r < b - 1e-9))     # RLDE-AFL better (lower energy)
            losses = int(np.sum(r > b + 1e-9))
            ties = int(n_inst - wins - losses)
            try:
                p = float(wilcoxon(r, b).pvalue)
            except Exception:
                p = float('nan')
            rlde_head[a] = {'rlde_wins': wins, 'ties': ties, 'rlde_losses': losses,
                            'wilcoxon_p': p}

    summary = {
        'n_instances': n_inst, 'seeds': seeds, 'maxFEs': args.maxFEs,
        'agents': agents,
        'aei_cost_mean': {k: float(v) for k, v in aei_mean.items()},
        'aei_cost_std': {k: float(v) for k, v in aei_std.items()},
        'mean_final_best_over_instances': {a: float(mean_final[a].mean()) for a in agents},
        'gnnde_head_to_head': head,
        'rlde_head_to_head': rlde_head,
        'baseline_cost_avg': float(baseline['cost_avg']),
        'reference_agent': args.reference,
    }
    print("\n==== Protein-docking AEI (cost arm, vs Random_search) ====", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(summary, open(args.out, 'w'), indent=2)
        print(f"[score] wrote {args.out}", flush=True)


if __name__ == '__main__':
    main()
