"""Merge per-function GNN-DE shard JSONs into one per-instance GNN_DE.json.

The CPU GNN-DE run shards by (instance, function) to dodge the GPU-slot bottleneck:
each task writes inst_<seed>/parts/GNN_DE_f<idx>.json holding ONE function's data.
This collapses those parts back into inst_<seed>/GNN_DE.json with the same schema
run_metabox_bbob.py would have produced (so score_bbob_15inst.py reads it unchanged).

Usage:
    python merge_gnnde_parts.py --root eval_metabox/results/bbob_15inst
"""
import argparse
import glob
import json
import os


def merge_instance(inst_dir):
    """Combine inst_dir/parts/GNN_DE_f*.json -> inst_dir/GNN_DE.json. Returns
    the number of functions merged (0 if no parts)."""
    parts = sorted(glob.glob(os.path.join(inst_dir, 'parts', 'GNN_DE_f*.json')))
    if not parts:
        return 0
    data, meta = {}, {}
    for p in parts:
        j = json.load(open(p))
        meta = j.get('meta', meta)
        data.update(j.get('data', {}))
    json.dump({'meta': meta, 'data': data},
              open(os.path.join(inst_dir, 'GNN_DE.json'), 'w'))
    return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    args = ap.parse_args()
    for inst_dir in sorted(glob.glob(os.path.join(args.root, 'inst_*'))):
        n = merge_instance(inst_dir)
        if n:
            print(f"{os.path.basename(inst_dir)}: merged {n} functions -> GNN_DE.json")


if __name__ == '__main__':
    main()
