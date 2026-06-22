"""Offline analysis of gradient direction conflicts across CEC2017 functions.

Usage:
    python -m encoder.analyze_grad_conflicts checkpoints/probe_fes_hitting/

Reads grad_projections.pt and diagnostics.jsonl from the given directory.
Outputs:
  1. Per-function-pair cosine similarity matrix (global + per-module)
  2. Temporal stability (consecutive same-function steps)
  3. Energy-weighted conflict score
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


def load_data(checkpoint_dir):
    """Load projections and diagnostics."""
    proj_path = checkpoint_dir / 'grad_projections.pt'
    diag_path = checkpoint_dir / 'diagnostics.jsonl'

    if not proj_path.exists():
        print(f"ERROR: {proj_path} not found. Run training first.")
        sys.exit(1)

    data = torch.load(proj_path, map_location='cpu', weights_only=False)

    # Load gradient norms from diagnostics for energy weighting
    gn_by_step = {}
    if diag_path.exists():
        with open(diag_path) as f:
            for line in f:
                d = json.loads(line)
                gn_by_step[d['step']] = d.get('grad_norm_total', 0)

    return data, gn_by_step


def cosine_matrix(projections_by_fn, fn_ids):
    """Compute pairwise cosine similarity of mean gradient directions."""
    # Mean direction per function
    means = {}
    for fid in fn_ids:
        vecs = projections_by_fn[fid]
        if len(vecs) == 0:
            continue
        stacked = torch.stack(vecs)
        means[fid] = stacked.mean(dim=0)

    active_fns = sorted(means.keys())
    n = len(active_fns)
    matrix = torch.zeros(n, n)

    for i, fi in enumerate(active_fns):
        for j, fj in enumerate(active_fns):
            matrix[i, j] = F.cosine_similarity(
                means[fi].unsqueeze(0), means[fj].unsqueeze(0)).item()

    return matrix, active_fns


def temporal_stability(projections_by_fn):
    """Cosine similarity between consecutive same-function gradient projections."""
    results = {}
    for fid, vecs in projections_by_fn.items():
        if len(vecs) < 2:
            continue
        cosines = []
        for i in range(1, len(vecs)):
            cos = F.cosine_similarity(
                vecs[i - 1].unsqueeze(0), vecs[i].unsqueeze(0)).item()
            cosines.append(cos)
        results[fid] = {
            'mean': sum(cosines) / len(cosines),
            'min': min(cosines),
            'max': max(cosines),
            'n': len(cosines),
        }
    return results


def print_conflict_matrix(matrix, fn_ids, label="Global"):
    """Print the cosine similarity matrix, highlighting conflicts."""
    n = len(fn_ids)
    print(f"\n{'=' * 60}")
    print(f"  {label} Cosine Similarity Matrix ({n} functions)")
    print(f"{'=' * 60}")

    # Header
    header = "     " + " ".join(f"F{f:02d}" for f in fn_ids)
    print(header)

    for i, fi in enumerate(fn_ids):
        row = f"F{fi:02d}  "
        for j in range(n):
            val = matrix[i, j].item()
            if i == j:
                row += "  .  "
            elif val < -0.1:
                row += f"{val:+.2f} "  # conflict
            else:
                row += f" {val:.2f} "
            # row += f" {val:+.2f}" if i != j else "   . "
        print(row)


def print_worst_conflicts(matrix, fn_ids, gn_by_fn, top_k=10):
    """Print the worst conflict pairs, weighted by gradient energy."""
    n = len(fn_ids)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            cos = matrix[i, j].item()
            fi, fj = fn_ids[i], fn_ids[j]
            energy = gn_by_fn.get(fi, 1.0) * gn_by_fn.get(fj, 1.0)
            pairs.append((cos, energy, fi, fj))

    # Sort by energy-weighted conflict (most negative cos * highest energy first)
    pairs.sort(key=lambda x: x[0] * x[1], reverse=False)

    print(f"\n{'=' * 60}")
    print(f"  Top {top_k} Gradient Conflicts (energy-weighted)")
    print(f"{'=' * 60}")
    print(f"  {'Pair':>10s}  {'cosine':>8s}  {'energy':>10s}  {'score':>10s}")
    for cos, energy, fi, fj in pairs[:top_k]:
        score = cos * energy
        print(f"  F{fi:02d}-F{fj:02d}  {cos:+8.3f}  {energy:10.0f}  {score:+10.0f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze gradient direction conflicts")
    parser.add_argument("checkpoint_dir", type=Path)
    args = parser.parse_args()

    data, gn_by_step = load_data(args.checkpoint_dir)

    fn_ids_list = data['fn_id']
    steps_list = data['step']

    # Group projections by function ID
    all_fn_ids = sorted(set(fn_ids_list))

    # Per-module analysis
    for module_name in ['global', 'backbone', 'heads', 'scorer']:
        projections = data[module_name]  # (N_steps, d_proj)
        by_fn = defaultdict(list)
        for i, fid in enumerate(fn_ids_list):
            by_fn[fid].append(projections[i])

        matrix, active_fns = cosine_matrix(by_fn, all_fn_ids)
        print_conflict_matrix(matrix, active_fns, label=module_name.capitalize())

    # Temporal stability (global projections only)
    global_by_fn = defaultdict(list)
    for i, fid in enumerate(fn_ids_list):
        global_by_fn[fid].append(data['global'][i])

    print(f"\n{'=' * 60}")
    print("  Temporal Stability (consecutive same-function cosine)")
    print(f"{'=' * 60}")
    stability = temporal_stability(global_by_fn)
    print(f"  {'FN':>4s}  {'mean':>8s}  {'min':>8s}  {'max':>8s}  {'n':>5s}")
    for fid in sorted(stability.keys()):
        s = stability[fid]
        print(f"  F{fid:02d}  {s['mean']:+8.3f}  {s['min']:+8.3f}  {s['max']:+8.3f}  {s['n']:5d}")

    # Energy-weighted conflicts
    gn_by_fn = defaultdict(list)
    for i, fid in enumerate(fn_ids_list):
        step = steps_list[i]
        gn = gn_by_step.get(step, 0)
        if isinstance(gn, (int, float)) and gn == gn:
            gn_by_fn[fid].append(gn)

    avg_gn_by_fn = {fid: (sum(gns) / len(gns)) ** 2 if gns else 1.0
                    for fid, gns in gn_by_fn.items()}

    matrix, active_fns = cosine_matrix(global_by_fn, all_fn_ids)
    print_worst_conflicts(matrix, active_fns, avg_gn_by_fn, top_k=15)

    # Summary statistics
    n = len(active_fns)
    off_diag = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diag.append(matrix[i, j].item())

    if off_diag:
        print(f"\n{'=' * 60}")
        print("  Summary")
        print(f"{'=' * 60}")
        print(f"  Functions with projections: {n}")
        print(f"  Total steps: {len(fn_ids_list)}")
        print(f"  Mean pairwise cosine: {sum(off_diag)/len(off_diag):+.3f}")
        print(f"  Min pairwise cosine:  {min(off_diag):+.3f}")
        print(f"  Max pairwise cosine:  {max(off_diag):+.3f}")
        n_negative = sum(1 for c in off_diag if c < 0)
        print(f"  Negative pairs: {n_negative}/{len(off_diag)} ({100*n_negative/len(off_diag):.0f}%)")

        # Hard function group analysis
        hard = {7, 21, 24, 26}
        hard_indices = [i for i, f in enumerate(active_fns) if f in hard]
        if len(hard_indices) >= 2:
            hard_pairs = []
            for ii in range(len(hard_indices)):
                for jj in range(ii + 1, len(hard_indices)):
                    hard_pairs.append(matrix[hard_indices[ii], hard_indices[jj]].item())
            print(f"\n  Hard function group (F07,F21,F24,F26):")
            print(f"    Mean pairwise cosine: {sum(hard_pairs)/len(hard_pairs):+.3f}")
            print(f"    Min: {min(hard_pairs):+.3f}  Max: {max(hard_pairs):+.3f}")


if __name__ == '__main__':
    main()
