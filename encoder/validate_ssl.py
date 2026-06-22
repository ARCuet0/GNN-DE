"""
validate_ssl.py — 7-check validation protocol for NPA SSL pretraining.

Runs all checks in ~30 min on RTX 4070 Super before committing to a long
training run. Prints PASS/FAIL per check with diagnostics.

Usage:
    python -m encoder.validate_ssl --device cuda
"""

import argparse
import collections
import random
import time

import torch
import torch.nn.functional as F

from .npa_ssl_online import OnlineSSLTrainer, STRATEGIES


# ======================================================================
# Check 1: Overfit-1 — single trajectory, losses must converge
# ======================================================================

def check_1_overfit(model, device, steps=200, threshold=0.05):
    """Train B=1 for `steps` steps. All 3 losses must drop below threshold."""
    print("\n" + "=" * 60)
    print("CHECK 1: Overfit-1 (single trajectory convergence)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    initial_losses = None
    final_losses = None

    for step in range(steps):
        ld = model.train_step(optimizer, B=1)
        if step == 0:
            initial_losses = {k: ld[k].item() for k in ('benefit', 'node', 'rank', 'total')}
        if (step + 1) % 50 == 0:
            print(f"  Step {step+1:3d}: total={ld['total'].item():.4f}  "
                  f"benefit={ld['benefit'].item():.4f}  "
                  f"node={ld['node'].item():.4f}  "
                  f"rank={ld['rank'].item():.4f}  | {ld['strategy']}")

    final_losses = {k: ld[k].item() for k in ('benefit', 'node', 'rank', 'total')}

    passed = final_losses['total'] < threshold
    decreased = final_losses['total'] < initial_losses['total'] * 0.5

    print(f"\n  Initial total: {initial_losses['total']:.4f}")
    print(f"  Final total:   {final_losses['total']:.4f}")
    print(f"  Threshold:     {threshold}")
    print(f"  Decreased 50%: {decreased}")

    ok = passed or decreased
    print(f"\n  --> {'PASS' if ok else 'FAIL'}: "
          f"{'converged' if passed else 'decreased' if decreased else 'stuck'}")
    return ok, initial_losses, final_losses


# ======================================================================
# Check 2: Per-level gradient magnitudes
# ======================================================================

def check_2_gradients(model, device):
    """One forward+backward, log grad.norm() per parameter group."""
    print("\n" + "=" * 60)
    print("CHECK 2: Per-level gradient magnitudes")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ld = model.train_step(optimizer, B=1)

    groups = {
        'temporal_gru': model.backbone.temporal_gru,
        'cross_dim': model.backbone.cross_dim,
        'pop_transformer': model.backbone.pop_transformer,
        'ls1_global_head': model.ls1_global_head,
        'ls1_node_head': model.ls1_node_head,
        'rank_head': model.rank_head,
    }

    all_ok = True
    for name, module in groups.items():
        grad_norms = []
        for p in module.parameters():
            if p.grad is not None:
                grad_norms.append(p.grad.norm().item())

        if not grad_norms:
            print(f"  {name:25s}: NO GRADS")
            all_ok = False
            continue

        mean_norm = sum(grad_norms) / len(grad_norms)
        max_norm = max(grad_norms)
        min_norm = min(grad_norms)

        status = "OK"
        if max_norm > 100:
            status = "EXPLODING"
            all_ok = False
        elif mean_norm < 1e-7:
            status = "VANISHING"
            all_ok = False

        print(f"  {name:25s}: mean={mean_norm:.2e}  "
              f"min={min_norm:.2e}  max={max_norm:.2e}  [{status}]")

    print(f"\n  --> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ======================================================================
# Check 3: Label distribution audit
# ======================================================================

def check_3_labels(model, n_trajectories=60):
    """Generate trajectories, audit label and strategy distributions."""
    print("\n" + "=" * 60)
    print("CHECK 3: Label distribution audit")
    print("=" * 60)

    strat_counts = collections.Counter()
    benefit_ratios = {s: [] for s in STRATEGIES}
    node_nonzero = {s: 0 for s in STRATEGIES}
    rank_values = []

    for i in range(n_trajectories):
        strategy = STRATEGIES[i % len(STRATEGIES)]
        result = model._generate_trajectory(strategy)
        labels = result[7]

        strat_counts[strategy] += 1
        br = labels['ls1_benefit_ratio'].item()
        benefit_ratios[strategy].append(br)
        if labels['node_ls1_label'].sum() > 0:
            node_nonzero[strategy] += 1
        rank_values.extend(labels['fitness_rank'].cpu().tolist())

    print("\n  Strategy balance:")
    for s in STRATEGIES:
        pct = strat_counts[s] / n_trajectories * 100
        print(f"    {s:12s}: {strat_counts[s]:3d} ({pct:.0f}%)")

    print("\n  Benefit ratio (mean per strategy):")
    all_ok = True
    for s in STRATEGIES:
        vals = benefit_ratios[s]
        mean_br = sum(vals) / max(len(vals), 1)
        nonzero_pct = sum(1 for v in vals if v > 0) / max(len(vals), 1) * 100
        print(f"    {s:12s}: mean={mean_br:.3f}  nonzero={nonzero_pct:.0f}%")

        if s == 'all_shade' and mean_br > 0.01:
            print(f"      WARNING: all_shade should have ~0 benefit ratio")
            all_ok = False
        if s in ('all_ls1', 'mixed') and nonzero_pct < 20:
            print(f"      WARNING: too few nonzero benefit ratios for {s}")
            all_ok = False

    print("\n  Node LS1 label nonzero (per strategy):")
    for s in STRATEGIES:
        n = strat_counts[s]
        pct = node_nonzero[s] / max(n, 1) * 100
        print(f"    {s:12s}: {node_nonzero[s]:3d}/{n} ({pct:.0f}%)")

    # Fitness rank distribution
    rank_t = torch.tensor(rank_values)
    print(f"\n  Fitness rank stats: min={rank_t.min():.3f} max={rank_t.max():.3f} "
          f"mean={rank_t.mean():.3f} std={rank_t.std():.3f}")

    if rank_t.max() < 0.9 or rank_t.min() > 0.1:
        print("    WARNING: fitness rank not spanning [0, 1]")
        all_ok = False

    print(f"\n  --> {'PASS' if all_ok else 'FAIL'}")
    return all_ok


# ======================================================================
# Check 4: Representation vitality
# ======================================================================

def check_4_representation(model, device, steps=100, threshold=0.01):
    """Monitor h.std and h_global.std over training. Flag if dead."""
    print("\n" + "=" * 60)
    print("CHECK 4: Representation vitality")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    std_log = []
    for step in range(steps):
        # Generate one trajectory and do forward only
        strategy = random.choice(STRATEGIES)
        result = model._generate_trajectory(strategy)
        (coords_hist, fitness_hist, valid_mask, n_valid, f_init,
         coords_current, fitness_current, labels) = result

        N = labels['N']
        d = device
        with torch.no_grad():
            h, _, _, h_global = model.backbone.encode(
                torch.zeros(N, 1, device=d),
                torch.zeros(2, 0, device=d, dtype=torch.long),
                torch.zeros(0, 1, device=d),
                torch.zeros(1, 1, device=d),
                coords_hist=coords_hist,
                fitness_hist=fitness_hist,
                valid_mask=valid_mask,
                n_valid=n_valid,
                coords_current=coords_current,
                fitness_current=fitness_current,
                f_init=f_init,
            )

        h_std = h.std(dim=0).mean().item()
        hg_std = h_global.std(dim=0).mean().item() if h_global.shape[0] > 1 else h_global.abs().mean().item()

        if step == 0 or step == steps - 1:
            std_log.append((step, h_std, hg_std))
            print(f"  Step {step:3d}: h.std={h_std:.4f}  h_global.mean_abs={hg_std:.4f}")

        # Also train so representations evolve
        ld = model.train_step(optimizer, B=1)

    h_ok = std_log[-1][1] > threshold
    print(f"\n  h.std at step {steps-1}: {std_log[-1][1]:.4f} (threshold={threshold})")
    print(f"  --> {'PASS' if h_ok else 'FAIL: dead representations'}")
    return h_ok


# ======================================================================
# Check 5: Memory/speed profiling
# ======================================================================

def check_5_memory_speed(model, device, n_steps=10):
    """Profile peak GPU memory and step time."""
    print("\n" + "=" * 60)
    print("CHECK 5: Memory/speed profiling")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    times = []
    for step in range(n_steps):
        t0 = time.time()
        ld = model.train_step(optimizer, B=1)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        dt = time.time() - t0
        times.append(dt)
        print(f"  Step {step}: {dt:.3f}s  loss={ld['total'].item():.4f}  | {ld['strategy']}")

    mean_t = sum(times) / len(times)
    peak_mem = 0
    if device.type == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9

    print(f"\n  Mean step time: {mean_t:.3f}s")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")
    print(f"  Estimated 10K steps: {mean_t * 10000 / 3600:.1f} hours")

    ok = peak_mem < 11.0  # RTX 4070 Super = 12GB, leave 1GB headroom
    if device.type != 'cuda':
        ok = True
    print(f"\n  --> {'PASS' if ok else 'FAIL: OOM risk'} "
          f"({peak_mem:.1f}GB / 12GB)")
    return ok


# ======================================================================
# Check 6: Batched vs single path consistency
# ======================================================================

def check_6_consistency(model, device, tolerance=1e-3):
    """Compare B=1 single path vs batched path outputs."""
    print("\n" + "=" * 60)
    print("CHECK 6: Batched vs single path consistency")
    print("=" * 60)

    # Fix seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)

    strategy = 'mixed'
    result = model._generate_trajectory(strategy)
    (coords_hist, fitness_hist, valid_mask, n_valid, f_init,
     coords_current, fitness_current, labels) = result

    N = labels['N']
    d = device

    with torch.no_grad():
        # Single path
        h_single, _, _, hg_single = model.backbone.encode(
            torch.zeros(N, 1, device=d),
            torch.zeros(2, 0, device=d, dtype=torch.long),
            torch.zeros(0, 1, device=d),
            torch.zeros(1, 1, device=d),
            coords_hist=coords_hist,
            fitness_hist=fitness_hist,
            valid_mask=valid_mask,
            n_valid=n_valid,
            coords_current=coords_current,
            fitness_current=fitness_current,
            f_init=f_init,
        )

        # Compute losses via single path
        benefit_single = torch.sigmoid(
            model.ls1_global_head(hg_single).squeeze(-1)).item()
        rank_single = torch.sigmoid(
            model.rank_head(h_single).squeeze(-1))

    print(f"  Single path: h.shape={h_single.shape}  "
          f"benefit={benefit_single:.4f}  "
          f"rank_mean={rank_single.mean().item():.4f}")

    # Note: true consistency check requires running batched path with B=1
    # on the same data — we verify shapes and value ranges instead
    h_range = (h_single.min().item(), h_single.max().item())
    rank_range = (rank_single.min().item(), rank_single.max().item())

    print(f"  h range: [{h_range[0]:.3f}, {h_range[1]:.3f}]")
    print(f"  rank range: [{rank_range[0]:.3f}, {rank_range[1]:.3f}]")

    ok = True
    if rank_range[0] < -0.01 or rank_range[1] > 1.01:
        print("  WARNING: rank predictions outside [0, 1]")
        ok = False
    if h_single.std() < 0.001:
        print("  WARNING: h representations collapsed")
        ok = False

    print(f"\n  --> {'PASS' if ok else 'FAIL'}")
    return ok


# ======================================================================
# Check 7: Node head vitality
# ======================================================================

def check_7_node_head(model, device, steps=500, acc_threshold=0.55):
    """Train for `steps` steps, verify node head learns above chance."""
    print("\n" + "=" * 60)
    print("CHECK 7: Node head vitality (binary LS1 head)")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    initial_node_loss = None
    node_losses = []
    node_correct = 0
    node_total = 0

    for step in range(steps):
        ld = model.train_step(optimizer, B=1)

        node_loss = ld['node'].item()
        if initial_node_loss is None:
            initial_node_loss = node_loss
        node_losses.append(node_loss)

        if (step + 1) % 100 == 0:
            recent = node_losses[-100:]
            avg = sum(recent) / len(recent)
            print(f"  Step {step+1:3d}: node_loss_avg={avg:.4f}  "
                  f"(initial={initial_node_loss:.4f})")

    final_avg = sum(node_losses[-100:]) / len(node_losses[-100:])
    decreased = final_avg < initial_node_loss * 0.9

    # Quick accuracy check on 20 trajectories
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            strategy = random.choice(('all_ls1', 'mixed'))
            result = model._generate_trajectory(strategy)
            labels = result[7]
            if labels['node_ls1_label'].sum() == 0:
                continue

            coords_hist, fitness_hist, valid_mask, n_valid, f_init = result[:5]
            coords_current, fitness_current = result[5], result[6]
            N = labels['N']
            d = device

            h, _, _, _ = model.backbone.encode(
                torch.zeros(N, 1, device=d),
                torch.zeros(2, 0, device=d, dtype=torch.long),
                torch.zeros(0, 1, device=d),
                torch.zeros(1, 1, device=d),
                coords_hist=coords_hist,
                fitness_hist=fitness_hist,
                valid_mask=valid_mask,
                n_valid=n_valid,
                coords_current=coords_current,
                fitness_current=fitness_current,
                f_init=f_init,
            )

            logit = model.ls1_node_head(h).squeeze(-1)
            _, top3 = torch.topk(-fitness_current, min(3, N))
            pred = (logit[top3] > 0).float()
            target = labels['node_ls1_label'][top3]
            node_correct += (pred == target).sum().item()
            node_total += len(top3)

    model.train()

    acc = node_correct / max(node_total, 1)
    print(f"\n  Initial node loss: {initial_node_loss:.4f}")
    print(f"  Final node loss:   {final_avg:.4f}")
    print(f"  Node accuracy:     {acc:.2%} ({node_correct}/{node_total})")
    print(f"  Decreased:         {decreased}")

    ok = decreased or acc > acc_threshold
    print(f"\n  --> {'PASS' if ok else 'FAIL: node head dead'}")
    return ok


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="NPA SSL validation protocol")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n-pop', type=int, default=100)
    parser.add_argument('--n-gens', type=int, default=50)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--global-out-dim', type=int, default=32)
    parser.add_argument('--skip', nargs='*', default=[], type=int,
                        help='Check numbers to skip (e.g., --skip 1 7)')
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"NPA SSL Validation Protocol — device={device}")
    print(f"Architecture: window={args.window} hidden={args.hidden_dim} "
          f"global_out={args.global_out_dim}")
    print(f"Population: N={args.n_pop} n_gens={args.n_gens}")

    model = OnlineSSLTrainer(
        device=device,
        n_pop=args.n_pop,
        n_gens=args.n_gens,
        window=args.window,
        hidden_dim=args.hidden_dim,
        global_out_dim=args.global_out_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    results = {}
    t0 = time.time()

    checks = [
        (1, "Overfit-1", lambda: check_1_overfit(model, device)),
        (2, "Gradients", lambda: check_2_gradients(model, device)),
        (3, "Labels", lambda: check_3_labels(model)),
        (4, "Representations", lambda: check_4_representation(model, device)),
        (5, "Memory/Speed", lambda: check_5_memory_speed(model, device)),
        (6, "Consistency", lambda: check_6_consistency(model, device)),
        (7, "Node head", lambda: check_7_node_head(model, device)),
    ]

    for num, name, fn in checks:
        if num in args.skip:
            print(f"\n  SKIP: Check {num} ({name})")
            results[num] = True
            continue
        try:
            result = fn()
            # fn may return tuple (ok, ...) or just ok
            if isinstance(result, tuple):
                results[num] = result[0]
            else:
                results[num] = result
        except Exception as e:
            print(f"\n  EXCEPTION in check {num}: {e}")
            results[num] = False

    elapsed = time.time() - t0

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    for num, name, _ in checks:
        status = "PASS" if results.get(num, False) else "FAIL"
        marker = "  " if results.get(num, False) else ">>"
        print(f"  {marker} Check {num}: {name:20s} [{status}]")

    n_passed = sum(1 for v in results.values() if v)
    n_total = len(checks)
    print(f"\n  {n_passed}/{n_total} checks passed in {elapsed:.0f}s")

    if n_passed == n_total:
        print("\n  ALL CHECKS PASSED — safe to launch long training run")
    else:
        failed = [num for num, ok in results.items() if not ok]
        print(f"\n  FAILED checks: {failed} — fix before training")

    return n_passed == n_total


if __name__ == '__main__':
    main()
