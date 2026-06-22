"""KSD-only short training loop with grad wrappers — diagnostic/ablation.

Tracks per step: w_geom value/grad, KSD loss, total grad norm, rolling
stability (std/mean of last 50 losses). Snapshots inter-fn cosines every
K steps over a subset of fids.

Outputs:
    <output>.json   — per-step training log + snapshot log.
    <output>.pth    — final checkpoint (when --save-checkpoint).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch

from analysis.gradient_decomposition import load_checkpoint
from cec2017 import CEC2017Torch
from l2o.ksd.measurement import (
    _run_generation_loop,
    _wrap_heads,
    aggregate_per_subnet,
    run_ksd_episode,
)

# Train fids: D=10 non-blacklist (23) minus outlier hybrids F16/F28/F18 that
# dominated the gradient norm in the multi-fid cosine measurement.
DEFAULT_TRAIN_FIDS = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      20, 22, 24, 25, 26, 27]

DEFAULT_SNAPSHOT_FIDS = [1, 3, 6, 8, 10, 12, 14, 24]


def _build_optimizer(gs, args, freeze_mode):
    """Construct optimizer with optional per-param LR groups.

    Returns the optimizer; prints group breakdown to stderr.
    Raises a no-op fallback when nothing is trainable so optimizer.step()
    still works (used by --freeze-mode all).
    """
    trainable = [p for p in gs.parameters() if p.requires_grad]
    if not trainable:
        print(f"[ksd-train] WARNING: no trainable params (freeze_mode={freeze_mode}). "
              f"Optimizer is a no-op; KSD reduction reflects architecture only.",
              file=sys.stderr, flush=True)
        return torch.optim.Adam([torch.zeros(1, requires_grad=True)], lr=args.lr)

    overrides = {'backbone': args.lr_backbone, 'wgeom': args.lr_wgeom,
                 'fcr_shared': args.lr_fcr_shared}
    if all(v is None for v in overrides.values()):
        return torch.optim.Adam(trainable, lr=args.lr)

    bb_p, wg_p, fcr_p, other_p = [], [], [], []
    for n, p in gs.named_parameters():
        if not p.requires_grad:
            continue
        if 'w_geom' in n:
            wg_p.append(p)
        elif 'adaptive_fcr.shared' in n:
            fcr_p.append(p)
        elif n.startswith('backbone.'):
            bb_p.append(p)
        else:
            other_p.append(p)

    def _lr(name, default):
        return overrides[name] if overrides[name] is not None else default

    groups = [
        {'params': bb_p, 'lr': _lr('backbone', args.lr), 'name': 'backbone'},
        {'params': wg_p, 'lr': _lr('wgeom', args.lr), 'name': 'wgeom'},
        {'params': fcr_p, 'lr': _lr('fcr_shared', args.lr), 'name': 'fcr_shared'},
        {'params': other_p, 'lr': args.lr, 'name': 'other'},
    ]
    groups = [g for g in groups if g['params']]
    print(f"[ksd-train] per-param LR groups:", file=sys.stderr, flush=True)
    for g in groups:
        print(f"             {g['name']}: lr={g['lr']:.2e}, n_params={len(g['params'])}",
              file=sys.stderr, flush=True)
    return torch.optim.Adam(groups)


def train_one_step(gen_step, backbone, variant, fn, config, device,
                    n_gens, T_temp=1.0):
    """Run one episode and return (total_loss, n_gens_completed, term_means).

    Caller does backward + grad-clip + optimizer.step after this returns.
    Skips axis_7/axis_8 collection — we only need the loss tensor and the
    per-gen term diagnostics on the training hot path.
    """
    ep = _run_generation_loop(
        gen_step, variant, fn, config, device,
        n_gens=n_gens, T_temp=T_temp,
        use_grad_wrapper=True,
        sign_coherence_log_state=None,
        collect_attn_diag=False)
    if ep['nan_seen'] or not ep['losses_per_gen']:
        return None, ep['n_gens_completed'], None
    total = torch.stack(ep['losses_per_gen']).mean()
    if not torch.isfinite(total):
        return None, ep['n_gens_completed'], None
    return total, ep['n_gens_completed'], ep['terms_per_gen']


def quick_snapshot(gen_step, backbone, variant, config, device, fids,
                    n_gens=4, seed_base=10000, T_temp=1.0):
    """Light cosine snapshot — runs run_ksd_episode on a subset of fids
    (1 seed each), returns axis_3 inter-fn cosines per group plus
    axis_7/axis_8 means.
    """
    D = config['D'][0] if isinstance(config['D'], list) else config['D']
    records = {}
    for i, fid in enumerate(fids):
        try:
            fn = CEC2017Torch(fid, D, device)
            rec = run_ksd_episode(gen_step, backbone, variant, fn, config,
                                   device, n_gens=n_gens,
                                   seed=seed_base + i,
                                   T_temp=T_temp,
                                   use_grad_wrapper=True,
                                   sign_coherence_log_state=None)
            records[(fid, 0)] = rec
        except Exception as e:
            records[(fid, 0)] = {'nan': True, 'reason': str(e)}
    axis_3 = aggregate_per_subnet(records, fids, seeds=1)
    cos = {}
    for g, v in axis_3.items():
        cos[g] = {
            'inter_fn_mean': v['inter_fn_mean'],
            'inter_fn_min': v['inter_fn_min'],
            'inter_fn_max': v['inter_fn_max'],
            'intra_fn_mean': v['intra_fn_mean'],
        }
    # Aggregate axis_7 / axis_8 across fids.
    a7, a8 = [], []
    for r in records.values():
        if r is None or r.get('nan', True):
            continue
        ad = r.get('attn_diag') or {}
        if 'axis_7_attn_entropy_grad_corr_mean' in ad:
            a7.append(ad['axis_7_attn_entropy_grad_corr_mean'])
        if 'axis_8_attn_descent_alignment_mean' in ad:
            a8.append(ad['axis_8_attn_descent_alignment_mean'])
    return {
        'cos': cos,
        'axis_7_mean': float(np.mean(a7)) if a7 else None,
        'axis_8_mean': float(np.mean(a8)) if a8 else None,
        'n_fids_completed': len(a8),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n-gens', type=int, default=8,
                        help='Generations per training episode.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--train-fids', type=int, nargs='+',
                        default=DEFAULT_TRAIN_FIDS,
                        help='Fids to sample from each step. Default '
                             'excludes F16/F28/F18 (outlier hybrids).')
    parser.add_argument('--snapshot-fids', type=int, nargs='+',
                        default=DEFAULT_SNAPSHOT_FIDS,
                        help='Subset for periodic cosine snapshots.')
    parser.add_argument('--snapshot-every', type=int, default=100,
                        help='Cosine snapshot cadence (steps).')
    parser.add_argument('--snapshot-n-gens', type=int, default=4,
                        help='n_gens during snapshot (cheap).')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Global grad clip max_norm.')
    parser.add_argument('--T-temp', type=float, default=1.0)
    parser.add_argument('--rng-seed', type=int, default=42)
    parser.add_argument('--output', type=str, required=True,
                        help='Output stem (writes .json + .pth).')
    parser.add_argument('--save-checkpoint', action='store_true',
                        help='Save final state_dicts to <output>.pth.')
    parser.add_argument('--save-checkpoint-every', type=int, default=0,
                        help='If > 0, save state_dicts every multiple of this '
                             'step count (plus step 0 + final).')
    parser.add_argument('--freeze-mode', type=str, default='none',
                        choices=['none', 'backbone', 'heads', 'all', 'wgeom_only'],
                        help='Ablation freeze mode. "backbone": freeze bb.* '
                             'except w_geom. "heads": freeze variant.heads.*. '
                             '"all": freeze everything (arch-only control). '
                             '"wgeom_only": only w_geom trainable.')
    parser.add_argument('--lr-backbone', type=float, default=None,
                        help='Per-param LR override for backbone.* (excl w_geom). '
                             'Use 1e-6 to protect pretrained backbone weights.')
    parser.add_argument('--lr-wgeom', type=float, default=None,
                        help='Per-param LR override for w_geom. Use 1e-2 (×100) '
                             'to let the geometric bias learn meaningfully.')
    parser.add_argument('--lr-fcr-shared', type=float, default=None,
                        help='Per-param LR override for the FCR head extended '
                             'Linear (with the new D+1 cols).')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print(f"[ksd-train] Loading: {args.checkpoint}", file=sys.stderr, flush=True)
    print(f"[ksd-train] steps={args.steps}, lr={args.lr}, n_gens={args.n_gens}, "
          f"device={device}", file=sys.stderr, flush=True)
    print(f"[ksd-train] train_fids={args.train_fids}", file=sys.stderr, flush=True)
    print(f"[ksd-train] snapshot every {args.snapshot_every}, "
          f"snapshot_fids={args.snapshot_fids}",
          file=sys.stderr, flush=True)

    gs, bb, var, _, config, base_step = load_checkpoint(args.checkpoint, device)
    D = config['D'][0] if isinstance(config['D'], list) else config['D']

    donor_wrap, fcr_wrap = _wrap_heads(bb, var, D)
    print(f"[ksd-train] Wrapped donor + fcr (D={D}). w_geom_init="
          f"{donor_wrap.w_geom.item():.4f}", file=sys.stderr, flush=True)

    n_frozen = 0
    n_trainable = 0
    if args.freeze_mode == 'all':
        for n, p in gs.named_parameters():
            p.requires_grad_(False); n_frozen += 1
    elif args.freeze_mode == 'backbone':
        for n, p in gs.named_parameters():
            if n.startswith('backbone.') and 'w_geom' not in n:
                p.requires_grad_(False); n_frozen += 1
            else:
                n_trainable += 1
    elif args.freeze_mode == 'heads':
        for n, p in gs.named_parameters():
            if n.startswith('variant.heads.'):
                p.requires_grad_(False); n_frozen += 1
            else:
                n_trainable += 1
    elif args.freeze_mode == 'wgeom_only':
        for n, p in gs.named_parameters():
            if 'w_geom' not in n:
                p.requires_grad_(False); n_frozen += 1
            else:
                n_trainable += 1
    else:
        n_trainable = sum(1 for _ in gs.parameters())
    if args.freeze_mode != 'none':
        print(f"[ksd-train] freeze_mode={args.freeze_mode}: "
              f"frozen={n_frozen}, trainable={n_trainable}",
              file=sys.stderr, flush=True)

    optimizer = _build_optimizer(gs, args, args.freeze_mode)
    rng = np.random.default_rng(args.rng_seed)
    recent_losses = deque(maxlen=50)

    log = []
    snapshot_log = []

    def _fmt_snap(snap):
        cos = snap.get('cos', {})
        bb_all = cos.get('bb.all', {}).get('inter_fn_mean')
        donor = cos.get('bb.donor_selector', {}).get('inter_fn_mean')
        fcr = cos.get('var.h0.adaptive_fcr', {}).get('inter_fn_mean')
        a7 = snap.get('axis_7_mean')
        a8 = snap.get('axis_8_mean')
        def _f(v):
            if v is None or not isinstance(v, (int, float)) or math.isnan(v):
                return '  nan'
            return f"{v:+.3f}"
        return f"bb={_f(bb_all)} donor={_f(donor)} fcr={_f(fcr)} ax7={_f(a7)} ax8={_f(a8)}"

    def _save_checkpoint(step):
        if not args.save_checkpoint_every:
            return
        ck = {
            'backbone_state_dict': bb.state_dict(),
            'variant_state_dict': var.state_dict(),
            'step': step,
        }
        if gs.surrogate is not None:
            ck['surrogate_state_dict'] = gs.surrogate.state_dict()
        path = f"{args.output}_step_{step:04d}.pth"
        torch.save(ck, path)
        print(f"[ksd-train]   ckpt → {path}", file=sys.stderr, flush=True)

    print(f"[ksd-train] Initial snapshot…", file=sys.stderr, flush=True)
    snap0 = quick_snapshot(gs, bb, var, config, device, args.snapshot_fids,
                            n_gens=args.snapshot_n_gens, seed_base=10000,
                            T_temp=args.T_temp)
    snapshot_log.append({'step': 0, **snap0})
    print(f"[ksd-train]   step 0 snap: {_fmt_snap(snap0)}",
          file=sys.stderr, flush=True)
    optimizer.zero_grad()        # quick_snapshot leaves grads populated
    _save_checkpoint(0)

    from encoder.cec2017_torch import CEC2017Torch
    skipped = 0
    for step in range(args.steps):
        fid = int(rng.choice(args.train_fids))
        try:
            fn = CEC2017Torch(fid, D, device)
        except Exception as e:
            skipped += 1
            continue

        loss, n_gens_done, terms = train_one_step(
            gs, bb, var, fn, config, device, args.n_gens, args.T_temp)
        if loss is None or not torch.isfinite(loss):
            skipped += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(gs.parameters(),
                                                     args.grad_clip)
        optimizer.step()

        loss_val = float(loss.item())
        recent_losses.append(loss_val)
        stab = (float(np.std(recent_losses) / max(np.abs(np.mean(recent_losses)),
                                                    1e-12))
                if len(recent_losses) >= 2 else float('nan'))

        w_grad = (float(donor_wrap.w_geom.grad.item())
                  if donor_wrap.w_geom.grad is not None else 0.0)
        log.append({
            'step': step,
            'fid': fid,
            'n_gens_completed': n_gens_done,
            'loss': loss_val,
            'w_geom_value': float(donor_wrap.w_geom.item()),
            'w_geom_grad_pre_clip': w_grad,
            'total_grad_norm_pre_clip': float(total_norm.item()
                                                if torch.is_tensor(total_norm)
                                                else total_norm),
            'stability': stab,
            'term_means_last_gen': terms[-1] if terms else None,
        })

        if step % 10 == 0 or step == args.steps - 1:
            print(f"[ksd-train] step {step:4d}  fid=F{fid:02d}  "
                  f"loss={loss_val:.4f}  w_geom={donor_wrap.w_geom.item():+.4f}  "
                  f"|g|={abs(w_grad):.2e}  stab={stab:.3f}  "
                  f"clip_norm={total_norm:.2e}",
                  file=sys.stderr, flush=True)

        if step > 0 and (step % args.snapshot_every == 0):
            print(f"[ksd-train] snapshot at step {step}…",
                  file=sys.stderr, flush=True)
            snap = quick_snapshot(gs, bb, var, config, device,
                                   args.snapshot_fids,
                                   n_gens=args.snapshot_n_gens,
                                   seed_base=10000,
                                   T_temp=args.T_temp)
            snapshot_log.append({'step': step, **snap})
            print(f"[ksd-train]   step {step} snap: {_fmt_snap(snap)}",
                  file=sys.stderr, flush=True)
            optimizer.zero_grad()
            if args.save_checkpoint_every and (step % args.save_checkpoint_every == 0):
                _save_checkpoint(step)

    # Final snapshot.
    print(f"[ksd-train] Final snapshot…", file=sys.stderr, flush=True)
    snap_final = quick_snapshot(gs, bb, var, config, device, args.snapshot_fids,
                                 n_gens=args.snapshot_n_gens, seed_base=10000,
                                 T_temp=args.T_temp)
    snapshot_log.append({'step': args.steps, **snap_final})
    print(f"[ksd-train]   step {args.steps} snap: {_fmt_snap(snap_final)}",
          file=sys.stderr, flush=True)
    _save_checkpoint(args.steps)

    out = {
        'checkpoint': str(args.checkpoint),
        'base_step': base_step,
        'steps': args.steps,
        'lr': args.lr,
        'n_gens': args.n_gens,
        'T_temp': args.T_temp,
        'grad_clip': args.grad_clip,
        'train_fids': list(args.train_fids),
        'snapshot_fids': list(args.snapshot_fids),
        'rng_seed': args.rng_seed,
        'skipped_steps': skipped,
        'final_w_geom': float(donor_wrap.w_geom.item()),
        'log': log,
        'snapshot_log': snapshot_log,
    }
    json_path = args.output + '.json'
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[ksd-train] wrote {json_path}", file=sys.stderr, flush=True)

    if args.save_checkpoint:
        ck = {
            'backbone_state_dict': bb.state_dict(),
            'variant_state_dict': var.state_dict(),
            'step': args.steps,
        }
        if gs.surrogate is not None:
            ck['surrogate_state_dict'] = gs.surrogate.state_dict()
        pt_path = args.output + '.pth'
        torch.save(ck, pt_path)
        print(f"[ksd-train] wrote {pt_path}", file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
