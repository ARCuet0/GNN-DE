"""
ssl_pretrain_multi.py — Shared-batch multi-model SSL training.

Trains 4 architecture variants on the SAME batches (identical data order)
with independent optimizers. Eliminates I/O competition that plagued
per-GPU-per-model training (91s data load vs 2s compute per step).

Usage:
    python -m Gesserit.ssl_pretrain_multi --device cuda \
        --data-dir D10,D30,D50 --output-dir checkpoints/full \
        --batch-size 1024 --num-workers 8 --seed 42
"""
import argparse
import gc
import glob as glob_mod
import json
import logging
import math
import os
import re
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

# ── Import from ssl_pretrain_pure (same package) ──
from Gesserit.ssl_pretrain_pure import (
    PureSSLDataset,
    DimGroupBatchSampler,
    PureSSLPretrainer,
    collate_pure_ssl,
    NODE_DIM, EDGE_DIM, GLOBAL_DIM, STORED_GLOBAL_DIM,
)
from Gesserit.encoder.sparse_temporal_backbone import TemporalSparseGATv2Backbone
from Gesserit.encoder.sparse_gatv2_backbone import TopologyMode

# ── Architecture configs ──
CONFIGS = [
    ("baseline",  dict(d_rnn=64,  temporal_layers=2, gatv2_hidden=128, n_gat_layers=3, n_heads=8, n_induced=8)),
    ("wide_temp", dict(d_rnn=128, temporal_layers=2, gatv2_hidden=128, n_gat_layers=3, n_heads=8, n_induced=8)),
    ("wide_gat",  dict(d_rnn=64,  temporal_layers=2, gatv2_hidden=256, n_gat_layers=2, n_heads=8, n_induced=8)),
    ("deep_temp", dict(d_rnn=64,  temporal_layers=4, gatv2_hidden=128, n_gat_layers=3, n_heads=8, n_induced=8)),
]


DEFAULT_LOSS_WEIGHTS = {
    'nextstep_node': 2.0, 'nextstep_global': 2.0, 'fitness_rank': 2.5,
    'node_recon': 0.5, 'edge_recon': 0.5, 'algo_class': 1.0,
}
_KNOWN_LOSSES = list(DEFAULT_LOSS_WEIGHTS.keys())


def _parse_losses(spec):
    """Parse --losses spec into a loss_weights dict.

    Examples:
        'all'                                      → all 6 with default weights
        'nextstep_node'                            → only nextstep_node at 2.0
        'all,-node_recon,-edge_recon'              → drop recon losses
        'all,node_recon:0.025,edge_recon:0.025'    → all 6, custom recon weights
        'nextstep_node:1.5,nextstep_global:1.0'    → only these two at custom weights

    Tokens in the comma-separated list:
        'all'         (only as first token)  — seed with DEFAULT_LOSS_WEIGHTS
        '-NAME'       (after 'all')          — drop NAME
        'NAME'                                — include NAME at default weight
        'NAME:WEIGHT'                         — include NAME at custom weight

    Unknown NAMEs are warned and skipped. Malformed WEIGHTs are warned and
    the token is skipped (the key is NOT inserted at the default).
    """
    parts = [p.strip() for p in spec.split(',')]
    seed_all = parts[0] == 'all'
    weights = DEFAULT_LOSS_WEIGHTS.copy() if seed_all else {}
    for tok in (parts[1:] if seed_all else parts):
        if tok.startswith('-'):
            target = tok[1:]
            if target not in DEFAULT_LOSS_WEIGHTS:
                log.warning("Unknown loss name in subtraction: -%s (known: %s)",
                            target, _KNOWN_LOSSES)
                continue
            weights.pop(target, None)
            continue
        name, sep, w_str = tok.partition(':')
        if name not in DEFAULT_LOSS_WEIGHTS:
            log.warning("Unknown loss name: %s (known: %s)", name, _KNOWN_LOSSES)
            continue
        if sep:
            try:
                weights[name] = float(w_str)
            except ValueError:
                log.warning("Malformed weight for %s: %r", name, w_str)
        else:
            weights[name] = DEFAULT_LOSS_WEIGHTS[name]
    if not weights:
        raise ValueError(
            f"Parsed --losses={spec!r} produced empty loss_weights dict. "
            "This would train with zero gradient. Check for typos.")
    return weights


def _build_model(cfg, device, loss_weights=None, pooler_type='induced'):
    """Build backbone + pretrainer from config dict."""
    backbone = TemporalSparseGATv2Backbone(
        d_rnn=cfg['d_rnn'], d_temporal=cfg['d_rnn'], gru_window=16,
        node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
        gatv2_hidden=cfg['gatv2_hidden'],
        gatv2_layers=cfg['n_gat_layers'], n_heads=cfg['n_heads'],
        global_out_dim=cfg['gatv2_hidden'],
        dropout=0.1,
        temporal_layers=cfg['temporal_layers'],
        n_induced=cfg['n_induced'],
        cross_dim_layers=0,
        unpooled_temporal=False,
        topology_mode=TopologyMode.COORDINATE_KNN,
        pooler_type=pooler_type,
        k_neighbors=8, device=device,
    ).to(device)

    pretrainer = PureSSLPretrainer(
        backbone,
        embed_dim=cfg['gatv2_hidden'],
        global_dim=cfg['gatv2_hidden'],
        device=device,
        loss_weights=loss_weights,
    )
    return backbone, pretrainer


def _current_lr(slot) -> float:
    """Return the current LR of a slot's scheduler as a scalar float.

    Reads from `optimizer.param_groups[0]['lr']` which reflects the most
    recent `scheduler.step()` (torch updates the param_group lr in place).
    First param group only — all pretrainer params share one group here.
    """
    return float(slot.optimizer.param_groups[0]['lr'])


def _make_scheduler(optimizer, kind: str, max_steps: int):
    """Build an LR scheduler.

    kind='warmrestarts': legacy CosineAnnealingWarmRestarts(T_0=10, T_mult=2).
      Epoch-based; restarts fire at epoch 10, 30, 70, 150. For runs shorter
      than one full epoch over large corpora this never triggers and the
      scheduler effectively does nothing useful — kept for backwards
      compatibility with Phase 1/2 runs.

    kind='cosine': CosineAnnealingLR(T_max=max_steps). Step-based. LR
      decays monotonically from lr to ~0 over exactly max_steps steps.
      Required for long single-epoch runs (Phase 3 SSL 340K steps).
    """
    if kind == 'warmrestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2)
    if kind == 'cosine':
        if max_steps <= 0:
            raise ValueError(
                f"cosine scheduler needs positive max_steps (got {max_steps}); "
                "pass --max-steps on the CLI for step-based decay.")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps)
    raise ValueError(f"unknown scheduler kind: {kind!r} "
                     "(valid: 'warmrestarts', 'cosine')")


class ModelSlot:
    """Holds one model variant's state: model, optimizer, scheduler, metrics."""
    __slots__ = ('name', 'cfg', 'backbone', 'pretrainer', 'optimizer',
                 'scheduler', 'best_val', 'patience_counter', 'alive',
                 'output_dir', 'epoch_train_loss', 'epoch_train_metrics',
                 'epoch_n_batches', 'n_params')

    def __init__(self, name, cfg, device, lr, wd, output_dir,
                 loss_weights=None, pooler_type='induced',
                 scheduler_kind='warmrestarts', max_steps=0):
        self.name = name
        self.cfg = cfg
        self.alive = True
        self.output_dir = os.path.join(output_dir, name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.backbone, self.pretrainer = _build_model(
            cfg, device, loss_weights=loss_weights, pooler_type=pooler_type)
        self.n_params = sum(p.numel() for p in self.pretrainer.parameters())
        self.optimizer = torch.optim.AdamW(
            self.pretrainer.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = _make_scheduler(
            self.optimizer, kind=scheduler_kind, max_steps=max_steps)
        self.best_val = float('inf')
        self.patience_counter = 0
        self.reset_epoch_stats()

    def reset_epoch_stats(self):
        self.epoch_train_loss = 0.0
        self.epoch_train_metrics = defaultdict(float)
        self.epoch_n_batches = 0

    def accumulate(self, metrics):
        self.epoch_train_loss += metrics['loss']
        for k, v in metrics.items():
            self.epoch_train_metrics[k] += v
        self.epoch_n_batches += 1


def _gpu_type():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return 'cpu'


def _full_checkpoint(s, epoch, global_step):
    """Build a full-state checkpoint dict for resume."""
    return {
        'backbone_state_dict': s.backbone.state_dict(),
        'pretrainer_state_dict': s.pretrainer.state_dict(),
        'optimizer_state_dict': s.optimizer.state_dict(),
        'scheduler_state_dict': s.scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'best_val': s.best_val,
        'patience_counter': s.patience_counter,
        'config': s.cfg,
        'gpu_type': _gpu_type(),
    }


def _find_resume_checkpoint(resume_dir):
    """Find the best checkpoint to resume from.

    Prefers ssl_nextstep_sparse_embed.pth (best-val),
    falls back to latest ssl_ep{N}.pth (periodic).
    """
    best = os.path.join(resume_dir, 'ssl_nextstep_sparse_embed.pth')
    if os.path.exists(best):
        return best
    periodics = sorted(
        glob_mod.glob(os.path.join(resume_dir, 'ssl_ep*.pth')),
        key=lambda p: int(re.search(r'ssl_ep(\d+)', p).group(1))
            if re.search(r'ssl_ep(\d+)', p) else -1)
    if periodics:
        return periodics[-1]
    # Also check sanity checkpoints from max_steps runs
    sanities = sorted(
        glob_mod.glob(os.path.join(resume_dir, 'sanity_*.pth')),
        key=os.path.getmtime)
    if sanities:
        return sanities[-1]
    return None


def _resume_slot(s, ckpt_path, device):
    """Restore a ModelSlot from a checkpoint. Returns (epoch, global_step)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'pretrainer_state_dict' in ckpt:
        s.pretrainer.load_state_dict(ckpt['pretrainer_state_dict'])
    elif 'backbone_state_dict' in ckpt:
        s.backbone.load_state_dict(ckpt['backbone_state_dict'], strict=False)
    if 'optimizer_state_dict' in ckpt:
        s.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        s.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    s.best_val = ckpt.get('best_val', float('inf'))
    s.patience_counter = ckpt.get('patience_counter', 0)
    epoch = ckpt.get('epoch', 0)
    global_step = ckpt.get('global_step', 0)
    log.info("[%s] Resumed from %s (epoch=%d, step=%d, best_val=%.4f, gpu=%s)",
             s.name, os.path.basename(ckpt_path), epoch, global_step,
             s.best_val, ckpt.get('gpu_type', '?'))
    return epoch, global_step


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)

    # ── Data (shared, loaded once) ──
    data_dirs = [d.strip() for d in args.data_dir.split(',') if d.strip()]
    train_data = PureSSLDataset(
        data_dirs, split='train', window_size=args.window_size,
        index_file=getattr(args, 'index_file', None))
    val_data = PureSSLDataset(
        data_dirs, split='val', window_size=args.window_size,
        index_file=getattr(args, 'index_file', None))

    if len(train_data) < 10:
        log.error("Too few training pairs (%d). Check data dirs.", len(train_data))
        return

    # ── Shared DataLoader ──
    # Models run sequentially on the same batch, so peak VRAM = largest
    # single model's activations + all 4 models' weights (~47 MB total).
    # The DimGroupBatchSampler adapts batch size per (ndim, N) group.
    batch_size = args.batch_size
    # Extract file_ids for cache-aware stripe sampling
    train_file_ids = [train_data._index[i][0] for i in range(len(train_data))]
    val_file_ids = [val_data._index[i][0] for i in range(len(val_data))]
    lru_size = getattr(train_data, '_LRU_SIZE', 512)
    train_sampler = DimGroupBatchSampler(
        train_data._group_keys, batch_size, drop_last=True, shuffle=True,
        file_ids=train_file_ids, lru_size=lru_size,
        calibration_path=args.calibration)
    val_sampler = DimGroupBatchSampler(
        val_data._group_keys, batch_size, drop_last=False, shuffle=False,
        file_ids=val_file_ids, lru_size=lru_size,
        calibration_path=args.calibration)

    # Sanity check: val sampler must produce batches
    val_batch_count = len(val_sampler)
    log.info("Val sampler: %d batches (from %d pairs)", val_batch_count, len(val_data))
    if val_batch_count == 0:
        raise RuntimeError(
            f"Val sampler produced 0 batches from {len(val_data)} pairs / "
            f"{len(set(val_data._group_keys))} groups. "
            f"Stripe threshold likely too high for val set.")

    worker_kwargs = {}
    if args.num_workers > 0:
        worker_kwargs['prefetch_factor'] = 4
        worker_kwargs['persistent_workers'] = True

    train_loader = DataLoader(
        train_data, batch_sampler=train_sampler,
        collate_fn=collate_pure_ssl,
        num_workers=args.num_workers, pin_memory=True,
        **worker_kwargs)
    val_loader = DataLoader(
        val_data, batch_sampler=val_sampler,
        collate_fn=collate_pure_ssl, num_workers=0)

    # Log main process memory before DataLoader forks workers
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    log.info("Main process VmRSS before workers: %s", line.strip())
                    break
    except OSError:
        pass

    log.info("Train: %d pairs, Val: %d pairs, Batch: %d, Workers: %d",
             len(train_data), len(val_data), batch_size, args.num_workers)

    # ── Build all models ──
    output_base = args.output_dir
    if args.seed != 42:
        output_base = f"{args.output_dir}_s{args.seed}"
    os.makedirs(output_base, exist_ok=True)

    configs = CONFIGS
    if args.config:
        configs = [(n, c) for n, c in CONFIGS if n == args.config]

    loss_weights = _parse_losses(args.losses)
    active_losses = [k for k, v in loss_weights.items() if v > 0]
    log.info("Loss config (--losses %s): %s", args.losses,
             ", ".join(f"{k}={v}" for k, v in loss_weights.items() if v > 0))
    if args.no_temporal:
        log.info("Temporal encoder DISABLED (--no-temporal)")
    if args.pooler_type != 'induced':
        log.info("Pooler type: %s", args.pooler_type)

    slots = []
    for name, cfg in configs:
        slot = ModelSlot(name, cfg, device, args.lr, args.wd, output_base,
                         loss_weights=loss_weights, pooler_type=args.pooler_type,
                         scheduler_kind=args.scheduler, max_steps=args.max_steps)
        log.info("[%s] %d params | d_rnn=%d gatv2_hidden=%d layers=%d temporal=%d",
                 name, slot.n_params, cfg['d_rnn'], cfg['gatv2_hidden'],
                 cfg['n_gat_layers'], cfg['temporal_layers'])
        slots.append(slot)

    total_params = sum(s.n_params for s in slots)
    log.info("Total params across %d models: %d (%.1f MB)",
             len(slots), total_params, total_params * 4 / 1e6)

    # ── Resume from checkpoint ──
    start_epoch = 0
    global_step = 0
    if args.resume:
        for s in slots:
            ckpt_path = _find_resume_checkpoint(
                os.path.join(args.resume, s.name) if os.path.isdir(os.path.join(args.resume, s.name))
                else args.resume)
            if ckpt_path:
                ep, gs = _resume_slot(s, ckpt_path, device)
                start_epoch = max(start_epoch, ep + 1)
                global_step = max(global_step, gs)
            else:
                log.warning("[%s] No checkpoint found in %s, starting fresh", s.name, args.resume)

    # ── Training loop ──
    max_steps = args.max_steps
    t_train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Reset per-model epoch stats
        for s in slots:
            if s.alive:
                s.pretrainer.train()
                s.reset_epoch_stats()

        t_epoch = time.time()
        t_data_start = time.time()

        for batch in train_loader:
            t_data = time.time() - t_data_start

            # Transfer batch to GPU once
            batch_gpu = {k: v.to(device, non_blocking=True)
                         if hasattr(v, 'to') else v
                         for k, v in batch.items()}

            # --no-temporal: strip history so backbone skips temporal encoder
            if args.no_temporal:
                batch_gpu.pop('coords_hist', None)
                batch_gpu.pop('fitness_hist', None)

            # Forward/backward for each alive model
            step_parts = []
            for s in slots:
                if not s.alive:
                    continue
                t_fwd = time.time()
                try:
                    loss, metrics = s.pretrainer(batch_gpu)
                    if torch.isnan(loss) or torch.isinf(loss):
                        log.warning("[%s] NaN/Inf loss at step %d (B=%d), skipping backward",
                                    s.name, global_step,
                                    batch_gpu['v_indices'][-1].item() + 1)
                        s.optimizer.zero_grad(set_to_none=True)
                        del loss
                        step_parts.append(f"{s.name}: SKIP (nan)")
                    else:
                        loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            s.pretrainer.parameters(), 1.0)
                        s.optimizer.step()
                        s.optimizer.zero_grad(set_to_none=True)
                        del loss  # release computation graph
                        s.accumulate(metrics)
                        dt = time.time() - t_fwd
                        loss_detail = " ".join(
                            f"{k}={v:.3f}" for k, v in metrics.items()
                            if k != 'loss' and isinstance(v, (int, float)) and v != -1)
                        step_parts.append(
                            f"{s.name}: L={metrics['loss']:.4f} "
                            f"lr={_current_lr(s):.2e} "
                            f"bwd={dt:.3f}s g={grad_norm:.1f} [{loss_detail}]")
                except RuntimeError as e:
                    err = str(e).lower()
                    if "out of memory" in err:
                        log.error("[%s] OOM at step %d — marking dead. %s",
                                  s.name, global_step, e)
                        s.alive = False
                        s.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        step_parts.append(f"{s.name}: DEAD (OOM)")
                    else:
                        raise
            # Free VRAM between models when sharing a GPU.
            if len(slots) > 1:
                gc.collect()
                torch.cuda.empty_cache()

            global_step += 1

            # Per-step logging
            do_log = (global_step <= 20
                      or (global_step <= 200 and global_step % 10 == 0)
                      or global_step % 500 == 0)
            if do_log:
                B = batch_gpu['v_indices'][-1].item() + 1
                N = batch_gpu['node_feat'].shape[0] // B
                vram = torch.cuda.memory_allocated() / 1e6 if device.type == 'cuda' else 0
                log.info("step %d | data %.3fs | B=%d N=%d | vram %.0fMB | %s",
                         global_step, t_data, B, N, vram,
                         " | ".join(step_parts))

            # Step-periodic checkpointing (every save_every steps)
            save_every = getattr(args, 'save_every', 5000)
            if save_every > 0 and global_step % save_every == 0 and global_step > 0:
                for s in slots:
                    if s.alive:
                        ckpt_path = os.path.join(s.output_dir, f'step_{global_step}.pth')
                        torch.save(_full_checkpoint(s, epoch, global_step), ckpt_path)
                        log.info("[%s] Periodic checkpoint at step %d → %s",
                                 s.name, global_step, ckpt_path)

            # Memory diagnostic — track RSS of main + workers + cgroup total
            if global_step % 500 == 0:
                try:
                    # 1. Main process RSS
                    mem_main = {}
                    with open('/proc/self/status') as f:
                        for line in f:
                            if line.startswith(('VmRSS:', 'RssFile:', 'RssAnon:')):
                                key, val = line.split(':')
                                mem_main[key] = val.strip()

                    # 2. Worker RSS via DataLoader's known worker PIDs
                    worker_rss = []
                    worker_mmaps = []
                    try:
                        worker_procs = getattr(
                            getattr(train_loader, '_iterator', None),
                            '_workers', [])
                        for w in worker_procs:
                            cpid = w.pid
                            try:
                                with open(f'/proc/{cpid}/status') as wf:
                                    for line in wf:
                                        if line.startswith('VmRSS:'):
                                            worker_rss.append(int(line.split()[1]))
                                            break
                            except OSError:
                                worker_rss.append(-1)
                            try:
                                with open(f'/proc/{cpid}/maps') as mf:
                                    worker_mmaps.append(
                                        sum(1 for l in mf if '.data.bin' in l))
                            except OSError:
                                worker_mmaps.append(-1)
                    except Exception:
                        pass

                    # 3. Cgroup memory usage — try all known paths
                    cgroup_mb = '?'
                    try:
                        with open('/proc/self/cgroup') as f:
                            for line in f:
                                cg_dir = line.strip().split(':')[-1]
                                for mem_file in ['memory.current', 'memory.usage_in_bytes']:
                                    p = f'/sys/fs/cgroup{cg_dir}/{mem_file}'
                                    if os.path.exists(p):
                                        with open(p) as uf:
                                            cgroup_mb = f'{int(uf.read().strip()) / 1e9:.1f}G'
                                        break
                                if cgroup_mb != '?':
                                    break
                    except OSError:
                        pass

                    w_rss_str = '+'.join(f'{r // 1024}M' for r in worker_rss) if worker_rss else 'none'
                    w_rss_total = sum(worker_rss) // 1024 if worker_rss else 0
                    w_mmaps_str = '+'.join(str(m) for m in worker_mmaps) if worker_mmaps else 'none'
                    log.info("MEM step %d: main=%s workers=%s (total=%dM, mmaps=%s) cgroup=%s",
                             global_step, mem_main.get('VmRSS', '?'),
                             w_rss_str, w_rss_total, w_mmaps_str, cgroup_mb)
                except Exception as e:
                    log.info("MEM step %d: diagnostic error: %s", global_step, e)

            if 0 < max_steps <= global_step:
                break

            t_data_start = time.time()

        if 0 < max_steps <= global_step:
            log.info("Reached max_steps=%d, stopping.", max_steps)
            for s in slots:
                if s.alive:
                    torch.save(_full_checkpoint(s, epoch, global_step),
                               os.path.join(s.output_dir, f'sanity_{global_step}.pth'))
            break

        # ── Schedulers ──
        for s in slots:
            if s.alive:
                s.scheduler.step()

        # ── Validation ──
        max_val = getattr(args, 'max_val_steps', 500)
        if max_val is None or max_val <= 0:
            max_val = float('inf')
        val_results = {}
        for s in slots:
            if not s.alive:
                val_results[s.name] = {'loss': float('inf')}
                continue
            s.pretrainer.eval()
            val_loss = 0.0
            val_metrics = defaultdict(float)
            val_n = 0
            with torch.no_grad():
                for vb in val_loader:
                    vb_gpu = {k: v.to(device, non_blocking=True)
                              if hasattr(v, 'to') else v
                              for k, v in vb.items()}
                    vl, vm = s.pretrainer(vb_gpu)
                    val_loss += vm['loss']
                    for k, v in vm.items():
                        val_metrics[k] += v
                    val_n += 1
                    if val_n >= max_val:
                        break
            if val_n == 0:
                log.error("[%s] ZERO val batches at epoch %d! Sampler bug. "
                          "Saving emergency checkpoint.", s.name, epoch)
                torch.save(_full_checkpoint(s, epoch, global_step),
                           os.path.join(s.output_dir, f'emergency_ep{epoch}.pth'))
                val_results[s.name] = {'loss': float('inf')}
                continue
            val_avg = val_loss / val_n
            val_m = {k: v / val_n for k, v in val_metrics.items()}
            val_m['loss'] = val_avg
            val_results[s.name] = val_m

        # ── Epoch summary ──
        epoch_dt = time.time() - t_epoch
        parts = []
        for s in slots:
            if not s.alive:
                parts.append(f"{s.name}: DEAD")
                continue
            nb = max(s.epoch_n_batches, 1)
            t_avg = s.epoch_train_loss / nb
            v = val_results[s.name]
            v_str = f"v={v['loss']:.4f}" if v['loss'] < float('inf') else "v=NONE"
            parts.append(
                f"{s.name} t={t_avg:.4f} {v_str} "
                f"fit={v.get('fitness_rank', 0):.4f}")
        log.info("Epoch %3d (%.0fs) | %s", epoch, epoch_dt, " | ".join(parts))

        # ── Checkpointing + patience ──
        for s in slots:
            if not s.alive:
                continue
            val_avg = val_results[s.name]['loss']
            if val_avg < s.best_val:
                s.best_val = val_avg
                s.patience_counter = 0
                ckpt_path = os.path.join(s.output_dir,
                                         'ssl_nextstep_sparse_embed.pth')
                ckpt = _full_checkpoint(s, epoch, global_step)
                ckpt['val_loss'] = val_avg
                ckpt['config'] = {
                    'backbone': 'sparse_coord',
                    'topology': 'COORDINATE_KNN',
                    'node_in': NODE_DIM, 'edge_in': EDGE_DIM,
                    'global_in': GLOBAL_DIM,
                    'stored_global_dim': STORED_GLOBAL_DIM,
                    'adaptive_k': 'max(2, min(8, N//4))',
                    'embed_dim': s.cfg['gatv2_hidden'],
                    'global_dim': s.cfg['gatv2_hidden'],
                    'gatv2_layers': s.cfg['n_gat_layers'],
                    **s.cfg,
                }
                torch.save(ckpt, ckpt_path)
                log.info("  [%s] → best val=%.4f saved to %s",
                         s.name, val_avg, ckpt_path)
            else:
                s.patience_counter += 1

            if epoch % 10 == 0 and epoch > 0:
                periodic = os.path.join(s.output_dir, f'ssl_ep{epoch}.pth')
                torch.save(_full_checkpoint(s, epoch, global_step), periodic)

        # ── Early stopping per model ──
        for s in slots:
            if s.alive and epoch >= args.min_epochs and s.patience_counter >= args.patience:
                log.info("[%s] Early stopping at epoch %d (patience=%d, best=%.4f)",
                         s.name, epoch, args.patience, s.best_val)
                s.alive = False

        # All dead? Stop.
        if not any(s.alive for s in slots):
            log.info("All models stopped. Ending training.")
            break

    # ── Final results ──
    wall_time = time.time() - t_train_start
    log.info("Training complete in %.0fs (%.1f hours)", wall_time, wall_time / 3600)

    for s in slots:
        results = {
            'config': s.cfg,
            'name': s.name,
            'param_count': s.n_params,
            'best_val_loss': s.best_val,
            'wall_seconds': wall_time,
            'seed': args.seed,
        }
        results_path = os.path.join(s.output_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        log.info("[%s] best_val=%.4f params=%d → %s",
                 s.name, s.best_val, s.n_params, results_path)


def main():
    parser = argparse.ArgumentParser(
        description="Shared-batch multi-model SSL training (4 configs, 1 GPU)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Comma-separated data directories")
    parser.add_argument("--index-file", type=str, default=None,
                        help="Pre-built index file (from build_index.py)")
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/full_multi")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N steps (0=full epochs)")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--config", type=str, default=None,
                        choices=[n for n, _ in CONFIGS],
                        help="Run single config (for parallel jobs)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Path to calibration JSON from calibrate_vram.py. "
                             "Enables per-shape adaptive batch sizing "
                             "(nearest-N fallback for unmeasured shapes). "
                             "If omitted, uses the BYTES_PER_ND env scalar.")
    parser.add_argument("--save-every", type=int, default=5000, dest='save_every',
                        help="Save checkpoint every N steps (0=epoch only)")
    parser.add_argument("--max-val-steps", type=int, default=500, dest='max_val_steps',
                        help="Max validation batches per epoch (0=unlimited)")
    parser.add_argument("--losses", type=str, default="all",
                        help="Comma-separated loss names (e.g. nextstep_node) or 'all' "
                             "or 'all,-node_recon,-edge_recon'")
    parser.add_argument("--scheduler", type=str, default="warmrestarts",
                        choices=["warmrestarts", "cosine"],
                        help="LR scheduler. 'warmrestarts' (default, legacy) uses "
                             "epoch-based CosineAnnealingWarmRestarts(T_0=10, T_mult=2); "
                             "'cosine' uses step-based CosineAnnealingLR(T_max=max_steps) — "
                             "required for long single-epoch runs (Phase 3+).")
    parser.add_argument("--no-temporal", action="store_true", dest='no_temporal',
                        help="Bypass temporal encoder (pass static snapshot to GAT)")
    parser.add_argument("--pooler-type", type=str, default="induced",
                        choices=["induced", "mean"], dest='pooler_type',
                        help="Temporal pooler: 'induced' (learned) or 'mean' (trivial)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    train(args)


if __name__ == '__main__':
    main()
