"""
npa_ssl_online.py — Online SSL pretraining for NPA backbone (K=2).

Generates trajectories on-the-fly with SHADE+LS1 GPU-native operators
on augmented CEC2017 functions.  No disk I/O, no DataLoader — data is
created fresh every training step.

Three allocation strategies (uniform random per trajectory):
  - all_shade:  SHADE only — pure exploration regime
  - all_ls1:    SHADE + LS1 on top-3 every gen — pure exploitation regime
  - mixed:      SHADE + LS1 50% prob after 50% gens — transition regime

SSL objectives:
  1. LS1 benefit (graph-level BCE): does the population benefit from LS1?
  2. Per-node LS1 need (node-level BCE): which individuals benefit?
  3. Fitness rank (node-level MSE): auxiliary population structure signal

Usage:
    python -m encoder.npa_ssl_online --device cuda --steps 10000

All operations GPU-resident.  Zero CPU transfers except n_valid.item()
for GRU slice (required by PyTorch API).
"""

import argparse
import json
import logging
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .npa_backbone import NPABackbone
from .ring_buffer import PopulationRingBuffer
from .augmented_cec2017 import AugmentedCEC2017
from .graph_features import soft_rank

from GNN_MOS_Classic.shared import SHADEMemoryGPU, shade_gpu, batched_mtsls1_gpu

log = logging.getLogger(__name__)

STRATEGIES = ('all_shade', 'all_ls1', 'mixed')


class OnlineSSLTrainer(nn.Module):
    """Online SSL pretraining: generate trajectories + train backbone."""

    def __init__(self, device, n_pop=100, n_gens=50, window=8,
                 ls1_evals=10, **backbone_kwargs):
        super().__init__()
        self.device = device
        self.n_pop = n_pop
        self.n_gens = n_gens
        self.window = window
        self.ls1_evals = ls1_evals

        # Backbone
        backbone_kwargs.setdefault('hidden_dim', 64)
        backbone_kwargs.setdefault('global_out_dim', 32)
        backbone_kwargs.setdefault('window', window)
        backbone_kwargs.setdefault('device', str(device))
        self.backbone = NPABackbone(**backbone_kwargs)

        hidden_dim = self.backbone.hidden_dim
        global_out_dim = self.backbone.global_out_dim

        # SSL heads
        self.ls1_global_head = nn.Linear(global_out_dim, 1)
        self.ls1_node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1))
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1))

        # Data generation (not nn parameters)
        self.aug = AugmentedCEC2017(device=str(device))

    @torch.no_grad()
    def _generate_trajectory(self, strategy):
        """Run SHADE+LS1 on one augmented function. Returns history + labels.

        Everything stays on GPU. No numpy conversions.
        """
        device = self.device
        N = self.n_pop
        n_gens = self.n_gens

        fn = self.aug.sample()
        D = fn.D

        # Init population
        x = torch.rand(N, D, device=device, dtype=torch.float64) * 200 - 100
        fitness = fn(x)

        # Ring buffer
        rb = PopulationRingBuffer(self.window, N, D, device)
        rb.reset(fitness.min().detach())

        shade_mem = SHADEMemoryGPU(H=10, device=device)

        # Label accumulators (GPU tensors)
        ls1_applied_flags = torch.zeros(n_gens, device=device, dtype=torch.bool)
        ls1_improved_flags = torch.zeros(n_gens, device=device, dtype=torch.bool)
        ls1_target_idx = torch.full((n_gens,), -1, device=device, dtype=torch.long)

        for gen in range(n_gens):
            gen_frac = gen / max(n_gens - 1, 1)
            rb.push(x, fitness)

            # SHADE
            children, _, _ = shade_gpu(
                x, fitness, x, shade_mem, gen_frac, lb=-100.0, ub=100.0)
            children_fit = fn(children)
            shade_mem.report_child_fitness(children_fit)

            improved = children_fit < fitness
            x_new = torch.where(improved.unsqueeze(1), children, x)
            f_new = torch.where(improved, children_fit, fitness)

            # LS1 decision
            do_ls1 = False
            if strategy == 'all_ls1':
                do_ls1 = True
            elif strategy == 'mixed':
                do_ls1 = gen_frac >= 0.5 and random.random() < 0.5

            if do_ls1:
                k = min(3, N)
                _, top_idx = torch.topk(-f_new, k)
                pick = top_idx[random.randint(0, k - 1)]

                x_t = x_new[pick:pick + 1]
                f_t = f_new[pick:pick + 1]

                x_ref, f_ref, _ = batched_mtsls1_gpu(
                    x_t, f_t, fn,
                    lb=-100.0, ub=100.0,
                    max_evals=self.ls1_evals, sr_frac=0.2)

                ls1_applied_flags[gen] = True
                ls1_target_idx[gen] = pick
                ls1_improved_flags[gen] = f_ref[0] < f_t[0]

                x_new[pick] = x_ref[0]
                f_new[pick] = f_ref[0]

            x = x_new
            fitness = f_new

        # Get history from ring buffer
        coords_hist, fitness_hist, valid_mask, n_valid = rb.get_history()

        # Aggregate labels
        n_applied = ls1_applied_flags.sum()
        n_improved = ls1_improved_flags.sum()
        ls1_benefit_ratio = (n_improved.float() / n_applied.float().clamp(min=1.0))

        # Per-node label: was this individual an LS1 target that improved?
        # Use the LAST generation's state for the per-node label
        last_target = ls1_target_idx[-1]
        node_ls1_label = torch.zeros(N, device=device)
        if ls1_applied_flags[-1] and ls1_improved_flags[-1]:
            node_ls1_label[last_target] = 1.0

        # Fitness rank of final population [0, 1]
        fitness_rank = soft_rank(fitness.float()) / max(N - 1, 1)

        labels = {
            'ls1_benefit_ratio': ls1_benefit_ratio,    # scalar
            'node_ls1_label': node_ls1_label,          # (N,)
            'fitness_rank': fitness_rank,               # (N,)
            'strategy': strategy,
            'N': N,
            'D': D,
        }

        return (coords_hist[:, :N, :D], fitness_hist[:, :N],
                valid_mask, n_valid, rb.f_init,
                x.float(), fitness.float(), labels)

    def train_step(self, optimizer, B=1):
        """Generate B trajectories + forward + backward. Returns loss dict."""
        device = self.device

        if B == 1:
            return self._train_step_single(optimizer)

        # B > 1: generate sequentially, forward batched
        all_coords = []
        all_fitness = []
        all_coords_current = []
        all_fitness_current = []
        all_f_init = []
        all_labels = []
        all_N = []
        all_D = []
        valid_mask = None
        n_valid = None

        for _ in range(B):
            strategy = random.choice(STRATEGIES)
            (ch, fh, vm, nv, fi,
             xc, fc, labels) = self._generate_trajectory(strategy)
            all_coords.append(ch)
            all_fitness.append(fh)
            all_coords_current.append(xc)
            all_fitness_current.append(fc)
            all_f_init.append(fi)
            all_labels.append(labels)
            all_N.append(labels['N'])
            all_D.append(labels['D'])
            valid_mask = vm
            n_valid = nv

        # All trajectories have same W but may differ in N, D
        # For simplicity, pad to max_N, max_D and use v_indices
        max_N = max(all_N)
        max_D = max(all_D)
        W = self.window

        # Pad and concatenate histories → (W, B*max_N, max_D)
        coords_flat = torch.zeros(W, B * max_N, max_D, device=device)
        fitness_flat = torch.zeros(W, B * max_N, device=device)
        coords_cur = torch.zeros(B * max_N, max_D, device=device)
        fitness_cur = torch.zeros(B * max_N, device=device)

        for b in range(B):
            N_b, D_b = all_N[b], all_D[b]
            offset = b * max_N
            coords_flat[:, offset:offset + N_b, :D_b] = all_coords[b]
            fitness_flat[:, offset:offset + N_b] = all_fitness[b]
            coords_cur[offset:offset + N_b, :D_b] = all_coords_current[b]
            fitness_cur[offset:offset + N_b] = all_fitness_current[b]

        # v_indices for PopulationTransformer
        v_indices = torch.arange(B, device=device).repeat_interleave(max_N)

        f_init_per_sample = torch.stack(all_f_init)  # (B,)

        h_temporal = self.backbone.temporal_gru(
            coords_flat, fitness_flat, n_valid,
            N_out=B * max_N, D_out=max_D)

        h_grid = self.backbone.cross_dim(h_temporal, fitness_cur)
        h, h_global, _ = self.backbone.pop_transformer(
            h_grid, coords_cur, v_indices)
        h = self.backbone.feature_injector(
            h, coords_cur, fitness_cur, v_indices)
        total_loss, loss_dict = self._compute_losses(
            h, h_global, all_labels, B, max_N, v_indices)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        loss_dict['grad_norm'] = grad_norm.detach()
        loss_dict['D'] = all_labels[0]['D']
        loss_dict['strategy'] = all_labels[0]['strategy']
        return loss_dict

    def _train_step_single(self, optimizer):
        """Optimized path for B=1."""
        strategy = random.choice(STRATEGIES)
        (coords_hist, fitness_hist, valid_mask, n_valid, f_init,
         coords_current, fitness_current, labels) = self._generate_trajectory(strategy)

        N = labels['N']
        h, _, _, h_global = self.backbone.encode_npa(
            coords_hist, fitness_hist, valid_mask, n_valid,
            coords_current, fitness_current, f_init)

        # Losses
        # 1. LS1 benefit (graph-level MSE on sigmoid output)
        benefit_pred = torch.sigmoid(
            self.ls1_global_head(h_global).squeeze(-1))            # (1,)
        benefit_target = labels['ls1_benefit_ratio'].unsqueeze(0)  # (1,)
        loss_benefit = F.mse_loss(benefit_pred, benefit_target)

        # 2. Per-node LS1 need (BCE, masked to top-3)
        node_logit = self.ls1_node_head(h).squeeze(-1).clamp(-10, 10)
        node_target = labels['node_ls1_label']           # (N,)
        _, top3 = torch.topk(-fitness_current, min(3, N))
        loss_node = F.binary_cross_entropy_with_logits(
            node_logit[top3], node_target[top3])

        # 3. Fitness rank (MSE)
        rank_pred = torch.sigmoid(self.rank_head(h).squeeze(-1))
        rank_target = labels['fitness_rank']
        loss_rank = F.mse_loss(rank_pred, rank_target)

        total_loss = 1.0 * loss_benefit + 1.0 * loss_node + 0.3 * loss_rank

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()

        return {
            'benefit': loss_benefit.detach(),
            'node': loss_node.detach(),
            'rank': loss_rank.detach(),
            'total': total_loss.detach(),
            'strategy': labels['strategy'],
            'D': labels['D'],
            'grad_norm': grad_norm.detach(),
            'h_std': h.std().detach(),
            'hg_std': h_global.std().detach(),
            'benefit_pred': benefit_pred.mean().detach(),
            'node_pred_mean': torch.sigmoid(node_logit).mean().detach(),
            'rank_pred_std': rank_pred.std().detach(),
        }

    def _compute_losses(self, h, h_global, all_labels, B, max_N, v_indices):
        """Compute SSL losses for batched forward."""
        device = h.device

        # 1. LS1 benefit (graph-level MSE on sigmoid output)
        benefit_pred = torch.sigmoid(
            self.ls1_global_head(h_global).squeeze(-1))            # (B,)
        benefit_targets = torch.stack(
            [lb['ls1_benefit_ratio'] for lb in all_labels])
        loss_benefit = F.mse_loss(benefit_pred, benefit_targets)

        # 2. Per-node LS1 need (BCE, top-3 per sample)
        node_logit = self.ls1_node_head(h).squeeze(-1).clamp(-10, 10)
        node_losses = []
        for b, lb in enumerate(all_labels):
            offset = b * max_N
            N_b = lb['N']
            _, top3 = torch.topk(-lb['fitness_rank'], min(3, N_b))
            logit_b = node_logit[offset + top3]
            target_b = lb['node_ls1_label'][top3]
            node_losses.append(F.binary_cross_entropy_with_logits(
                logit_b, target_b))
        loss_node = torch.stack(node_losses).mean()

        # 3. Fitness rank
        rank_pred = torch.sigmoid(self.rank_head(h).squeeze(-1))
        rank_targets = torch.zeros(B * max_N, device=device)
        valid_mask = torch.zeros(B * max_N, device=device, dtype=torch.bool)
        for b, lb in enumerate(all_labels):
            offset = b * max_N
            N_b = lb['N']
            rank_targets[offset:offset + N_b] = lb['fitness_rank']
            valid_mask[offset:offset + N_b] = True

        loss_rank = F.mse_loss(rank_pred[valid_mask], rank_targets[valid_mask])

        # 4. VICReg anti-collapse: variance + covariance on per-node h
        # Per-graph to avoid cross-graph statistics
        vic_losses = []
        for b in range(B):
            offset = b * max_N
            N_b = all_labels[b]['N']
            h_b = h[offset:offset + N_b]                         # (N_b, H)
            # Variance hinge: force std >= 1 per feature dim
            std_b = torch.sqrt(h_b.var(dim=0) + 1e-4)
            var_loss = F.relu(1.0 - std_b).mean()
            # Covariance: penalize off-diagonal correlations
            h_c = h_b - h_b.mean(dim=0)
            cov = (h_c.T @ h_c) / max(N_b - 1, 1)
            cov.fill_diagonal_(0)
            cov_loss = (cov ** 2).sum() / h_b.shape[1]
            vic_losses.append(var_loss + 0.04 * cov_loss)
        loss_vic = torch.stack(vic_losses).mean()

        total = (1.0 * loss_benefit + 1.0 * loss_node
                 + 0.3 * loss_rank + 1.0 * loss_vic)
        return total, {
            'benefit': loss_benefit.detach(),
            'node': loss_node.detach(),
            'rank': loss_rank.detach(),
            'vic': loss_vic.detach(),
            'total': total.detach(),
            'h_std': h.std().detach(),
            'hg_std': h_global.std().detach(),
            'benefit_pred': benefit_pred.mean().detach(),
            'node_pred_mean': torch.sigmoid(node_logit).mean().detach(),
            'rank_pred_std': rank_pred.std().detach(),
        }


# ======================================================================
# Training loop
# ======================================================================

def cosine_lr(optimizer, step, total_steps, lr_max, warmup=500):
    if step < warmup:
        lr = lr_max * step / max(warmup, 1)
    else:
        progress = (step - warmup) / max(total_steps - warmup, 1)
        lr = lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = max(lr, 1e-6)
    return lr


def calibrate_batch_size(model, optimizer, device, max_B=32):
    """Find the largest batch size that fits in GPU memory.

    Tries B=1, 2, 4, 8, ... doubling each time. On OOM, backs off to
    the last successful value. Returns the safe B.
    """
    log.info("Calibrating batch size (max_B=%d)...", max_B)
    safe_B = 1
    B = 1
    while B <= max_B:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            _ = model.train_step(optimizer, B=B)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            free = torch.cuda.get_device_properties(device).total_memory / 1e9 - peak
            log.info("  B=%d: OK (peak=%.1fGB, free=%.1fGB)", B, peak, free)
            safe_B = B
            # If less than 1GB headroom, stop
            if free < 1.0:
                break
            B *= 2
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            log.info("  B=%d: OOM — backing off to B=%d", B, safe_B)
            break
    log.info("Calibrated batch size: B=%d", safe_B)
    return safe_B


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    model = OnlineSSLTrainer(
        device=device,
        n_pop=args.n_pop,
        n_gens=args.n_gens,
        window=args.window,
        ls1_evals=args.ls1_evals,
        d_model=args.d_model,
        d_rnn=args.d_rnn,
        d_ind=args.d_ind,
        hidden_dim=args.hidden_dim,
        global_out_dim=args.global_out_dim,
        n_heads=args.n_heads,
        level2_layers=args.level2_layers,
        level3_layers=args.level3_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("OnlineSSLTrainer: %d params", n_params)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Auto-calibrate batch size if requested
    if args.batch_trajectories <= 0:
        args.batch_trajectories = calibrate_batch_size(
            model, optimizer, device, max_B=32)
        # Reset optimizer state after calibration
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    best_loss = float('inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt.get('step', 0) + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        log.info("Resumed from %s (step %d)", args.resume, start_step)

    strat_counts = {s: 0 for s in STRATEGIES}
    running = {k: 0.0 for k in
               ['benefit', 'node', 'rank', 'vic', 'total',
                'grad_norm', 'h_std', 'hg_std',
                'benefit_pred', 'node_pred_mean', 'rank_pred_std']}
    t_start = time.time()

    diag_path = os.path.join(args.save_dir, 'diagnostics.jsonl')
    diag_file = open(diag_path, 'a')
    log.info("Diagnostics → %s", diag_path)

    for step in range(start_step, args.steps):
        lr = cosine_lr(optimizer, step, args.steps, args.lr, args.warmup)

        t0 = time.time()
        loss_dict = model.train_step(optimizer, B=args.batch_trajectories)
        dt = time.time() - t0

        strat = loss_dict.get('strategy', '?')
        if isinstance(strat, str):
            strat_counts[strat] = strat_counts.get(strat, 0) + 1

        for k in running:
            v = loss_dict.get(k)
            if v is not None:
                running[k] += (v.item() if hasattr(v, 'item') else float(v))

        if (step + 1) % args.log_every == 0:
            n = args.log_every
            elapsed = time.time() - t_start
            avg = {k: v / n for k, v in running.items()}

            log.info(
                "Step %d/%d (%.1fs, %.2fs/step, lr=%.2e) | "
                "loss=%.4f ben=%.4f node=%.4f rank=%.4f | "
                "gn=%.3f h=%.3f hg=%.3f | "
                "bp=%.3f np=%.3f rp_std=%.4f | %s",
                step + 1, args.steps, elapsed, dt, lr,
                avg['total'], avg['benefit'], avg['node'], avg['rank'],
                avg['grad_norm'], avg['h_std'], avg['hg_std'],
                avg['benefit_pred'], avg['node_pred_mean'], avg['rank_pred_std'],
                {k: v for k, v in strat_counts.items()})

            # Write JSONL for later analysis
            diag = {
                'step': step + 1, 'lr': lr, 'dt': dt,
                'elapsed': elapsed, **avg,
                'strats': dict(strat_counts),
                'B': args.batch_trajectories,
            }
            diag_file.write(json.dumps(diag) + '\n')
            diag_file.flush()

            running = {k: 0.0 for k in running}

        if (step + 1) % args.save_every == 0:
            ckpt = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'backbone_state_dict': model.backbone.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_loss': best_loss,
                'args': vars(args),
            }
            path = os.path.join(args.save_dir, f'npa_ssl_step{step+1}.pth')
            torch.save(ckpt, path)
            log.info("  Saved %s", path)

    # Final save
    path = os.path.join(args.save_dir, 'npa_ssl_final.pth')
    torch.save({
        'step': args.steps - 1,
        'model_state_dict': model.state_dict(),
        'backbone_state_dict': model.backbone.state_dict(),
        'args': vars(args),
    }, path)
    log.info("Final checkpoint: %s", path)
    diag_file.close()


def _pairwise_cosine(h, n_pairs=25):
    """Mean pairwise cosine similarity over random pairs. (N, H) -> scalar."""
    N = h.shape[0]
    idx = torch.randperm(N, device=h.device)[:min(2 * n_pairs, N)]
    n = len(idx) // 2
    if n == 0:
        return 0.0
    a, b = h[idx[:n]], h[idx[n:2*n]]
    cos = F.cosine_similarity(a, b, dim=1)
    return cos.mean().item()


@torch.no_grad()
def _cosine_diagnostics(model, coords_flat, fitness_flat, coords_cur,
                        fitness_cur, n_valid, v_indices, B, max_N, max_D):
    """Compute per-stage cosine similarity using one graph from the batch."""
    # Use first graph only (indices 0..max_N-1)
    cf1 = coords_flat[:, :max_N, :]           # (W, N, D)
    ff1 = fitness_flat[:, :max_N]              # (W, N)
    cc1 = coords_cur[:max_N]                   # (N, D)
    fc1 = fitness_cur[:max_N]                  # (N,)

    h_temporal = model.backbone.temporal_gru(cf1, ff1, n_valid)
    cos_gru = _pairwise_cosine(h_temporal.reshape(max_N, -1))

    h_grid = model.backbone.cross_dim(h_temporal, fc1)
    # Cosine on the grid: flatten (N, D, h) → (N, D*h) per-individual
    cos_twostage = _pairwise_cosine(h_grid.reshape(max_N, -1))

    h, _, _ = model.backbone.pop_transformer(h_grid, cc1)
    cos_pop = _pairwise_cosine(h)

    h = model.backbone.feature_injector(h, cc1, fc1)
    cos_final = _pairwise_cosine(h)

    return {
        'cos_gru': round(cos_gru, 4),
        'cos_twostage': round(cos_twostage, 4),
        'cos_pop': round(cos_pop, 4),
        'cos_final': round(cos_final, 4),
    }


def train_offline(args):
    """Train from pre-collected snapshots on disk. Much faster than online."""
    from torch.utils.data import DataLoader
    from .npa_gpu_dataset import NPAMemmapDataset

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    dataset = NPAMemmapDataset(args.data_dir, dim_filter=args.dim_filter)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        collate_fn=dataset.collate,
                        shuffle=True, num_workers=0,
                        pin_memory=True, drop_last=True)
    log.info("DataLoader: %d samples, batch=%d, %d batches/epoch",
             len(dataset), args.batch_size, len(loader))

    # Model — reuse OnlineSSLTrainer for backbone + heads
    model = OnlineSSLTrainer(
        device=device, n_pop=100, n_gens=50, window=args.window,
        d_model=args.d_model, d_rnn=args.d_rnn, d_ind=args.d_ind,
        hidden_dim=args.hidden_dim, global_out_dim=args.global_out_dim,
        n_heads=args.n_heads, level2_layers=args.level2_layers,
        level3_layers=args.level3_layers, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("OnlineSSLTrainer: %d params", n_params)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        log.info("Resumed from %s", args.resume)

    total_steps = args.epochs * len(loader)
    diag_path = os.path.join(args.save_dir, 'diagnostics.jsonl')
    diag_file = open(diag_path, 'a')
    log.info("Diagnostics → %s", diag_path)

    running = {k: 0.0 for k in
               ['benefit', 'node', 'rank', 'vic', 'total',
                'grad_norm', 'h_std', 'hg_std',
                'benefit_pred', 'node_pred_mean', 'rank_pred_std']}
    global_step = 0
    n_skipped = 0
    t_start = time.time()

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(loader):
            lr = cosine_lr(optimizer, global_step, total_steps,
                           args.lr, args.warmup)

            # Move to GPU
            B = batch['coords_hist'].shape[0]
            W = batch['coords_hist'].shape[1]
            max_N = batch['max_N']
            max_D = batch['max_D']

            ch = batch['coords_hist'].to(device, dtype=torch.float32)  # (B, W, N, D)
            fh = batch['fitness_hist'].to(device)                      # (B, W, N)
            coords_flat = ch.permute(1, 0, 2, 3).reshape(W, B * max_N, max_D)
            fitness_flat = fh.permute(1, 0, 2).reshape(W, B * max_N)
            coords_cur = ch[:, -1].reshape(B * max_N, max_D)
            fitness_cur = fh[:, -1].reshape(B * max_N)
            n_valid = torch.tensor(W, device=device, dtype=torch.long)
            v_indices = torch.arange(B, device=device).repeat_interleave(max_N)

            # fes_ratio: per-sample scalar → expand to per-individual
            fes_ratio = batch['fes_ratio'].to(device).repeat_interleave(max_N)

            h_temporal = model.backbone.temporal_gru(
                coords_flat, fitness_flat, n_valid,
                N_out=B * max_N, D_out=max_D)
            h_grid = model.backbone.cross_dim(h_temporal, fitness_cur, fes_ratio)
            h, h_global, _ = model.backbone.pop_transformer(
                h_grid, coords_cur, v_indices)
            h = model.backbone.feature_injector(
                h, coords_cur, fitness_cur, v_indices)

            # Build labels in the format _compute_losses expects
            all_labels = []
            for b in range(B):
                N_b = batch['valid_N'][b].item()
                all_labels.append({
                    'N': N_b,
                    'node_ls1_label': batch['oracle_switch'][b, :N_b].to(device),
                    'ls1_benefit_ratio': batch['optimal_ls1_frac'][b].to(device),
                    'fitness_rank': batch['fitness_rank'][b, :N_b].to(device),
                    'strategy': 'offline',
                })

            total_loss, loss_dict = model._compute_losses(
                h, h_global, all_labels, B, max_N, v_indices)

            # Skip NaN/Inf loss batches
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                n_skipped += 1
                global_step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Skip if gradients are NaN or extreme (prevents AdamW state corruption)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                n_skipped += 1
                global_step += 1
                continue

            optimizer.step()

            # Accumulate diagnostics
            loss_dict['grad_norm'] = grad_norm.detach()
            for k in running:
                v = loss_dict.get(k)
                if v is not None:
                    running[k] += (v.item() if hasattr(v, 'item') else float(v))

            if (global_step + 1) % args.log_every == 0:
                n = args.log_every
                elapsed = time.time() - t_start
                dt = elapsed / (global_step + 1)
                avg = {k: v / n for k, v in running.items()}
                log.info(
                    "E%d S%d/%d (%.1fs, %.3fs/step, lr=%.2e) | "
                    "loss=%.4f ben=%.4f node=%.4f rank=%.4f | "
                    "gn=%.3f h=%.3f hg=%.3f",
                    epoch, global_step + 1, total_steps, elapsed, dt, lr,
                    avg['total'], avg['benefit'], avg['node'], avg['rank'],
                    avg['grad_norm'], avg['h_std'], avg['hg_std'])

                # Per-stage cosine diagnostics (use last batch, cheap)
                with torch.no_grad():
                    cos_diag = _cosine_diagnostics(
                        model, coords_flat, fitness_flat, coords_cur,
                        fitness_cur, n_valid, v_indices, B, max_N, max_D)

                diag = {'step': global_step + 1, 'epoch': epoch,
                        'lr': lr, 'dt': dt, 'elapsed': elapsed,
                        **avg, **cos_diag, 'B': B, 'skipped': n_skipped}
                diag_file.write(json.dumps(diag) + '\n')
                diag_file.flush()
                running = {k: 0.0 for k in running}

            if (global_step + 1) % args.save_every == 0:
                ckpt = {
                    'step': global_step, 'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'backbone_state_dict': model.backbone.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
                }
                path = os.path.join(args.save_dir,
                                    f'npa_ssl_offline_step{global_step+1}.pth')
                torch.save(ckpt, path)
                log.info("  Saved %s", path)

            global_step += 1

    # Final save
    path = os.path.join(args.save_dir, 'npa_ssl_offline_final.pth')
    torch.save({
        'step': global_step - 1,
        'model_state_dict': model.state_dict(),
        'backbone_state_dict': model.backbone.state_dict(),
        'args': vars(args),
    }, path)
    log.info("Final checkpoint: %s (%d steps)", path, global_step)
    diag_file.close()


def main():
    parser = argparse.ArgumentParser(
        description="SSL pretraining for NPA K=2 (online or offline)")

    parser.add_argument("--mode", choices=["online", "offline"],
                        default="offline")

    # Training (shared)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-dir", type=str, default="encoder/npa_checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    # Online mode
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-trajectories", type=int, default=0,
                        help="B trajectories per step (0=auto-calibrate)")
    parser.add_argument("--n-pop", type=int, default=100)
    parser.add_argument("--n-gens", type=int, default=50)
    parser.add_argument("--ls1-evals", type=int, default=10)

    # Offline mode
    parser.add_argument("--data-dir", type=str, default="DATASETS/NPA_GPU")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dim-filter", type=int, default=None,
                        help="Only load specific dimensionality (10, 30, 50)")

    # Architecture (shared)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-rnn", type=int, default=32)
    parser.add_argument("--d-ind", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--global-out-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--level2-layers", type=int, default=2)
    parser.add_argument("--level3-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    if args.mode == 'online':
        train(args)
    else:
        train_offline(args)


if __name__ == '__main__':
    main()
