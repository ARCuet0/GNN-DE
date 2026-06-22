"""
npa_ssl_pretrain.py — SSL pretraining script for the NPA backbone.

Usage:
    python -m encoder.npa_ssl_pretrain --device cuda \
        --data-dir DATASETS/SOFT_K4_ORACLE_v2 --epochs 30 --batch-size 32

    # Single dimension:
    python -m encoder.npa_ssl_pretrain --device cuda \
        --data-dir DATASETS/SOFT_K4_ORACLE_v2 --dim-filter 30

    # Resume:
    python -m encoder.npa_ssl_pretrain --device cuda \
        --resume encoder/npa_checkpoints/npa_ssl_best.pth
"""

import argparse
import logging
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from .npa_ssl_pretrainer import NPASSLPretrainer
from .npa_trajectory_dataset import TrajectoryDataset

log = logging.getLogger(__name__)


def cosine_lr(optimizer, step, total_steps, lr_max, warmup_steps=500):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        lr = lr_max * step / max(warmup_steps, 1)
    else:
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        lr = lr_max * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = max(lr, 1e-6)
    return lr


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # -- Data --
    log.info("Loading datasets from %s ...", args.data_dir)
    ds_train = TrajectoryDataset(
        args.data_dir, window=args.window,
        dim_filter=args.dim_filter, max_N=args.max_N, split='train')
    ds_val = TrajectoryDataset(
        args.data_dir, window=args.window,
        dim_filter=args.dim_filter, max_N=args.max_N, split='val')

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    loader_val = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    log.info("Train: %d samples (%d batches), Val: %d samples",
             len(ds_train), len(loader_train), len(ds_val))

    # -- Model --
    model = NPASSLPretrainer(
        K=4,
        window=args.window,
        d_model=args.d_model,
        d_rnn=args.d_rnn,
        d_ind=args.d_ind,
        hidden_dim=args.hidden_dim,
        global_out_dim=args.global_out_dim,
        n_heads=args.n_heads,
        level2_layers=args.level2_layers,
        level3_layers=args.level3_layers,
        max_D=args.max_D,
        dropout=args.dropout,
        device=str(device),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log.info("NPASSLPretrainer: %d params", n_params)

    # -- Optimizer --
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(loader_train)
    global_step = 0
    start_epoch = 0
    best_val_loss = float('inf')

    # -- Resume --
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        log.info("Resumed from %s (epoch %d, step %d, val_loss=%.4f)",
                 args.resume, start_epoch, global_step, best_val_loss)

    # -- Training loop --
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_losses = {k: 0.0 for k in ['alloc', 'eff', 'rank', 'total']}
        n_batches = 0
        t0 = time.time()

        for batch in loader_train:
            lr = cosine_lr(optimizer, global_step, total_steps,
                           args.lr, args.warmup_steps)

            loss, loss_dict = model(batch, device)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()
            n_batches += 1
            global_step += 1

        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        dt = time.time() - t0

        # -- Validation --
        model.eval()
        val_losses = {k: 0.0 for k in ['alloc', 'eff', 'rank', 'total']}
        n_val = 0
        with torch.no_grad():
            for batch in loader_val:
                _, loss_dict = model(batch, device)
                for k, v in loss_dict.items():
                    val_losses[k] += v.item()
                n_val += 1
        for k in val_losses:
            val_losses[k] /= max(n_val, 1)

        log.info("Epoch %d/%d (%.1fs) lr=%.2e | "
                 "train: total=%.4f alloc=%.4f eff=%.4f rank=%.4f | "
                 "val: total=%.4f alloc=%.4f eff=%.4f rank=%.4f",
                 epoch + 1, args.epochs, dt, lr,
                 epoch_losses['total'], epoch_losses['alloc'],
                 epoch_losses['eff'], epoch_losses['rank'],
                 val_losses['total'], val_losses['alloc'],
                 val_losses['eff'], val_losses['rank'])

        # -- Checkpoint --
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']

        ckpt_data = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'backbone_state_dict': model.backbone.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_losses['total'],
            'best_val_loss': best_val_loss,
            'args': vars(args),
        }

        if (epoch + 1) % args.save_every == 0:
            path = os.path.join(args.save_dir, f'npa_ssl_ep{epoch+1}.pth')
            torch.save(ckpt_data, path)
            log.info("  Saved checkpoint: %s", path)

        if is_best:
            path = os.path.join(args.save_dir, 'npa_ssl_best.pth')
            torch.save(ckpt_data, path)
            log.info("  New best val_loss=%.4f → %s", best_val_loss, path)

    log.info("Training complete. Best val_loss=%.4f", best_val_loss)


def main():
    parser = argparse.ArgumentParser(
        description="SSL pretrain NPA backbone on trajectory data")

    # Data
    parser.add_argument("--data-dir", type=str,
                        default="DATASETS/SOFT_K4_ORACLE_v2")
    parser.add_argument("--dim-filter", type=int, default=None,
                        help="Only use this dimensionality (10, 30, or 50)")
    parser.add_argument("--max-N", type=int, default=300,
                        help="Pad populations to this size")
    parser.add_argument("--window", type=int, default=8,
                        help="Trajectory window length W")

    # Training
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-dir", type=str,
                        default="encoder/npa_checkpoints")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)

    # Architecture
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-rnn", type=int, default=32)
    parser.add_argument("--d-ind", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--global-out-dim", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--level2-layers", type=int, default=2)
    parser.add_argument("--level3-layers", type=int, default=2)
    parser.add_argument("--max-D", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    train(args)


if __name__ == '__main__':
    main()
