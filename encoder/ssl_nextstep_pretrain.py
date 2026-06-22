"""
ssl_nextstep_pretrain.py — SSL pretrainer for next-step node feature prediction.

Compares 3 backbones:
  - pna_gatv2:      PNAGATv2Backbone (GPU-resident data)
  - temporal_gatv2:  TemporalGATv2Backbone (DataLoader, temporal data)
  - npa:             NPABackbone (DataLoader, temporal data, no edges)

All trained end-to-end (no freezing).

Usage:
    python -m encoder.ssl_nextstep_pretrain --backbone pna_gatv2 --device cuda
    python -m encoder.ssl_nextstep_pretrain --backbone temporal_gatv2 --device cuda
    python -m encoder.ssl_nextstep_pretrain --backbone npa --device cuda
"""

import argparse
import logging
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM
from .ssl_heads import _make_mlp, EdgeReconHead, MSESigmoidHead
from .ssl_nextstep_dataset import (
    PREDICT_FEATURES, GPUDataset, load_gpu_dataset,
    NextStepPairDataset, collate_nextstep,
)

log = logging.getLogger(__name__)

TEMPORAL_BACKBONES = ('temporal_gatv2', 'npa', 'npa_edges')

LOSS_WEIGHTS = {
    'nextstep_node': 1.0,
    'nextstep_global': 0.3,
    'fitness_rank': 0.2,
    'edge_recon': 0.2,
}


# ======================================================================
# Pretrainer
# ======================================================================

class NextStepPretrainer(nn.Module):
    """SSL pretrainer: next-step prediction + auxiliary objectives."""

    def __init__(self, backbone, embed_dim=64, global_dim=32,
                 edge_dim=EDGE_DIM, mask_ratio=0.15, has_edges=True,
                 device='cpu'):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.has_edges = has_edges

        n_predict = len(PREDICT_FEATURES)
        self.node_head = _make_mlp(embed_dim, embed_dim, n_predict, act='gelu')
        self.global_head = _make_mlp(global_dim, global_dim, GLOBAL_DIM, act='gelu')
        self.fitness_head = MSESigmoidHead(embed_dim, hidden=embed_dim)
        if has_edges:
            self.edge_head = EdgeReconHead(embed_dim, edge_dim, hidden=embed_dim)

        self.to(device)

    def forward(self, batch):
        edge_mask = None
        original_edge_attr = batch['edge_attr']

        if self.has_edges:
            E = original_edge_attr.shape[0]
            edge_mask = torch.rand(E, device=original_edge_attr.device) < self.mask_ratio
            edge_attr_input = original_edge_attr.clone()
            edge_attr_input[edge_mask] = 0.0
        else:
            edge_attr_input = original_edge_attr

        # Build encode kwargs (temporal fields passed if present)
        encode_kw = dict(
            v_indices=batch['v_indices'],
            e_indices=batch['e_indices'],
        )
        for key in ('coords_hist', 'fitness_hist', 'n_valid',
                     'coordinates', 'fitness'):
            if key in batch:
                if key == 'coordinates':
                    encode_kw['coords_current'] = batch[key]
                elif key == 'fitness' and 'fitness_hist' in batch:
                    encode_kw['fitness_current'] = batch[key]
                else:
                    encode_kw[key] = batch[key]

        h, e, h_per_head, h_global = self.backbone.encode(
            batch['node_feat'], batch['edge_index'], edge_attr_input,
            batch['global_feat'], **encode_kw,
        )

        pred_nodes = self.node_head(h)
        target_nodes = batch['target_node_feat'][:, PREDICT_FEATURES]
        l_node = F.mse_loss(pred_nodes, target_nodes)

        pred_global = self.global_head(h_global)
        l_global = F.mse_loss(pred_global, batch['target_global'])

        l_fitness = self.fitness_head.loss(h, batch['fitness_rank'])

        E = original_edge_attr.shape[0]
        if (self.has_edges and e is not None and edge_mask is not None
                and e.shape[0] == E):
            l_edge = self.edge_head.loss(e, original_edge_attr, edge_mask)
        else:
            l_edge = torch.tensor(0.0, device=h.device)

        total = (LOSS_WEIGHTS['nextstep_node'] * l_node
                 + LOSS_WEIGHTS['nextstep_global'] * l_global
                 + LOSS_WEIGHTS['fitness_rank'] * l_fitness
                 + LOSS_WEIGHTS['edge_recon'] * l_edge)

        metrics = {
            'loss': total.item(),
            'nextstep_node': l_node.item(),
            'nextstep_global': l_global.item(),
            'fitness_rank': l_fitness.item(),
            'edge_recon': l_edge.item(),
        }
        return total, metrics, pred_nodes.detach(), pred_global.detach()


# ======================================================================
# R² (stays on GPU)
# ======================================================================

def compute_r2_per_feature(pred, target):
    ss_res = ((pred - target) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return 1 - ss_res / (ss_tot + 1e-8)


@torch.no_grad()
def evaluate_gpu(pretrainer, gpu_ds, batch_size):
    """Evaluate on GPU-resident dataset."""
    pretrainer.eval()
    all_pred, all_target, all_current = [], [], []
    all_pred_g, all_target_g, all_current_g = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in gpu_ds.iter_batches(batch_size, shuffle=False):
        loss, _, pred_n, pred_g = pretrainer(batch)
        total_loss += loss.item()
        n_batches += 1
        all_pred.append(pred_n)
        all_target.append(batch['target_node_feat'][:, PREDICT_FEATURES])
        all_current.append(batch['node_feat'][:, PREDICT_FEATURES])
        all_pred_g.append(pred_g)
        all_target_g.append(batch['target_global'])
        all_current_g.append(batch['global_feat'])

    return _compute_eval_result(all_pred, all_target, all_current,
                                all_pred_g, all_target_g, all_current_g,
                                total_loss, n_batches)


@torch.no_grad()
def evaluate_loader(pretrainer, val_loader, device):
    """Evaluate with DataLoader (temporal backbones)."""
    pretrainer.eval()
    all_pred, all_target, all_current = [], [], []
    all_pred_g, all_target_g, all_current_g = [], [], []
    total_loss = 0.0
    n_batches = 0

    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, _, pred_n, pred_g = pretrainer(batch)
        total_loss += loss.item()
        n_batches += 1
        all_pred.append(pred_n)
        all_target.append(batch['target_node_feat'][:, PREDICT_FEATURES])
        all_current.append(batch['node_feat'][:, PREDICT_FEATURES])
        all_pred_g.append(pred_g)
        all_target_g.append(batch['target_global'])
        all_current_g.append(batch['global_feat'])

    return _compute_eval_result(all_pred, all_target, all_current,
                                all_pred_g, all_target_g, all_current_g,
                                total_loss, n_batches)


def _compute_eval_result(all_pred, all_target, all_current,
                         all_pred_g, all_target_g, all_current_g,
                         total_loss, n_batches):
    r2_model = compute_r2_per_feature(torch.cat(all_pred), torch.cat(all_target))
    r2_persist = compute_r2_per_feature(torch.cat(all_current), torch.cat(all_target))
    r2_g_model = compute_r2_per_feature(torch.cat(all_pred_g), torch.cat(all_target_g))
    r2_g_persist = compute_r2_per_feature(torch.cat(all_current_g), torch.cat(all_target_g))
    return {
        'avg_loss': total_loss / max(n_batches, 1),
        'r2_node_model': r2_model,
        'r2_node_persist': r2_persist,
        'r2_global_model': r2_g_model,
        'r2_global_persist': r2_g_persist,
    }


def log_r2(eval_result):
    r2_m = eval_result['r2_node_model']
    r2_p = eval_result['r2_node_persist']
    n_feat = r2_m.shape[0]
    from .similarity_graph import NODE_NAMES
    names = [NODE_NAMES[i] for i in PREDICT_FEATURES[:n_feat]]

    log.info("  Node R² (model / persistence):")
    for i in range(n_feat):
        log.info("    %-25s  model=%.4f  persist=%.4f  Δ=%+.4f",
                 names[i], r2_m[i].item(), r2_p[i].item(),
                 (r2_m[i] - r2_p[i]).item())
    log.info("    %-25s  model=%.4f  persist=%.4f  Δ=%+.4f",
             "MEAN", r2_m.mean().item(), r2_p.mean().item(),
             (r2_m.mean() - r2_p.mean()).item())

    r2_gm = eval_result['r2_global_model']
    r2_gp = eval_result['r2_global_persist']
    log.info("  Global R²: model=%.4f  persist=%.4f  Δ=%+.4f",
             r2_gm.mean().item(), r2_gp.mean().item(),
             (r2_gm.mean() - r2_gp.mean()).item())


# ======================================================================
# Backbone factory
# ======================================================================

def _create_backbone(name, device):
    if name == 'pna_gatv2':
        from .backbone import PNAGATv2Backbone
        return PNAGATv2Backbone(
            node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
            pna_hidden=64, pna_out=32, pna_layers=4,
            gatv2_hidden=64, gatv2_layers=2, n_heads=4,
            device=device)
    elif name == 'temporal_gatv2':
        from .temporal_gatv2_backbone import TemporalGATv2Backbone
        return TemporalGATv2Backbone(
            node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
            d_rnn=32, d_temporal=32, gru_window=8,
            gatv2_hidden=64, gatv2_layers=2, n_heads=4,
            global_out_dim=32, dropout=0.1, device=device)
    elif name == 'npa':
        from .npa_backbone import NPABackbone
        return NPABackbone(
            hidden_dim=64, global_out_dim=32, n_heads=4,
            d_rnn=32, d_model=32, d_ind=64,
            device=device)
    elif name == 'npa_edges':
        from .npa_edge_backbone import NPAEdgeBackbone
        return NPAEdgeBackbone(
            edge_in=EDGE_DIM,
            hidden_dim=64, global_out_dim=32, n_heads=4,
            d_rnn=32, d_model=32, d_ind=64,
            device=device)
    else:
        raise ValueError(f"Unknown backbone: {name}")


def _is_temporal(name):
    return name in TEMPORAL_BACKBONES


# ======================================================================
# Training
# ======================================================================

def train(args):
    device = torch.device(args.device)
    temporal = _is_temporal(args.backbone)
    has_edges = args.backbone not in ('npa',)

    # --- Data ---
    if temporal:
        train_data = NextStepPairDataset(
            args.data_dir, split='train', temporal=True,
            window_size=args.window_size)
        val_data = NextStepPairDataset(
            args.data_dir, split='val', temporal=True,
            window_size=args.window_size)

        while len(train_data) < 100:
            log.info("Waiting for data... %d pairs", len(train_data))
            time.sleep(30)
            train_data.refresh()
            val_data.refresh()

        log.info("Starting [%s]: %d train, %d val pairs (DataLoader)",
                 args.backbone, len(train_data), len(val_data))
    else:
        train_ds = load_gpu_dataset(args.data_dir, 'train', device,
                                    max_pairs=args.max_pairs)
        val_ds = load_gpu_dataset(args.data_dir, 'val', device,
                                  max_pairs=args.max_val_pairs)
        log.info("Starting [%s]: %d train, %d val pairs (GPU-resident)",
                 args.backbone, len(train_ds), len(val_ds))

    # --- Model ---
    backbone = _create_backbone(args.backbone, device)
    pretrainer = NextStepPretrainer(
        backbone, embed_dim=64, global_dim=32,
        edge_dim=EDGE_DIM, has_edges=has_edges, device=device)

    optimizer = torch.optim.AdamW(
        pretrainer.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir,
                             f'ssl_nextstep_{args.backbone}.pth')

    for epoch in range(args.epochs):
        # --- Train ---
        pretrainer.train()
        epoch_loss = 0.0
        epoch_node_loss = 0.0
        n_batches = 0

        if temporal:
            # DimGroupBatchSampler for D-homogeneous batches
            from .ssl_pretrain_k2 import DimGroupBatchSampler
            train_sampler = DimGroupBatchSampler(
                train_data._ndims, args.batch_size,
                drop_last=True, shuffle=True)
            train_loader = DataLoader(
                train_data, batch_sampler=train_sampler,
                collate_fn=collate_nextstep, num_workers=0)
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss, metrics, _, _ = pretrainer(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += metrics['loss']
                epoch_node_loss += metrics['nextstep_node']
                n_batches += 1
        else:
            for batch in train_ds.iter_batches(args.batch_size):
                loss, metrics, _, _ = pretrainer(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += metrics['loss']
                epoch_node_loss += metrics['nextstep_node']
                n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_node = epoch_node_loss / max(n_batches, 1)

        # --- Validate ---
        if temporal:
            val_sampler = DimGroupBatchSampler(
                val_data._ndims, args.batch_size,
                drop_last=False, shuffle=False)
            val_loader = DataLoader(
                val_data, batch_sampler=val_sampler,
                collate_fn=collate_nextstep, num_workers=0)
            eval_result = evaluate_loader(pretrainer, val_loader, device)
        else:
            eval_result = evaluate_gpu(pretrainer, val_ds, args.batch_size)

        val_loss = eval_result['avg_loss']
        r2_mean = eval_result['r2_node_model'].mean().item()
        r2_persist = eval_result['r2_node_persist'].mean().item()

        log.info(
            "Epoch %3d | train %.4f (node=%.4f) | val %.4f | "
            "R²=%.4f (persist=%.4f) | lr=%.1e | %s",
            epoch, avg_loss, avg_node, val_loss,
            r2_mean, r2_persist, optimizer.param_groups[0]['lr'],
            args.backbone)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            log_r2(eval_result)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'pretrainer_state_dict': pretrainer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_r2_mean': r2_mean,
                'config': {
                    'backbone': args.backbone,
                    'node_in': NODE_DIM, 'edge_in': EDGE_DIM,
                    'global_in': GLOBAL_DIM,
                    'embed_dim': 64, 'global_dim': 32,
                },
            }, ckpt_path)
            log.info("  Saved best → %s (R²=%.4f)", ckpt_path, r2_mean)
        else:
            patience_counter += 1

        if (epoch >= args.min_epochs
                and patience_counter >= args.patience):
            log.info("Early stopping after %d epochs", args.patience)
            break

    log.info("Done [%s]. Best val loss: %.4f", args.backbone, best_val_loss)


# ======================================================================
# Persistence baseline
# ======================================================================

def eval_persistence_only(args):
    from torch.utils.data import DataLoader as DL
    val_data = NextStepPairDataset(args.data_dir, split='val')
    if len(val_data) == 0:
        log.error("No validation pairs found in %s", args.data_dir)
        return

    val_loader = DL(val_data, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_nextstep, num_workers=0)

    all_current, all_target = [], []
    all_current_g, all_target_g = [], []
    for batch in val_loader:
        all_current.append(batch['node_feat'][:, PREDICT_FEATURES])
        all_target.append(batch['target_node_feat'][:, PREDICT_FEATURES])
        all_current_g.append(batch['global_feat'])
        all_target_g.append(batch['target_global'])

    r2_persist = compute_r2_per_feature(torch.cat(all_current), torch.cat(all_target))
    r2_persist_g = compute_r2_per_feature(torch.cat(all_current_g), torch.cat(all_target_g))

    from .similarity_graph import NODE_NAMES
    n_feat = len(PREDICT_FEATURES)
    names = [NODE_NAMES[i] for i in PREDICT_FEATURES[:n_feat]]

    log.info("Persistence baseline R² (val set, %d pairs):", len(val_data))
    log.info("  Node features:")
    for i in range(n_feat):
        log.info("    %-25s  R²=%.4f", names[i], r2_persist[i].item())
    log.info("    %-25s  R²=%.4f", "MEAN", r2_persist.mean().item())
    log.info("  Global features: mean R²=%.4f", r2_persist_g.mean().item())


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SSL next-step prediction — 3 backbone comparison")
    parser.add_argument("--backbone", type=str, default="pna_gatv2",
                        choices=["pna_gatv2", "temporal_gatv2", "npa", "npa_edges"])
    parser.add_argument("--data-dir", type=str, default="DATASETS/NPA_GPU")
    parser.add_argument("--output-dir", type=str,
                        default="checkpoints/ssl_nextstep")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Cap GPU-resident train pairs (0=all)")
    parser.add_argument("--max-val-pairs", type=int, default=0,
                        help="Cap GPU-resident val pairs (0=all)")
    parser.add_argument("--window-size", type=int, default=8,
                        help="Temporal window W for coords_hist (8,12,17,24,35,50)")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=15)
    parser.add_argument("--eval-persistence-only", action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    if args.eval_persistence_only:
        eval_persistence_only(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
