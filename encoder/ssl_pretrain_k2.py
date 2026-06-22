"""
ssl_pretrain_k2.py — Streaming SSL pretrainer for K=2 (SHADE+LS1).

Trains any backbone (PNAGATv2, Transformer, PureTransformer, ...) on
data being written live by collect_npa_gpu.py. Re-scans the data
directory every --refresh-every epochs to incorporate new snapshots.

5 SSL objectives:
  1. SwitchDecision (node-level, primary):   predict oracle_switch_adjusted
  2. LS1Benefit (node-level, auxiliary):      predict ls1_delta magnitude
  3. BudgetAllocation (graph-level):          predict optimal_ls1_frac
  4. FitnessRank (node-level, structural):    predict fitness_rank
  5. EdgeReconstruction (edge-level):         reconstruct 15% masked edges

Usage:
    # Start alongside running collect_npa_gpu.py:
    python -m encoder.ssl_pretrain_k2 --device cuda --data-dir DATASETS/NPA_GPU

    # With specific backbone:
    python -m encoder.ssl_pretrain_k2 --device cuda --backbone transformer
"""

import argparse
import hashlib
import logging
import os
import pickle
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from .similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM

log = logging.getLogger(__name__)

# ======================================================================
# Dataset
# ======================================================================

class StreamingK2Dataset(Dataset):
    """Lazy-loading dataset: stores only (file_path, offset) per snapshot.

    Loads snapshots from disk on __getitem__ via an LRU file cache.
    Re-scans the data directory on refresh() to discover new pkl files.
    Val = snapshots from fids {3, 8, 18} (one per CEC2017 category).
    """

    VAL_FIDS = frozenset({3, 8, 18})
    _FILE_CACHE_SIZE = 16  # number of pkl files to keep in RAM

    def __init__(self, data_dir, split='train', device='cpu',
                 require_history=False):
        self.data_dir = data_dir
        self.split = split
        self.device = device
        self.require_history = require_history
        # Index: list of (filepath, offset_within_file)
        self._index = []
        self._ndims = []  # parallel to _index: ndim per snapshot
        self._loaded_files = set()
        # LRU cache: filepath → list of snapshot dicts
        self._cache = {}
        self._cache_order = []
        self.refresh()

    def refresh(self):
        """Scan for new pkl files and index them. Returns count of new files."""
        if not os.path.isdir(self.data_dir):
            return 0
        pkl_files = sorted(f for f in os.listdir(self.data_dir)
                           if f.endswith('.pkl') and not f.endswith('.tmp'))
        n_new = 0
        for fname in pkl_files:
            if fname in self._loaded_files:
                continue
            fpath = os.path.join(self.data_dir, fname)
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    data = [data]
                for i, s in enumerate(data):
                    fid = s.get('fid', 0)
                    is_val = fid in self.VAL_FIDS
                    if self.require_history and not s.get('has_history', False):
                        continue
                    if (self.split == 'val') == is_val:
                        self._index.append((fpath, i))
                        self._ndims.append(s.get('ndim', 10))
                self._loaded_files.add(fname)
                n_new += 1
            except (EOFError, pickle.UnpicklingError, OSError):
                pass  # file still being written
        return n_new

    def _load_file(self, fpath):
        """Load a pkl file with LRU eviction."""
        if fpath in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(fpath)
            self._cache_order.append(fpath)
            return self._cache[fpath]
        # Evict oldest if cache full
        while len(self._cache_order) >= self._FILE_CACHE_SIZE:
            evict = self._cache_order.pop(0)
            del self._cache[evict]
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            data = [data]
        self._cache[fpath] = data
        self._cache_order.append(fpath)
        return data

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        fpath, offset = self._index[idx]
        s = self._load_file(fpath)[offset]

        # All tensors created on CPU — DataLoader transfers to GPU via collate
        node_feat = torch.as_tensor(s['node_feat'], dtype=torch.float32)
        edge_index = torch.as_tensor(s['edge_index'], dtype=torch.long)
        edge_attr = torch.as_tensor(s['edge_attr'], dtype=torch.float32)
        gf = s['global_feat']
        global_feat = torch.as_tensor(
            gf.squeeze(0) if gf.ndim > 1 else gf, dtype=torch.float32)

        # Oracle labels
        switch_labels = torch.as_tensor(
            s['oracle_switch_adjusted'], dtype=torch.float32)
        ls1_delta = torch.as_tensor(s['ls1_delta'], dtype=torch.float32)
        fitness_rank = torch.as_tensor(s['fitness_rank'], dtype=torch.float32)
        optimal_ls1_frac = torch.tensor(
            s['optimal_ls1_frac'], dtype=torch.float32)

        result = {
            'node_feat': node_feat,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'global_feat': global_feat,
            'switch_labels': switch_labels,
            'ls1_delta': ls1_delta,
            'fitness_rank': fitness_rank,
            'optimal_ls1_frac': optimal_ls1_frac,
        }

        # Temporal data (present when has_history=True)
        if s.get('has_history', False) and 'coords_hist' in s:
            coords_hist = torch.as_tensor(
                s['coords_hist'], dtype=torch.float32)   # (W, N, D)
            fitness_hist = torch.as_tensor(
                s['fitness_hist'], dtype=torch.float32)   # (W, N)
            result['coords_hist'] = coords_hist
            result['fitness_hist'] = fitness_hist
            result['n_valid'] = torch.tensor(
                coords_hist.shape[0], dtype=torch.long)

        return result


def collate_graphs(batch):
    """Collate variable-size graphs into a single batched graph."""
    node_offset = 0
    all_nodes, all_edges, all_edge_attr = [], [], []
    all_global = []
    all_switch, all_ls1_delta, all_fitness_rank = [], [], []
    all_ls1_frac = []
    v_indices, e_indices = [], []

    has_temporal = 'coords_hist' in batch[0]
    all_coords_hist, all_fitness_hist = [], []

    for i, g in enumerate(batch):
        N = g['node_feat'].shape[0]
        E = g['edge_index'].shape[1]

        all_nodes.append(g['node_feat'])
        all_edges.append(g['edge_index'] + node_offset)
        all_edge_attr.append(g['edge_attr'])
        all_global.append(g['global_feat'])

        all_switch.append(g['switch_labels'])
        all_ls1_delta.append(g['ls1_delta'])
        all_fitness_rank.append(g['fitness_rank'])
        all_ls1_frac.append(g['optimal_ls1_frac'])

        if has_temporal:
            all_coords_hist.append(g['coords_hist'])    # (W, N_i, D)
            all_fitness_hist.append(g['fitness_hist'])   # (W, N_i)

        v_indices.append(torch.full((N,), i, dtype=torch.long,
                                    device=g['node_feat'].device))
        e_indices.append(torch.full((E,), i, dtype=torch.long,
                                    device=g['node_feat'].device))
        node_offset += N

    result = {
        'node_feat': torch.cat(all_nodes),
        'edge_index': torch.cat(all_edges, dim=1),
        'edge_attr': torch.cat(all_edge_attr),
        'global_feat': torch.stack(all_global),
        'v_indices': torch.cat(v_indices),
        'e_indices': torch.cat(e_indices),
        'switch_labels': torch.cat(all_switch),
        'ls1_delta': torch.cat(all_ls1_delta),
        'fitness_rank': torch.cat(all_fitness_rank),
        'optimal_ls1_frac': torch.stack(all_ls1_frac),
    }

    if has_temporal:
        # All graphs in batch share same D (guaranteed by DimGroupBatchSampler)
        result['coords_hist'] = torch.cat(all_coords_hist, dim=1)   # (W, N_total, D)
        result['fitness_hist'] = torch.cat(all_fitness_hist, dim=1)  # (W, N_total)
        result['n_valid'] = batch[0]['n_valid']                      # scalar

    return result


# ======================================================================
# Batch sampler for D-homogeneous batches
# ======================================================================

class DimGroupBatchSampler(torch.utils.data.Sampler):
    """Yields batches where all samples share the same ndim.

    Required for temporal backbones where TemporalGRUEncoder needs a
    single D per call: (W, N_total, D).
    """

    def __init__(self, ndim_list, batch_size, drop_last=True, shuffle=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        # Group indices by ndim
        self._groups = defaultdict(list)
        for idx, d in enumerate(ndim_list):
            self._groups[d].append(idx)

    def __iter__(self):
        all_batches = []
        for d, indices in self._groups.items():
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        total = 0
        for indices in self._groups.values():
            n = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size:
                n += 1
            total += n
        return total


# ======================================================================
# SSL Heads
# ======================================================================

class SwitchDecisionHead(nn.Module):
    """Node-level: predict oracle_switch_adjusted (binary)."""
    def __init__(self, embed_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1))

    def forward(self, h_nodes, labels):
        logits = self.head(h_nodes).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        with torch.no_grad():
            acc = ((logits > 0).float() == labels).float().mean().item()
        return loss, acc


class LS1BenefitHead(nn.Module):
    """Node-level: predict ls1_delta (log1p-scaled, already in [0, ~5])."""
    def __init__(self, embed_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1))

    def forward(self, h_nodes, ls1_delta):
        pred = self.head(h_nodes).squeeze(-1)
        # Normalize target to ~[0,1] for stable MSE
        target = ls1_delta.clamp(max=10.0) / 10.0
        return F.mse_loss(pred, target)


class BudgetAllocationHead(nn.Module):
    """Graph-level: predict optimal_ls1_frac ∈ [0, 1]."""
    def __init__(self, embed_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, h_global, target_frac):
        pred = self.head(h_global).squeeze(-1)
        return F.mse_loss(pred, target_frac)


class FitnessRankHead(nn.Module):
    """Node-level: predict fitness_rank ∈ [0, 1]."""
    def __init__(self, embed_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, h_nodes, ranks):
        pred = self.head(h_nodes).squeeze(-1)
        return F.mse_loss(pred, ranks)


class EdgeReconstructionHead(nn.Module):
    """Edge-level: reconstruct masked edge features."""
    def __init__(self, embed_dim, edge_dim=EDGE_DIM, mask_ratio=0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, edge_dim))

    def forward(self, h_edges, original_edge_attr, edge_mask):
        if edge_mask.sum() == 0:
            return torch.tensor(0.0, device=h_edges.device, requires_grad=True)
        pred = self.head(h_edges[edge_mask])
        target = original_edge_attr[edge_mask]
        return F.mse_loss(pred, target)


# ======================================================================
# Pretrainer
# ======================================================================

LOSS_WEIGHTS = {
    'switch': 1.0,
    'ls1_benefit': 0.3,
    'budget': 0.5,
    'fitness_rank': 0.3,
    'edge_recon': 0.3,
}


class K2SSLPretrainer(nn.Module):
    """SSL pretrainer wrapping any backbone with 5 objective heads."""

    def __init__(self, backbone, embed_dim=64, global_dim=32,
                 edge_dim=EDGE_DIM, mask_ratio=0.15, device='cpu'):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio

        # Node-level heads use h (gatv2_hidden = embed_dim)
        self.switch_head = SwitchDecisionHead(embed_dim)
        self.ls1_head = LS1BenefitHead(embed_dim)
        self.fitness_head = FitnessRankHead(embed_dim)

        # Graph-level head uses h_global (pna_out = global_dim)
        self.budget_head = BudgetAllocationHead(global_dim)

        # Edge-level head uses e (gatv2_hidden = embed_dim)
        self.edge_head = EdgeReconstructionHead(embed_dim, edge_dim, mask_ratio)

        self.to(device)

    def forward(self, batch):
        """Compute all SSL losses on a batched graph.

        Returns:
            total_loss, metrics_dict
        """
        # Edge masking
        E = batch['edge_attr'].shape[0]
        edge_mask = torch.rand(E, device=batch['edge_attr'].device) < self.mask_ratio
        original_edge_attr = batch['edge_attr']
        edge_attr_masked = original_edge_attr.clone()
        edge_attr_masked[edge_mask] = 0.0

        # Backbone forward (temporal kwargs ignored by non-temporal backbones)
        h, e, h_per_head, h_global = self.backbone.encode(
            batch['node_feat'],
            batch['edge_index'],
            edge_attr_masked,
            batch['global_feat'],
            v_indices=batch['v_indices'],
            e_indices=batch['e_indices'],
            coords_hist=batch.get('coords_hist'),
            fitness_hist=batch.get('fitness_hist'),
            n_valid=batch.get('n_valid'),
        )

        # Compute losses
        l_switch, switch_acc = self.switch_head(h, batch['switch_labels'])
        l_ls1 = self.ls1_head(h, batch['ls1_delta'])
        l_budget = self.budget_head(h_global, batch['optimal_ls1_frac'])
        l_fitness = self.fitness_head(h, batch['fitness_rank'])
        l_edge = self.edge_head(e, original_edge_attr, edge_mask)

        total = (LOSS_WEIGHTS['switch'] * l_switch
                 + LOSS_WEIGHTS['ls1_benefit'] * l_ls1
                 + LOSS_WEIGHTS['budget'] * l_budget
                 + LOSS_WEIGHTS['fitness_rank'] * l_fitness
                 + LOSS_WEIGHTS['edge_recon'] * l_edge)

        metrics = {
            'loss': total.item(),
            'switch': l_switch.item(),
            'switch_acc': switch_acc,
            'ls1_benefit': l_ls1.item(),
            'budget': l_budget.item(),
            'fitness_rank': l_fitness.item(),
            'edge_recon': l_edge.item(),
        }
        return total, metrics


# ======================================================================
# Training loop
# ======================================================================

def _create_backbone(name, device):
    """Instantiate a backbone by name."""
    if name == 'pna_gatv2':
        from .backbone import PNAGATv2Backbone
        return PNAGATv2Backbone(
            node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
            pna_hidden=64, pna_out=32, pna_layers=4,
            gatv2_hidden=64, gatv2_layers=2, n_heads=4,
            device=device)
    elif name == 'transformer':
        from .transformer_backbone import TransformerBackbone
        return TransformerBackbone(
            node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
            device=device)
    elif name == 'pure_transformer':
        from .pure_transformer_backbone import PureTransformerBackbone
        return PureTransformerBackbone(
            node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
            device=device)
    elif name == 'temporal_gatv2':
        from .temporal_gatv2_backbone import TemporalGATv2Backbone
        return TemporalGATv2Backbone(
            node_in=NODE_DIM, edge_in=EDGE_DIM, global_in=GLOBAL_DIM,
            d_rnn=32, d_temporal=32, gru_window=8,
            gatv2_hidden=64, gatv2_layers=2, n_heads=4,
            global_out_dim=32, dropout=0.1, device=device)
    else:
        raise ValueError(f"Unknown backbone: {name}")


def _is_temporal(backbone_name):
    return backbone_name in ('temporal_gatv2', 'temporal_pna_gatv2')


def train(args):
    device = torch.device(args.device)
    temporal = _is_temporal(args.backbone)

    # Dataset
    train_data = StreamingK2Dataset(args.data_dir, split='train',
                                    require_history=temporal)
    val_data = StreamingK2Dataset(args.data_dir, split='val',
                                  require_history=temporal)

    # Wait for minimum data
    while len(train_data) < args.min_snapshots:
        log.info("Waiting for data... %d/%d snapshots",
                 len(train_data), args.min_snapshots)
        time.sleep(30)
        train_data.refresh()
        val_data.refresh()

    log.info("Starting training: %d train, %d val snapshots",
             len(train_data), len(val_data))

    # Model
    backbone = _create_backbone(args.backbone, device)
    pretrainer = K2SSLPretrainer(
        backbone, embed_dim=64, global_dim=32,
        edge_dim=EDGE_DIM, device=device)

    optimizer = torch.optim.AdamW(pretrainer.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)

    best_val_loss = float('inf')
    patience_counter = 0
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir,
                             f'ssl_k2_{args.backbone}.pth')

    for epoch in range(args.epochs):
        # Refresh data periodically
        if epoch > 0 and epoch % args.refresh_every == 0:
            n_tr = train_data.refresh()
            n_va = val_data.refresh()
            if n_tr > 0 or n_va > 0:
                log.info("  [Refresh] +%d train files, +%d val files "
                         "(total: %d train, %d val)",
                         n_tr, n_va, len(train_data), len(val_data))

        # Train
        if temporal:
            train_sampler = DimGroupBatchSampler(
                train_data._ndims, args.batch_size,
                drop_last=True, shuffle=True)
            train_loader = DataLoader(
                train_data, batch_sampler=train_sampler,
                collate_fn=collate_graphs, num_workers=0)
        else:
            train_loader = DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True,
                collate_fn=collate_graphs, num_workers=0, drop_last=True)

        pretrainer.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        n_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, metrics = pretrainer(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrainer.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_m = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}

        # Val
        val_loss = 0.0
        val_metrics = {}
        n_val = 0
        if len(val_data) > 0:
            if temporal:
                val_sampler = DimGroupBatchSampler(
                    val_data._ndims, args.batch_size,
                    drop_last=False, shuffle=False)
                val_loader = DataLoader(
                    val_data, batch_sampler=val_sampler,
                    collate_fn=collate_graphs, num_workers=0)
            else:
                val_loader = DataLoader(
                    val_data, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate_graphs, num_workers=0)
            pretrainer.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss, metrics = pretrainer(batch)
                    val_loss += loss.item()
                    for k, v in metrics.items():
                        val_metrics[k] = val_metrics.get(k, 0.0) + v
                    n_val += 1
            val_loss /= max(n_val, 1)
            val_m = {k: v / max(n_val, 1) for k, v in val_metrics.items()}
        else:
            val_m = {}

        log.info("Epoch %3d | train %.4f (sw_acc=%.1f%%) | val %.4f (sw_acc=%.1f%%) | "
                 "lr=%.1e | %d train, %d val snaps",
                 epoch, avg_loss, avg_m.get('switch_acc', 0) * 100,
                 val_loss, val_m.get('switch_acc', 0) * 100,
                 optimizer.param_groups[0]['lr'],
                 len(train_data), len(val_data))

        # Checkpoint
        if val_loss < best_val_loss and len(val_data) > 0:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'pretrainer_state_dict': pretrainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_switch_acc': val_m.get('switch_acc', 0),
                'config': {
                    'backbone': args.backbone,
                    'node_in': NODE_DIM, 'edge_in': EDGE_DIM,
                    'global_in': GLOBAL_DIM,
                    'embed_dim': 64, 'global_dim': 32,
                },
            }, ckpt_path)
            log.info("  Saved best checkpoint → %s", ckpt_path)
        else:
            patience_counter += 1

        if (epoch >= args.min_epochs
                and patience_counter >= args.patience
                and len(val_data) > 0):
            log.info("Early stopping after %d epochs without improvement", args.patience)
            break

    log.info("Training complete. Best val loss: %.4f", best_val_loss)


def main():
    parser = argparse.ArgumentParser(
        description="Streaming SSL pretrainer for K=2 (SHADE+LS1)")
    parser.add_argument("--data-dir", type=str, default="DATASETS/NPA_GPU")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ssl_k2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backbone", type=str, default="pna_gatv2",
                        choices=["pna_gatv2", "transformer", "pure_transformer",
                                 "temporal_gatv2"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-snapshots", type=int, default=100)
    parser.add_argument("--refresh-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=15)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S")

    train(args)


if __name__ == '__main__':
    main()
