"""
ssl_nextstep_dataset.py — Dataset for next-step node feature prediction.

Pairs consecutive-generation snapshots from NPA_GPU data:
  (graph_t, node_feat_{t+1}) where gen_{t+1} = gen_t + 1.

Three modes:
  - CPU lazy: NextStepPairDataset (persistence baseline, debugging)
  - GPU-resident: load_gpu_dataset() for PNAGATv2 (no temporal)
  - CPU temporal: NextStepPairDataset(temporal=True) + DataLoader for NPA/TemporalGATv2
"""

import logging
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from .similarity_graph import NODE_DIM, EDGE_DIM, GLOBAL_DIM

log = logging.getLogger(__name__)

VAL_FIDS = frozenset({3, 8, 18})

# Exclude parent_fitness_rank (idx 7): always -1 in batched GPU collection.
PREDICT_FEATURES = list(range(min(NODE_DIM, 7)))  # [0..6]


# ======================================================================
# CPU lazy dataset (all backbones, temporal optional)
# ======================================================================

class NextStepPairDataset(Dataset):
    """Lazy-loading pairs from NPA_GPU pkl files with LRU file cache.

    When temporal=True, only includes pairs with has_history=True and
    returns coords_hist, fitness_hist, coordinates, fitness for
    NPA/TemporalGATv2 backbones.
    """

    _FILE_CACHE_SIZE = 32

    # Log-equidistant W ablation values: 8, 12, 17, 24, 35, 50
    WINDOW_SIZES = [8, 12, 17, 24, 35, 49]

    def __init__(self, data_dir, split='train', temporal=False, window_size=8):
        self.data_dir = data_dir
        self.split = split
        self.temporal = temporal
        self.window_size = window_size  # slice coords_hist to this W
        self._index = []
        self._ndims = []  # parallel: ndim per pair (for DimGroupBatchSampler)
        self._cache = {}
        self._cache_order = []
        self._loaded_files = set()
        self.refresh()

    def refresh(self):
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
                # Identify run boundaries (gen resets)
                run_start = 0
                for i in range(len(data) - 1):
                    s_t, s_t1 = data[i], data[i + 1]
                    if s_t1.get('gen', -1) != s_t.get('gen', -2) + 1:
                        run_start = i + 1
                        continue
                    if self.temporal:
                        # How many prior snapshots in this run?
                        n_hist = i - run_start + 1  # includes current
                        if n_hist < self.window_size:
                            continue
                    fid = s_t.get('fid', 0)
                    is_val = fid in VAL_FIDS
                    if (self.split == 'val') == is_val:
                        self._index.append((fpath, i, i + 1,
                                            run_start if self.temporal else 0))
                        self._ndims.append(s_t.get('ndim', 10))
                self._loaded_files.add(fname)
                n_new += 1
            except (EOFError, pickle.UnpicklingError, OSError):
                pass
        if n_new > 0:
            log.info("[%s%s] +%d files → %d pairs total",
                     self.split, '+temporal' if self.temporal else '',
                     n_new, len(self._index))
        return n_new

    def _load_file(self, fpath):
        if fpath in self._cache:
            self._cache_order.remove(fpath)
            self._cache_order.append(fpath)
            return self._cache[fpath]
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
        fpath, off_t, off_t1, run_start = self._index[idx]
        file_data = self._load_file(fpath)
        s_t, s_t1 = file_data[off_t], file_data[off_t1]

        gf = s_t['global_feat']
        gf1 = s_t1['global_feat']
        result = {
            'node_feat': torch.as_tensor(s_t['node_feat'], dtype=torch.float32),
            'edge_index': torch.as_tensor(s_t['edge_index'], dtype=torch.long),
            'edge_attr': torch.as_tensor(s_t['edge_attr'], dtype=torch.float32),
            'global_feat': torch.as_tensor(
                gf.squeeze(0) if gf.ndim > 1 else gf, dtype=torch.float32),
            'target_node_feat': torch.as_tensor(
                s_t1['node_feat'], dtype=torch.float32),
            'target_global': torch.as_tensor(
                gf1.squeeze(0) if gf1.ndim > 1 else gf1, dtype=torch.float32),
            'fitness_rank': torch.as_tensor(
                s_t['fitness_rank'], dtype=torch.float32),
        }

        if self.temporal:
            W = self.window_size
            # Reconstruct history from consecutive snapshots in same run
            hist_start = max(run_start, off_t - W + 1)
            coords_list = []
            fitness_list = []
            for j in range(hist_start, off_t + 1):
                sj = file_data[j]
                coords_list.append(torch.as_tensor(
                    sj['coordinates'], dtype=torch.float32))
                fitness_list.append(torch.as_tensor(
                    sj['fitness'], dtype=torch.float32))

            result['coords_hist'] = torch.stack(coords_list)    # (W, N, D)
            result['fitness_hist'] = torch.stack(fitness_list)   # (W, N)
            result['coordinates'] = coords_list[-1]              # (N, D)
            result['fitness'] = fitness_list[-1]                  # (N,)
            result['n_valid'] = torch.tensor(len(coords_list), dtype=torch.long)

        return result


def collate_nextstep(batch):
    """Collate variable-size graphs into a single batched graph.

    Handles both non-temporal and temporal modes.
    """
    node_offset = 0
    all_nodes, all_edges, all_edge_attr = [], [], []
    all_global = []
    all_target_nodes, all_target_global = [], []
    all_fitness_rank = []
    v_indices, e_indices = [], []

    has_temporal = 'coords_hist' in batch[0]
    all_coords_hist, all_fitness_hist = [], []
    all_coords, all_fitness = [], []

    for i, g in enumerate(batch):
        N = g['node_feat'].shape[0]
        E = g['edge_index'].shape[1]

        all_nodes.append(g['node_feat'])
        all_edges.append(g['edge_index'] + node_offset)
        all_edge_attr.append(g['edge_attr'])
        all_global.append(g['global_feat'])

        all_target_nodes.append(g['target_node_feat'])
        all_target_global.append(g['target_global'])
        all_fitness_rank.append(g['fitness_rank'])

        v_indices.append(torch.full((N,), i, dtype=torch.long))
        e_indices.append(torch.full((E,), i, dtype=torch.long))

        if has_temporal:
            all_coords_hist.append(g['coords_hist'])    # (W, N_i, D)
            all_fitness_hist.append(g['fitness_hist'])   # (W, N_i)
            all_coords.append(g['coordinates'])          # (N_i, D)
            all_fitness.append(g['fitness'])              # (N_i,)

        node_offset += N

    result = {
        'node_feat': torch.cat(all_nodes),
        'edge_index': torch.cat(all_edges, dim=1),
        'edge_attr': torch.cat(all_edge_attr),
        'global_feat': torch.stack(all_global),
        'v_indices': torch.cat(v_indices),
        'e_indices': torch.cat(e_indices),
        'target_node_feat': torch.cat(all_target_nodes),
        'target_global': torch.stack(all_target_global),
        'fitness_rank': torch.cat(all_fitness_rank),
    }

    if has_temporal:
        # D-homogeneous batches (guaranteed by DimGroupBatchSampler)
        result['coords_hist'] = torch.cat(all_coords_hist, dim=1)   # (W, N_total, D)
        result['fitness_hist'] = torch.cat(all_fitness_hist, dim=1)  # (W, N_total)
        result['coordinates'] = torch.cat(all_coords)                # (N_total, D)
        result['fitness'] = torch.cat(all_fitness)                   # (N_total,)
        result['n_valid'] = batch[0]['n_valid']                      # scalar

    return result


# ======================================================================
# GPU-resident dataset (PNAGATv2 only, no temporal)
# ======================================================================

def _scan_pairs(data_dir, split):
    """Scan pkl files and yield (filepath, offset_t, offset_t1, s_t, s_t1)."""
    pkl_files = sorted(f for f in os.listdir(data_dir)
                       if f.endswith('.pkl') and not f.endswith('.tmp'))
    for fname in pkl_files:
        fpath = os.path.join(data_dir, fname)
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                data = [data]
            for i in range(len(data) - 1):
                s_t, s_t1 = data[i], data[i + 1]
                if s_t1.get('gen', -1) != s_t.get('gen', -2) + 1:
                    continue
                fid = s_t.get('fid', 0)
                is_val = fid in VAL_FIDS
                if (split == 'val') == is_val:
                    yield fpath, i, i + 1, s_t, s_t1
        except (EOFError, pickle.UnpicklingError, OSError):
            pass


def load_gpu_dataset(data_dir, split, device, max_pairs=0):
    """Load all pairs into padded GPU tensors. Returns a GPUDataset."""
    node_feats_t, node_feats_t1 = [], []
    edge_indices, edge_attrs, edge_counts = [], [], []
    global_feats_t, global_feats_t1 = [], []
    fitness_ranks = []
    n_loaded = 0

    log.info("[%s] Scanning %s for GPU preload...", split, data_dir)
    for _fpath, _i, _j, s_t, s_t1 in _scan_pairs(data_dir, split):
        node_feats_t.append(s_t['node_feat'].astype('float32'))
        node_feats_t1.append(s_t1['node_feat'].astype('float32'))

        ei = s_t['edge_index'].astype('int64')
        ea = s_t['edge_attr'].astype('float32')
        edge_counts.append(ei.shape[1])
        edge_indices.append(ei)
        edge_attrs.append(ea)

        gf = s_t['global_feat']
        gf = gf.squeeze(0) if gf.ndim > 1 else gf
        global_feats_t.append(gf.astype('float32'))
        gf1 = s_t1['global_feat']
        gf1 = gf1.squeeze(0) if gf1.ndim > 1 else gf1
        global_feats_t1.append(gf1.astype('float32'))

        fitness_ranks.append(s_t['fitness_rank'].astype('float32'))

        n_loaded += 1
        if max_pairs > 0 and n_loaded >= max_pairs:
            break
        if n_loaded % 100_000 == 0:
            log.info("  loaded %d pairs...", n_loaded)

    if n_loaded == 0:
        raise RuntimeError(f"No {split} pairs found in {data_dir}")

    log.info("[%s] %d pairs loaded from disk. Packing to GPU...", split, n_loaded)

    S = n_loaded
    N = node_feats_t[0].shape[0]
    E_max = max(edge_counts)

    t_node_t = torch.tensor(np.stack(node_feats_t), dtype=torch.float32)
    t_node_t1 = torch.tensor(np.stack(node_feats_t1), dtype=torch.float32)
    t_global_t = torch.tensor(np.stack(global_feats_t), dtype=torch.float32)
    t_global_t1 = torch.tensor(np.stack(global_feats_t1), dtype=torch.float32)
    t_fitness = torch.tensor(np.stack(fitness_ranks), dtype=torch.float32)
    t_edge_count = torch.tensor(edge_counts, dtype=torch.long)

    t_edge_index = torch.zeros(S, 2, E_max, dtype=torch.long)
    t_edge_attr = torch.zeros(S, E_max, EDGE_DIM, dtype=torch.float32)
    for i in range(S):
        ec = edge_counts[i]
        t_edge_index[i, :, :ec] = torch.tensor(edge_indices[i])
        t_edge_attr[i, :ec, :] = torch.tensor(edge_attrs[i])

    del node_feats_t, node_feats_t1, edge_indices, edge_attrs
    del global_feats_t, global_feats_t1, fitness_ranks

    ds = GPUDataset(
        node_feat_t=t_node_t.to(device),
        node_feat_t1=t_node_t1.to(device),
        edge_index=t_edge_index.to(device),
        edge_attr=t_edge_attr.to(device),
        edge_count=t_edge_count.to(device),
        global_feat_t=t_global_t.to(device),
        global_feat_t1=t_global_t1.to(device),
        fitness_rank=t_fitness.to(device),
    )
    mem_gb = sum(t.element_size() * t.nelement()
                 for t in [ds.node_feat_t, ds.node_feat_t1, ds.edge_index,
                           ds.edge_attr, ds.global_feat_t, ds.global_feat_t1,
                           ds.fitness_rank, ds.edge_count]) / 1e9
    log.info("[%s] GPU dataset ready: %d pairs, E_max=%d, %.2f GB on %s",
             split, S, E_max, mem_gb, device)
    return ds


class GPUDataset:
    """GPU-resident dataset. All tensors live on device; batching is pure indexing."""

    def __init__(self, *, node_feat_t, node_feat_t1, edge_index, edge_attr,
                 edge_count, global_feat_t, global_feat_t1, fitness_rank):
        self.node_feat_t = node_feat_t
        self.node_feat_t1 = node_feat_t1
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_count = edge_count
        self.global_feat_t = global_feat_t
        self.global_feat_t1 = global_feat_t1
        self.fitness_rank = fitness_rank
        self.S = node_feat_t.shape[0]
        self.N = node_feat_t.shape[1]
        self.device = node_feat_t.device

    def __len__(self):
        return self.S

    def iter_batches(self, batch_size, shuffle=True):
        if shuffle:
            perm = torch.randperm(self.S, device=self.device)
        else:
            perm = torch.arange(self.S, device=self.device)
        for start in range(0, self.S, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < batch_size and shuffle:
                continue
            yield self._build_batch(idx)

    def _build_batch(self, idx):
        B = len(idx)
        N = self.N

        node_feat = self.node_feat_t[idx].reshape(B * N, -1)
        target_node_feat = self.node_feat_t1[idx].reshape(B * N, -1)
        fitness_rank = self.fitness_rank[idx].reshape(B * N)
        global_feat = self.global_feat_t[idx]
        target_global = self.global_feat_t1[idx]

        offsets = torch.arange(B, device=self.device).unsqueeze(1) * N
        ei = self.edge_index[idx]
        ei_offset = ei + offsets.unsqueeze(1)

        ea = self.edge_attr[idx]
        ec = self.edge_count[idx]

        E_max = ei.shape[2]
        edge_mask = torch.arange(E_max, device=self.device).unsqueeze(0) < ec.unsqueeze(1)

        edge_index_flat = ei_offset.permute(1, 0, 2).reshape(2, B * E_max)
        edge_attr_flat = ea.reshape(B * E_max, -1)
        edge_valid = edge_mask.reshape(B * E_max)

        edge_index = edge_index_flat[:, edge_valid]
        edge_attr = edge_attr_flat[edge_valid]

        v_indices = torch.arange(B, device=self.device).repeat_interleave(N)
        e_indices = torch.arange(B, device=self.device).repeat_interleave(ec)

        return {
            'node_feat': node_feat,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'global_feat': global_feat,
            'v_indices': v_indices,
            'e_indices': e_indices,
            'target_node_feat': target_node_feat,
            'target_global': target_global,
            'fitness_rank': fitness_rank,
        }
