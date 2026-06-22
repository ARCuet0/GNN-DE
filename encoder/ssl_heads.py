"""
ssl_heads.py — Shared SSL prediction heads for encoder pretraining.

All heads follow the pattern: Linear → act → Linear, with a configurable
loss method.  Used by GNN_MOS_Classic/pretrain_encoder.py,
TRANSFORMER_SSL/ssl_pretrainer.py, NEURAL_META_K4/ssl_pretrain_pna.py,
and ENSEMBLE_K4/ssl_objectives_ensemble.py.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2 = math.log(2)


# ======================================================================
# Base
# ======================================================================

def _make_mlp(d_in, d_hidden, d_out, act='relu'):
    acts = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'gelu': nn.GELU}
    return nn.Sequential(nn.Linear(d_in, d_hidden), acts[act](), nn.Linear(d_hidden, d_out))


# ======================================================================
# Node-level heads
# ======================================================================

class MSESigmoidHead(nn.Module):
    """MSE on sigmoid output vs [0,1] target (e.g. fitness_rank, progress)."""
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = _make_mlp(d, hidden, 1)
    def forward(self, h): return torch.sigmoid(self.net(h).squeeze(-1))
    def loss(self, h, target): return F.mse_loss(self.forward(h), target)


class BCEHead(nn.Module):
    """BCE on logits, normalized by log(2) → ~[0,1] scale (e.g. shade_binary)."""
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = _make_mlp(d, hidden, 1)
    def forward(self, h): return self.net(h).squeeze(-1)
    def loss(self, h, target):
        return F.binary_cross_entropy_with_logits(self.forward(h), target) / LOG2


class MaskedBCEHead(nn.Module):
    """BCE that ignores uncertain targets (0.5), only trains on clear 0/1."""
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = _make_mlp(d, hidden, 1)
    def forward(self, h): return self.net(h).squeeze(-1)
    def loss(self, h, target):
        valid = (target != 0.5)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=h.device)
        logits = self.forward(h)[valid]
        return F.binary_cross_entropy_with_logits(logits, target[valid]) / LOG2


class MaskedMSEHead(nn.Module):
    """MSE only on valid entries (e.g. ls1_log_imp where ls1_valid)."""
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = _make_mlp(d, hidden, 1)
    def loss(self, h, target, valid):
        if valid.sum() == 0: return torch.tensor(0.0, device=h.device)
        return F.mse_loss(self.net(h).squeeze(-1)[valid], target[valid])


class ScaledMSEHead(nn.Module):
    """MSE on raw output vs pre-scaled target (e.g. grad_mag/20, n_basins/250)."""
    def __init__(self, d, hidden=32):
        super().__init__()
        self.net = _make_mlp(d, hidden, 1)
    def loss(self, h, target):
        return F.mse_loss(self.net(h).squeeze(-1), target)


# ======================================================================
# Edge-level heads
# ======================================================================

class EdgeReconHead(nn.Module):
    """Masked edge feature reconstruction (15% masking)."""
    def __init__(self, d, edge_dim, hidden=32):
        super().__init__()
        self.net = _make_mlp(d, hidden, edge_dim)
    def loss(self, h_edges, target, mask):
        if mask is None or mask.sum() == 0:
            return torch.tensor(0.0, device=h_edges.device)
        return F.mse_loss(self.net(h_edges[mask]), target[mask])


class MaskedNodeReconHead(nn.Module):
    """BERT-like node reconstruction: predict masked node features from context."""
    def __init__(self, d, node_dim, hidden=None, act='gelu'):
        super().__init__()
        hidden = hidden or d
        self.net = _make_mlp(d, hidden, node_dim, act=act)
    def loss(self, h_nodes, target, mask):
        if mask is None or mask.sum() == 0:
            return torch.tensor(0.0, device=h_nodes.device)
        return F.mse_loss(self.net(h_nodes[mask]), target[mask])


# ======================================================================
# Graph-level heads
# ======================================================================

class KLAllocationHead(nn.Module):
    """KL divergence for budget allocation prediction (K classes)."""
    def __init__(self, d, K, hidden=None):
        super().__init__()
        hidden = hidden or d
        self.net = _make_mlp(d, hidden, K)
    def forward(self, h_global):
        return F.log_softmax(self.net(h_global), dim=-1)
    def loss(self, h_global, target_allocation):
        log_probs = self.forward(h_global)
        target_safe = target_allocation.clamp(min=1e-6)
        target_safe = target_safe / target_safe.sum(dim=-1, keepdim=True)
        return F.kl_div(log_probs, target_safe, reduction='batchmean')


class MultiOutputSigmoidHead(nn.Module):
    """MSE on sigmoid output for K independent channels (e.g. per-subpop efficiency)."""
    def __init__(self, d, K, hidden=None):
        super().__init__()
        hidden = hidden or d
        self.net = _make_mlp(d, hidden, K)
    def forward(self, h_global):
        return torch.sigmoid(self.net(h_global))
    def loss(self, h_global, target):
        return F.mse_loss(self.forward(h_global), target)
