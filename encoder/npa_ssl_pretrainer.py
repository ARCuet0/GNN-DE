"""
npa_ssl_pretrainer.py — SSL pretraining wrapper for the NPA backbone.

Wraps NPABackbone with 3 SSL heads:
  1. Allocation prediction   (graph-level, KL divergence)
  2. Efficiency prediction   (graph-level, MSE)
  3. Fitness rank prediction (node-level,  MSE)

All operations are GPU-resident and fully batched.  Batch dimension is handled
by concatenating B populations into (B*N_pad, ...) with v_indices, following
the standard PyG batching pattern.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .npa_backbone import NPABackbone


class NPASSLPretrainer(nn.Module):
    """SSL pretraining model wrapping NPABackbone + prediction heads."""

    def __init__(self, K: int = 4, **backbone_kwargs):
        super().__init__()
        self.backbone = NPABackbone(**backbone_kwargs)

        hidden_dim = self.backbone.hidden_dim
        global_out_dim = self.backbone.global_out_dim

        # Graph-level heads (from h_global)
        self.alloc_head = nn.Linear(global_out_dim, K)
        self.eff_head = nn.Linear(global_out_dim, K)

        # Node-level head (from h)
        self.rank_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict, device: torch.device):
        """Run SSL forward pass on a batch of trajectory windows.

        Args:
            batch: dict from TrajectoryDataset DataLoader with keys:
                coords_window:  (B, W, N_pad, D)
                fitness_window: (B, W, N_pad)
                valid_N:        (B, W)
                oracle_alloc:   (B, K)
                subpop_eff:     (B, K)
                fitness_rank:   (B, N_pad)
                ndim:           (B,)  — actual D per sample
            device: target device

        Returns:
            total_loss: scalar
            loss_dict:  {name: scalar} for logging
        """
        coords_w = batch['coords_window'].to(device)       # (B, W, N_pad, D)
        fitness_w = batch['fitness_window'].to(device)      # (B, W, N_pad)
        valid_N = batch['valid_N'].to(device)               # (B, W)
        oracle_alloc = batch['oracle_alloc'].to(device)     # (B, K)
        subpop_eff = batch['subpop_eff'].to(device)         # (B, K)
        fitness_rank = batch['fitness_rank'].to(device)     # (B, N_pad)

        B, W, N_pad, D = coords_w.shape

        # -- Compute f_init per sample: population best at first timestep --
        # Use valid_N to mask padding before taking min
        fitness_first = fitness_w[:, 0, :]                  # (B, N_pad)
        # Create mask: True for valid individuals
        idx_range = torch.arange(N_pad, device=device).unsqueeze(0)  # (1, N_pad)
        valid_first = idx_range < valid_N[:, 0:1]          # (B, N_pad)
        # Set padding to inf before min
        fitness_first_masked = fitness_first.clone()
        fitness_first_masked[~valid_first] = float('inf')
        f_init_per_sample = fitness_first_masked.amin(dim=1)  # (B,)

        # -- Prepare batched inputs for backbone --
        # Reshape: (B, W, N_pad, D) → (W, B*N_pad, D)
        coords_flat = coords_w.permute(1, 0, 2, 3).reshape(W, B * N_pad, D)
        fitness_flat = fitness_w.permute(1, 0, 2).reshape(W, B * N_pad)

        # f_init: repeat per individual
        f_init_flat = f_init_per_sample.repeat_interleave(N_pad)  # (B*N_pad,)

        # valid_mask: all W timesteps are valid (dataset guarantees full windows)
        valid_mask = torch.ones(W, device=device, dtype=torch.bool)
        n_valid = torch.tensor(W, device=device, dtype=torch.long)

        # v_indices: maps each individual to its graph/sample
        v_indices = torch.arange(B, device=device).repeat_interleave(N_pad)

        # coords_current = last timestep
        coords_current = coords_flat[-1]                    # (B*N_pad, D)
        fitness_current = fitness_flat[-1]                   # (B*N_pad,)

        # -- Forward through backbone --
        # The backbone's TemporalGRUEncoder computes f_init internally from
        # the per-individual fitness. But we need to pass the correct f_init
        # per sample. Since individuals from the same sample share f_init,
        # we pass the per-sample value repeated.
        # However, the current backbone.encode expects a single f_init scalar.
        # For batched SSL, we compute features manually and call internal levels.

        h_temporal = self.backbone.temporal_gru(
            coords_flat, fitness_flat, n_valid,
            N_out=B * N_pad, D_out=D)                       # (B*N_pad, D, d_rnn)

        h_grid = self.backbone.cross_dim(
            h_temporal, fitness_current)                     # (B*N_pad, D, d_rnn)
        h, h_global, h_per_head = self.backbone.pop_transformer(
            h_grid, coords_current, v_indices)               # h: (B*N_pad, H)
        h = self.backbone.feature_injector(
            h, coords_current, fitness_current, v_indices)
        # h_global: (B, global_out_dim)

        # -- SSL losses --
        # 1. Allocation prediction (KL divergence)
        alloc_logits = self.alloc_head(h_global)            # (B, K)
        alloc_pred = F.log_softmax(alloc_logits, dim=-1)
        alloc_target = oracle_alloc.clamp(min=1e-8)
        alloc_target = alloc_target / alloc_target.sum(dim=-1, keepdim=True)
        loss_alloc = F.kl_div(alloc_pred, alloc_target, reduction='batchmean',
                              log_target=False)

        # 2. Efficiency prediction (MSE on log-scale, raw values span 0..10^7)
        eff_pred = self.eff_head(h_global)                  # (B, K)
        eff_target_log = torch.log1p(subpop_eff)            # log(1 + eff)
        loss_eff = F.mse_loss(eff_pred, eff_target_log)

        # 3. Fitness rank prediction (node-level MSE, masked for padding)
        rank_pred = torch.sigmoid(
            self.rank_head(h).squeeze(-1))                  # (B*N_pad,)
        rank_target = fitness_rank.reshape(B * N_pad)       # (B*N_pad,)

        # Mask: only compute loss on valid individuals of the last timestep
        valid_last = idx_range.expand(B, -1) < valid_N[:, -1:]  # (B, N_pad)
        valid_mask_flat = valid_last.reshape(B * N_pad)     # (B*N_pad,)

        if valid_mask_flat.any():
            loss_rank = F.mse_loss(
                rank_pred[valid_mask_flat],
                rank_target[valid_mask_flat])
        else:
            loss_rank = torch.tensor(0.0, device=device)

        # Weighted total
        total_loss = 1.0 * loss_alloc + 0.5 * loss_eff + 0.3 * loss_rank

        loss_dict = {
            'alloc': loss_alloc.detach(),
            'eff': loss_eff.detach(),
            'rank': loss_rank.detach(),
            'total': total_loss.detach(),
        }
        return total_loss, loss_dict

