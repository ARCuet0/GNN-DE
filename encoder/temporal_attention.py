"""
temporal_attention.py — Drop-in replacement for TemporalGRUEncoder using 1D self-attention.

Same interface: (coords_hist, fitness_hist, n_valid) → h_temporal (N, D, d_model).
Each (individual, dimension) pair gets W tokens with positional encoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SelfAttnBlock(nn.Module):
    """Pre-norm self-attention + FFN using F.scaled_dot_product_attention.

    vmap-compatible: uses only ops with native batching rules,
    unlike nn.TransformerEncoderLayer which falls back to slow
    element-wise dispatch under vmap.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, edge_bias=None):
        """x: (B, L, d_model) → (B, L, d_model).

        edge_bias: optional (B, 1, L, L) additive bias on attention logits
            (broadcast over heads). Used by the B2 (set-attn + edge bias) arm
            to inject explicit pairwise inductive bias on top of all-to-all
            attention. None for the standard set-attn / temporal-attn paths.
        """
        B, L, D = x.shape
        x_norm = self.norm1(x)
        qkv = self.W_qkv(x_norm).reshape(B, L, 3, self.n_heads, self.head_dim)
        Q, K, V = qkv.unbind(dim=2)  # each (B, L, n_heads, head_dim)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)

        # Manual scaled dot-product attention (avoids CUDA kernel issues
        # with F.scaled_dot_product_attention on head_dim=16)
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
        if edge_bias is not None:
            # (B, 1, L, L) broadcasts over the n_heads dim.
            attn_weights = attn_weights + edge_bias
        attn_weights = F.softmax(attn_weights, dim=-1)
        if self.training and self.drop.p > 0:
            attn_weights = F.dropout(attn_weights, p=self.drop.p, training=True)
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.out_proj(attn_out))

        x = x + self.ffn(self.norm2(x))
        return x


class TemporalAttentionEncoder(nn.Module):
    """1D self-attention over temporal window per (individual, dimension).

    Input:  coords_hist (W, N, D), fitness_hist (W, N), n_valid
    Output: h_temporal (N, D, d_model)

    6 per-timestep features (same as TemporalGRUEncoder):
      1. x_norm:  coords / 100
      2. dx:      displacement / 100
      3. f_norm:  window-normalized fitness [0,1]
      4. df_norm: |Δf_norm|
      5. vel_ema: EMA of |dx| (α=0.3)
      6. fw_disp: dx * sign(-Δf_norm)
    """

    N_FEATURES = 6

    def __init__(self, d_model=32, n_layers=2, n_heads=4, max_W=64,
                 vel_ema_alpha=0.3, dropout=0.1, coord_range=None):
        super().__init__()
        self.d_model = d_model
        self.d_rnn = d_model  # compatibility with TemporalDimPooler
        self.vel_ema_alpha = vel_ema_alpha
        # Search-space half-range for input normalization. Required explicit:
        # coords / coord_range and dx / coord_range break silently if the box
        # at eval differs from the box at training.
        assert coord_range is not None, (
            "TemporalAttentionEncoder requires explicit coord_range")
        self.coord_range = float(coord_range)

        self.feature_norm = nn.LayerNorm(self.N_FEATURES)
        self.input_proj = nn.Linear(self.N_FEATURES, d_model)
        self.pos_embed = nn.Embedding(max_W, d_model)

        self.layers = nn.ModuleList([
            _SelfAttnBlock(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _window_normalize_fitness(fitness):
        fmax = torch.finfo(fitness.dtype).max
        # Finite guard only: the min-max below is sign-agnostic, so non-positive
        # (BBOB f_opt<0) fitness must NOT be floored to 1e-30 — that collapses the
        # window to a constant. See finding_bbob_fitness_blindness_clamp_2026_06_05.
        f_safe = fitness.clamp(min=-fmax, max=fmax)
        f_best = f_safe.min()
        f_worst = f_safe.max()
        spread = f_worst - f_best
        # Converged population: uniform 0.5 instead of 0/epsilon
        if spread < 1e-20:
            f_norm = torch.full_like(f_safe, 0.5)
        else:
            f_norm = (f_safe - f_best) / spread
        df_norm = F.pad((f_norm[1:] - f_norm[:-1]).abs(), (0, 0, 1, 0))
        return f_norm, df_norm

    def _build_features(self, x_norm, dx, f_norm, df_norm):
        alpha = self.vel_ema_alpha
        abs_dx = dx.abs()
        vel_ema = torch.zeros_like(x_norm)
        vel_ema[0] = abs_dx[0]
        for t in range(1, x_norm.shape[0]):
            vel_ema[t] = alpha * abs_dx[t] + (1 - alpha) * vel_ema[t - 1]

        f_diff = F.pad(f_norm[1:] - f_norm[:-1], (0, 0, 0, 0, 1, 0))
        improvement_sign = torch.sign(-f_diff[:, :, 0:1]).expand_as(dx)
        fw_disp = dx * improvement_sign

        return torch.stack([x_norm, dx, f_norm, df_norm, vel_ema, fw_disp], dim=-1)

    def forward(self, coords, fitness, n_valid,
                N_out=None, D_out=None, **_ignored):
        W, N_total, D = coords.shape
        if N_out is None:
            N_out = N_total
        if D_out is None:
            D_out = D

        x_norm = coords / self.coord_range
        dx = F.pad(coords[1:] - coords[:-1], (0, 0, 0, 0, 1, 0)) / self.coord_range

        f_norm, df_norm = self._window_normalize_fitness(fitness)
        f_norm = f_norm.unsqueeze(-1).expand_as(x_norm)
        df_norm = df_norm.unsqueeze(-1).expand_as(x_norm)

        features = self._build_features(x_norm, dx, f_norm, df_norm)

        # n_valid can be int (vmap-safe) or tensor (legacy)
        nv = n_valid if isinstance(n_valid, int) else n_valid.item()
        # (nv, N, D, 6) → (N*D, nv, 6)
        features = features[:nv].permute(1, 2, 0, 3).reshape(N_total * D, nv, self.N_FEATURES)
        features = self.feature_norm(features)
        h = self.input_proj(features)  # (N*D, nv, d_model)

        # Add positional encoding
        pos = self.pos_embed(torch.arange(nv, device=h.device))
        h = h + pos.unsqueeze(0)

        # Self-attention over temporal dimension
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)

        # Mean pool over temporal dimension
        h = h.mean(dim=1)  # (N*D, d_model)

        return h.view(N_out, D_out, self.d_model)
