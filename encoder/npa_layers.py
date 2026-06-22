"""
npa_layers.py — Three-level Neural Population Analyzer layers.

Level 1: TemporalGRUEncoder   — shared-weight GRU over (individual, dim) pairs
Level 2: CrossDimTransformer   — per-individual transformer over D dimension tokens
Level 3: PopulationTransformer — cross-individual transformer (permutation equivariant)

All operations are GPU-resident, fully batched, zero Python loops over
timesteps/individuals/dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Level 1 — Temporal GRU per dimension (shared weights)
# ======================================================================

class TemporalGRUEncoder(nn.Module):
    """Shared-weight GRU over the temporal axis for every (individual, dim).

    Input:  ring-buffer history (W, N, D) coords + (W, N) fitness
    Output: h_temporal (N, D, d_rnn)

    6 per-timestep features:
      1. x_norm:  coords / 100                             per (t, i, d)
      2. dx:      displacement / 100                        per (t, i, d)
      3. f_norm:  window-normalized fitness [0,1]           per (t, i) → broadcast D
      4. df_norm: |Δf_norm|                                per (t, i) → broadcast D
      5. vel_ema: EMA of |dx| per dim (α=0.3)             per (t, i, d)
      6. fw_disp: dx * sign(-Δf_norm)                     per (t, i, d)
    """
    N_FEATURES = 6

    def __init__(self, d_model: int = 32, d_rnn: int = 32,
                 vel_ema_alpha: float = 0.3):
        super().__init__()
        self.d_model = d_model
        self.d_rnn = d_rnn
        self.vel_ema_alpha = vel_ema_alpha
        self.feature_norm = nn.LayerNorm(self.N_FEATURES)
        self.input_proj = nn.Linear(self.N_FEATURES, d_model)
        self.gru = nn.GRU(d_model, d_rnn, batch_first=True)

    @staticmethod
    def _window_normalize_fitness(fitness):
        """Normalize fitness to [0, 1] within the window.

        f_norm[t, i] = (f[t,i] - f_best) / (f_worst - f_best)
        where f_best/f_worst are min/max across ALL timesteps and individuals.

        Args:
            fitness: (W, N) raw fitness, may contain inf
        Returns:
            f_norm:  (W, N) in [0, 1]
            df_norm: (W, N) |Δf_norm| in [0, 1]
        """
        fmax = torch.finfo(fitness.dtype).max
        f_safe = fitness.clamp(min=1e-30, max=fmax)

        f_best = f_safe.min()
        f_worst = f_safe.max()
        denom = (f_worst - f_best).clamp(min=1e-30)

        f_norm = (f_safe - f_best) / denom  # [0, 1]

        df_norm = F.pad((f_norm[1:] - f_norm[:-1]).abs(), (0, 0, 1, 0))

        return f_norm, df_norm

    def _build_features(self, x_norm, dx, f_norm, df_norm):
        """Build 6 GRU features. All vary per-individual per-dimension.

        Args (all (W, N_total, D)):
            x_norm:  coords / 100                    [-1, 1]
            dx:      Δcoords / 100                   [-2, 2]
            f_norm:  window-normalized fitness        [0, 1]
            df_norm: |Δf_norm|                       [0, 1]

        Returns: (N_total*D, W, 6)
        """
        W = x_norm.shape[0]
        alpha = self.vel_ema_alpha

        # Feature 5: Velocity EMA per-dim
        abs_dx = dx.abs()
        vel_ema = torch.zeros_like(x_norm)
        vel_ema[0] = abs_dx[0]
        for t in range(1, W):
            vel_ema[t] = alpha * abs_dx[t] + (1 - alpha) * vel_ema[t - 1]

        # Feature 6: Fitness-weighted displacement per-dim
        f_diff = F.pad(f_norm[1:] - f_norm[:-1], (0, 0, 0, 0, 1, 0))
        improvement_sign = torch.sign(-f_diff[:, :, 0:1]).expand_as(dx)
        fw_disp = dx * improvement_sign

        features = torch.stack(
            [x_norm, dx, f_norm, df_norm, vel_ema, fw_disp], dim=-1)
        N_total, D = x_norm.shape[1], x_norm.shape[2]
        return features.permute(1, 2, 0, 3).reshape(N_total * D, W, self.N_FEATURES)

    def forward(self, coords, fitness, n_valid, N_out=None, D_out=None):
        """Encode coordinates + fitness into per-(individual, dimension) hidden states.

        Works for both single-sample (W, N, D) and batched (W, B*N, D).

        Args:
            coords:   (W, N_total, D)  coordinates (raw, [-100, 100])
            fitness:  (W, N_total)     fitness (raw, may contain inf)
            n_valid:  0-dim long       valid timesteps in window
            N_out:    reshape hint — N for single, B*N_pad for batched
            D_out:    reshape hint — D

        Returns:
            h_temporal: (N_out, D_out, d_rnn)
        """
        W, N_total, D = coords.shape
        if N_out is None:
            N_out = N_total
        if D_out is None:
            D_out = D

        x_norm = coords / 100.0
        dx = F.pad(coords[1:] - coords[:-1], (0, 0, 0, 0, 1, 0)) / 100.0

        # Window-normalized fitness: [0, 1], varies per-individual
        f_norm, df_norm = self._window_normalize_fitness(fitness)
        f_norm = f_norm.unsqueeze(-1).expand_as(x_norm)
        df_norm = df_norm.unsqueeze(-1).expand_as(x_norm)

        features = self._build_features(x_norm, dx, f_norm, df_norm)

        nv = n_valid.item()
        features = self.feature_norm(features[:, :nv, :])
        projected = self.input_proj(features)
        _, h_last = self.gru(projected)
        return h_last.squeeze(0).view(N_out, D_out, self.d_rnn)


class TemporalDimPooler(nn.Module):
    """Pool GRU hidden states (N, D, d_rnn) -> (N, d_out) via mean + projection.

    D varies across problems (10/30/50), so mean pool gives a D-invariant
    summary. The projection + LayerNorm lets the downstream PNA learn to
    weight temporal features appropriately.
    """

    def __init__(self, d_rnn: int = 32, d_out: int = 32):
        super().__init__()
        self.proj = nn.Linear(d_rnn, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h_temporal):
        """h_temporal: (N, D, d_rnn) -> (N, d_out)"""
        return self.norm(self.proj(h_temporal.mean(dim=1)))


# ======================================================================
# Level 2 — Two-Stage Attention (NeurELA-style)
# ======================================================================

class _PreNormAttnBlock(nn.Module):
    """Pre-norm self-attention + FFN block. Used by both stages."""

    def __init__(self, d_model, n_heads, dropout=0.1):
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
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h):
        """h: (batch, seq, d_model) → (batch, seq, d_model)"""
        B, S, H = h.shape

        h_norm = self.norm1(h)
        qkv = self.W_qkv(h_norm).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, S, H)
        h = h + self.drop(self.out_proj(attn_out))

        h = h + self.ffn(self.norm2(h))
        return h


class TwoStageAttention(nn.Module):
    """NeurELA-style two-stage attention over (N, D) grid.

    Stage 1 (cross-individual): for each dimension d, N individuals
             attend to each other. Individuals learn their relative
             position in the population per-dimension.
    Stage 2 (cross-dimension): for each individual i, D dimensions
             attend to each other. Each individual aggregates its
             own dimensional information.

    Input:  h_temporal (N, D, d_rnn) + fitness_current (N,)
    Output: h_ind (N, d_ind)
    """

    def __init__(self, d_rnn: int = 32, d_ind: int = 64,
                 n_layers: int = 2, n_heads: int = 4,
                 max_D: int = 100, dropout: float = 0.1):
        super().__init__()
        self.d_rnn = d_rnn
        self.d_ind = d_ind

        # Positional encoding for dimension axis (Stage 2 only)
        self.pos_embed = nn.Embedding(max_D, d_rnn)

        # FiLM conditioning: (mantissa, exponent, fes_ratio) → (gamma, beta)
        self.film_proj = nn.Linear(3, 2 * d_rnn)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        self.film_proj.bias.data[:d_rnn] = 1.0  # gamma=1, beta=0

        # n_layers pairs of (cross-individual, cross-dimension)
        self.cross_ind_layers = nn.ModuleList([
            _PreNormAttnBlock(d_rnn, n_heads, dropout) for _ in range(n_layers)
        ])
        self.cross_dim_layers = nn.ModuleList([
            _PreNormAttnBlock(d_rnn, n_heads, dropout) for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_rnn)

    def forward(self, h_temporal, fitness_current, fes_ratio=None,
                n_pop=None):
        """
        Args:
            h_temporal:      (N, D, d_rnn) from GRU. N may be B*N_pop.
            fitness_current: (N,) raw fitness (may contain inf)
            fes_ratio:       (N,) or scalar — FES/MAX_FES budget progress
            n_pop:           int or None — individuals per population.
                             If provided, Stage 1 attention is batched per
                             population: (D*B, N_pop, H) instead of (D, N, H).
        Returns:
            h_grid: (N, D, d_rnn) — NO pooling, full grid preserved
        """
        N, D, H = h_temporal.shape
        device = h_temporal.device

        # FiLM from (mantissa, exponent, fes_ratio)
        fmax = torch.finfo(fitness_current.dtype).max
        f_safe = fitness_current.clamp(min=1e-30, max=fmax)
        mant, exp_raw = torch.frexp(f_safe)
        if fes_ratio is None:
            fes = torch.full((N,), 0.5, device=device)
        elif fes_ratio.dim() == 0:
            fes = fes_ratio.expand(N)
        else:
            fes = fes_ratio
        film_input = torch.stack([mant, exp_raw.float() / 20.0, fes], dim=-1)
        film = self.film_proj(film_input)                         # (N, 2H)
        gamma, beta = film[:, :H], film[:, H:]                   # (N, H)

        # Add positional encoding on D axis
        pos_ids = torch.arange(D, device=device, dtype=torch.long)
        h = h_temporal + self.pos_embed(pos_ids)[None, :, :]     # (N, D, H)

        if n_pop is not None and n_pop < N:
            B_pop = N // n_pop
        else:
            B_pop = 1
            n_pop = N

        for cross_ind, cross_dim in zip(self.cross_ind_layers, self.cross_dim_layers):
            # Stage 1: cross-individual per dimension, batched per population
            # (N, D, H) → (B_pop, n_pop, D, H) → (D, B_pop, n_pop, H) → (D*B_pop, n_pop, H)
            h = h.view(B_pop, n_pop, D, H).permute(2, 0, 1, 3).reshape(D * B_pop, n_pop, H)
            h = cross_ind(h)                                      # (D*B_pop, n_pop, H)
            h = h.view(D, B_pop, n_pop, H).permute(1, 2, 0, 3).reshape(N, D, H)

            # FiLM modulation after cross-individual
            h = gamma[:, None, :] * h + beta[:, None, :]

            # Stage 2: cross-dimension per individual (already (N, D, H), no cross-pop)
            h = cross_dim(h)                                      # (N, D, H)

        return self.final_norm(h)


# ======================================================================
# Level 3 — Population Transformer (cross-individual)
# ======================================================================

class PopulationTransformerLayer(nn.Module):
    """Single population-level layer with soft distance bias."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.W_qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

        # GCNII-style initial residual: anchor to pre-transformer features
        self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, h, dist_bias, batch_mask=None, h_initial=None,
                tok_ind=None, batch_shape=None):
        """
        Args:
            h:            (T, hidden_dim)  T = B * N_pop * D tokens
            dist_bias:    (N_ind, N_ind, n_heads)  compact individual-level bias
            batch_mask:   unused (kept for interface compat)
            h_initial:    (T, hidden_dim) or None  for GCNII anchor
            tok_ind:      (T_pop,) long — maps token-within-pop → individual index
            batch_shape:  (B, T_pop) tuple — enables batched attention.
                          If None, falls back to flat (T, T) attention.
        Returns:
            h: (T, hidden_dim)
        """
        T, H = h.shape

        h_norm = self.norm1(h)

        if batch_shape is not None:
            # -- Batched attention: (B, T_pop, T_pop) instead of (T, T) --
            B_pop, T_pop = batch_shape
            qkv = self.W_qkv(h_norm).reshape(B_pop, T_pop, 3,
                                              self.n_heads, self.head_dim)
            Q = qkv[:, :, 0]  # (B, T_pop, heads, head_dim)
            K = qkv[:, :, 1]
            V = qkv[:, :, 2]

            logits = torch.einsum('bihd,bjhd->bijh', Q, K) * self.scale

            # Add distance bias: (N_pop, N_pop, heads) indexed by tok_ind
            logits = logits + dist_bias[tok_ind][:, tok_ind].unsqueeze(0)

            weights = torch.softmax(logits, dim=2)
            weights = self.drop(weights)

            msg = torch.einsum('bijh,bjhd->bihd', weights, V)
            msg = msg.reshape(T, H)
        else:
            # -- Flat fallback (single graph or no batching) --
            qkv = self.W_qkv(h_norm).reshape(T, 3, self.n_heads, self.head_dim)
            Q, K, V = qkv[:, 0], qkv[:, 1], qkv[:, 2]

            logits = torch.einsum('ihd,jhd->ijh', Q, K) * self.scale

            if tok_ind is not None:
                logits = logits + dist_bias[tok_ind][:, tok_ind]
            else:
                logits = logits + dist_bias

            weights = torch.softmax(logits, dim=1)
            weights = self.drop(weights)

            msg = torch.einsum('ijh,jhd->ihd', weights, V)
            msg = msg.reshape(T, H)

        msg = self.out_proj(msg)
        h = h + self.drop(msg)

        # -- FFN --
        h = h + self.ffn(self.norm2(h))

        # -- GCNII initial residual --
        if h_initial is not None:
            a = self.alpha.sigmoid()
            h = (1 - a) * h + a * h_initial

        return h


class PopulationTransformer(nn.Module):
    """Cross-individual Transformer operating on N*D tokens (zero pooling).

    Input:  h_grid (N, D, d_rnn) + coords_current (N, D)
    Output: h (N, hidden_dim), h_global (B, global_out_dim),
            h_per_head (N, n_heads, head_dim)

    Attention is over all N*D tokens within each graph. Each token
    attends to all tokens from all individuals — cross-individual
    context is preserved at the token level. Per-individual output
    is obtained by mean-pooling D tokens AFTER cross-individual
    attention (the pool is safe because tokens already encode
    population context).
    """

    def __init__(self, d_ind: int = 64, hidden_dim: int = 64,
                 global_out_dim: int = 32, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.global_out_dim = global_out_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        # d_ind is now d_rnn (tokens come directly from TwoStage, no pool_proj)
        self.d_in = d_ind

        self.input_proj = nn.Linear(d_ind, hidden_dim)
        self.layers = nn.ModuleList([
            PopulationTransformerLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.global_proj = nn.Linear(hidden_dim, global_out_dim)

        # Soft distance bias: scalar distance → per-head bias
        self.dist_bias_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, n_heads),
        )

        # Gated skip
        self.skip_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.skip_gate[0].weight)
        nn.init.zeros_(self.skip_gate[0].bias)

    def forward(self, h_grid, coords_current, v_indices=None):
        """
        Args:
            h_grid:         (N, D, d_rnn) from TwoStageAttention (no pool)
            coords_current: (N, D) raw coordinates
            v_indices:      (N,) long or None — per-individual graph mapping
        Returns:
            h:          (N, hidden_dim)    per-individual (D tokens pooled post-attn)
            h_global:   (B, global_out_dim)
            h_per_head: (N, n_heads, head_dim)
        """
        N, D, H_in = h_grid.shape
        device = h_grid.device

        # Determine batching: if v_indices provided, use batched attention
        if v_indices is not None:
            B = v_indices.max().item() + 1
            N_pop = N // B  # individuals per population
        else:
            B = 1
            N_pop = N
        T_pop = N_pop * D  # tokens per population

        # Flatten to tokens
        h_tokens = h_grid.reshape(N * D, H_in)                   # (T, d_rnn)
        h_tokens = self.input_proj(h_tokens)                      # (T, hidden_dim)
        h_pre = h_tokens

        # Token-to-individual mapping WITHIN each population
        tok_ind = torch.arange(N_pop, device=device).repeat_interleave(D)  # (T_pop,)

        # -- Distance bias: per-population (N_pop, N_pop, heads) --
        # Reshape coords to (B, N_pop, D), compute dist per population
        coords_pop = coords_current.float().view(B, N_pop, D)
        dist = torch.cdist(coords_pop, coords_pop)               # (B, N_pop, N_pop)
        # Use median from first population (all share similar scale)
        triu_idx = torch.triu_indices(N_pop, N_pop, offset=1, device=device)
        triu_dists = dist[0, triu_idx[0], triu_idx[1]]
        median_dist = triu_dists.median().detach()
        # Average distances across B for shared bias (populations have similar structure)
        dist_mean = dist.mean(dim=0)                              # (N_pop, N_pop)
        dist_norm = (dist_mean / median_dist.clamp(min=1e-6)).unsqueeze(-1)
        dist_bias_nn = self.dist_bias_mlp(dist_norm)              # (N_pop, N_pop, n_heads)

        batch_shape = (B, T_pop)

        # -- Transformer layers with GCNII --
        for layer in self.layers:
            h_tokens = layer(h_tokens, dist_bias_nn, h_initial=h_pre,
                             tok_ind=tok_ind, batch_shape=batch_shape)

        h_tokens = self.final_norm(h_tokens)

        # -- Gated skip --
        gate = self.skip_gate(torch.cat([h_tokens, h_pre], dim=-1))
        h_tokens = gate * h_tokens + (1 - gate) * h_pre

        # -- Pool D tokens per individual (AFTER cross-individual attention) --
        h_tokens_3d = h_tokens.view(N, D, self.hidden_dim)
        h = h_tokens_3d.mean(dim=1)                              # (N, hidden_dim)

        # -- Global readout --
        if v_indices is not None:
            h_global = torch.zeros(B, self.hidden_dim,
                                   device=device, dtype=h.dtype)
            h_global.scatter_reduce_(
                0, v_indices.unsqueeze(-1).expand_as(h),
                h, reduce='mean', include_self=False)
        else:
            h_global = h.mean(dim=0, keepdim=True)

        h_global = self.global_proj(h_global)
        h_per_head = h.view(N, self.n_heads, self.head_dim)

        return h, h_global, h_per_head


# ======================================================================
# Per-individual feature injection (between Level 2 and Level 3)
# ======================================================================

class IndividualFeatureInjector(nn.Module):
    """Compute and inject per-individual features that survive D-pooling.

    4 features computed from coords_current and fitness_current:
      0. fitness_rank:        soft_rank(fitness) / (N-1)         [0, 1]
      1. dist_to_best_rank:   soft_rank(||x_i - x_best||) / N   [0, 1]
      2. dist_to_centroid_rank: soft_rank(||x_i - centroid||) / N [0, 1]
      3. local_density_inv:   1 - mean_knn_dist / max_knn_dist   [0, 1]

    These match the first 4 node features of the GNN backbone that
    prevent representation collapse.
    """

    N_FEATURES = 4

    def __init__(self, d_ind: int, k_neighbors: int = 8):
        super().__init__()
        self.k = k_neighbors
        self.fuse = nn.Linear(d_ind + self.N_FEATURES, d_ind)

    def _compute_features_single(self, coords, fitness):
        """Compute 4 features for one population.

        Args:
            coords:  (N, D) coordinates
            fitness: (N,)   fitness values (may contain inf)
        Returns:
            features: (N, 4)
        """
        from .graph_features import soft_rank

        N = coords.shape[0]
        device = coords.device

        # Clamp fitness for frexp safety
        fmax = torch.finfo(fitness.dtype).max
        f_safe = fitness.clamp(min=1e-30, max=fmax)

        # 0. Fitness rank [0, 1] — lower fitness = lower rank = better
        fit_rank = soft_rank(f_safe) / max(N - 1, 1)

        # Pairwise distances (reused for features 1, 2, 3)
        coords_f = coords.float()
        dist = torch.cdist(coords_f, coords_f)                   # (N, N)

        # 1. Distance to best individual
        best_idx = f_safe.argmin()
        dist_to_best = dist[:, best_idx]
        dtb_rank = soft_rank(dist_to_best) / max(N - 1, 1)

        # 2. Distance to centroid
        centroid = coords_f.mean(dim=0, keepdim=True)             # (1, D)
        dist_to_cent = (coords_f - centroid).norm(dim=1)          # (N,)
        dtc_rank = soft_rank(dist_to_cent) / max(N - 1, 1)

        # 3. Local density inverse (k-NN mean distance)
        k = min(self.k, N - 1)
        knn_dists = dist.topk(k + 1, dim=1, largest=False).values[:, 1:]  # skip self
        mean_knn = knn_dists.mean(dim=1)                          # (N,)
        max_knn = mean_knn.max().clamp(min=1e-6)
        ld_inv = 1.0 - mean_knn / max_knn

        return torch.stack([fit_rank, dtb_rank, dtc_rank, ld_inv], dim=-1)

    def _compute_features_batched(self, coords, fitness, B, N):
        """Vectorized feature computation for B equal-sized populations.

        Args:
            coords:  (B*N, D) coordinates
            fitness: (B*N,)   fitness values
            B: number of populations
            N: population size
        Returns:
            features: (B*N, 4)
        """
        from .graph_features import soft_rank

        D = coords.shape[1]
        fmax = torch.finfo(fitness.dtype).max
        f_safe = fitness.clamp(min=1e-30, max=fmax).view(B, N)

        # 0. Fitness rank — soft_rank supports (B, N) natively
        fit_rank = soft_rank(f_safe) / max(N - 1, 1)              # (B, N)

        # Pairwise distances — cdist supports (B, N, D)
        coords_f = coords.float().view(B, N, D)
        dist = torch.cdist(coords_f, coords_f)                    # (B, N, N)

        # 1. Distance to best individual per population
        best_idx = f_safe.argmin(dim=1)                            # (B,)
        b_idx = torch.arange(B, device=coords.device)
        dist_to_best = dist[b_idx, :, best_idx]                   # (B, N)
        dtb_rank = soft_rank(dist_to_best) / max(N - 1, 1)        # (B, N)

        # 2. Distance to centroid
        centroid = coords_f.mean(dim=1, keepdim=True)              # (B, 1, D)
        dist_to_cent = (coords_f - centroid).norm(dim=2)           # (B, N)
        dtc_rank = soft_rank(dist_to_cent) / max(N - 1, 1)        # (B, N)

        # 3. Local density inverse (k-NN)
        k = min(self.k, N - 1)
        knn_dists = dist.topk(k + 1, dim=2, largest=False).values[:, :, 1:]
        mean_knn = knn_dists.mean(dim=2)                           # (B, N)
        max_knn = mean_knn.amax(dim=1, keepdim=True).clamp(min=1e-6)
        ld_inv = 1.0 - mean_knn / max_knn                         # (B, N)

        return torch.stack(
            [fit_rank, dtb_rank, dtc_rank, ld_inv], dim=-1
        ).reshape(B * N, 4)

    def forward(self, h_ind, coords_current, fitness_current, v_indices=None):
        """
        Args:
            h_ind:           (N_total, d_ind)
            coords_current:  (N_total, D)
            fitness_current: (N_total,)
            v_indices:       (N_total,) long or None
        Returns:
            h_ind_enriched:  (N_total, d_ind)
        """
        N_total = h_ind.shape[0]

        if v_indices is None:
            features = self._compute_features_single(
                coords_current, fitness_current)
        else:
            # Check for equal-sized populations (common case: batched rollouts)
            B = v_indices[-1].item() + 1
            if N_total % B == 0:
                N = N_total // B
                features = self._compute_features_batched(
                    coords_current, fitness_current, B, N)
            else:
                # Fallback for unequal populations
                features = torch.zeros(
                    N_total, self.N_FEATURES, device=h_ind.device)
                for b in range(B):
                    mask = v_indices == b
                    features[mask] = self._compute_features_single(
                        coords_current[mask], fitness_current[mask])

        return self.fuse(torch.cat([h_ind, features], dim=-1))


# ======================================================================
# Cross-attention block
# ======================================================================

class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention + FFN. Queries attend to separate keys/values."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_kv = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv):
        """q: (B, Lq, d), kv: (B, Lkv, d) → (B, Lq, d)"""
        B, Lq, H = q.shape
        Lkv = kv.shape[1]

        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)

        Q = self.W_q(q_norm).reshape(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        kv_proj = self.W_kv(kv_norm).reshape(B, Lkv, 2, self.n_heads, self.head_dim)
        K, V = kv_proj.unbind(dim=2)
        K, V = K.transpose(1, 2), V.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.drop.p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).reshape(B, Lq, H)
        q = q + self.drop(self.out_proj(attn_out))

        q = q + self.ffn(self.norm2(q))
        return q


class InducedPointPooler(nn.Module):
    """Induced-point cross-attention pooler: (N, D, d_rnn) -> (N, d_out).

    R learnable queries attend to D keys via cross-attention.
    O(R*D) in D — dim-agnostic, vmap-compatible, GPU-only.
    Drop-in replacement for TemporalDimPooler.
    """

    def __init__(self, d_rnn: int = 32, d_out: int = 32,
                 n_induced: int = 8, n_heads: int | None = None,
                 dropout: float = 0.0):
        super().__init__()
        if n_heads is None:
            n_heads = max(1, d_rnn // 16)
            while d_rnn % n_heads != 0:
                n_heads -= 1
        self.queries = nn.Parameter(torch.empty(n_induced, d_rnn))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.cross_attn = _CrossAttnBlock(d_model=d_rnn, n_heads=n_heads,
                                          dropout=dropout)
        self.proj = nn.Linear(d_rnn, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, h_temporal):
        """h_temporal: (N, D, d_rnn) -> (N, d_out)"""
        N = h_temporal.shape[0]
        h_temporal = h_temporal.to(self.queries.dtype)
        q = self.queries.unsqueeze(0).expand(N, -1, -1)  # (N, R, d_rnn)
        attended = self.cross_attn(q, h_temporal)          # (N, R, d_rnn)
        pooled = attended.mean(dim=1)                      # (N, d_rnn)
        return self.norm(self.proj(pooled))                # (N, d_out)


# ======================================================================
# Three-stage factored attention (replaces GRU + TwoStageAttention)
# ======================================================================

class ThreeStageAttention(nn.Module):
    """Three-stage factored attention over (N, D) population grid.

    Stage 1: Self-attention over W*D per individual  O(N·(W·D)²)
             Processes temporal + dimensional structure, then pools W.
    Stage 2: Self-attention over N per dimension     O(N²)
             + learnable readout tokens for h_global
    Stage 3: Cross-attention broadcast               O(N·D·n_readout)
             Each individual's D tokens attend to readout (population context).

    Replaces TemporalGRUEncoder + TwoStageAttention.
    """

    N_FEATURES = 7  # 6 temporal features + fes_ratio

    def __init__(self, n_feat=7, d_model=32, d_global_out=64,
                 n_layers=2, n_heads=4, max_W=16,
                 max_D=100, n_readout=4, vel_ema_alpha=0.3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_readout = n_readout
        self.vel_ema_alpha = vel_ema_alpha

        # Feature projection
        self.input_norm = nn.LayerNorm(n_feat)
        self.input_proj = nn.Linear(n_feat, d_model)

        # Positional encodings
        self.dim_embed = nn.Embedding(max_D, d_model)
        self.time_embed = nn.Embedding(max_W, d_model)

        # Readout tokens for population description
        self.readout_tokens = nn.Parameter(
            torch.randn(n_readout, d_model) * 0.02)

        # 3 stages × n_layers
        self.dim_layers = nn.ModuleList([
            _PreNormAttnBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)])
        self.ind_layers = nn.ModuleList([
            _PreNormAttnBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)])
        self.cross_layers = nn.ModuleList([
            _CrossAttnBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)])

        self.final_norm = nn.LayerNorm(d_model)
        self.readout_proj = nn.Linear(d_model, d_global_out)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _window_normalize_fitness(fitness):
        """Normalize fitness to [0, 1] within the window."""
        fmax = torch.finfo(fitness.dtype).max
        f_safe = fitness.clamp(min=1e-30, max=fmax)
        f_best = f_safe.min()
        f_worst = f_safe.max()
        denom = (f_worst - f_best).clamp(min=1e-30)
        f_norm = (f_safe - f_best) / denom
        df_norm = F.pad((f_norm[1:] - f_norm[:-1]).abs(), (0, 0, 1, 0))
        return f_norm, df_norm

    def _build_features(self, coords_hist, fitness_hist, n_valid,
                        fes_ratio, device):
        """Build 7 features from ring buffer, keep temporal structure.

        Returns:
            features: (N, W*D, 7) — all timesteps preserved as tokens
            W_valid: int — number of valid timesteps
        """
        W, N, D = coords_hist.shape
        nv = n_valid.item()

        x_norm = coords_hist / 100.0
        dx = F.pad(coords_hist[1:] - coords_hist[:-1],
                    (0, 0, 0, 0, 1, 0)) / 100.0

        f_norm, df_norm = self._window_normalize_fitness(fitness_hist)
        f_norm = f_norm.unsqueeze(-1).expand_as(x_norm)
        df_norm = df_norm.unsqueeze(-1).expand_as(x_norm)

        # Velocity EMA
        alpha = self.vel_ema_alpha
        abs_dx = dx.abs()
        vel_ema = torch.zeros_like(x_norm)
        vel_ema[0] = abs_dx[0]
        for t in range(1, W):
            vel_ema[t] = alpha * abs_dx[t] + (1 - alpha) * vel_ema[t - 1]

        # Fitness-weighted displacement
        f_diff = F.pad(f_norm[1:] - f_norm[:-1], (0, 0, 0, 0, 1, 0))
        improvement_sign = torch.sign(-f_diff[:, :, 0:1]).expand_as(dx)
        fw_disp = dx * improvement_sign

        # 7th feature: fes_ratio broadcast to (W, N, D)
        if fes_ratio is None:
            fes = torch.full((N,), 0.5, device=device)
        elif fes_ratio.dim() == 0:
            fes = fes_ratio.expand(N)
        else:
            fes = fes_ratio
        fes_broad = fes[None, :, None].expand(W, N, D)

        # Stack: (W, N, D, 7) — keep only valid timesteps
        features = torch.stack(
            [x_norm, dx, f_norm, df_norm, vel_ema, fw_disp, fes_broad],
            dim=-1)[:nv]                                  # (nv, N, D, 7)

        # Reshape to (N, nv*D, 7) — each individual gets nv×D tokens
        features = (features.permute(1, 0, 2, 3)
                            .reshape(N, nv * D, self.N_FEATURES))
        return features, nv

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def _stage2_with_readout(self, h, readout, ind_attn, B_pop, n_pop, D):
        """Stage 2: self-attn over N per dimension, with readout tokens."""
        H = self.d_model
        N = B_pop * n_pop
        n_r = self.n_readout

        # Reshape to (D*B_pop, n_pop, H)
        h_2d = (h.view(B_pop, n_pop, D, H)
                 .permute(2, 0, 1, 3)
                 .reshape(D * B_pop, n_pop, H))

        # Tile readout across D: (B_pop, n_r, H) → (D*B_pop, n_r, H)
        readout_exp = (readout.unsqueeze(0)
                       .expand(D, -1, -1, -1)
                       .reshape(D * B_pop, n_r, H))

        h_with_r = torch.cat([h_2d, readout_exp], dim=1)
        h_with_r = ind_attn(h_with_r)

        h_2d = h_with_r[:, :n_pop]
        readout_out = h_with_r[:, n_pop:]

        # Reshape back
        h = (h_2d.view(D, B_pop, n_pop, H)
                  .permute(1, 2, 0, 3)
                  .reshape(N, D, H))

        # Aggregate readout across D
        readout = (readout_out.view(D, B_pop, n_r, H)
                              .mean(dim=0))             # (B_pop, n_r, H)
        return h, readout

    def _stage3_broadcast(self, h, readout, cross_attn, B_pop, n_pop, D):
        """Stage 3: each individual's D tokens cross-attend to readout tokens.

        No pooling — readout tokens serve as population context keys.
        Each individual produces different output because its D queries differ.
        """
        H = self.d_model
        N = B_pop * n_pop

        # q: each individual's D tokens → (B_pop*n_pop, D, H)
        q = h.view(N, D, H)

        # kv: readout tokens, shared across individuals within population
        # (B_pop, n_readout, H) → (B_pop*n_pop, n_readout, H)
        kv = (readout.unsqueeze(1)
                     .expand(B_pop, n_pop, self.n_readout, H)
                     .reshape(N, self.n_readout, H))

        # Cross-attention: D queries attend to n_readout keys
        h = h + cross_attn(q, kv).reshape(N, D, H)
        return h

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, coords_hist, fitness_hist, n_valid,
                fitness_current, fes_ratio=None, n_pop=None):
        """
        Args:
            coords_hist:     (W, N_total, D)
            fitness_hist:    (W, N_total)
            n_valid:         scalar tensor
            fitness_current: (N_total,)
            fes_ratio:       (N_total,) or scalar or None
            n_pop:           int — individuals per population (None = single pop)

        Returns:
            h_grid:  (N_total, D, d_model)
            h_global: (B, d_global_out)
        """
        N = coords_hist.shape[1]
        D = coords_hist.shape[2]
        device = coords_hist.device

        # Feature extraction — keep temporal structure: (N, W*D, 7)
        features, W_valid = self._build_features(
            coords_hist, fitness_hist, n_valid, fes_ratio, device)
        h = self.input_proj(self.input_norm(features))  # (N, W*D, d_model)

        # Positional encodings: dim + time
        dim_ids = torch.arange(D, device=device, dtype=torch.long)
        time_ids = torch.arange(W_valid, device=device, dtype=torch.long)
        # Tile: dim repeats W times, time repeats D times
        pos = (self.dim_embed(dim_ids).unsqueeze(0).expand(W_valid, -1, -1)
               + self.time_embed(time_ids).unsqueeze(1).expand(-1, D, -1))
        pos = pos.reshape(W_valid * D, self.d_model)     # (W*D, d_model)
        h = h + pos[None, :, :]

        # Population batching
        if n_pop is not None and n_pop < N:
            B_pop = N // n_pop
        else:
            B_pop = 1
            n_pop = N

        # Init readout tokens
        readout = self.readout_tokens.unsqueeze(0).expand(
            B_pop, -1, -1).clone()

        for dim_attn, ind_attn, cross_attn in zip(
                self.dim_layers, self.ind_layers, self.cross_layers):
            # Stage 1: self-attn over W*D per individual
            h = dim_attn(h)                               # (N, W*D, d_model)

            # Stage 2 needs (N, D, d_model) — pool W after Stage 1
            if h.shape[1] > D:
                h = (h.view(N, W_valid, D, self.d_model)
                      .mean(dim=1))                       # (N, D, d_model)

            # Stage 2: self-attn over N per dimension + readout
            h, readout = self._stage2_with_readout(
                h, readout, ind_attn, B_pop, n_pop, D)

            # Stage 3: D tokens cross-attend to readout (population context)
            h = self._stage3_broadcast(
                h, readout, cross_attn, B_pop, n_pop, D)

        h = self.final_norm(h)
        h_global = self.readout_proj(readout.mean(dim=1))
        return h, h_global
