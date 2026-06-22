"""DE-family operator heads: BatchedDiffDE, BatchedDiffAttDE, AdaptiveFCRBeta."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.operators._base import _ParamMLP, _make_proj


def _safe_conc(x):
    """Sanitize a Beta concentration tensor: non-finite -> finite, strictly
    positive. Prevents a hard ValueError in torch.distributions.Beta when an
    upstream NaN reaches the F/CR head (bug-prevention pattern #9). Healthy
    concentrations (>= 1, from 1 + softplus) pass through unchanged; the
    resulting NaN loss is then handled by the optimizer-step NaN guard, turning
    an unrecoverable crash into a recoverable skipped step."""
    return torch.nan_to_num(x, nan=1.0, posinf=1e4, neginf=1.0).clamp(min=1e-3)


class AdaptiveFCRBeta(nn.Module):
    """Beta-parameterized adaptive F/CR predictor.

    Produces per-individual Beta distribution parameters for F and CR,
    structurally bounded to prevent sigmoid saturation collapse.

    Input: cat(h_individual, h_global) = 2*h_dim
    Output: f_alpha, f_beta, cr_alpha, cr_beta ∈ [1, 1+alpha_beta_max]
    """

    def __init__(self, h_dim=128, hidden=64, alpha_beta_max=5.0):
        super().__init__()
        self.alpha_beta_max = alpha_beta_max

        self.shared = nn.Sequential(
            nn.Linear(2 * h_dim, hidden),
            nn.SiLU(),
        )
        self.f_head = nn.Linear(hidden, 2)
        self.cr_head = nn.Linear(hidden, 2)

        # Init to produce alpha=beta=3 → Beta(3,3) centered at 0.5
        # softplus(1.54) ≈ 2.0, so 1 + 2.0 = 3.0
        with torch.no_grad():
            self.f_head.weight.zero_()
            self.f_head.bias.fill_(1.54)
            self.cr_head.weight.zero_()
            self.cr_head.bias.fill_(1.54)

    def forward(self, h_ind, h_global):
        """
        Args:
            h_ind: (B, N, h_dim) per-individual backbone embedding
            h_global: (B, h_dim) global population embedding
        Returns:
            f_alpha, f_beta, cr_alpha, cr_beta: each (B, N)
        """
        h_global_exp = h_global.unsqueeze(1).expand(-1, h_ind.size(1), -1)
        x = torch.cat([h_ind, h_global_exp], dim=-1)  # (B, N, 2*h_dim)

        shared = self.shared(x)  # (B, N, hidden)

        f_raw = self.f_head(shared)    # (B, N, 2)
        cr_raw = self.cr_head(shared)  # (B, N, 2)

        f_alpha = 1 + F.softplus(f_raw[..., 0]).clamp(max=self.alpha_beta_max)
        f_beta = 1 + F.softplus(f_raw[..., 1]).clamp(max=self.alpha_beta_max)
        cr_alpha = 1 + F.softplus(cr_raw[..., 0]).clamp(max=self.alpha_beta_max)
        cr_beta = 1 + F.softplus(cr_raw[..., 1]).clamp(max=self.alpha_beta_max)

        return f_alpha, f_beta, cr_alpha, cr_beta


def _compute_crossover_mask(cr_b, u, crossover_temp, use_ste=True):
    """Crossover mask. STE: forward hard {0,1}, backward soft sigmoid grad
    through CR (matches L-SHADE binomial swap semantics, introduced d2bb237).
    `use_ste=False` reverts to pure soft sigmoid (pre-d2bb237 behavior) — used
    for A/B testing and to evaluate pre-STE checkpoints under their training
    regime.
    """
    if use_ste:
        mask_soft = torch.sigmoid(crossover_temp * (cr_b - u))
        mask_hard = (cr_b > u).to(cr_b.dtype)
        return mask_hard + (mask_soft - mask_soft.detach())
    return torch.sigmoid(crossover_temp * (cr_b - u))


def _sample_shade_fcr(F_i, CR_i, params_dict, M, B, N, device, dtype):
    """SHADE-style F/CR override for the fcr_shade lesions (eval-time only).

    Keeps the learned donor selection intact and replaces ONLY the Beta-sampled
    F/CR with a canonical SHADE draw, so the lesion isolates the F/CR head
    rather than the fcr_static strawman (constant F/CR also removes all spread):
      - `_fcr_shade_memory` (LShadeMemory): full success-history adaptation.
      - `_fcr_shade_static`: Cauchy(0.5, 0.1) / Normal(0.5, 0.1), no adaptation.
    Returns (F_i, CR_i) unchanged when no shade flag is set.
    """
    mem = params_dict.get('_fcr_shade_memory')
    if mem is not None:
        F_s, CR_s, _r = mem.sample(N, M=M)
        return (F_s.to(device=device, dtype=dtype),
                CR_s.to(device=device, dtype=dtype))
    if params_dict.get('_fcr_shade_static'):
        u = torch.rand(M, B, N, device=device, dtype=dtype)
        F_s = (0.5 + 0.1 * torch.tan(math.pi * (u - 0.5))).clamp(0.01, 1.0)
        CR_s = (0.5 + 0.1 * torch.randn(M, B, N, device=device, dtype=dtype)
                ).clamp(0.0, 1.0)
        return F_s, CR_s
    return F_i, CR_i


def _de_sample_batch(params_dict, coords, M, crossover_temp, memory_gate=None,
                     use_ste=True):
    """Shared DE mutation + crossover. Returns (M, B, N, D).

    Supports two F/CR parameterizations:
    - Legacy (sigmoid): uses F_mu, F_logsig, CR_mu, CR_logsig
    - Beta: uses _f_alpha, _f_beta, _cr_alpha, _cr_beta

    If memory_gate is provided and params_dict contains F_prior/CR_prior,
    blends neural F/CR with SHADE memory prior (legacy path only).

    Side effect: writes '_realized_F' and '_realized_CR' (detached) into
    params_dict so callers can read the actually-used F/CR values.
    """
    B, N, D = coords.shape
    dtype, device = coords.dtype, coords.device

    fcr_mode = params_dict.get('_fcr_mode', 'beta' if '_f_alpha' in params_dict else 'legacy')

    if fcr_mode == 'lshade':
        # E13 distillation path: teacher's L-SHADE memory drives F, CR.
        # The teacher is attached as `params_dict['_lshade_teacher']`; sample()
        # returns per-(M, B, N) values. _realized_F/CR are detached (no grad
        # through teacher's stochastic sampling). Distillation loss reads
        # `_mu_F_pred`/`_mu_CR_pred` from the head + `_realized_F`/`CR` realized.
        teacher = params_dict.get('_lshade_teacher')
        if teacher is None:
            raise RuntimeError(
                "fcr_mode='lshade' requires `_lshade_teacher` in params_dict "
                "(attach LShadeMemory instance to head._lshade_teacher).")
        F_i, CR_i, _r = teacher.sample(N, M=M)
        F_i = F_i.to(device=device, dtype=dtype)
        CR_i = CR_i.to(device=device, dtype=dtype)
    elif fcr_mode == 'cauchy_neural':
        # Inference path: GNN's μ_F_pred/μ_CR_pred drive sampling directly.
        # Cauchy(μ_F_pred, 0.1) for F; Normal(μ_CR_pred, 0.1) for CR.
        mu_F = params_dict['_mu_F_pred']  # (B, N)
        mu_CR = params_dict['_mu_CR_pred']
        mu_F_e = mu_F.unsqueeze(0).expand(M, -1, -1)  # (M, B, N)
        mu_CR_e = mu_CR.unsqueeze(0).expand(M, -1, -1)
        # Sigma for the F Cauchy: learned per-individual when present in
        # params (falsification arm C, --fcr-learn-sigma); else legacy 0.1.
        sigma_F_pred = params_dict.get('_sigma_F_pred')
        if sigma_F_pred is not None:
            sigma_F_e = sigma_F_pred.unsqueeze(0).expand(M, -1, -1)
        else:
            sigma_F_e = 0.1
        # Cauchy sampling with resample-on-non-positive (max 10 iters).
        F_i = mu_F_e + sigma_F_e * torch.tan(math.pi * (torch.rand_like(mu_F_e) - 0.5))
        for _ in range(10):
            bad = F_i <= 0.0
            if not bad.any():
                break
            new = mu_F_e + sigma_F_e * torch.tan(math.pi * (torch.rand_like(mu_F_e) - 0.5))
            F_i = torch.where(bad, new, F_i)
        # E3 — F clipping mode (eval-time only).
        # 'clamp_005' (default): clamp F to [0.05, 1.0] — TersQ legacy.
        # 'reject_resample_pyade': pyade behaviour. F>1 → 0 → resample loop.
        #     Removes both 0.05 floor and 1.0 saturation; mass at F just over 0
        #     after resamples settles per-Cauchy distribution.
        fcr_clip_mode = params_dict.get('_fcr_clip_mode', 'clamp_005')
        if fcr_clip_mode == 'reject_resample_pyade':
            F_i = torch.where(F_i > 1.0, torch.zeros_like(F_i), F_i)
            for _ in range(10):
                bad = F_i <= 0.0
                if not bad.any():
                    break
                new = mu_F_e + sigma_F_e * torch.tan(
                    math.pi * (torch.rand_like(mu_F_e) - 0.5))
                F_i = torch.where(bad, new, F_i)
            # Hard backstop: any remaining non-positive → clip to ε (1e-3).
            F_i = F_i.clamp(min=1e-3).to(dtype)
        else:
            F_i = F_i.clamp(min=0.05, max=1.0).to(dtype)
        CR_i = (mu_CR_e + 0.1 * torch.randn_like(mu_CR_e)).clamp(0.0, 1.0).to(dtype)
        # Exp 1 Arm 2 — eval-time forced F/CR override. When `_force_F`
        # or `_force_CR` are set on the head (via head._force_F_attr /
        # _force_CR_attr → params_dict), replace the sampled values with
        # constants. Tests whether AdaptiveFCRCauchy earns its keep vs a
        # fixed scheduler under atomic donor.
        force_F = params_dict.get('_force_F')
        if force_F is not None:
            F_i = torch.full_like(F_i, float(force_F))
        force_CR = params_dict.get('_force_CR')
        if force_CR is not None:
            CR_i = torch.full_like(CR_i, float(force_CR))
    elif fcr_mode == 'beta' or '_f_alpha' in params_dict:
        # Beta parameterization path
        f_dist = torch.distributions.Beta(_safe_conc(params_dict['_f_alpha']),
                                          _safe_conc(params_dict['_f_beta']))
        cr_dist = torch.distributions.Beta(_safe_conc(params_dict['_cr_alpha']),
                                           _safe_conc(params_dict['_cr_beta']))
        F_i = 0.1 + 0.8 * f_dist.rsample([M])   # (M, B, N)
        CR_i = 0.1 + 0.8 * cr_dist.rsample([M])  # (M, B, N)
        # fcr_shade lesions: swap the Beta draw for a canonical SHADE F/CR draw.
        F_i, CR_i = _sample_shade_fcr(F_i, CR_i, params_dict, M, B, N,
                                      device, dtype)
        # Action 2 (forced F/CR schedule): override neural F or CR with
        # externally supplied scalar (LSHADE-style schedule from eval). Tests
        # whether the operator can use variable F/CR when head prediction is
        # bypassed. Each can be overridden independently.
        force_F = params_dict.get('_force_F')
        if force_F is not None:
            F_i = torch.full_like(F_i, float(force_F))
        force_CR = params_dict.get('_force_CR')
        if force_CR is not None:
            CR_i = torch.full_like(CR_i, float(force_CR))
    else:
        # Legacy sigmoid parameterization
        noise_F = torch.randn(M, B, N, device=device)
        noise_CR = torch.randn(M, B, N, device=device)
        F_neural = 0.1 + 0.7 * torch.sigmoid(
            params_dict['F_mu'] + F.softplus(params_dict['F_logsig']) * noise_F)
        CR_neural = 0.1 + 0.8 * torch.sigmoid(
            params_dict['CR_mu'] + F.softplus(params_dict['CR_logsig']) * noise_CR)

        if memory_gate is not None and 'F_prior' in params_dict:
            gate = torch.sigmoid(memory_gate)
            F_i = (1 - gate) * F_neural + gate * params_dict['F_prior'].unsqueeze(0)
            CR_i = (1 - gate) * CR_neural + gate * params_dict['CR_prior'].unsqueeze(0)
        else:
            F_i = F_neural
            CR_i = CR_neural

    params_dict['_realized_F'] = F_i.detach()
    params_dict['_realized_CR'] = CR_i.detach()

    # Option A — per-M donor resampling (BatchedDiffAttDE path only).
    # By default, x_pbest and x_diff are (B, N, D) — ONE donor triple per parent,
    # broadcast over M. That makes the M proposals for a parent lie on a 1D ray
    # (same direction, varying F magnitude + crossover mask). When
    # params_dict['_per_m_donors'] is True, we resample the attention gumbel-softmax
    # over M to get genuinely different donor triples per M.
    if params_dict.get('_per_m_donors') and '_A_pbest' in params_dict:
        A_pbest = params_dict['_A_pbest']  # (B, N, N_pool) or (B, N, k_donor)
        A_r1 = params_dict['_A_r1']
        A_r2 = params_dict['_A_r2']
        # Augmented donor pool when graph-native archive is active.
        # Falls back to active coords when '_donor_coords' is None or absent.
        pool_coords = params_dict.get('_donor_coords')
        if pool_coords is None:
            pool_coords = coords
        N_pool = pool_coords.shape[1]
        # tau may be a scalar float or a 0-d tensor; gumbel_softmax accepts both.
        tau_mut = params_dict.get('_tau', 1.0)
        # D1000 line: when present, A_* axis 2 is the kNN-restricted local
        # slot (k_donor << N_pool) and cand_idx maps local→global.
        cand_idx = params_dict.get('_donor_cand_idx')
        knn_donor = cand_idx is not None
        K_axis = A_pbest.shape[-1]

        A_pbest_m = A_pbest.unsqueeze(0).expand(M, -1, -1, -1)
        A_r1_m = A_r1.unsqueeze(0).expand(M, -1, -1, -1)
        A_r2_m = A_r2.unsqueeze(0).expand(M, -1, -1, -1).clone()

        w_pbest_m = F.gumbel_softmax(A_pbest_m, tau=tau_mut, hard=True)
        w_r1_m = F.gumbel_softmax(A_r1_m, tau=tau_mut, hard=True)
        r1_idx_m = w_r1_m.argmax(dim=-1)                               # (M, B, N)
        r1_mask_m = F.one_hot(r1_idx_m, K_axis).bool()
        A_r2_m = A_r2_m.masked_fill(r1_mask_m, -1e9)
        w_r2_m = F.gumbel_softmax(A_r2_m, tau=tau_mut, hard=True)

        if knn_donor:
            # cand_coords (B, N, k_donor, D) was gathered ONCE in
            # compute_params and cached in params_dict — reuse here. Falls
            # back to a re-gather if the single-shot path didn't populate it
            # (e.g. when a caller bypasses compute_params).
            cand_coords = params_dict.get('_cand_coords')
            if cand_coords is None:
                B_cd, N_cd, k_donor = cand_idx.shape
                D_pool = pool_coords.shape[-1]
                cand_flat = cand_idx.reshape(B_cd, N_cd * k_donor)
                cand_coords = pool_coords.gather(
                    1, cand_flat.unsqueeze(-1).expand(B_cd, N_cd * k_donor, D_pool)
                ).reshape(B_cd, N_cd, k_donor, D_pool).to(dtype)
            x_pbest_m = torch.einsum('mbnk,bnkd->mbnd',
                                     w_pbest_m.to(dtype), cand_coords)
            x_r1_m = torch.einsum('mbnk,bnkd->mbnd',
                                  w_r1_m.to(dtype), cand_coords)
            x_r2_m = torch.einsum('mbnk,bnkd->mbnd',
                                  w_r2_m.to(dtype), cand_coords)
        else:
            x_pbest_m = torch.einsum('mbnk,bkd->mbnd', w_pbest_m.to(dtype),
                                     pool_coords)
            x_r1_m = torch.einsum('mbnk,bkd->mbnd', w_r1_m.to(dtype),
                                  pool_coords)
            x_r2_m = torch.einsum('mbnk,bkd->mbnd', w_r2_m.to(dtype),
                                  pool_coords)

        x_diff_m = x_r1_m - x_r2_m                                     # (M, B, N, D)

        # ── gather-bypass override (eval-time intervention) ──
        # Falsification follow-up (docs/falsification_2026_04_28_results.md):
        # tests whether the deployed operator's hard-gather to N_pool points
        # restricts proposal geometry to a discrete lattice. Two modes:
        #   'pbest_jitter' — keep gumbel-selected x_pbest, add per-axis Gaussian
        #     noise scaled to ε·σ_pop. Tests off-lattice escape with attention
        #     selection preserved.
        #   'rand' — replace x_pbest with uniform random gather from pool, then
        #     add same jitter (or none if eps=0). Destroys both attention
        #     selection and lattice restriction.
        # Only x_pbest is modified; x_r1/x_r2 stay on-lattice so the F·(r1−r2)
        # diff term keeps its DE semantics.
        _gb_mode = params_dict.get('_gather_bypass_mode')
        _gb_eps  = float(params_dict.get('_gather_bypass_eps') or 0.0)
        if _gb_mode and _gb_mode != 'off':
            sigma_pop = coords.std(dim=1, keepdim=True).clamp(min=1e-12)  # (B, 1, D)
            sigma_pop_b = sigma_pop.unsqueeze(0).expand(M, B, N, D).to(dtype)
            if _gb_mode == 'rand':
                rand_idx = torch.randint(N_pool, (M, B, N), device=device)
                rand_idx_exp = rand_idx.unsqueeze(-1).expand(-1, -1, -1, D)
                pool_m = pool_coords.unsqueeze(0).expand(M, -1, -1, -1).to(dtype)
                x_pbest_m = torch.gather(pool_m, 2, rand_idx_exp)
            elif _gb_mode != 'pbest_jitter':
                raise ValueError(
                    f"_gather_bypass_mode must be off|pbest_jitter|rand, got {_gb_mode}")
            if _gb_eps > 0.0:
                eps_gauss = torch.randn(M, B, N, D, device=device, dtype=dtype)
                x_pbest_m = x_pbest_m + (_gb_eps * sigma_pop_b * eps_gauss)

        diff = x_pbest_m - coords.unsqueeze(0) + x_diff_m              # (M, B, N, D)

        # Expose per-m donor indices for supervised donor-oracle loss.
        # compute_donor_oracle_loss grades these against best-m fitness.
        params_dict['_pbest_idx_m'] = w_pbest_m.argmax(dim=-1).detach()  # (M, B, N)
        params_dict['_r1_idx_m'] = r1_idx_m.detach()                     # (M, B, N)
        params_dict['_r2_idx_m'] = w_r2_m.argmax(dim=-1).detach()        # (M, B, N)
    else:
        diff = (params_dict['x_pbest'] - coords + params_dict['x_diff']).unsqueeze(0)
    mutation = F_i.unsqueeze(-1).to(dtype) * diff

    # E2 — eval-time operator-scale hook. Tests whether smaller moves help
    # convergence on hybrids/composition. mode∈{off,scalar,lin_diam,sqrt_diam}.
    # Scalar: mutation *= constant. lin_diam: mutation *= (sigma_pop_now /
    # sigma_pop_init) per batch. sqrt_diam: same but sqrt'd.
    op_scale_mode = params_dict.get('_operator_scale_mode')
    if op_scale_mode and op_scale_mode != 'off':
        if op_scale_mode == 'scalar':
            scale = float(params_dict.get('_operator_scale_value', 1.0))
            mutation = mutation * scale
        elif op_scale_mode in ('lin_diam', 'sqrt_diam', 'ratio_init'):
            sigma_now = coords.std(dim=1).mean(dim=-1).clamp(min=1e-6)  # (B,)
            sigma_init = params_dict.get('_sigma_pop_init')
            if sigma_init is None:
                sigma_init = sigma_now.detach()  # fallback: identity
            elif not torch.is_tensor(sigma_init):
                sigma_init = torch.tensor(sigma_init, device=sigma_now.device,
                                           dtype=sigma_now.dtype)
            ratio = (sigma_now / sigma_init.to(sigma_now)).clamp(max=1.0)
            if op_scale_mode == 'sqrt_diam':
                ratio = ratio.sqrt()
            elif op_scale_mode == 'ratio_init':
                ratio = ratio.clamp(min=0.1)
            mutation = mutation * ratio.view(1, B, 1, 1).to(dtype)

    u = torch.rand(M, B, N, D, device=device, dtype=dtype)
    cr_b = CR_i.unsqueeze(-1).to(dtype)
    mask = _compute_crossover_mask(cr_b, u, crossover_temp, use_ste=use_ste)
    j_rand = torch.randint(0, D, (M, B, N), device=device)
    mask = mask.scatter(-1, j_rand.unsqueeze(-1), 1.0)

    return mask * mutation


class BatchedDiffDE(nn.Module):
    """Learned DE/current-to-pbest/1 with per-population pbest attention.

    NOT L-SHADE (no LPSR, no archive, no success-history memory).
    Predicts F, CR per node from embeddings. Mutation topology:
    delta = F * (x_pbest - x + x_diff) with soft crossover.

    All ops are (B, N, ...) native — no loops, no scatter.
    """

    def __init__(self, embed_dim=16, head_idx=0, crossover_temp=10.0, p_best=0.15,
                 backbone_dim=0):
        super().__init__()
        self.embed_dim = embed_dim

        self.head_idx = head_idx
        self.crossover_temp = crossover_temp
        self.p_best = p_best

        self.backbone_dim = backbone_dim
        self.proj, self.proj_norm = _make_proj(backbone_dim, embed_dim)

        full_dim = embed_dim + backbone_dim
        self.param_mlp = _ParamMLP(full_dim, 4, hidden=64)  # F_mu, F_logsig, CR_mu, CR_logsig
        with torch.no_grad():
            self.param_mlp.mlp[-1].bias[2] = 0.8  # CR_mu warm-start
        self.diff_key = nn.Linear(embed_dim, 8, bias=False)
        self.diff_query = nn.Linear(embed_dim, 8, bias=False)
        self.subpop_scale = nn.Parameter(torch.tensor(0.0))
        self.memory_gate = nn.Parameter(torch.tensor(0.0))

    def get_embedding(self, h_backbone):
        if self.proj is not None:
            return self.proj_norm(self.proj(h_backbone))
        return h_backbone[..., :self.embed_dim]

    def compute_params(self, h_out, coords, fitness, adj=None,
                       route_probs=None, bounds_span=200.0,
                       knn_idx=None, h_backbone=None, **_kwargs):
        B, N, D = coords.shape
        hd = self.embed_dim

        h_in = torch.cat([h_out, h_backbone], dim=-1) if h_backbone is not None else h_out
        params_raw = self.param_mlp(h_in)
        F_mu = params_raw[..., 0]
        F_logsig = params_raw[..., 1]
        CR_mu = params_raw[..., 2]
        CR_logsig = params_raw[..., 3]

        # Batched pbest: topk per population
        k_pbest = max(2, int(self.p_best * N))
        # Z-score so fit_logits is rank-preserving regardless of fitness mean offset.
        # Previous `-fitness/fit_std` saturated to ±20 when fitness had large
        # absolute magnitude (CEC2017 f_optimal > 0), losing within-elite rank
        # signal at L178 (attn_logits += fit_logits.gather(...)). Same bug as
        # fixed in BatchedDiffAttDE.compute_params. Clamp tightened to ±5.
        fit_mean = fitness.mean(dim=-1, keepdim=True)
        fit_std = fitness.std(dim=-1, keepdim=True).clamp(min=1e-8)
        fit_logits = (-(fitness - fit_mean) / fit_std).clamp(min=-5, max=5)

        _, topk_idx = torch.topk(fit_logits, k_pbest, dim=-1)
        h_topk = torch.gather(h_out, 1, topk_idx.unsqueeze(-1).expand(-1, -1, hd))

        scale = math.sqrt(hd)
        attn_logits = torch.bmm(h_out, h_topk.transpose(1, 2)) / scale
        attn_logits = attn_logits + fit_logits.gather(1, topk_idx).unsqueeze(1)

        if route_probs is not None:
            rp = route_probs[..., self.head_idx].clamp(min=1e-8)
            log_base = math.log(1.0 / N)
            bias = self.subpop_scale * (torch.log(rp) - log_base)
            topk_bias = bias.gather(1, topk_idx)
            attn_logits = attn_logits + topk_bias.unsqueeze(1)

        topk_coords = torch.gather(coords, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        gumbel_w = F.gumbel_softmax(attn_logits, tau=1.0, hard=True)
        x_pbest = torch.bmm(gumbel_w.to(coords.dtype), topk_coords)

        # Random pair differential
        r1 = torch.randint(N, (B, N), device=coords.device)
        r2 = torch.randint(N - 1, (B, N), device=coords.device)
        r2 = r2 + (r2 >= r1).long()
        x_r1 = coords.gather(1, r1.unsqueeze(-1).expand(-1, -1, D))
        x_r2 = coords.gather(1, r2.unsqueeze(-1).expand(-1, -1, D))
        x_diff = x_r1 - x_r2

        return {
            'F_mu': F_mu, 'F_logsig': F_logsig,
            'CR_mu': CR_mu, 'CR_logsig': CR_logsig,
            'x_pbest': x_pbest, 'x_diff': x_diff,
        }

    def sample_batch(self, params_dict, coords, bounds_span, M):
        return _de_sample_batch(params_dict, coords, M, self.crossover_temp,
                                memory_gate=self.memory_gate,
                                use_ste=getattr(self, 'use_ste', True))


class BatchedDiffAttDE(nn.Module):
    """Stateless DE/current-to-pbest/1 — donor selection lives in the backbone.

    Post-refactor contract:
        - Donor logits (pbest/r1/r2) come from `backbone.donor_selector`
          (DonorSelectionGATv2) via the `donor_logits` kwarg.
        - The operator only does: gumbel-softmax sampling, r1/r2 exclusion,
          gather coords, and F/CR sampling via `AdaptiveFCRBeta`.
        - NO per-op projection (_make_proj), NO internal Q/K, NO alpha
          scalars, NO global_cond. The backbone operates on the full
          `gatv2_hidden` embedding without a 128→16 funnel.
        - `tau` is kept (Gumbel temperature) with init=2.0 for softer
          selection at warm-start, when the backbone's donor_selector has
          just been initialized from scratch.
    """

    def __init__(self, embed_dim=128, head_idx=0, crossover_temp=10.0, p_best=0.15,
                 backbone_dim=0, fcr_mode='beta', **_ignored):
        super().__init__()
        self.embed_dim = embed_dim
        self.head_idx = head_idx
        self.crossover_temp = crossover_temp
        self.backbone_dim = backbone_dim if backbone_dim else embed_dim
        # F/CR sampling mode (E13). 'beta' = original AdaptiveFCRBeta head.
        # 'lshade' = AdaptiveFCRCauchy head + teacher LShadeMemory drives F/CR
        # (used during distillation training). 'cauchy_neural' = AdaptiveFCRCauchy
        # head drives F/CR via Cauchy(μ_F_pred, 0.1) (used at inference).
        if fcr_mode not in ('beta', 'lshade', 'cauchy_neural'):
            raise ValueError(f"fcr_mode must be beta|lshade|cauchy_neural, got {fcr_mode}")
        self.fcr_mode = fcr_mode
        # Option A flag — when True, resample donor gumbel over the M axis
        # in _de_sample_batch (uses _A_pbest/_A_r1/_A_r2 stored in params).
        self.per_m_donors = False
        # STE crossover toggle (d2bb237). True = forward hard {0,1} + soft
        # backward grad (deployed). False = pre-d2bb237 soft sigmoid forward,
        # used to evaluate pre-STE checkpoints under their training regime.
        self.use_ste = True
        # Donor selection mode. 'neural' (default) uses backbone donor_selector
        # logits as-is. 'lshade' overrides them with hand-crafted L-SHADE rules
        # (per-individual top-p_i pbest, r1 uniform over pop, r2 uniform over
        # pop ∪ archive). p_max sampled from U(2/N, lshade_pbest_max).
        self.donor_mode = 'neural'
        self.lshade_pbest_max = 0.11

        if self.fcr_mode == 'beta':
            self.adaptive_fcr = AdaptiveFCRBeta(h_dim=self.backbone_dim)
        else:  # 'lshade' or 'cauchy_neural'
            from encoder.operators.adaptive_fcr_cauchy import AdaptiveFCRCauchy
            self.adaptive_fcr = AdaptiveFCRCauchy(h_dim=self.backbone_dim)

        # Gumbel temperature — init 2.0 for softer selection while the
        # backbone's donor_selector converges from scratch.
        self.tau = nn.Parameter(torch.tensor(2.0))
        # Parameterization of the effective Gumbel τ. 'softplus' (default)
        # uses τ_eff = 0.1 + softplus(self.tau).clamp(max=5.0) so ∂L/∂tau
        # never vanishes at the lower bound. 'clamp' is the legacy behavior
        # τ_eff = self.tau.clamp(0.1, 5.0) — gradient is zero outside [0.1,
        # 5.0]. The legacy mode lives on for the control arm of the
        # 2026-04-28 falsification experiment (docs/falsification_2026_04_28.md);
        # in production use 'softplus' to keep the F/CR head trainable.
        self.tau_mode = 'softplus'

        # proj / proj_norm set to None so NeuralK4Variant.head_projs filters
        # this head out of legacy per-head projection collections.
        self.proj = None
        self.proj_norm = None

    def effective_tau(self):
        """Effective Gumbel temperature, with gradient-preserving lower bound.

        Under `tau_mode='softplus'` (default) this is `0.1 + softplus(tau)`,
        capped at 5.0. The softplus floor keeps the gradient w.r.t. `self.tau`
        non-zero everywhere, so the F/CR head can recover from a
        below-floor `self.tau` value during training (the bug observed in
        E13: `self.tau` drifted to ≈0.05 between step 16k–17k and the
        legacy clamp froze the gradient there, blocking F/CR learning).

        Under `tau_mode='clamp'` this is the legacy `tau.clamp(0.1, 5.0)`,
        used by the control arm of the 2026-04-28 falsification experiment.
        """
        if self.tau_mode == 'softplus':
            return (0.1 + F.softplus(self.tau)).clamp(max=5.0)
        elif self.tau_mode == 'clamp':
            return self.tau.clamp(0.1, 5.0)
        else:
            raise ValueError(f"tau_mode must be 'softplus' or 'clamp', got {self.tau_mode!r}")

    def get_embedding(self, h_backbone):
        # Slice when embed_dim < backbone_dim so mixed K>1 variants can stack
        # heads of different embed dims. The sliced h_out is inert here — the
        # operator's substantive compute uses `h_backbone` (full dim) directly.
        if self.embed_dim < h_backbone.shape[-1]:
            return h_backbone[..., :self.embed_dim]
        return h_backbone

    def compute_params(self, h_out, coords, fitness, adj=None,
                       route_probs=None, bounds_span=200.0,
                       knn_idx=None, h_backbone=None, donor_logits=None,
                       donor_coords=None, donor_cand_idx=None, **kwargs):
        """
        Args:
            h_out:        (B, N, backbone_dim) backbone embedding (no proj).
            coords:       (B, N, D) population positions (active parents).
            fitness:      (B, N) fitness values (unused at selection time;
                          the backbone already encoded fitness into donor_logits).
            donor_logits: (B, N, N_pool, 3) when donor_cand_idx is None
                          (legacy all-to-all); (B, N, k_donor, 3) when
                          donor_cand_idx is provided (D1000 kNN-restricted).
                          Roles [pbest, r1, r2]. REQUIRED kwarg.
            h_backbone:   (B, N, backbone_dim) — same as h_out now; kept for
                          API compatibility with other heads.
            donor_coords: Optional (B, N_pool, D) augmented donor pool.
            donor_cand_idx: Optional (B, N, k_donor) long. Per-parent global
                          donor indices into pool_coords. When provided, the
                          gather-based path activates (strict O(N*k_donor)
                          memory; D1000 lema). When None, the legacy
                          (B, N, N_pool) bmm path runs unchanged.

        Returns:
            params dict consumed by `_de_sample_batch`, containing:
              x_pbest, x_diff, _A_pbest, _A_r1, _A_r2, _f_alpha/_f_beta,
              _cr_alpha/_cr_beta, _F_mean, _CR_mean, _diff_vector,
              _tau, _per_m_donors, _donor_coords, _donor_cand_idx.
        """
        if donor_logits is None:
            raise KeyError(
                "BatchedDiffAttDE.compute_params requires `donor_logits` "
                "kwarg produced by backbone.donor_selector.")
        B, N, D = coords.shape
        if N < 3:
            raise ValueError(f"BatchedDiffAttDE requires N >= 3 (got N={N}): "
                             f"r1/r2 exclusion needs at least 3 individuals")
        device = coords.device

        # Donor pool: archive-augmented when donor_coords supplied.
        # N_pool == N in the legacy path; N+K when archive is integrated.
        pool_coords = donor_coords if donor_coords is not None else coords
        N_pool = pool_coords.shape[1]
        # D1000 line: kNN-restricted donor cand_idx (B, N, k_donor) maps
        # each per-parent local logit slot to its global pool index. When
        # set, the L-SHADE pool override and donor_oracle CE paths are
        # bypassed (their dense (B,N,N_pool) shape would defeat the lema).
        knn_donor = donor_cand_idx is not None
        k_donor = donor_cand_idx.shape[-1] if knn_donor else None

        # ── F/CR head: branch by fcr_mode ──
        h_global = kwargs.get('h_global')
        h_fcr = h_backbone if h_backbone is not None else h_out
        if h_global is None:
            h_global = h_fcr.mean(dim=1)
        sigma_F_pred = None
        if self.fcr_mode == 'beta':
            f_alpha, f_beta, cr_alpha, cr_beta = self.adaptive_fcr(h_fcr, h_global)
            # Diagnostic means for supervised loss
            f_mean_raw = f_alpha / (f_alpha + f_beta)
            cr_mean_raw = cr_alpha / (cr_alpha + cr_beta)
            f_mean = 0.1 + 0.8 * f_mean_raw
            cr_mean = 0.1 + 0.8 * cr_mean_raw
        else:  # 'lshade' or 'cauchy_neural'
            adaptive_out = self.adaptive_fcr(h_fcr, h_global)
            if len(adaptive_out) == 3:
                mu_F, mu_CR, sigma_F_pred = adaptive_out
            else:
                mu_F, mu_CR = adaptive_out
            # No Beta keys; downstream loss reads `_mu_F_pred`/`_mu_CR_pred`
            # (and `_sigma_F_pred` when the head learns sigma).
            f_alpha = f_beta = cr_alpha = cr_beta = None
            f_mean = mu_F
            cr_mean = mu_CR

        # ── Donor selection: unbind logits into the 3 roles ──
        # Two shapes accepted:
        #   (B, N, N_pool, 3) — legacy all-to-all (donor_cand_idx is None)
        #   (B, N, k_donor, 3) — D1000 kNN-restricted (donor_cand_idx given)
        if knn_donor:
            assert donor_logits.shape == (B, N, k_donor, 3), (
                f"donor_logits shape mismatch under knn donor: got "
                f"{tuple(donor_logits.shape)}, expected ({B}, {N}, "
                f"{k_donor}, 3)")
        else:
            assert donor_logits.shape == (B, N, N_pool, 3), (
                f"donor_logits shape mismatch: got {tuple(donor_logits.shape)}, "
                f"expected ({B}, {N}, {N_pool}, 3)")
        A_pbest, A_r1, A_r2 = donor_logits.unbind(-1)
        # Snapshot the neural logits BEFORE any donor_mode override. These
        # are what compute_donor_oracle_loss supervises (off-policy CE).
        # Keep a separate variable so the per-M sampler in _de_sample_batch
        # can still read the (possibly overridden) sampling logits.
        A_pbest_neural = A_pbest
        A_r1_neural = A_r1
        A_r2_neural = A_r2

        # ── L-SHADE pool override (donor_mode in {'lshade', 'lshade_masked'}) ──
        # Both modes use the SAME pool definition:
        #   A_pbest: per-individual top-p_i pool (p_i ~ U(2/N, lshade_pbest_max));
        #            archive slots forbidden.
        #   A_r1:    active pop, ≠ self; archive forbidden.
        #   A_r2:    active pop ∪ valid archive, ≠ self.
        # The two modes differ ONLY in what fills the allowed slots:
        #   'lshade'        → flat 0 logits → Gumbel-softmax samples uniformly
        #                     within pool (E13 distillation teacher).
        #   'lshade_masked' → preserves the neural logits within the pool
        #                     (E12 inductive bias: neural picks within pool).
        _donor_mode = getattr(self, 'donor_mode', 'neural')
        if knn_donor and _donor_mode != 'neural':
            raise NotImplementedError(
                f"donor_mode={_donor_mode!r} is not implemented for "
                f"kNN-restricted donor (donor_cand_idx given). The L-SHADE "
                f"pool override needs (B, N, N_pool) shapes; switching it on "
                f"would defeat the D1000 lema. Use --donor-mode neural with "
                f"--donor-kind knn.")
        if _donor_mode in ('lshade', 'lshade_masked', 'lshade_uniform_pbest'):
            arc_mask = None
            if N_pool > N:
                # Derive archive-slot validity from the original donor_logits.
                # DonorSelectionGATv2.forward_asym applies cand_mask by setting
                # ALL 3 role channels to -1e9 at invalid (un-filled) archive
                # slots; live archive slots have finite logits. We detect
                # validity per-slot by checking whether ANY role channel is
                # finite at that slot. Take .any() over the parent axis (all
                # parents see the same archive validity).
                arc_logits = donor_logits[:, :, N:, :]  # (B, N, K, 3)
                arc_mask = (arc_logits > -1e8).any(dim=-1).any(dim=1)  # (B, K)
                # Optional override: caller may inject explicit donor_mask via kwargs.
                arc_mask_kw = kwargs.get('donor_mask')
                if arc_mask_kw is not None:
                    arc_mask = arc_mask_kw[:, N:]
            # Per-individual p_i ~ U(2/N, p_max). Each individual samples its own k_i.
            p_min = 2.0 / N
            p_max = float(getattr(self, 'lshade_pbest_max', 0.11))
            p_max_eff = max(p_max, p_min)  # guard for tiny N
            p_i = torch.empty(B, N, device=device).uniform_(p_min, p_max_eff)
            k_i = (p_i * N).round().clamp(min=2).to(torch.long)  # (B, N)
            # Ranks in fitness (ascending; rank 0 = best).
            ranks = fitness.argsort(dim=-1).argsort(dim=-1)  # (B, N)
            diag = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
            diag = diag.expand(B, -1, -1)
            # allowed[b, i, j] = True iff rank[b, j] < k_i[b, i]  AND  j != i.
            # Excluding self matches the neural A_pbest's diagonal mask
            # (-1e9 from donor_selector cand_mask). Without the diagonal
            # exclusion, the L-SHADE Gumbel sample can land on j == i; the
            # downstream donor_oracle CE then targets a position the neural
            # masks at -1e9 → loss blows up to ~1e7. Tanabe 2014 admits self
            # in current-to-pbest (degenerate update x_pbest - x_i = 0); we
            # exclude it here to keep the supervised target in the support
            # of the neural attention.
            allowed_pbest = (ranks.unsqueeze(1) < k_i.unsqueeze(-1)) & ~diag
            # Exp 2 — uniform-pbest ablation: pbest sampled UNIFORMLY over all
            # N (excluding self), no rank-based top-p% constraint. Tests
            # whether the rank prior is the load-bearing piece of L-SHADE's
            # atomic donor rule, vs "any non-broken rule recovers."
            if _donor_mode == 'lshade_uniform_pbest':
                allowed_pbest = ~diag  # only exclude self; no rank filter
            neg_inf = torch.tensor(-1e9, device=device, dtype=A_pbest.dtype)

            if _donor_mode in ('lshade', 'lshade_uniform_pbest'):
                # Flat logits within pool → uniform sampling.
                zero = torch.tensor(0.0, device=device, dtype=A_pbest.dtype)
                A_pbest_new = torch.full((B, N, N_pool), -1e9, device=device,
                                         dtype=A_pbest.dtype)
                A_pbest_new[..., :N] = torch.where(allowed_pbest, zero, neg_inf)
                A_pbest = A_pbest_new

                A_r1_new = torch.full((B, N, N_pool), -1e9, device=device,
                                      dtype=A_r1.dtype)
                A_r1_new[..., :N] = torch.where(diag, neg_inf, zero)
                A_r1 = A_r1_new

                A_r2_new = torch.full((B, N, N_pool), -1e9, device=device,
                                      dtype=A_r2.dtype)
                A_r2_new[..., :N] = torch.where(diag, neg_inf, zero)
                if N_pool > N:
                    arc_allow = arc_mask.unsqueeze(1).expand(-1, N, -1)
                    A_r2_new[..., N:] = torch.where(arc_allow, zero, neg_inf)
                A_r2 = A_r2_new
            else:  # 'lshade_masked'
                # Mask neural logits to the L-SHADE pool, preserving values inside.
                # Same allocation shape as the 'lshade' branch above (full +
                # slice-assign + where) — no bool intermediates on the hot path.
                A_pbest_new = A_pbest.new_full((B, N, N_pool), -1e9)
                A_pbest_new[..., :N] = torch.where(allowed_pbest,
                                                   A_pbest[..., :N], neg_inf)
                A_pbest = A_pbest_new

                A_r1_new = A_r1.new_full((B, N, N_pool), -1e9)
                A_r1_new[..., :N] = torch.where(diag, neg_inf, A_r1[..., :N])
                A_r1 = A_r1_new

                A_r2_new = A_r2.new_full((B, N, N_pool), -1e9)
                A_r2_new[..., :N] = torch.where(diag, neg_inf, A_r2[..., :N])
                if N_pool > N and arc_mask is not None:
                    arc_allow = arc_mask.unsqueeze(1).expand(-1, N, -1)
                    A_r2_new[..., N:] = torch.where(arc_allow,
                                                    A_r2[..., N:], neg_inf)
                A_r2 = A_r2_new

        tau = self.effective_tau()

        cand_coords = None
        if knn_donor:
            # Gather donor coords per parent ONCE (B, N, k_donor, D); cached
            # in params_dict so the per-M sampler in _de_sample_batch reuses
            # it instead of re-gathering. einsum matches the legacy bmm form
            # ('mbnk,bkd->mbnd' over a dense N_pool axis).
            cand_flat = donor_cand_idx.reshape(B, N * k_donor)
            cand_coords = pool_coords.gather(
                1, cand_flat.unsqueeze(-1).expand(B, N * k_donor, D)
            ).reshape(B, N, k_donor, D).to(pool_coords.dtype)

            w_pbest = F.gumbel_softmax(A_pbest, tau=tau, hard=True)
            x_pbest = torch.einsum('bnk,bnkd->bnd',
                                   w_pbest.to(pool_coords.dtype), cand_coords)
            w_r1 = F.gumbel_softmax(A_r1, tau=tau, hard=True)
            x_r1 = torch.einsum('bnk,bnkd->bnd',
                                w_r1.to(pool_coords.dtype), cand_coords)
            r1_idx = w_r1.argmax(dim=-1)
            r1_mask = F.one_hot(r1_idx, k_donor).bool()
            A_r2_masked = A_r2.masked_fill(r1_mask, -1e9)
            w_r2 = F.gumbel_softmax(A_r2_masked, tau=tau, hard=True)
            x_r2 = torch.einsum('bnk,bnkd->bnd',
                                w_r2.to(pool_coords.dtype), cand_coords)
        else:
            # Legacy all-to-all path — bit-exact preserved.
            w_pbest = F.gumbel_softmax(A_pbest, tau=tau, hard=True)
            x_pbest = torch.bmm(w_pbest.to(pool_coords.dtype), pool_coords)

            w_r1 = F.gumbel_softmax(A_r1, tau=tau, hard=True)
            x_r1 = torch.bmm(w_r1.to(pool_coords.dtype), pool_coords)

            r1_idx = w_r1.argmax(dim=-1)
            r1_mask = F.one_hot(r1_idx, N_pool).bool()
            A_r2_masked = A_r2.masked_fill(r1_mask, -1e9)
            w_r2 = F.gumbel_softmax(A_r2_masked, tau=tau, hard=True)
            x_r2 = torch.bmm(w_r2.to(pool_coords.dtype), pool_coords)

        x_diff = x_r1 - x_r2
        diff_vector = x_pbest - coords + x_diff

        out = {
            '_F_mean': f_mean, '_CR_mean': cr_mean,
            'x_pbest': x_pbest, 'x_diff': x_diff,
            '_diff_vector': diff_vector,
            # Two views of the per-role logits:
            #   _A_pbest / _A_r1 / _A_r2: SAMPLING logits (with donor_mode
            #     override applied). Constants under 'lshade', masked-neural
            #     under 'lshade_masked', neural under 'neural'. Consumed by
            #     _de_sample_batch's per-M Gumbel-ST resampling. Existing
            #     semantics — preserved for backward compatibility.
            #   _A_pbest_neural / _A_r1_neural / _A_r2_neural: NEURAL logits
            #     (pre-override). Always have grad in lshade/lshade_masked.
            #     Consumed by compute_donor_oracle_loss as the supervised
            #     target. Pre-2026-04-27 there was only the sampling view,
            #     leaving donor_oracle CE without grad under 'lshade'.
            '_A_pbest': A_pbest, '_A_r1': A_r1, '_A_r2': A_r2_masked,
            '_A_pbest_neural': A_pbest_neural,
            '_A_r1_neural': A_r1_neural,
            '_A_r2_neural': A_r2_neural,
            '_tau': tau,  # tensor or float; _de_sample_batch accepts both
            '_per_m_donors': bool(self.per_m_donors),
            # Action 2: forced-F injection. When set externally on the head
            # via head._force_F_attr (eval-time only), inject into params_dict
            # so _de_sample_batch overrides the Beta-sampled F.
            '_force_F': getattr(self, '_force_F_attr', None),
            '_force_CR': getattr(self, '_force_CR_attr', None),
            # fcr_shade lesions (eval-time only). Set externally on the head via
            # head._fcr_shade_static_attr / _fcr_shade_memory_attr.
            '_fcr_shade_static': bool(getattr(self, '_fcr_shade_static_attr', False)),
            '_fcr_shade_memory': getattr(self, '_fcr_shade_memory_attr', None),
            # Gather-bypass probe (falsif follow-up, eval-time only). Set
            # externally via head._gather_bypass_mode_attr / _gather_bypass_eps_attr.
            # _de_sample_batch overrides x_pbest_m post-Gumbel.
            '_gather_bypass_mode': getattr(self, '_gather_bypass_mode_attr', None),
            '_gather_bypass_eps':  float(getattr(self, '_gather_bypass_eps_attr', 0.0)),
            # E2 — operator-scale hook (eval-time only). Multiplies `mutation`
            # by a constant or σ_pop_now/σ_pop_init ratio. See _de_sample_batch.
            '_operator_scale_mode': getattr(self, '_operator_scale_mode_attr', None),
            '_operator_scale_value': float(getattr(self, '_operator_scale_value_attr', 1.0)),
            '_sigma_pop_init': getattr(self, '_sigma_pop_init_attr', None),
            # E3 — F clipping mode. 'clamp_005' (default) | 'reject_resample_pyade'.
            '_fcr_clip_mode': getattr(self, '_fcr_clip_mode_attr', 'clamp_005'),
            # Forward augmented pool to per_m_donors path so its einsum
            # samples from the same N_pool as the single-shot path.
            '_donor_coords': pool_coords if donor_coords is not None else None,
            # When set, the per-M sampler in _de_sample_batch uses
            # gather-based aggregation over a (B, N, k_donor) sparse axis.
            # _cand_coords is the (B, N, k_donor, D) gather computed once in
            # the single-shot path above; reused per-M to avoid re-gather.
            '_donor_cand_idx': donor_cand_idx,
            '_cand_coords': cand_coords,
            # E13: fcr_mode dispatch + L-SHADE teacher attached for distillation.
            '_fcr_mode': self.fcr_mode,
            '_lshade_teacher': getattr(self, '_lshade_teacher', None),
        }
        if self.fcr_mode == 'beta':
            out['_f_alpha'] = f_alpha
            out['_f_beta'] = f_beta
            out['_cr_alpha'] = cr_alpha
            out['_cr_beta'] = cr_beta
        else:  # 'lshade' or 'cauchy_neural'
            out['_mu_F_pred'] = f_mean   # (B, N) with grad
            out['_mu_CR_pred'] = cr_mean
            if sigma_F_pred is not None:
                out['_sigma_F_pred'] = sigma_F_pred  # (B, N) with grad
        return out

    def sample_batch(self, params_dict, coords, bounds_span, M):
        return _de_sample_batch(params_dict, coords, M, self.crossover_temp,
                                use_ste=self.use_ste)


# Backward compatibility alias
BatchedDiffLSHADE = BatchedDiffDE
