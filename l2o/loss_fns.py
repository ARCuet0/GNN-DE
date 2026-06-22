"""Loss functions and benchmark dispatchers for L2O training."""
import logging
import math

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


def _fes_reduce(per_elem, fes_w):
    """FES-weighted mean: mean_B(sum_t(per_elem * fes_w) / sum_t(fes_w)).

    Generations where more individuals are active contribute proportionally
    more to the loss. When active_frac is constant across gens (typical
    under F_MIN=0.1), equivalent to plain mean.
    """
    w_sum = fes_w.sum(dim=0)  # (B,)
    if w_sum.min() < 1.0:
        log.warning("FES w_sum < 1.0 — all-inactive chunk?")
    w_sum = w_sum.clamp(min=1e-8)
    return ((per_elem * fes_w).sum(dim=0) / w_sum).mean()


def compute_hitting_loss(chunk_best, f_optimal, log_tgt, args,
                         gap_init=None, chunk_fes=None):
    """Dispatch to the configured hitting loss variant.

    When chunk_fes is provided, weight each generation's contribution
    by its FES cost (per-batch, detached).
    """
    best_stack = torch.stack(chunk_best)  # (T, B)
    gap = (best_stack - f_optimal).clamp(min=1e-30)

    _use_fes = chunk_fes is not None
    if _use_fes:
        fes_w = torch.stack(chunk_fes).detach()
    _reduce = (lambda x: _fes_reduce(x, fes_w)) if _use_fes else torch.Tensor.mean

    if args.loss == 'log1p_linear':
        from encoder.grad_stabilizers import log1p_linear
        log_gap = torch.log1p(gap)
        sf = (log_gap - log_tgt).clamp(min=0)
        hitting = _reduce(log1p_linear(sf, knee=args.loss_knee))

        imp_w = getattr(args, 'improvement_weight', 0.0)
        if imp_w > 0 and gap_init is not None and gap_init > 1e-6:
            log_gap_init = math.log1p(gap_init)
            improvement = (log_gap_init - log_gap).clamp(min=0) / log_gap_init
            hitting = hitting - imp_w * _reduce(improvement)

        return hitting

    elif args.loss == 'log_gap':
        from encoder.unified_loss import log_gap_loss
        return _reduce(log_gap_loss(best_stack, f_optimal, reduce='none'))

    elif args.loss == 'adaptive_log1p':
        from encoder.unified_loss import adaptive_log1p_loss
        tau = gap.mean().detach().item()
        per_elem, _ = adaptive_log1p_loss(
            best_stack, f_optimal, tau_ema=tau, reduce='none')
        return _reduce(per_elem)

    else:
        raise ValueError(f"Unknown loss: {args.loss}")


class _RankTransformSTE(torch.autograd.Function):
    """Rank-transform forward, identity backward (straight-through).

    Forward: replace x[i] with its normalized rank in [-1, 1] based on
    sort order over the FLATTENED tensor. Heavy-tail killer: extreme
    outliers receive only the top-rank score (=+1), not their raw scale.

    Backward: gradient is passed through unchanged (STE). The downstream
    optimizer sees a uniform gradient direction "increase imp here" but
    the loss VALUE is scale-invariant per fid — so when summed across
    many training steps with different fids, no single fid dominates by
    absolute magnitude.
    """

    @staticmethod
    def forward(ctx, x):
        flat = x.detach().flatten()
        n = flat.numel()
        if n <= 1:
            return torch.zeros_like(x)
        ranks = flat.argsort(stable=True).argsort(stable=True).to(x.dtype)
        normalized = (ranks / (n - 1)) * 2.0 - 1.0  # [-1, 1]
        return normalized.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def compute_cf_improvement_loss(parent_fit, off_fit, weight=1.0,
                                normalize='rank_ste'):
    """Per-slot counterfactual-improvement loss (Arm D, ~2026-05-01).

    Replaces hitting loss as the primary BPTT signal under the
    discussion-derived plan. Bloque A+B (22 fids) measured cf_improvement
    raw at cos≈+0.40 inter-fn vs hitting at +0.14 — 3× more coherent
    signal to the backbone. Pairwise (cos≈+0.74) is the upper bound;
    cf_improvement is the unprocessed precursor pairwise distills via
    `(imp_diff > 0)` binarization.

    Args:
        parent_fit: (B, N) tensor — fitness BEFORE the generation step.
        off_fit:    (B, N) tensor — fitness AFTER the generation step
                    (post greedy-keep). Differentiable side that carries
                    gradient back to the operator + backbone.
        weight: scalar loss weight.
        normalize: heavy-tail control. F05/F09 produce 1.9M× / 649223×
            ratios on raw improvements (Phase 0) — without normalization
            a single fid monopolizes optimizer updates.
            * 'rank_ste': forward = rank-transform to [-1, 1], backward
                          = identity (STE). Loss VALUE invariant to per-
                          fid scale; gradient direction = "raise imp here".
            * 'sigma_batch': divide imp by within-batch detached std.
                             Gradient magnitude = 1/(N·σ) per slot —
                             smaller on high-σ fids (F05). Differentiable
                             through both forward and backward, no STE.

    Returns:
        Scalar loss. Sign convention: improvement (parent_fit > off_fit)
        produces a NEGATIVE loss; lower loss is better.
    """
    imp = parent_fit - off_fit  # (B, N), positive = improved
    if normalize == 'rank_ste':
        normalized = _RankTransformSTE.apply(imp)
    elif normalize == 'sigma_batch':
        sigma = imp.detach().std().clamp(min=1e-6)
        normalized = imp / sigma
    else:
        raise ValueError(
            f"Unknown normalize='{normalize}'. "
            f"Expected 'rank_ste' or 'sigma_batch'.")
    return -weight * normalized.mean()


def make_eval_fn(fid, D, device, benchmark, aug_cache=None):
    """Create evaluation function for the given benchmark."""
    from encoder.cec2017_torch import CEC2017Torch

    if benchmark == 'cec2017':
        return CEC2017Torch(fid, D, device)

    elif benchmark == 'augmented':
        from encoder.augmented_cec2017 import AugmentedCEC2017
        if aug_cache is None:
            aug_cache = AugmentedCEC2017(device=device, dims=(D,))
        return aug_cache.sample(fid=fid, D=D)


def compute_geo_losses(extras, old_coords, fitness, fn, args, device):
    """Compute geometric auxiliary losses. Returns list of loss tensors."""
    if args.geo_weight <= 0:
        return []

    from encoder.grad_oracle import compute_grad_f, compute_alignment_target

    B, N, D = old_coords.shape
    diag = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
    geo_losses = []

    gf = None
    if args.lupi_grad:
        gf = compute_grad_f(old_coords, fn)
        grad_target = compute_alignment_target(
            old_coords.detach().float(), gf.float(), diag, tau=0.5)
    else:
        grad_target = None

    # Precompute coord distances once (shared by DE and CMA-ES branches)
    _need_cdist = (grad_target is None and
                   (extras.get('_A_pbest') is not None or
                    extras.get('attn_logits_cmaes') is not None))
    if _need_cdist:
        cd_f = old_coords.detach().float()
        coord_dist = torch.cdist(cd_f, cd_f)
    else:
        coord_dist = None

    # DE A_pbest attention
    A_pbest = extras.get('_A_pbest')
    if A_pbest is not None:
        if grad_target is not None:
            tgt = grad_target
        else:
            tau = coord_dist.median(dim=-1, keepdim=True).values.clamp(min=1e-8)
            cd_m = coord_dist.masked_fill(diag, float('inf'))
            tgt = torch.softmax(-cd_m / tau, dim=-1)
        logA = A_pbest.log_softmax(dim=-1)
        log_tgt = tgt.clamp(min=1e-8).log()
        g1 = (tgt * (log_tgt - logA)).sum(dim=-1).mean()
        if torch.isfinite(g1):
            geo_losses.append(args.geo_weight * 0.1 * g1)

    # CoordLS dim_bias
    db_geo = extras.get('dim_bias_coordls')
    if db_geo is not None:
        if args.lupi_grad and gf is not None:
            st = gf.abs()
        else:
            st = extras.get('sensitivity_target_coordls')
        if st is not None:
            tau_s = st.mean(dim=-1, keepdim=True).clamp(min=1e-8) * 2.0
            td = torch.softmax(st / tau_s, dim=-1)
            pl = db_geo.clamp(min=-50, max=50).log_softmax(dim=-1)
            log_td = td.clamp(min=1e-8).log()
            lc = (td * (log_td - pl)).sum(dim=-1).mean()
            if torch.isfinite(lc):
                geo_losses.append(args.geo_weight * 0.2 * lc)

    # CMA-ES attn_logits
    al = extras.get('attn_logits_cmaes')
    if al is not None:
        if grad_target is not None:
            comb = grad_target
        else:
            tau_c = coord_dist.median(dim=-1, keepdim=True).values.clamp(min=1e-8)
            cdm = coord_dist.masked_fill(diag, float('inf'))
            prox = torch.softmax(-cdm / tau_c, dim=-1)
            ff = fitness.detach().float()
            fs = ff.std(dim=-1, keepdim=True).clamp(min=1e-8)
            fa = torch.softmax((-ff / fs).unsqueeze(1).expand_as(prox), dim=-1)
            comb = (prox * fa).masked_fill(diag, 0.0)
            comb = comb / comb.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        lp = al.clamp(min=-50, max=50).log_softmax(dim=-1)
        log_comb = comb.clamp(min=1e-8).log()
        lc2 = (comb * (log_comb - lp)).sum(dim=-1).mean()
        gc2 = 0.1 * lc2
        if torch.isfinite(gc2):
            geo_losses.append(args.geo_weight * gc2)

    return geo_losses


def compute_lupi_loss(dk_live, c_graph, x_star, contra_weight):
    """LUPI per-head distance+alignment loss (zero CEC2017 evals).

    dist_imp is normalized by the search domain diagonal (200*sqrt(D))
    so all individuals contribute on the same scale regardless of distance.

    Returns:
        loss: scalar loss (or None if not finite)
        dist_imp: (K,) mean normalized distance improvement per head (detached)
        alignment: (K,) mean alignment per head (detached)
    """
    vec_to_star = x_star - c_graph
    dist_parent = vec_to_star.norm(dim=-1)  # (B, N)
    diff_k = vec_to_star.unsqueeze(0).unsqueeze(3) - dk_live  # (M,B,N,K,D)
    dist_child = diff_k.norm(dim=-1)  # (M,B,N,K)
    # Normalize by domain diagonal: 200*sqrt(D) for [-100,100]^D
    D = dk_live.shape[-1]
    domain_scale = 200.0 * (D ** 0.5)
    dist_imp = (dist_parent.unsqueeze(0).unsqueeze(-1) - dist_child) / domain_scale
    dot = (dk_live * vec_to_star.unsqueeze(0).unsqueeze(3)).sum(dim=-1)
    dk_norm = dk_live.norm(dim=-1)
    denom = (dk_norm * dist_parent.unsqueeze(0).unsqueeze(-1)).clamp(min=1e-8)
    alignment = dot / denom
    lupi_total = -contra_weight * (dist_imp + alignment).mean()
    loss = lupi_total if torch.isfinite(lupi_total) else None
    with torch.no_grad():
        di = dist_imp.mean(dim=(0, 1, 2))  # (K,)
        al = alignment.mean(dim=(0, 1, 2))  # (K,)
    return loss, di, al


def compute_oracle_router_loss(extras, fitness):
    """Oracle CE loss for router: per-individual supervised signal.

    Uses fit_per_k (fitness under each head) to determine which head
    was best for each individual. Only includes individuals where:
    1. The best head actually improved over parent fitness
    2. The best head is distinguishable from second-best

    Returns:
        (loss_or_None, oracle_agreement_frac)
        oracle_agreement_frac: fraction of valid individuals where
        the router already picked the oracle-best head.
    """
    fit_per_k = extras.get('fit_per_k')  # (M, B, N, K) detached
    logits_live = extras.get('logits_live')  # (B, N, K) with gradient
    if fit_per_k is None or logits_live is None:
        return None, None

    # Best of M samples
    if fit_per_k.dim() == 4:
        fit_per_k = fit_per_k.mean(dim=0)  # (B, N, K)

    B, N, K = fit_per_k.shape
    parent_fit = fitness.detach().unsqueeze(-1)  # (B, N, 1)

    # Oracle: best head per individual
    oracle_best_k = fit_per_k.argmin(dim=-1)  # (B, N)
    best_k_fit = fit_per_k.gather(-1, oracle_best_k.unsqueeze(-1)).squeeze(-1)  # (B, N)

    # Filter 1: best head actually improved
    improved = best_k_fit < parent_fit.squeeze(-1)

    # Filter 2: best head is distinguishable from second-best
    topk2 = fit_per_k.topk(2, dim=-1, largest=False)
    second_best_fit = topk2.values[..., 1]
    margin = (second_best_fit - best_k_fit).abs()
    scale = parent_fit.squeeze(-1).abs().clamp(min=1.0)
    distinguishable = margin > 0.01 * scale

    valid = improved & distinguishable  # (B, N)

    if valid.sum() < 2:
        return None, None

    # CE loss on valid individuals only (gradient flows to scorer)
    valid_logits = logits_live[valid]  # (n_valid, K)
    valid_targets = oracle_best_k[valid]  # (n_valid,)
    loss = F.cross_entropy(valid_logits, valid_targets)

    # Agreement metric: how often does router already pick oracle-best?
    with torch.no_grad():
        router_choice = logits_live.detach().argmax(dim=-1)  # (B, N)
        agreement = (router_choice[valid] == valid_targets).float().mean().item()

    return loss, agreement


def _improved_distinguishable_valid(off_all, fitness, margin_frac=0.01):
    """Shared oracle-valid mask for M-axis supervised losses.

    Uses a single topk(2) pass over M to produce best-m index, best fitness
    and runner-up margin — avoiding duplicate reductions.

    Args:
        off_all: (M, B, N) per-proposal fitness (detached OK).
        fitness: (B, N) parent fitness.
        margin_frac: require margin > margin_frac * |parent_fit| (with floor 1.0).

    Returns:
        (m_star, valid)
        m_star: (B, N) long — index of best-m per individual (from topk.indices[0]).
        valid:  (B, N) bool — True where best-m improved AND is distinguishable.
    """
    topk2 = off_all.topk(2, dim=0, largest=False)
    m_star = topk2.indices[0]
    best_fit = topk2.values[0]
    improved = best_fit < fitness.detach()
    margin = (topk2.values[1] - topk2.values[0]).abs()
    scale = fitness.detach().abs().clamp(min=1.0)
    valid = improved & (margin > margin_frac * scale)
    return m_star, valid


def compute_donor_oracle_loss(extras, fitness, weight=0.0,
                              w_pbest=1.0, w_r1=1.0, w_r2=1.0,
                              r2_mode='ce', r2_soft_frac=0.3):
    """Oracular loss on donor-attention logits, supervised by best-m fitness.

    For each individual with an improved-and-distinguishable best-m among the
    M proposals, push A_pbest/A_r1/A_r2 to assign high probability to the
    pbest/r1/r2 donor indices that m_star effectively used.

    Requires per_m_donors=True on the DE head (otherwise all m proposals share
    the same donor triple, making the oracle targets trivial). Uses only the
    M fitness values already paid in FES — no extra evaluations.

    The trajectory is NOT modified; this is pure side-supervision.

    Args:
        weight: global scalar multiplier (0 → diag-only, no loss contribution).
        w_pbest, w_r1, w_r2: per-component linear weights on the three CE/
            soft losses. All-zero → loss=None. The weighted mean keeps the
            overall magnitude invariant to rebalancing.
        r2_mode: supervision mode for A_r2.
            'ce'   — CE against tgt_r2 = r2_idx_m[m*] (original behavior).
            'soft' — CE against a uniform distribution over the bottom-K
                     (by fitness) parents, matching r2's bad-region-donor
                     inductive bias. K = max(2, int(r2_soft_frac*N)).
            'off'  — loss_r2 ≡ 0; equivalent to w_r2=0 but self-documenting.
        r2_soft_frac: fraction of population in the soft target's bottom-K.

    Expected keys in ``extras``:
        _A_pbest, _A_r1, _A_r2 : (B, N, N) live logits with gradient.
            _A_r2 here has only the diagonal masked; the per-m r1 exclusion
            applied during actual selection is NOT reflected (approximation).
        _pbest_idx_m, _r1_idx_m, _r2_idx_m : (M, B, N) detached long tensors,
            the hard-argmax donor choices at each m.
        off_fitness_all : (M, B, N) detached fitness of each proposal.
    """
    if r2_mode not in ('ce', 'soft', 'off'):
        raise ValueError(f"r2_mode must be ce|soft|off, got {r2_mode!r}")

    # Prefer the NEURAL logits view (pre-donor_mode-override) so the CE
    # supervises donor_selector with grad. Fall back to the sampling view
    # for backward compat: under donor_mode='neural' (or pre-2026-04-27
    # checkpoints) the two views are identical. Under 'lshade' the sampling
    # view holds detached constants — using it produces grad-less loss.
    A_pbest = extras.get('_A_pbest_neural', extras.get('_A_pbest'))
    pbest_idx_m = extras.get('_pbest_idx_m')
    off_all = extras.get('off_fitness_all')
    if A_pbest is None or pbest_idx_m is None or off_all is None:
        return None, {}

    A_r1 = extras.get('_A_r1_neural', extras.get('_A_r1'))
    A_r2 = extras.get('_A_r2_neural', extras.get('_A_r2'))
    r1_idx_m = extras.get('_r1_idx_m')
    r2_idx_m = extras.get('_r2_idx_m')
    if A_r1 is None or A_r2 is None or r1_idx_m is None or r2_idx_m is None:
        return None, {}

    # Shape guard: off_fitness_all must be (M, B, N) matching pbest_idx_m.
    # Under --gate-type surrogate, opt_variant flattens proposals into
    # (1, B, N_prop) — incompatible semantics for per-m oracle supervision.
    if off_all.shape != pbest_idx_m.shape or off_all.shape[0] < 2:
        return None, {}

    m_star, valid = _improved_distinguishable_valid(off_all, fitness)
    if valid.sum() < 2:
        return None, {}

    B, N = valid.shape
    # Restrict A_* to the active candidate slice [:N] when the graph-native
    # archive widens donor logits to (B, N, N+K). pbest is constrained to the
    # active pop (mask in donor_selector), so its targets are always in
    # [0, N). For r1/r2 the slicing matches the per-m oracle semantics: the
    # active pool is the supervisable region. Archive-targeted r1/r2 samples
    # (target >= N) are excluded from the loss via valid_r1/valid_r2 below.
    if A_pbest.shape[-1] != N:
        A_pbest = A_pbest[:, :, :N]
        A_r1 = A_r1[:, :, :N]
        A_r2 = A_r2[:, :, :N]
    device = valid.device
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    n_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    tgt_pbest = pbest_idx_m[m_star, b_idx, n_idx]          # (B, N)
    tgt_r1 = r1_idx_m[m_star, b_idx, n_idx]
    tgt_r2 = r2_idx_m[m_star, b_idx, n_idx]

    loss_pbest = F.cross_entropy(A_pbest[valid], tgt_pbest[valid])
    valid_r1 = valid & (tgt_r1 < N)
    if valid_r1.sum() >= 2:
        loss_r1 = F.cross_entropy(A_r1[valid_r1], tgt_r1[valid_r1])
    else:
        loss_r1 = torch.zeros((), device=device, dtype=A_r1.dtype)

    if r2_mode == 'ce':
        valid_r2 = valid & (tgt_r2 < N)
        if valid_r2.sum() >= 2:
            loss_r2 = F.cross_entropy(A_r2[valid_r2], tgt_r2[valid_r2])
        else:
            loss_r2 = torch.zeros((), device=device, dtype=A_r2.dtype)
    elif r2_mode == 'off':
        # Override w_r2 so the component fully drops out of both numerator
        # AND total_w — equivalent to explicitly passing w_r2=0.0 with 'ce'.
        loss_r2 = torch.zeros((), device=device, dtype=A_r2.dtype)
        w_r2 = 0.0
    else:  # 'soft'
        # Bottom-K by fitness = worst individuals (largest fitness in minimization).
        # Soft target puts uniform mass on those K parents, EXCLUDING:
        #   (a) self (diagonal — A_r2 has diag at -1e9)
        #   (b) the per-row r1 position — A_r2 also masks r1 to -1e9 inside
        #       compute_params; if r1 happens to be in the bottom-K and we
        #       leave its mass non-zero, log_softmax(-1e9) blows the loss up
        #       to ~1e7 (gradient is still 0 at the masked entry but the
        #       diagnostic becomes useless and grad-clip sees a noisy total).
        # Generalize to any A_r2 entry the operator masked to -1e9.
        k_bad = max(2, int(r2_soft_frac * N))
        if k_bad > N:
            k_bad = N
        _, bad_idx = fitness.detach().topk(k_bad, dim=-1, largest=True)  # (B, K)
        soft_tgt = torch.zeros(B, N, N, device=device, dtype=A_r2.dtype)
        bad_bn = bad_idx.unsqueeze(1).expand(B, N, k_bad)
        soft_tgt.scatter_(-1, bad_bn, 1.0)
        diag_mask = torch.eye(N, device=device, dtype=torch.bool).unsqueeze(0)
        soft_tgt = soft_tgt.masked_fill(diag_mask, 0.0)
        soft_tgt = soft_tgt.masked_fill(A_r2 < -1e8, 0.0)
        row_sum = soft_tgt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        soft_tgt = soft_tgt / row_sum
        log_p = F.log_softmax(A_r2, dim=-1)
        loss_r2 = -(soft_tgt[valid] * log_p[valid]).sum(dim=-1).mean()

    total_w = float(w_pbest) + float(w_r1) + float(w_r2)
    if weight > 0 and total_w > 1e-8:
        loss = weight * (
            w_pbest * loss_pbest + w_r1 * loss_r1 + w_r2 * loss_r2
        ) / total_w
        if not torch.isfinite(loss):
            return None, {}
    else:
        # Diag-only mode (weight=0) or all weights zero → no loss contribution.
        loss = None

    with torch.no_grad():
        ap = A_pbest.detach().argmax(dim=-1)
        ar1 = A_r1.detach().argmax(dim=-1)
        ar2 = A_r2.detach().argmax(dim=-1)
        # Diag values kept as tensors; .item() deferred to build_step_diagnostics.
        diag = {
            'donor_agreement_pbest': (ap[valid] == tgt_pbest[valid]).float().mean(),
            'donor_agreement_r1':    (ar1[valid] == tgt_r1[valid]).float().mean(),
            'donor_agreement_r2':    (ar2[valid] == tgt_r2[valid]).float().mean(),
            'donor_valid_frac':      valid.float().mean(),
            'donor_loss_pbest':      loss_pbest.detach(),
            'donor_loss_r1':         loss_r1.detach(),
            'donor_loss_r2':         loss_r2.detach(),
        }

    return loss, diag


def compute_kl_lshade_distill_loss(extras, fitness, fes_progress, weight=0.0,
                                   p_max=0.2, p_min=0.05,
                                   w_pbest=1.0, w_r1=1.0, w_r2=1.0):
    """KL distillation: donor_selector logits -> L-SHADE atomic soft targets.

    Forward-KL: D_KL(target || softmax(A)) = -sum(target * log_softmax(A)) - H(target).
    Only the cross-entropy term has gradient w.r.t. A; H(target) is constant.

    Targets:
        pbest: uniform 1/K mass on top-K parents by fitness rank,
               K = max(2, int(p * N)),
               p = p_min + (p_max - p_min) * (1 - fes_progress).
        r1:    uniform 1/(N-1) over off-diagonal (any non-self).
        r2:    uniform 1/(N-1) over off-diagonal (matches A_r2's diag mask).

    Args:
        extras: dict with _A_pbest_neural / _A_r1_neural / _A_r2_neural (B,N,*) live logits.
                Falls back to _A_pbest / _A_r1 / _A_r2 for compatibility.
        fitness: (B, N) lower=better.
        fes_progress: float in [0, 1], cumulative_fes / step_fes for current step.
        weight: global scalar (0 -> short-circuit, returns (None, {})).
        p_max, p_min: pbest top-p% adaptive schedule (L-SHADE-canonical 0.2 -> 0.05).
        w_pbest, w_r1, w_r2: per-component weights (mean-normalized to keep magnitude
                             invariant under reweighting).

    Returns:
        (loss, diag) with `weight * total_loss`, or (None, {}) when short-circuited.
    """
    if weight == 0.0:
        return None, {}

    A_pbest = extras.get('_A_pbest_neural', extras.get('_A_pbest'))
    A_r1 = extras.get('_A_r1_neural', extras.get('_A_r1'))
    A_r2 = extras.get('_A_r2_neural', extras.get('_A_r2'))
    if A_pbest is None or A_r1 is None or A_r2 is None:
        return None, {}

    B, N = fitness.shape
    if A_pbest.shape[-1] > N:
        A_pbest = A_pbest[:, :, :N]
        A_r1 = A_r1[:, :, :N]
        A_r2 = A_r2[:, :, :N]

    device = A_pbest.device
    dtype = A_pbest.dtype

    fp = float(max(0.0, min(1.0, fes_progress)))
    p_adapt = p_min + (p_max - p_min) * (1.0 - fp)
    K = max(2, int(p_adapt * N))

    # pbest target: per-row uniform over top-K parents excluding self (the diag of
    # A_pbest is masked to -1e9, so any mass on the diagonal would inject -inf
    # cross-entropy and explode the loss). Renormalize per row to keep mass=1.
    rank = fitness.argsort(dim=1).argsort(dim=1)        # (B, N) rank within batch
    topk_mask = (rank < K).to(dtype)                     # (B, N): 1 on top-K
    diag_bool_pre = torch.eye(N, device=device, dtype=torch.bool)
    pbest_raw = topk_mask.unsqueeze(1).expand(B, N, N) \
        * (~diag_bool_pre).to(dtype).unsqueeze(0)
    row_sum = pbest_raw.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    pbest_target = pbest_raw / row_sum

    log_pbest = F.log_softmax(A_pbest, dim=-1)
    loss_pbest = -(pbest_target * log_pbest).sum(dim=-1).mean()

    # r1/r2 target: uniform 1/(N-1) over off-diagonal.
    diag_bool = torch.eye(N, device=device, dtype=torch.bool)
    off_diag = (~diag_bool).to(dtype) / float(N - 1)
    off_diag = off_diag.unsqueeze(0).expand(B, N, N)

    log_r1 = F.log_softmax(A_r1, dim=-1)
    log_r2 = F.log_softmax(A_r2, dim=-1)
    loss_r1 = -(off_diag * log_r1).sum(dim=-1).mean()
    loss_r2 = -(off_diag * log_r2).sum(dim=-1).mean()

    total_w = max(w_pbest + w_r1 + w_r2, 1e-8)
    total = (w_pbest * loss_pbest + w_r1 * loss_r1 + w_r2 * loss_r2) / total_w

    diag = {
        'kl_loss_pbest': loss_pbest.detach(),
        'kl_loss_r1': loss_r1.detach(),
        'kl_loss_r2': loss_r2.detach(),
        'kl_p_adaptive': p_adapt,
        'kl_K_topk': K,
    }
    return weight * total, diag


def compute_fcr_oracle_from_m_loss(extras, fitness, weight=0.3):
    """MSE oracular F/CR, target = (F_m*, CR_m*) realized at the best-m.

    FES-free replacement for compute_fcr_grid_loss. Instead of evaluating a
    separate grid (2*n_grid*B*N extra evaluations, not tracked in FES
    accounting), uses the realized F/CR already sampled by the DE head
    during the current generation's M proposals.

    For each individual with an improved-and-distinguishable best-m, the
    learned Beta means F_mean/CR_mean (B, N) are pushed toward the F/CR
    values that m_star used.

    Expected keys in ``extras``:
        _F_mean, _CR_mean : (B, N) live mean of Beta with gradient.
            _CR_mean may be None if CR is not learned.
        _realized_F, _realized_CR : (M, B, N) detached realizations.
        off_fitness_all : (M, B, N) detached per-proposal fitness.

    Returns:
        (loss_or_None, diag_dict)
    """
    F_mean = extras.get('_F_mean')
    CR_mean = extras.get('_CR_mean')
    F_real = extras.get('_realized_F')
    CR_real = extras.get('_realized_CR')
    off_all = extras.get('off_fitness_all')
    if F_mean is None or F_real is None or off_all is None:
        return None, {}

    # Need M >= 2 along dim 0 for the top-2 reduction that selects best-m and
    # measures its runner-up margin. Surrogate path stores off_fitness_all as
    # (1, B, N_prop) — that degenerate axis means "M absent" and the oracle
    # target is undefined here.
    if off_all.shape[0] < 2 or F_real.shape[0] != off_all.shape[0]:
        return None, {}

    m_star, valid = _improved_distinguishable_valid(off_all, fitness)
    if valid.sum() < 2:
        return None, {}

    B, N = valid.shape
    device = valid.device
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)
    n_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, N)
    tgt_F = F_real[m_star, b_idx, n_idx]                    # (B, N)

    loss_F = (F_mean[valid] - tgt_F[valid]).pow(2).mean()

    use_CR = CR_mean is not None
    if use_CR:
        tgt_CR = CR_real[m_star, b_idx, n_idx]
        loss_CR = (CR_mean[valid] - tgt_CR[valid]).pow(2).mean()
    else:
        tgt_CR = None
        loss_CR = torch.zeros((), device=device, dtype=loss_F.dtype)

    loss = weight * (loss_F + loss_CR)
    if not torch.isfinite(loss):
        return None, {}

    with torch.no_grad():
        # Key names mirror compute_fcr_grid_loss so build_step_diagnostics
        # (which expects fcr_loss_F / fcr_F_mean / fcr_F_optimal_{mean,std} /
        # fcr_loss_CR / fcr_CR_mean / fcr_CR_optimal_mean) works drop-in.
        tgt_F_valid = tgt_F[valid]
        diag = {
            'fcr_loss_F':          loss_F.detach(),
            'fcr_F_mean':          F_mean.detach().mean(),
            'fcr_F_optimal_mean':  tgt_F_valid.mean(),
            'fcr_F_optimal_std':   (tgt_F_valid.std() if tgt_F_valid.numel() >= 2
                                    else torch.zeros((), device=device, dtype=tgt_F_valid.dtype)),
            'fcr_valid_frac':      valid.float().mean().item(),
        }
        if use_CR:
            diag['fcr_loss_CR']          = loss_CR.detach()
            diag['fcr_CR_mean']          = CR_mean.detach().mean()
            diag['fcr_CR_optimal_mean']  = tgt_CR[valid].mean()

    return loss, diag


_LOG_PI = math.log(math.pi)
_SIGMA_FLOOR = 1e-3  # autocast-bf16 safe (>> 2^-14 minimum normal)


def _cauchy_nll(x, mu, sigma):
    """Negative log-likelihood of x under Cauchy(mu, sigma), elementwise.

    NLL = log(π) + log(σ) + log1p(((x-μ)/σ)²). Defense-in-depth σ floor at
    1e-3 keeps the form numerically safe under autocast/bf16 even if the
    head's own softplus floor underflows.
    """
    sigma = sigma.clamp(min=_SIGMA_FLOOR)
    z = (x - mu) / sigma
    return _LOG_PI + torch.log(sigma) + torch.log1p(z * z)


def compute_fcr_distill_loss(extras, weight=0.0, mode='mse'):
    """Online L-SHADE → GNN distillation on per-individual μ_F/μ_CR.

    Used when `--fcr-mode lshade`: the teacher (LShadeMemory) drives the
    realized F, CR per (M, B, N); the GNN's AdaptiveFCRCauchy head predicts
    per-individual μ_F_pred, μ_CR_pred (B, N) and optionally σ_F_pred (B, N).

    `mode` selects the loss form:
      - 'mse' (default, legacy E13): MSE between pred and the M-mean of the
        teacher's realized values. Converges to E[F | state] — collapses
        heavy-tail Cauchy structure to its mean (the mode-collapse fault
        diagnosed for E13 in `docs/distillation_lshade_report.md` §5.2).
      - 'cauchy_nll' (2026-04-28 falsification arm C): NLL of every
        realized F under Cauchy(μ_F_pred, σ_F_pred). Pulls μ_F toward the
        median (robust to outliers / heavy tails) and lets σ_F absorb the
        scale, so the head can reproduce M_F drift + heavy-tail. CR stays
        on MSE (Normal teacher, σ_CR fixed at 0.1 — no falsification gain).

    Args:
        extras: dict with `_realized_F`, `_realized_CR` (M,B,N detached) and
                `_mu_F_pred`, `_mu_CR_pred` (B,N with grad). Under
                `mode='cauchy_nll'`, also requires `_sigma_F_pred` (B,N).
        weight: scalar; loss is weighted internally (caller appends as-is).
        mode:   'mse' | 'cauchy_nll' — see above.

    Returns:
        (loss, diag) — loss is None if weight==0 or required keys missing
        (under cauchy_nll, missing `_sigma_F_pred` returns None).
    """
    if weight <= 0:
        return None, {}
    F_real = extras.get('_realized_F')
    CR_real = extras.get('_realized_CR')
    mu_F_pred = extras.get('_mu_F_pred')
    mu_CR_pred = extras.get('_mu_CR_pred')
    if F_real is None or CR_real is None or mu_F_pred is None or mu_CR_pred is None:
        return None, {}
    F_target = F_real.detach().to(mu_F_pred.dtype)               # (M, B, N)
    CR_target = CR_real.detach().to(mu_CR_pred.dtype).mean(dim=0)  # (B, N) — CR always MSE

    if mode == 'cauchy_nll':
        sigma_F_pred = extras.get('_sigma_F_pred')
        if sigma_F_pred is None:
            return None, {}
        # NLL of each realized F sample under Cauchy(mu_F_pred, sigma_F_pred).
        # Broadcast pred (B, N) → (M, B, N) over the M-sample axis.
        mu_e = mu_F_pred.unsqueeze(0).expand_as(F_target)
        sg_e = sigma_F_pred.unsqueeze(0).expand_as(F_target)
        loss_F = _cauchy_nll(F_target, mu_e, sg_e).mean()
    else:  # 'mse'
        F_target_m = F_target.mean(dim=0)  # (B, N)
        loss_F = (mu_F_pred - F_target_m).pow(2).mean()
    loss_CR = (mu_CR_pred - CR_target).pow(2).mean()
    loss = weight * (loss_F + loss_CR)
    # Align keys with the Beta-oracle schema so build_step_diagnostics
    # (l2o/loss_fns.py:765-779) aggregates this loss without modification.
    # The "optimal" mean/std fields here are the teacher's realized F/CR
    # statistics — analogous to the Beta-oracle's "optimal" Beta target.
    # F_target is (M, B, N) under cauchy_nll, (M, B, N) also under mse (the
    # M-mean reduction lives in F_target_m above). The "optimal" mean/std
    # are reported on the M-meaned (B, N) view so the metric semantics
    # match the pre-2026-04-28 schema (then F_target was already M-meaned).
    F_target_bn = F_target.mean(dim=0)
    diag = {
        'fcr_loss_F':           loss_F.detach(),
        'fcr_F_mean':           mu_F_pred.detach().mean(),
        'fcr_F_optimal_mean':   F_target_bn.mean(),
        'fcr_F_optimal_std':    F_target_bn.std(),
        'fcr_loss_CR':          loss_CR.detach(),
        'fcr_CR_mean':          mu_CR_pred.detach().mean(),
        'fcr_CR_optimal_mean':  CR_target.mean(),
    }
    sigma_F_pred = extras.get('_sigma_F_pred')
    if sigma_F_pred is not None:
        diag['fcr_F_sigma_mean'] = sigma_F_pred.detach().mean()
    return loss, diag


def compute_gate_auc(logits, labels):
    """Fast AUC approximation for gate monitoring (no sklearn dependency).

    Uses the Mann-Whitney U statistic: AUC = P(score_pos > score_neg).
    Returns 0.5 if all labels are the same (undefined AUC).
    """
    pos = logits[labels > 0.5]
    neg = logits[labels < 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    # Vectorized: count how often pos > neg
    comparisons = (pos.unsqueeze(1) > neg.unsqueeze(0)).float()
    ties = (pos.unsqueeze(1) == neg.unsqueeze(0)).float()
    return (comparisons + 0.5 * ties).mean().item()


def pairwise_ranking_loss(scores, improvements, n_pairs=500, threshold=1e-3,
                          threshold_quantile=0.0):
    """Pairwise BCE ranking loss for RankerGate.

    Samples random pairs (i, j). Label = 1 if improvement_i > improvement_j.
    Loss = BCE(sigmoid(score_i - score_j), label). Pairs with
    |imp_i - imp_j| < threshold are filtered as ambiguous.

    Args:
        scores: (B, N) raw gate scores (with gradient)
        improvements: (B, N) parent_fit - off_fit (positive = improved)
        n_pairs: number of random pairs to sample per batch
        threshold: minimum |imp_diff| to include a pair (fixed)
        threshold_quantile: if > 0, use per-batch adaptive threshold at
            this quantile of |imp_diff|. Overrides fixed threshold.
            E.g., 0.25 excludes the bottom 25% of pairs per batch.

    Returns:
        Scalar loss (0.0 if no valid pairs).
    """
    B, N = scores.shape
    device = scores.device

    # Sample random pairs
    idx_i = torch.randint(0, N, (B, n_pairs), device=device)
    idx_j = torch.randint(0, N, (B, n_pairs), device=device)

    score_i = scores.gather(1, idx_i)
    score_j = scores.gather(1, idx_j)
    imp_i = improvements.gather(1, idx_i)
    imp_j = improvements.gather(1, idx_j)

    imp_diff = imp_i - imp_j
    abs_diff = imp_diff.abs()

    # Adaptive or fixed threshold
    if threshold_quantile > 0:
        thr = abs_diff.quantile(threshold_quantile, dim=1, keepdim=True)
        valid = (abs_diff > thr).float()
    else:
        valid = (abs_diff > threshold).float()
    n_valid = valid.sum()

    score_diff = score_i - score_j
    label = (imp_diff > 0).float()

    # Weighted BCE avoids CUDA sync from valid.any() on hot path
    per_pair = F.binary_cross_entropy_with_logits(
        score_diff, label, reduction='none')
    return (per_pair * valid).sum() / n_valid.clamp(min=1.0)


def build_gate_diag(logits_flat, labels_flat, active_mask=None):
    """Build gate diagnostic dict (AUC, AUC active, pos_rate).

    Shared by compute_gate_bce and ranker pairwise loss paths.
    """
    auc = compute_gate_auc(logits_flat, labels_flat)
    diag = {'auc': auc}
    if active_mask is not None:
        act_flat = (active_mask > 0.5).reshape(-1)
        diag['auc_active'] = compute_gate_auc(
            logits_flat[act_flat], labels_flat[act_flat])
    pos_mask = labels_flat > 0.5
    diag['pos_rate'] = pos_mask.float().mean().item()
    if pos_mask.any() and (~pos_mask).any():
        diag['logit_pos'] = logits_flat[pos_mask].mean().item()
        diag['logit_neg'] = logits_flat[~pos_mask].mean().item()
    return diag


def compute_gate_bce(variant, extras, fitness, gate_bce_weight, gate_bce_scale_ema):
    """Gate contrafactual BCE: per-individual W=1 labels.

    Returns:
        (loss_or_None, updated_gate_bce_scale_ema, bce_detached_or_None,
         gate_auc_or_None)
        Always a 4-tuple.
    """
    _off_all = extras.get('off_fitness_all')
    _h_live = extras.get('h_live')
    if _off_all is None or _h_live is None:
        return None, gate_bce_scale_ema, None, None
    _off_reduced = _off_all.min(dim=0).values if _off_all.dim() == 3 else _off_all
    _h_global = extras.get('h_global')
    _node_feat = extras.get('node_feat')
    gate_bce = variant.activity_gate.contrafactual_bce_loss(
        _h_live, fitness.detach(), _off_reduced, scale_ema=gate_bce_scale_ema,
        h_global=_h_global, node_feat=_node_feat)
    with torch.no_grad():
        _abs_change = (fitness.detach() - _off_reduced).abs().mean()
        gate_bce_scale_ema = 0.99 * gate_bce_scale_ema + 0.01 * _abs_change.clamp(min=1e-6)
        # Gate AUC: how well do logits predict improvability?
        _logits = variant.activity_gate.get_logits(
            _h_live.detach(), h_global=_h_global, node_feat=_node_feat)
        _labels = (fitness.detach() > _off_reduced).float()
        _flat_logits = _logits.reshape(-1)
        _flat_labels = _labels.reshape(-1)
        _active = extras.get('active_mask')
        _gate_diag = build_gate_diag(_flat_logits, _flat_labels, _active)
        # Improvement quality by active/inactive bucket
        if _active is not None:
            _imp = (fitness.detach() - _off_reduced).reshape(-1)
            _act_mask = (_active > 0.5).reshape(-1)
            _inact_mask = ~_act_mask
            if _act_mask.any():
                _gate_diag['imp_active'] = _imp[_act_mask].mean().item()
            if _inact_mask.any():
                _gate_diag['imp_inactive'] = _imp[_inact_mask].mean().item()
    _finite = torch.isfinite(gate_bce)
    loss = gate_bce_weight * gate_bce if _finite else None
    return loss, gate_bce_scale_ema, gate_bce.detach() if _finite else None, _gate_diag


def build_step_diagnostics(*, step, rank, fn_id, D, N, task_id, aug_seed,
                           target_hit, gap_init, gap_final, gap_ratio,
                           total_gens, n_bptt_gens, bptt_w, step_fes,
                           cumulative_fes, n_chunks, dt, grad_norm,
                           diag_every, named_groups, get_grad_norm_fn,
                           acc):
    """Build per-step diagnostics dict.

    Args:
        acc: dict with optional keys: entropy, active_frac, lupi_dist_k,
             lupi_align_k, hit_loss, geo_loss, gate_bce,
             improved_count, improved_total, winner_counts.
    """
    _gn = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
    diag = {
        'step': step, 'rank': rank, 'fn': fn_id, 'D': D, 'N': N,
        'task_id': task_id, 'aug_seed': aug_seed, 'target_hit': target_hit,
        'gap_init': round(gap_init, 4), 'gap_final': round(gap_final, 4),
        'gap_ratio': round(gap_ratio, 6), 'total_gens': total_gens,
        'bptt_gens': n_bptt_gens, 'bptt_w': int(bptt_w),
        'step_fes': step_fes, 'cumulative_fes': round(cumulative_fes, 1),
        'n_chunks': n_chunks, 'dt': round(dt, 2),
        'grad_norm': round(_gn, 6),
    }
    if step % diag_every == 0:
        diag['grad_norms'] = {k: round(get_grad_norm_fn(v).item(), 6)
                              for k, v in named_groups.items()}
    if acc.get('entropy'):
        diag['entropy'] = round(torch.stack(acc['entropy']).mean().item(), 4)
    if acc.get('active_frac'):
        diag['active_frac'] = round(torch.stack(acc['active_frac']).mean().item(), 4)
    if acc.get('lupi_dist_k'):
        _dk = torch.stack(acc['lupi_dist_k']).mean(dim=0)
        _ak = torch.stack(acc['lupi_align_k']).mean(dim=0)
        for k in range(_dk.shape[0]):
            diag[f'lupi/dist_k{k}'] = round(_dk[k].item(), 6)
            diag[f'lupi/align_k{k}'] = round(_ak[k].item(), 6)
    if acc.get('hit_loss'):
        diag['loss/hitting'] = round(torch.stack(acc['hit_loss']).mean().item(), 6)
    if acc.get('geo_loss'):
        diag['loss/geo_lupi'] = round(torch.stack(acc['geo_loss']).mean().item(), 6)
    if acc.get('gate_bce'):
        diag['loss/gate_bce'] = round(torch.stack(acc['gate_bce']).mean().item(), 6)
    # `surrogate_pw` is the dedicated diag key for the PairwiseSurrogate
    # ranking loss (decoupled from `gate_bce` so the legacy ActivityGate /
    # RankerGate paths can be analysed independently from the surrogate path).
    if acc.get('surrogate_pw'):
        diag['loss/surrogate_pw'] = round(
            torch.stack(acc['surrogate_pw']).mean().item(), 6)
    if acc.get('gate_auc'):
        _gd = acc['gate_auc']  # list of dicts
        diag['gate_auc'] = round(sum(d['auc'] for d in _gd) / len(_gd), 4)
        if 'auc_active' in _gd[0]:
            diag['gate_auc_active'] = round(sum(d['auc_active'] for d in _gd) / len(_gd), 4)
        if 'pos_rate' in _gd[0]:
            diag['gate_pos_rate'] = round(sum(d['pos_rate'] for d in _gd) / len(_gd), 4)
        if 'logit_pos' in _gd[0]:
            _lp = [d['logit_pos'] for d in _gd if 'logit_pos' in d]
            _ln = [d['logit_neg'] for d in _gd if 'logit_neg' in d]
            if _lp:
                diag['gate_logit_pos'] = round(sum(_lp) / len(_lp), 4)
                diag['gate_logit_neg'] = round(sum(_ln) / len(_ln), 4)
        _ia = [d['imp_active'] for d in _gd if 'imp_active' in d]
        _ii = [d['imp_inactive'] for d in _gd if 'imp_inactive' in d]
        if _ia:
            diag['imp_active'] = round(sum(_ia) / len(_ia), 6)
        if _ii:
            diag['imp_inactive'] = round(sum(_ii) / len(_ii), 6)
    if acc.get('oracle_agreement'):
        _oa = acc['oracle_agreement']
        diag['oracle_agreement'] = round(sum(_oa) / len(_oa), 4)
    if acc.get('improved_total', 0) > 0:
        diag['improved_pct'] = round(acc['improved_count'].item() / acc['improved_total'], 4)
    wc = acc.get('winner_counts')
    if wc is not None and wc.sum() > 0:
        diag['route_pct'] = [round(x.item(), 3) for x in wc / wc.sum()]
    _st = acc.get('surr_total_count', 0)
    if _st > 0:
        diag['surr_parent_frac'] = round(acc['surr_parent_count'] / _st, 4)
    if acc.get('surr_sel_imp'):
        _si = acc['surr_sel_imp']
        diag['surr/sel_improve_rate'] = round(sum(d['sel_imp_rate'] for d in _si) / len(_si), 4)
        diag['surr/rej_improve_rate'] = round(sum(d['rej_imp_rate'] for d in _si) / len(_si), 4)
        diag['surr/sel_med_imp'] = round(sum(d['sel_med_imp'] for d in _si) / len(_si), 6)
        diag['surr/rej_med_imp'] = round(sum(d['rej_med_imp'] for d in _si) / len(_si), 6)
    if acc.get('fcr_diag'):
        _fd = acc['fcr_diag']
        n = len(_fd)
        _s = lambda v: v.item() if torch.is_tensor(v) else v
        _avg = lambda key: round(sum(_s(d[key]) for d in _fd) / n, 6)
        for src, dst in [('fcr_loss_F', 'fcr/loss_F'),
                         ('fcr_F_mean', 'fcr/F_mean'),
                         ('fcr_F_optimal_mean', 'fcr/F_optimal_mean'),
                         ('fcr_F_optimal_std', 'fcr/F_optimal_std')]:
            diag[dst] = _avg(src)
        if 'fcr_loss_CR' in _fd[0]:
            for src, dst in [('fcr_loss_CR', 'fcr/loss_CR'),
                             ('fcr_CR_mean', 'fcr/CR_mean'),
                             ('fcr_CR_optimal_mean', 'fcr/CR_optimal_mean')]:
                diag[dst] = _avg(src)
        if 'fcr_F_sigma_mean' in _fd[0]:
            diag['fcr/F_sigma_mean'] = _avg('fcr_F_sigma_mean')
    if acc.get('donor_diag'):
        _dd = acc['donor_diag']
        n = len(_dd)
        _s = lambda v: v.item() if torch.is_tensor(v) else v
        _avg_d = lambda key: round(sum(_s(d[key]) for d in _dd if key in d) / n, 6)
        for src, dst in [('donor_agreement_pbest', 'donor/agreement_pbest'),
                         ('donor_agreement_r1',    'donor/agreement_r1'),
                         ('donor_agreement_r2',    'donor/agreement_r2'),
                         ('donor_valid_frac',      'donor/valid_frac'),
                         ('donor_loss_pbest',      'donor/loss_pbest'),
                         ('donor_loss_r1',         'donor/loss_r1'),
                         ('donor_loss_r2',         'donor/loss_r2'),
                         # KL distillation diagnostics (Etapa A): same accumulator
                         # since both write to donor_diag, but distinct keys.
                         ('kl_loss_pbest',         'kl/loss_pbest'),
                         ('kl_loss_r1',            'kl/loss_r1'),
                         ('kl_loss_r2',            'kl/loss_r2'),
                         ('kl_p_adaptive',         'kl/p_adaptive'),
                         ('kl_K_topk',             'kl/K_topk')]:
            diag[dst] = _avg_d(src)
    if acc.get('attn_diag'):
        _ad = acc['attn_diag']
        n = len(_ad)
        _s = lambda v: v.item() if torch.is_tensor(v) else v
        for key in ('attn/entropy_pbest', 'attn/pearson_pbest_fit',
                    'attn/pearson_F_rank',
                    # [2026-05-04 disentangle] aggregate per-step from list of per-gen dicts
                    'disentangle_L_e', 'disentangle_L_x', 'disentangle_L_hsic',
                    'disentangle_R2_explor', 'disentangle_R2_exploit',
                    'disentangle_total',
                    # [2026-05-06 arm C] anti-leak verification when random_target=True
                    'antileak_cor_explor', 'antileak_cor_exploit',
                    # [2026-05-07 arm C] anti-collapse: std of head predictions
                    'disentangle_predstd_explor', 'disentangle_predstd_exploit'):
            vals = [_s(d[key]) for d in _ad if key in d]
            if vals:
                diag[key] = round(sum(vals) / len(vals), 6)
    return diag


# ── Structural diagnostics on attention + F/CR ──

def _pearson_flat(x, y, eps=1e-12):
    """Pearson correlation over flattened 1D tensors. Zero if any std is zero."""
    x = x.reshape(-1)
    y = y.reshape(-1)
    if x.numel() < 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    denx = xm.pow(2).sum().sqrt()
    deny = ym.pow(2).sum().sqrt()
    den = denx * deny
    if den.item() < eps:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return num / den


def compute_attn_diag(extras, fitness):
    """Structural probes on donor attention and F_mean calibration.

    Zero-weight diagnostics — no gradient contribution. Outputs scalar tensors
    so .item() can be deferred to build_step_diagnostics.

    Metrics:
        attn/entropy_pbest : mean entropy of softmax(A_pbest) over rows.
            High → uniform pbest selection. Low → peaked.
        attn/pearson_pbest_fit : Pearson(A_pbest[b, i, j], f_j) flattened over
            off-diagonal (i, j) pairs, averaged over B. If A_pbest collapses
            to fitness bias, |corr| → 1 — diagnoses whether the h·h^T term
            is contributing anything beyond the bias.
        attn/pearson_F_rank : Pearson(F_mean[b, n], fitness_rank[b, n]).
            Tests whether F/CR adapts to per-individual fitness rank.
            (rank 0 = best, N-1 = worst)
    """
    diag = {}
    A_pbest = extras.get('_A_pbest')
    if A_pbest is not None:
        B, N_q, N_pool = A_pbest.shape
        # Restrict diagnostics to the active candidate slice [:N] when the
        # graph-native archive widens A_pbest to (B, N, N+K). pbest is
        # constrained to active parents (D4: archive slots are -1e9 on this
        # channel), so the active slice carries the meaningful distribution.
        if N_pool != N_q:
            A_pbest = A_pbest[:, :, :N_q]
        N = N_q
        device = A_pbest.device
        dtype = A_pbest.dtype
        # Softmax over last dim (the -1e9 diag mask drives that column to 0).
        # Compute entropy only over the N-1 off-diagonal entries for each row.
        with torch.no_grad():
            probs = F.softmax(A_pbest, dim=-1)  # (B, N, N)
            # -sum p*log(p) with p>0 safety
            ent = -(probs.clamp(min=1e-12) * probs.clamp(min=1e-12).log()).sum(dim=-1)
            # mean over (B, N), excluding diagonal self-rows (they're fine).
            diag['attn/entropy_pbest'] = ent.mean()

        # Pearson(A_pbest[i,j], f_j) over off-diagonal pairs, per batch.
        with torch.no_grad():
            diag_mask = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
            off_mask = ~diag_mask  # (1, N, N)
            f_j = fitness.detach().unsqueeze(1).expand(B, N, N)  # broadcasts over i
            corrs = []
            for b in range(B):
                a_flat = A_pbest[b][off_mask[0]].detach()
                f_flat = f_j[b][off_mask[0]]
                # Guard: clamp -inf residues (shouldn't appear since diag
                # already removed, but A_pbest could have -1e9 from r1-mask
                # elsewhere — not here for pbest, but defensive)
                a_flat = a_flat.clamp(min=-1e8)
                if a_flat.numel() > 1 and f_flat.std() > 1e-8 and a_flat.std() > 1e-8:
                    corrs.append(_pearson_flat(a_flat, f_flat))
            if corrs:
                diag['attn/pearson_pbest_fit'] = torch.stack(corrs).mean()
            else:
                diag['attn/pearson_pbest_fit'] = torch.zeros((), device=device, dtype=dtype)

    F_mean = extras.get('_F_mean')
    if F_mean is not None:
        B, N = F_mean.shape
        device = F_mean.device
        dtype = F_mean.dtype
        with torch.no_grad():
            # Fitness rank: 0 = best (lowest fitness), N-1 = worst.
            rank_idx = fitness.detach().argsort(dim=-1)  # (B, N) — rank_idx[b,k]=n
            inv_rank = torch.zeros_like(rank_idx)
            arange = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
            inv_rank.scatter_(-1, rank_idx, arange)  # inv_rank[b,n] = rank of n
            corrs = []
            for b in range(B):
                fm = F_mean[b].detach().float()
                r = inv_rank[b].float()
                if fm.std() > 1e-8 and r.std() > 1e-8:
                    corrs.append(_pearson_flat(fm, r))
            if corrs:
                diag['attn/pearson_F_rank'] = torch.stack(corrs).mean().to(dtype)
            else:
                diag['attn/pearson_F_rank'] = torch.zeros((), device=device, dtype=dtype)

    return diag


# ── Adaptive F/CR grid supervision ──

_FCR_GRID_DEFAULT = (0.2, 0.35, 0.5, 0.65, 0.8)


def compute_fcr_grid_loss(extras, coords, fitness, eval_fn, weight=0.3,
                          grid=_FCR_GRID_DEFAULT,
                          lb: float = -100.0, ub: float = 100.0):
    """Evaluate fixed F/CR grid, compute MSE(F_mean, F_optimal) + same for CR.

    Single eval_fn call for both F and CR grids (merged batch).

    Args:
        extras: dict with '_F_mean', '_CR_mean', '_diff_vector' (B, N, D)
        coords: (B, N, D) current population coordinates
        fitness: (B, N) current population fitness
        eval_fn: CEC2017Torch callable, takes (batch, D) → (batch,)
        weight: loss weight multiplier
        grid: F/CR values to evaluate

    Returns:
        (loss, diag_dict) or (None, {}) if required keys missing
    """
    F_mean = extras.get('_F_mean')
    CR_mean = extras.get('_CR_mean')
    diff_vec = extras.get('_diff_vector')

    if F_mean is None or diff_vec is None:
        return None, {}

    B, N, D = coords.shape
    device = coords.device
    g = torch.tensor(grid, device=device, dtype=coords.dtype)
    n_grid = len(g)

    diff_expanded = diff_vec.unsqueeze(0)       # (1, B, N, D)
    F_grid = g.view(n_grid, 1, 1, 1)            # (n_grid, 1, 1, 1)
    coords_0 = coords.unsqueeze(0)              # (1, B, N, D)

    # F grid offspring: coords + F_k * diff
    offspring_F = (coords_0 + F_grid * diff_expanded).clamp(lb, ub)

    # CR grid offspring: coords + crossover_mask * (F_mean * diff)
    F_mean_det = F_mean.detach().unsqueeze(0).unsqueeze(-1)  # (1, B, N, 1)
    mutation = F_mean_det.to(coords.dtype) * diff_expanded    # (1, B, N, D)
    CR_grid = g.view(n_grid, 1, 1, 1)
    u = torch.rand(n_grid, B, N, D, device=device, dtype=coords.dtype)
    cr_mask = torch.sigmoid(10.0 * (CR_grid - u))
    j_rand = torch.randint(0, D, (n_grid, B, N), device=device)
    cr_mask = cr_mask.scatter(-1, j_rand.unsqueeze(-1), 1.0)
    offspring_CR = (coords_0 + cr_mask * mutation).clamp(lb, ub)

    # Single eval_fn call for both grids
    all_offspring = torch.cat([offspring_F, offspring_CR], dim=0)  # (2*n_grid, B, N, D)
    all_fit = eval_fn(all_offspring.reshape(2 * n_grid * B * N, D))
    all_fit = all_fit.reshape(2 * n_grid, B, N)

    fit_F = all_fit[:n_grid]          # (n_grid, B, N)
    fit_CR = all_fit[n_grid:]         # (n_grid, B, N)

    parent_fit = fitness.unsqueeze(0)
    F_optimal = g[(parent_fit - fit_F).argmax(dim=0)].float()
    CR_optimal = g[(parent_fit - fit_CR).argmax(dim=0)].float()

    loss_F = (F_mean - F_optimal.detach()).pow(2).mean()
    loss_CR = (CR_mean - CR_optimal.detach()).pow(2).mean() if CR_mean is not None else 0.0

    total = weight * (loss_F + loss_CR)

    # Diagnostics — no .item() on hot path, defer to logging
    with torch.no_grad():
        diag = {
            'fcr_loss_F': loss_F.detach(),
            'fcr_F_mean': F_mean.detach().mean(),
            'fcr_F_optimal_mean': F_optimal.mean(),
            'fcr_F_optimal_std': F_optimal.std(),
        }
        if CR_mean is not None:
            diag['fcr_loss_CR'] = loss_CR.detach() if torch.is_tensor(loss_CR) else 0.0
            diag['fcr_CR_mean'] = CR_mean.detach().mean()
            diag['fcr_CR_optimal_mean'] = CR_optimal.mean()

    return total if torch.isfinite(total) else None, diag
