"""KSD measurement runner — ∂KSD/∂θ inter-fn cosine over episodes.

Reuses helpers from analysis/phase0_measurement.py and
analysis/gradient_decomposition.py read-only. Output schema mirrors
phase0_measurement.py so existing aggregation tooling can ingest the
JSON without changes.

Usage:
    python -m l2o.ksd.measurement <checkpoint> \\
        --device cpu --seeds 3 --n-gens 8 \\
        --output diagnostics/ksd_a3
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np
import torch

from analysis.gradient_decomposition import load_checkpoint
from analysis.phase0_measurement import (
    get_extended_named_groups,
    grad_vec_for_group,
    cos_sim_matrix,
)
from cec2017 import CEC2017Torch
from encoder.opt_variant import _clamp_fitness
from encoder.similarity_graph_gpu import build_sparse_graphs_gpu
from l2o.ksd.loss import ksd_loss_batched
from l2o.ksd.grad_feature import (
    GeometricBiasDonorWrapper, GradFeatureFCRWrapper,
    KSDState, sign_coherence_score,
)


def _named_groups_with_donor_selector(backbone, variant, gen_step):
    """Wrap get_extended_named_groups to handle the wrapped backbone case.

    `phase0_measurement.get_extended_named_groups` checks
    `hasattr(backbone, 'donor_selector')`, but in TemporalSparseGATv2Backbone
    the donor_selector lives at `backbone.backbone.donor_selector`. We patch
    locally without modifying the upstream helper.

    Additionally exposes `var.h0.adaptive_fcr` granularly so the F/CR head
    cosine can be reported separately from the rest of var.h0.
    """
    ng = get_extended_named_groups(backbone, variant, gen_step)
    if 'bb.donor_selector' not in ng and hasattr(backbone, 'backbone') \
            and hasattr(backbone.backbone, 'donor_selector'):
        ds = backbone.backbone.donor_selector
        ng['bb.donor_selector'] = [n for n, _ in ds.named_parameters()]
        bb_all = []
        for k in ('bb.gat', 'bb.temporal', 'bb.pooler', 'bb.donor_selector'):
            if k in ng:
                bb_all.extend(ng[k])
        if bb_all:
            ng['bb.all'] = bb_all
    # var.h0.adaptive_fcr granular group.
    if (hasattr(variant, 'heads') and len(variant.heads) > 0
            and hasattr(variant.heads[0], 'adaptive_fcr')):
        ng['var.h0.adaptive_fcr'] = [
            n for n, _ in variant.heads[0].adaptive_fcr.named_parameters()
        ]
    return ng


# Subnetwork groups reported per-fn. bb.donor_selector is conditional
# (added by get_extended_named_groups when present).
SUBNET_GROUPS = ('bb.all', 'bb.gat', 'bb.temporal', 'bb.pooler',
                 'bb.donor_selector', 'var.h0', 'var.h0.adaptive_fcr',
                 'var.surrogate', 'var.router')


def _compute_g_dir_g_mag(coords_detached, eval_fn, eps=1e-8):
    """Compute (g_raw, g_dir, g_mag) from a detached (B, N, D) coords tensor.

    g_raw  = +∇f at coords (raw gradient, retained for sign-coherence diag).
    g_dir  = -g_raw / ‖g_raw‖   (descent-aligned unit direction).
    g_mag  = log1p(‖g_raw‖).clamp(min=1e-2).

    eval_fn must accept (M, D) flat batches and return (M,) fitness; we
    reshape internally to maintain the (B, N, D) → ∂/∂coords graph.
    """
    B, N, D = coords_detached.shape
    c = coords_detached.detach().clone().requires_grad_(True)
    f_flat = eval_fn(c.reshape(B * N, D))                          # (B*N,)
    f_sum = f_flat.sum()
    g_raw = torch.autograd.grad(f_sum, c)[0]                       # (B, N, D), +∇f
    g_norm = torch.linalg.norm(g_raw, dim=-1, keepdim=True) + eps
    g_dir = (-g_raw / g_norm).detach()
    g_mag = torch.log1p(g_norm).clamp(min=1e-2).detach()
    return g_raw.detach(), g_dir, g_mag


def _wrap_heads(backbone, variant, D):
    """Replace bb.backbone.donor_selector and variant.heads[0].adaptive_fcr
    with grad-feature wrappers. Returns (donor_wrap, fcr_wrap) so callers
    can mutate w_geom or restore.

    The wrappers share a thunk that reads `variant._ksd_state` (set by
    measurement loop before each generation). When state is None,
    wrappers passthrough — preserves baseline behaviour outside KSD.
    """
    state_provider = lambda: getattr(variant, '_ksd_state', None)

    base_donor = backbone.backbone.donor_selector
    if isinstance(base_donor, GeometricBiasDonorWrapper):
        donor_wrap = base_donor
    else:
        donor_wrap = GeometricBiasDonorWrapper(base_donor, state_provider).to(
            next(base_donor.parameters()).device)
        backbone.backbone.donor_selector = donor_wrap

    base_fcr = variant.heads[0].adaptive_fcr
    if isinstance(base_fcr, GradFeatureFCRWrapper):
        fcr_wrap = base_fcr
    else:
        fcr_wrap = GradFeatureFCRWrapper(base_fcr, D=D,
                                          state_provider=state_provider).to(
            next(base_fcr.parameters()).device)
        variant.heads[0].adaptive_fcr = fcr_wrap

    return donor_wrap, fcr_wrap


def _attn_diagnostics(A_pbest_logits, coords, g_dir, g_mag):
    """axis_7 (entropy ↔ ‖∇f‖ correlation) + axis_8 (attention-descent alignment).

    A_pbest_logits: (B, N, N_pool) raw logits (post-bias if wrappers active).
    coords:         (B, N, D), detached.
    g_dir:          (B, N, D), descent direction, detached, unit-norm.
    g_mag:          (B, N, 1), log1p magnitude, detached.

    Returns dict with per-step scalars (axis_7 and axis_8).
    """
    with torch.no_grad():
        A_logits = A_pbest_logits.detach()
        # Mask -1e9 entries (self-mask, archive-pad). Softmax still well-defined.
        A_attn = torch.softmax(A_logits, dim=-1)              # (B, N, N_pool)
        # Restrict to active pop (first N pool entries).
        N_pop = coords.shape[1]
        A_attn = A_attn[..., :N_pop]
        # Renormalise after slicing, for entropy + alignment to use a proper distribution.
        A_attn = A_attn / A_attn.sum(dim=-1, keepdim=True).clamp(min=1e-12)

        # axis_7: entropy of A_attn[i, :] vs ||g_raw||_i.
        eps = 1e-12
        ent = -(A_attn * (A_attn + eps).log()).sum(-1)        # (B, N)
        # ||g_raw|| = exp(g_mag) - 1 if g_mag > 1e-2, else effectively 0; we just use g_mag as proxy.
        gnorm_proxy = g_mag.squeeze(-1)                        # (B, N)
        # Pearson per batch, then average.
        def _pearson(x, y):
            x_c = x - x.mean()
            y_c = y - y.mean()
            den = (x_c.std() * y_c.std() + 1e-12)
            return (x_c * y_c).mean() / den
        pearsons = [_pearson(ent[b].flatten(), gnorm_proxy[b].flatten()).item()
                    for b in range(ent.shape[0])]
        axis_7 = float(np.mean(pearsons))

        # axis_8: Σ_j A_attn[i,j] · cos(g_dir[i], coords[j] - coords[i]).
        # diff[b, i, j, :] = coords[b, j] - coords[b, i]
        diff = coords.unsqueeze(1) - coords.unsqueeze(2)       # (B, N, N, D)
        diff_n = diff / (diff.norm(dim=-1, keepdim=True) + 1e-12)
        # g_dir is already unit-norm.
        # cos = diff_n · g_dir[i]: (B, N, N, D) · (B, N, 1, D) summed over D
        cos_match = (diff_n * g_dir.unsqueeze(2)).sum(-1)      # (B, N, N)
        align = (A_attn * cos_match).sum(-1).mean()            # scalar
        axis_8 = float(align.item())

    return {'axis_7_attn_entropy_grad_corr': axis_7,
            'axis_8_attn_descent_alignment': axis_8}


def _ksd_over_population(coords, eval_fn, h_ema_per_batch, T=1.0,
                          alpha=0.9, return_terms=False):
    """Compute KSD on a (B, N, D) population — single vectorised call.

    coords may carry grad_fn from the variant forward — autograd flows back
    to model params via this chain. Under freeze-mode='all' or any setup
    where the variant produces grad-free output, we re-leaf the tensor with
    requires_grad so ksd_loss_batched's internal `autograd.grad(f, X)` can
    still compute the score (the resulting backward simply won't reach any
    model params, which is the intended behaviour).
    """
    if not coords.requires_grad:
        coords = coords.detach().clone().requires_grad_(True)
    out = ksd_loss_batched(coords, eval_fn, T=T,
                            h_ema=h_ema_per_batch, alpha=alpha,
                            return_terms=return_terms)
    if return_terms:
        loss, h_new_list, t1, t2, t3, t4 = out
        return loss, h_new_list, [t1.detach(), t2.detach(),
                                   t3.detach(), t4.detach()]
    loss, h_new_list = out
    return loss, h_new_list


def _run_generation_loop(gen_step, variant, fn, config, device,
                          n_gens, T_temp, use_grad_wrapper,
                          sign_coherence_log_state=None,
                          collect_attn_diag=True):
    """Shared per-generation episode used by both measurement and training.

    Drives the generation loop, KSDState plumbing, and per-step KSD
    accumulation. Caller decides what to do with the returned losses
    (cosine measurement vs optimizer.step).

    Returns dict:
        losses_per_gen:      list of scalar tensors with grad chain.
        terms_per_gen:       list of [t1, t2, t3, t4] floats (one per gen).
        attn_diag_per_gen:   list of {axis_7, axis_8} dicts (when use_grad_wrapper
                             and collect_attn_diag).
        nan_seen:            bool — non-finite loss or grad oracle failure.
        nan_reason:          str or None.
        n_gens_completed:    int.
    """
    D = config['D'][0] if isinstance(config['D'], list) else config['D']
    N = config['N']
    B = config.get('b_per_gpu', 4)
    M = config.get('m_samples', 20)
    MAX_FES = config.get('budget_mult', 1000) * D
    gru_W = config.get('gru_window', 16)
    k_neighbors = config.get('k_neighbors', 8)
    gumbel_tau = config.get('gumbel_tau', 1.0)

    gen_step.zero_grad()
    gen_step.train()
    gen_step.eval_fn = fn
    if hasattr(variant, '_oracle_best_k'):
        variant._oracle_best_k = None

    def eval_fn(x):
        return fn(x.to(torch.float64))

    coords = (torch.rand(B, N, D, device=device) * 200 - 100).to(torch.float64)
    fitness = _clamp_fitness(fn(coords.reshape(-1, D)).reshape(B, N))

    coords_ring = torch.zeros(B, gru_W, N, D, dtype=torch.float32, device=device)
    fitness_ring = torch.zeros(B, gru_W, N, dtype=torch.float32, device=device)
    stag = torch.zeros(B, device=device)
    dfit = torch.zeros(B, device=device)
    crat = torch.zeros(B, device=device)
    prev_best = prev_spread = None
    cum_fes = 0

    h_ema_per_batch = [None] * B
    losses_per_gen = []
    terms_per_gen = []
    attn_diag_per_gen = []
    nan_seen = False
    nan_reason = None

    for gen in range(min(n_gens, 800)):
        if cum_fes >= MAX_FES:
            break
        ri = gen % gru_W
        coords_ring[:, ri] = coords.detach().float()
        fitness_ring[:, ri] = fitness.detach().float()
        nv = min(gen + 1, gru_W)
        if gen < gru_W:
            idx = list(range(gen + 1))
        else:
            s = (gen + 1) % gru_W
            idx = [(s + i) % gru_W for i in range(gru_W)]
        ch, fh = coords_ring[:, idx], fitness_ring[:, idx]
        pc = coords_ring[:, (ri - 1) % gru_W].float() if gen > 0 else None
        pf = fitness_ring[:, (ri - 1) % gru_W].float() if gen > 0 else None

        cb = fitness.min(dim=1).values
        cs = coords.float().std(dim=1).mean(dim=1)
        if prev_best is not None:
            d = prev_best - cb
            dfit = d.clamp(-1, 1)
            stag = torch.where(d.abs() < 1e-10, stag + 1, torch.zeros_like(stag))
            crat = ((prev_spread - cs) / prev_spread.clamp(min=1e-8)).clamp(-1, 1)
        prev_best, prev_spread = cb, cs

        cache = build_sparse_graphs_gpu(
            coords.float(), fitness.float(),
            step_num=cum_fes, max_steps=MAX_FES, ndim=D,
            k_neighbors=k_neighbors,
            stagnation_counters=stag, delta_fitnesses=dfit,
            contraction_rates=crat, prev_coords=pc, prev_fitnesses=pf)

        # Side-channel: write KSDState before variant forward so the
        # donor and FCR wrappers can read it. When use_grad_wrapper is
        # False, leave _ksd_state cleared so wrappers passthrough.
        g_dir_pre = g_mag_pre = g_raw_pre = None
        if use_grad_wrapper:
            try:
                g_raw_pre, g_dir_pre, g_mag_pre = _compute_g_dir_g_mag(
                    coords.detach(), eval_fn)
            except RuntimeError as e:
                print(f"  [KSD] gen {gen}: grad oracle failed: {e}",
                      file=sys.stderr)
                nan_seen = True
                nan_reason = f'grad oracle failed: {e}'
                break
            state = KSDState(coords=coords.detach(),
                             g_dir=g_dir_pre, g_mag=g_mag_pre)
            state.assert_detached()
            variant._ksd_state = state
            if sign_coherence_log_state is not None and \
                    not sign_coherence_log_state.get('logged', False):
                coh = sign_coherence_score(g_dir_pre * g_mag_pre, g_raw_pre)
                print(f"[ksd] sign_coherence (expect strongly negative): "
                      f"{coh:.4f}", file=sys.stderr, flush=True)
                sign_coherence_log_state['logged'] = True
        else:
            variant._ksd_state = None

        result = gen_step.run(
            coords=coords, fitness=fitness, cache=cache,
            f_optimal=fn.f_optimal, M=M, gumbel_tau=gumbel_tau,
            node_feat=cache.node_feat, global_feat=cache.global_feat,
            coords_hist=ch, fitness_hist=fh,
            n_valid=nv, fes_frac=cum_fes / MAX_FES, surrogate_M=N)

        new_coords = result['new_coords']
        new_fitness = result['new_fitness']

        try:
            ksd, h_new_list, terms_means = _ksd_over_population(
                new_coords, eval_fn, h_ema_per_batch=h_ema_per_batch,
                T=T_temp, alpha=0.9, return_terms=True)
        except RuntimeError as e:
            print(f"  [KSD] gen {gen}: forward error: {e}", file=sys.stderr)
            nan_seen = True
            nan_reason = f'forward error: {e}'
            break

        if not torch.isfinite(ksd).item():
            nan_seen = True
            nan_reason = 'non-finite ksd'
            break

        h_ema_per_batch = h_new_list
        losses_per_gen.append(ksd)
        terms_per_gen.append([float(t.item()) for t in terms_means])

        extras = result.get('extras', {})
        if collect_attn_diag and use_grad_wrapper \
                and '_A_pbest' in extras and g_dir_pre is not None:
            try:
                diag = _attn_diagnostics(extras['_A_pbest'].detach(),
                                          coords.detach(),
                                          g_dir_pre, g_mag_pre)
                attn_diag_per_gen.append(diag)
            except RuntimeError as e:
                print(f"  [KSD] attn diagnostic gen {gen} failed: {e}",
                      file=sys.stderr)

        coords = new_coords
        fitness = new_fitness
        fes = extras.get('fes_used', float(N))
        cum_fes += (fes.item() if torch.is_tensor(fes) else fes)

    return {
        'losses_per_gen': losses_per_gen,
        'terms_per_gen': terms_per_gen,
        'attn_diag_per_gen': attn_diag_per_gen,
        'nan_seen': nan_seen,
        'nan_reason': nan_reason,
        'n_gens_completed': len(losses_per_gen),
    }


def run_ksd_episode(gen_step, backbone, variant, fn, config, device,
                    n_gens=8, seed=0, T_temp=1.0,
                    use_grad_wrapper=False,
                    sign_coherence_log_state=None):
    """Single-fid, single-seed KSD episode.

    When `use_grad_wrapper=True`, the KSDState is written to
    `variant._ksd_state` before each generation. Wrappers must already
    be installed (via _wrap_heads) — we assume the caller did so.

    `sign_coherence_log_state`: optional dict with key 'logged' to log
    the sign-coherence diagnostic exactly once across all episodes.

    Returns dict with bb_grad_vec, per-group grad vectors, term means,
    plus a 'nan' flag if any forward/backward produced non-finite values.
    Adds axis_7 / axis_8 means when use_grad_wrapper=True (A_pbest in extras).
    """
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    named_groups = _named_groups_with_donor_selector(backbone, variant, gen_step)

    ep = _run_generation_loop(
        gen_step, variant, fn, config, device,
        n_gens=n_gens, T_temp=T_temp,
        use_grad_wrapper=use_grad_wrapper,
        sign_coherence_log_state=sign_coherence_log_state,
        collect_attn_diag=True)
    losses_per_gen = ep['losses_per_gen']
    terms_per_gen = ep['terms_per_gen']
    attn_diag_per_gen = ep['attn_diag_per_gen']
    nan_seen = ep['nan_seen']

    if not losses_per_gen:
        return {'nan': True,
                'reason': ep.get('nan_reason') or 'no generations completed'}

    total_loss = torch.stack(losses_per_gen).mean()
    if not torch.isfinite(total_loss).item():
        nan_seen = True

    grad_norms = {}
    grad_vecs = {}
    if not nan_seen:
        gen_step.zero_grad()
        try:
            total_loss.backward()
        except RuntimeError as e:
            print(f"  [KSD] backward error: {e}", file=sys.stderr)
            nan_seen = True

        for gname, pnames in named_groups.items():
            v = grad_vec_for_group(gen_step, pnames)
            grad_vecs[gname] = v
            grad_norms[gname] = float(v.norm().item()) if v is not None else 0.0

    if nan_seen:
        gen_step.zero_grad()
        gen_step.eval()
        return {'nan': True, 'reason': 'nan in forward/backward',
                'n_gens_completed': len(losses_per_gen)}

    # Term means averaged over generations.
    term_means = np.array(terms_per_gen).mean(axis=0).tolist()  # [t1, t2, t3, t4]

    # axis_7 / axis_8 means.
    attn_diag = {}
    if attn_diag_per_gen:
        attn_diag = {
            'axis_7_attn_entropy_grad_corr_mean': float(np.mean(
                [d['axis_7_attn_entropy_grad_corr'] for d in attn_diag_per_gen])),
            'axis_8_attn_descent_alignment_mean': float(np.mean(
                [d['axis_8_attn_descent_alignment'] for d in attn_diag_per_gen])),
        }

    # w_geom value + grad norm. w_geom.grad must be non-zero when
    # use_grad_wrapper=True; otherwise a silent detach has broken the chain.
    w_geom_diag = {}
    if use_grad_wrapper and hasattr(backbone, 'backbone') \
            and hasattr(backbone.backbone, 'donor_selector') \
            and hasattr(backbone.backbone.donor_selector, 'w_geom'):
        w = backbone.backbone.donor_selector.w_geom
        w_geom_diag = {
            'w_geom_value': float(w.item()),
            'w_geom_grad': float(w.grad.item()) if w.grad is not None else 0.0,
        }

    out = {
        'nan': False,
        'loss_val': float(total_loss.item()),
        'n_gens_completed': len(losses_per_gen),
        'grad_norms': grad_norms,
        'bb_grad_vec': grad_vecs.get('bb.all'),
        'bb_grad_norm': grad_norms.get('bb.all', 0.0),
        'grad_vecs_per_group': grad_vecs,
        'term_means': term_means,
        'attn_diag': attn_diag,
        'w_geom_diag': w_geom_diag,
    }

    gen_step.zero_grad()
    gen_step.eval()
    # Clear side-channel so subsequent calls (next fid/seed) start clean.
    if hasattr(variant, '_ksd_state'):
        variant._ksd_state = None
    return out


def aggregate_per_subnet(records, fids, seeds, groups=SUBNET_GROUPS):
    """Inter-fn / intra-fn cosine per subnetwork group, KSD only."""
    out = {}
    for g in groups:
        all_vecs = {fid: [] for fid in fids}
        for fid in fids:
            for seed in range(seeds):
                r = records.get((fid, seed))
                if r is None or r.get('nan', True):
                    continue
                v = (r.get('grad_vecs_per_group') or {}).get(g)
                if v is not None:
                    all_vecs[fid].append(v)

        fn_mean_vecs = []
        for fid in fids:
            vs = all_vecs[fid]
            fn_mean_vecs.append(torch.stack(vs).mean(dim=0) if vs else None)

        if not any(v is not None for v in fn_mean_vecs):
            continue   # group not present (e.g. donor_selector absent)

        inter_M = cos_sim_matrix(fn_mean_vecs)
        n = len(fids)
        mask = ~torch.eye(n, dtype=torch.bool)
        off = inter_M[mask]
        finite = off[torch.isfinite(off)]

        intra_sims = []
        for fid in fids:
            vs = all_vecs[fid]
            if len(vs) < 2:
                continue
            pair_M = cos_sim_matrix(vs)
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    s = pair_M[i, j].item()
                    if np.isfinite(s):
                        intra_sims.append(s)

        out[g] = {
            'inter_fn_mean': float(finite.mean()) if len(finite) else float('nan'),
            'inter_fn_max': float(finite.max()) if len(finite) else float('nan'),
            'inter_fn_min': float(finite.min()) if len(finite) else float('nan'),
            'intra_fn_mean': float(np.mean(intra_sims)) if intra_sims else float('nan'),
            'matrix': inter_M.tolist(),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--n-gens', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--fids', type=int, nargs='+', default=list(range(1, 30)))
    parser.add_argument('--T-temp', type=float, default=1.0,
                        help='Temperature T for the Gibbs target p* ∝ exp(-f/T).')
    parser.add_argument('--use-grad-wrapper', action='store_true',
                        help='Wrap donor_selector + adaptive_fcr with ∇f-aware '
                             'wrappers (Tier 1 of the head injection plan). '
                             'Adds geometric bias on pbest channel + extends '
                             'F/CR head input by D+1 features. At init the '
                             'wrappers are bit-exact to baseline (zero w_geom '
                             'addition, zero-init new D+1 columns).')
    parser.add_argument('--ssl-warmstart-bb', type=Path, default=None,
                        help='Path to a SSL backbone-only checkpoint '
                             '(e.g. checkpoints/backbone/step_15000.pth). '
                             'When provided, after load_checkpoint the '
                             'backbone weights are overwritten by this '
                             'checkpoint while variant + surrogate stay '
                             'as in the main checkpoint (so architecture is '
                             'preserved). Used to measure on "SSL warmstart".')
    parser.add_argument('--output', type=str, required=True,
                        help='Output stem (writes .json + _grads.pt)')
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    print(f"[ksd] Loading: {args.checkpoint}", file=sys.stderr, flush=True)
    print(f"[ksd] device={device}, fids={len(args.fids)}, seeds={args.seeds}, "
          f"n_gens={args.n_gens}, T={args.T_temp}, "
          f"use_grad_wrapper={args.use_grad_wrapper}, "
          f"ssl_warmstart_bb={args.ssl_warmstart_bb}",
          file=sys.stderr, flush=True)

    gs, bb, var, _, config, step = load_checkpoint(args.checkpoint, device)

    if args.ssl_warmstart_bb is not None:
        ssl_ck = torch.load(args.ssl_warmstart_bb, map_location=device,
                            weights_only=False)
        bb_state = ssl_ck.get('backbone_state_dict')
        if bb_state is None:
            raise KeyError(
                f"--ssl-warmstart-bb checkpoint at {args.ssl_warmstart_bb} "
                f"has no 'backbone_state_dict' key. Found: {list(ssl_ck.keys())}")
        missing, unexpected = bb.load_state_dict(bb_state, strict=False)
        print(f"[ksd] SSL backbone overlay: "
              f"{len(missing)} missing, {len(unexpected)} unexpected keys",
              file=sys.stderr, flush=True)
        if missing:
            print(f"[ksd]   missing[:5]: {missing[:5]}", file=sys.stderr)
        if unexpected:
            print(f"[ksd]   unexpected[:5]: {unexpected[:5]}", file=sys.stderr)

    D = config['D'][0] if isinstance(config['D'], list) else config['D']

    if args.use_grad_wrapper:
        donor_wrap, fcr_wrap = _wrap_heads(bb, var, D)
        print(f"[ksd] Wrapped heads: donor=GeometricBiasDonorWrapper, "
              f"fcr=GradFeatureFCRWrapper (D={D})",
              file=sys.stderr, flush=True)

    records = {}
    fids_skipped = []
    total = len(args.fids) * args.seeds
    idx = 0
    sign_log = {'logged': False}    # one-time across the whole run
    for seed in range(args.seeds):
        for fid in args.fids:
            idx += 1
            print(f"[ksd] [{idx}/{total}] F{fid:02d} seed={seed}",
                  file=sys.stderr, flush=True)
            try:
                fn = CEC2017Torch(fid, D, device)
                rec = run_ksd_episode(gs, bb, var, fn, config, device,
                                       n_gens=args.n_gens,
                                       seed=seed * 1000 + fid,
                                       T_temp=args.T_temp,
                                       use_grad_wrapper=args.use_grad_wrapper,
                                       sign_coherence_log_state=sign_log)
                records[(fid, seed)] = rec
                if rec.get('nan'):
                    if fid not in fids_skipped:
                        fids_skipped.append(fid)
                    print(f"[ksd]   NaN: {rec.get('reason')}",
                          file=sys.stderr, flush=True)
            except Exception as e:
                print(f"[ksd]   FAILED: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                records[(fid, seed)] = {'nan': True, 'reason': f'exception: {e}'}
                if fid not in fids_skipped:
                    fids_skipped.append(fid)

    # axis_3 cos-sim per subnetwork.
    axis_3 = aggregate_per_subnet(records, args.fids, args.seeds)

    # axis_1 per-fn norm at bb.all (F09 dominance check).
    per_fn_norm = {}
    for fid in args.fids:
        norms = []
        for seed in range(args.seeds):
            r = records.get((fid, seed))
            if r is not None and not r.get('nan', True):
                norms.append(r.get('bb_grad_norm', 0.0))
        per_fn_norm[fid] = float(np.mean(norms)) if norms else 0.0
    axis_1 = {}
    if per_fn_norm:
        max_fid = max(per_fn_norm, key=per_fn_norm.get)
        max_norm = per_fn_norm[max_fid]
        others = [v for k, v in per_fn_norm.items() if k != max_fid]
        median = float(np.median(others)) if others else 0.0
        axis_1 = {
            'per_fn_norm': per_fn_norm,
            'max_fn': int(max_fid),
            'max_norm': float(max_norm),
            'median_others': median,
            'max_over_median': float(max_norm / max(median, 1e-12)),
        }

    # axis_5 term hierarchy — averaged across all (fid, seed) that didn't NaN.
    term_stack = []
    for r in records.values():
        if r is not None and not r.get('nan', True):
            term_stack.append(r.get('term_means', [0, 0, 0, 0]))
    axis_5 = {}
    if term_stack:
        ts = np.array(term_stack)
        axis_5 = {
            'term1_mean': float(ts[:, 0].mean()),
            'term2_mean': float(ts[:, 1].mean()),
            'term3_mean': float(ts[:, 2].mean()),
            'term4_mean': float(ts[:, 3].mean()),
            'abs_t1_over_t4': float(np.abs(ts[:, 0]).mean() /
                                     max(np.abs(ts[:, 3]).mean(), 1e-12)),
        }

    # axis_7 / axis_8 aggregation (only meaningful when --use-grad-wrapper).
    axis_7_vals = []
    axis_8_vals = []
    w_geom_vals = []
    w_geom_grads = []
    for r in records.values():
        if r is None or r.get('nan', True):
            continue
        ad = r.get('attn_diag') or {}
        if 'axis_7_attn_entropy_grad_corr_mean' in ad:
            axis_7_vals.append(ad['axis_7_attn_entropy_grad_corr_mean'])
        if 'axis_8_attn_descent_alignment_mean' in ad:
            axis_8_vals.append(ad['axis_8_attn_descent_alignment_mean'])
        wd = r.get('w_geom_diag') or {}
        if 'w_geom_value' in wd:
            w_geom_vals.append(wd['w_geom_value'])
            w_geom_grads.append(wd['w_geom_grad'])
    attn_summary = {}
    if axis_7_vals:
        attn_summary['axis_7_attn_entropy_grad_corr'] = {
            'mean': float(np.mean(axis_7_vals)),
            'min': float(np.min(axis_7_vals)),
            'max': float(np.max(axis_7_vals)),
            'n': len(axis_7_vals),
        }
    if axis_8_vals:
        attn_summary['axis_8_attn_descent_alignment'] = {
            'mean': float(np.mean(axis_8_vals)),
            'min': float(np.min(axis_8_vals)),
            'max': float(np.max(axis_8_vals)),
            'n': len(axis_8_vals),
        }
    w_geom_summary = {}
    if w_geom_vals:
        w_geom_summary = {
            'value_mean': float(np.mean(w_geom_vals)),
            'value_std': float(np.std(w_geom_vals)),
            'grad_abs_mean': float(np.mean(np.abs(w_geom_grads))),
            'grad_nonzero_fraction': float(np.mean(
                [abs(x) > 1e-12 for x in w_geom_grads])),
        }

    out_dict = {
        'ckpt': str(args.checkpoint),
        'step': step,
        'loss': 'ksd',
        'T_temp': args.T_temp,
        'n_gens': args.n_gens,
        'seeds': args.seeds,
        'fids': args.fids,
        'fids_skipped': fids_skipped,
        'use_grad_wrapper': args.use_grad_wrapper,
        'ssl_warmstart_bb': str(args.ssl_warmstart_bb) if args.ssl_warmstart_bb else None,
        'config_critical': {
            'D': D,
            'N': config.get('N'),
            'B': config.get('b_per_gpu'),
            'M': config.get('m_samples'),
            'budget_mult': config.get('budget_mult'),
            'operators': config.get('operators'),
            'gate_type': config.get('gate_type'),
        },
        'axis_1_F09_dominance': {'ksd': axis_1},
        'axis_3_cos_sim': {'ksd': axis_3},
        'axis_5_term_hierarchy': axis_5,
        'attn_diagnostics': attn_summary,
        'w_geom_summary': w_geom_summary,
    }

    out_json = args.output + '.json'
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as fh:
        json.dump(out_dict, fh, indent=2, default=str)
    print(f"[ksd] wrote {out_json}", file=sys.stderr, flush=True)

    # Raw gradient vectors → .pt for re-aggregation.
    raw = {}
    for (fid, seed), r in records.items():
        if r is None or r.get('nan', True):
            continue
        for g, v in (r.get('grad_vecs_per_group') or {}).items():
            if v is not None:
                raw[f'ksd_{g}_F{fid:02d}_s{seed}'] = v
    out_pt = args.output + '_grads.pt'
    torch.save(raw, out_pt)
    print(f"[ksd] wrote {out_pt} ({len(raw)} tensors)",
          file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
