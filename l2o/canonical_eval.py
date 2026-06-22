"""Canonical CEC2017 evaluation for L2O checkpoints.

CRITICAL: the eval loop must mirror the train loop's state tracking
(stagnation_counters / delta_fitnesses / contraction_rates) and pass the same
hyperparameters (gumbel_tau, M, k_neighbors) that were used at train time,
otherwise the model receives features/routing distributions it was never
trained on and eval numbers stall independently of the model's actual quality.
"""
import os

import torch

from encoder.operators.adaptive_fcr_cauchy import AdaptiveFCRCauchy
from encoder.opt_variant import _clamp_fitness
from encoder.operators.surrogate_selection import DEPLOYABLE_DEFAULT_SPEC
from l2o.schedules import (PopulationGenState, compute_lpsr_n_target,
                           compute_surrogate_m, gather_pop, lpsr_keep_indices)


def load_backbone_checkpoint(backbone, path, device):
    if not os.path.exists(path):
        return 0, set()
    ck = torch.load(path, map_location=device, weights_only=False)
    model_sd = backbone.state_dict()
    ckpt_sd = {k: v for k, v in ck['backbone_state_dict'].items()
               if k in model_sd and v.shape == model_sd[k].shape}
    skipped = set(ck['backbone_state_dict'].keys()) - set(ckpt_sd.keys())
    backbone.load_state_dict(ckpt_sd, strict=False)
    del ck
    return len(ckpt_sd), skipped


def _install_neural_overrides(variant):
    """Flip head attrs `donor_mode='lshade'`→`'neural'` and
    `fcr_mode='lshade'`→`'cauchy_neural'` for post-distillation eval.

    `donor_mode='lshade_masked'` (E12 inductive bias) is intentional and
    NOT flipped. Returns a list of (head, attr_name, prev_value) for restore.

    Exception-safe: any failure to set an attribute (e.g. property without
    setter, frozen module) triggers restoration of partial flips before
    re-raising. Validates that fcr_mode='lshade' heads carry an
    AdaptiveFCRCauchy adaptive_fcr — flipping to 'cauchy_neural' otherwise
    would silently break compute_params (no _mu_F_pred populated under
    fcr_mode='beta').
    """
    if variant is None:
        return []
    overrides = []
    try:
        for h in variant.modules():
            prev_donor = getattr(h, 'donor_mode', None)
            if prev_donor == 'lshade':
                h.donor_mode = 'neural'
                overrides.append((h, 'donor_mode', prev_donor))
            prev_fcr = getattr(h, 'fcr_mode', None)
            if prev_fcr == 'lshade':
                adaptive_fcr = getattr(h, 'adaptive_fcr', None)
                if not isinstance(adaptive_fcr, AdaptiveFCRCauchy):
                    raise RuntimeError(
                        f'_install_neural_overrides: head {type(h).__name__} '
                        f'has fcr_mode="lshade" but adaptive_fcr is '
                        f'{type(adaptive_fcr).__name__!r} '
                        f'(not AdaptiveFCRCauchy). Flipping to '
                        f'"cauchy_neural" would crash on missing _mu_F_pred. '
                        f'This indicates a misconfigured head — instantiate '
                        f'with fcr_mode="lshade" or "cauchy_neural".')
                h.fcr_mode = 'cauchy_neural'
                overrides.append((h, 'fcr_mode', prev_fcr))
    except Exception:
        _restore_overrides(overrides)
        raise
    return overrides


def _restore_overrides(overrides):
    for h, attr, prev in overrides:
        setattr(h, attr, prev)


def _snapshot_archive_state(gen_step):
    """Capture a deep-copy snapshot of gen_step's archive state.

    The archive is shared between train and eval (single GenerationStep
    instance); without snapshot/restore, eval's _enqueue_archive calls
    pollute training-era archive contents. Cheap clone vs. running an
    eval gen (~0.1ms on GPU for typical K=468, B<=32, D<=30).
    """
    return {
        'coords': gen_step.archive_coords.clone()
                  if gen_step.archive_coords is not None else None,
        'fitness': gen_step.archive_fitness.clone()
                   if gen_step.archive_fitness is not None else None,
        'mask': gen_step.archive_mask.clone()
                if gen_step.archive_mask is not None else None,
        'age': gen_step.archive_age.clone()
               if gen_step.archive_age is not None else None,
        'gen_counter': gen_step._archive_gen_counter,
        'has_entries': gen_step._archive_has_entries,
    }


def _restore_archive_state(gen_step, snap):
    """Reverse of _snapshot_archive_state. No-op if archive was unallocated."""
    if snap['coords'] is None:
        return
    gen_step.archive_coords.copy_(snap['coords'])
    gen_step.archive_fitness.copy_(snap['fitness'])
    gen_step.archive_mask.copy_(snap['mask'])
    gen_step.archive_age.copy_(snap['age'])
    gen_step._archive_gen_counter = snap['gen_counter']
    gen_step._archive_has_entries = snap['has_entries']


def run_canonical_eval(gen_step, all_fn_ids, D, N, B, MAX_FES, gru_W,
                       device, build_graph_fn,
                       gumbel_tau=1.0, m_samples=1, k_neighbors=8,
                       variant=None,
                       selection_spec=DEPLOYABLE_DEFAULT_SPEC,
                       surrogate_M_init=0, surrogate_m_final=0,
                       gate_target_frac=0.5,
                       lpsr_n=False, lpsr_n_min=4):
    """No-grad eval on raw CEC2017 functions. Returns per-fid gap_ratios.

    Mirrors the training generation loop exactly — only differences are
    torch.no_grad() and absence of backprop. Every forward-pass kwarg must
    be supplied by the caller to match its own train-time args (gumbel_tau,
    m_samples, k_neighbors, selection_spec, surrogate_M_init/m_final,
    gate_target_frac, lpsr_n/lpsr_n_min).

    `per_m_donors` (OptA) is a persistent attribute on the DE head set at
    training init; eval inherits it.

    With `lpsr_n=True`, the active population shrinks from `N` (init) to
    `lpsr_n_min` linearly with `cumulative_fes / MAX_FES`, mirroring
    train_distributed.py's LPSR-N block. The surrogate_M schedule is clamped
    to the active N to avoid topk(M>N).

    Eval-time L-SHADE → neural override: any head with `donor_mode='lshade'`
    is flipped to `'neural'` for the duration of the eval, and
    `fcr_mode='lshade'` to `'cauchy_neural'`. `donor_mode='lshade_masked'`
    (E12) is left unchanged.
    """
    from encoder.cec2017_torch import CEC2017Torch

    results = {}
    gen_step_backup_fn = gen_step.eval_fn
    overrides = _install_neural_overrides(variant)
    archive_snap = _snapshot_archive_state(gen_step)

    try:
        with torch.no_grad():
            for fid in all_fn_ids:
                fn = CEC2017Torch(fid, D, device)
                gen_step.eval_fn = fn
                if variant is not None and hasattr(variant, '_oracle_best_k'):
                    variant._oracle_best_k = None

                coords = (torch.rand(B, N, D, device=device) * 200 - 100).to(torch.float64)
                fitness = _clamp_fitness(fn(coords.reshape(-1, D)).reshape(B, N))

                # Match train's gap_init semantics: per-batch min, clamped, then mean
                gap_init = (fitness.min(dim=1).values - fn.f_optimal
                            ).clamp(min=1.0).mean().item()

                coords_ring = torch.zeros(B, gru_W, N, D, dtype=torch.float32, device=device)
                fitness_ring = torch.zeros(B, gru_W, N, dtype=torch.float32, device=device)

                # State tracking — shared helper, identical to train.
                pop_state = PopulationGenState(B, device)

                cumulative_fes = 0

                for gen in range(800):
                    if cumulative_fes >= MAX_FES:
                        break

                    # LPSR-N: shared helper with train and eval_e7d_parallel
                    # so the 3 implementations cannot drift.
                    if lpsr_n:
                        _N_target = compute_lpsr_n_target(
                            N, lpsr_n_min, cumulative_fes / MAX_FES)
                        if _N_target < coords.shape[1]:
                            _keep_idx = lpsr_keep_indices(fitness, _N_target)
                            coords = gather_pop(coords, _keep_idx, dim=1)
                            fitness = gather_pop(fitness, _keep_idx, dim=1)
                            coords_ring = gather_pop(coords_ring, _keep_idx, dim=2)
                            fitness_ring = gather_pop(fitness_ring, _keep_idx, dim=2)
                            pop_state.reset_baseline()

                    ri = gen % gru_W
                    coords_ring[:, ri] = coords.float()
                    fitness_ring[:, ri] = fitness.float()
                    n_valid = min(gen + 1, gru_W)
                    if gen < gru_W:
                        idx = list(range(gen + 1))
                    else:
                        start = (gen + 1) % gru_W
                        idx = [(start + i) % gru_W for i in range(gru_W)]
                    coords_hist = coords_ring[:, idx]
                    fitness_hist = fitness_ring[:, idx]
                    prev_c = coords_ring[:, (ri - 1) % gru_W].float() if gen > 0 else None
                    prev_f = fitness_ring[:, (ri - 1) % gru_W].float() if gen > 0 else None

                    pop_state.update(coords, fitness)

                    cache = build_graph_fn(
                        coords.float(), fitness.float(),
                        step_num=cumulative_fes, max_steps=MAX_FES, ndim=D,
                        k_neighbors=k_neighbors,
                        stagnation_counters=pop_state.stagnation_counters,
                        delta_fitnesses=pop_state.delta_fitnesses,
                        contraction_rates=pop_state.contraction_rates,
                        prev_coords=prev_c, prev_fitnesses=prev_f)
                    fes_frac = cumulative_fes / MAX_FES
                    N_active = coords.shape[1]
                    surr_M_now = compute_surrogate_m(
                        surrogate_M_init, surrogate_m_final, fes_frac, N_active)
                    # LPSR-N can drop N_active below the surrogate-M schedule;
                    # gumbel_topk(k=M_sel) over (B, N_active) parents requires
                    # M_sel ≤ N_active. Mirrors train_distributed.py:672/889.
                    surr_M_now = min(surr_M_now, N_active)
                    result = gen_step.run(
                        coords=coords, fitness=fitness, cache=cache,
                        f_optimal=fn.f_optimal, M=m_samples, gumbel_tau=gumbel_tau,
                        node_feat=cache.node_feat, global_feat=cache.global_feat,
                        coords_hist=coords_hist, fitness_hist=fitness_hist,
                        n_valid=n_valid, fes_frac=fes_frac,
                        gate_target_frac=gate_target_frac,
                        k_neighbors=k_neighbors,
                        step_num=cumulative_fes, max_steps=MAX_FES,
                        surrogate_M=surr_M_now,
                        selection_spec=selection_spec)
                    coords = result['new_coords']
                    fitness = result['new_fitness']
                    extras = result.get('extras', {})
                    _fes = extras.get('fes_used', float(N_active))
                    cumulative_fes += (_fes.item() if torch.is_tensor(_fes) else _fes)

                gap_final = (fitness.min(dim=1).values - fn.f_optimal).mean().item()
                results[fid] = round(gap_final / max(gap_init, 1e-12), 6)
    finally:
        gen_step.eval_fn = gen_step_backup_fn
        _restore_archive_state(gen_step, archive_snap)
        _restore_overrides(overrides)

    return results
