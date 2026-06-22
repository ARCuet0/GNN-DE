"""Parallel eval of E7d_lpsr_K1 best_eval.pth on full CEC2017 D=10.

Matches E7d training config:
  operators=k1, gate_type=surrogate, surrogate_M=50→5 (LPSR schedule),
  N=50, D=10, budget=1000*D=10000.

Usage:
  python eval_e7d_parallel.py --ckpt checkpoints/E7d_lpsr_K1/best_eval.pth \
      --seeds 5 --workers 4
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

# Timing-clean BLAS pinning. When --log-timing is in argv, force single-threaded
# BLAS *before* numpy/scipy/torch import so per-(fid,seed) wall_seconds is not
# polluted by thread-thrash (see finding_blas_oversubscription_cmaes_2026_05_28).
# Idempotent: only sets vars that aren't already set, so non-timing runs are
# untouched and a user-supplied OMP_NUM_THREADS=4 isn't clobbered when timing.
if '--log-timing' in sys.argv:
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(_v, '1')

multiprocessing.set_start_method('spawn', force=True)

# Per-worker module globals, populated by _worker_init once per worker process.
_WORKER_STATE = {}


def _build_args(archive_capacity=0, lb=-100.0, ub=100.0):
    """Reconstruct the SimpleNamespace with fields model_factory.build_model expects.

    `lb`/`ub` default to CEC2017 (the deployed regime). BBOB callers (e.g.
    eval_bbob_smoke) override post-construction via `gen_step.lb = ...`, but
    for callers that rely on the SimpleNamespace path the explicit kwarg
    avoids relying on the now-removed default in GenerationStep.
    """
    return SimpleNamespace(
        topology='embedding_knn',
        d_rnn=64, temporal_layers=2,
        gatv2_hidden=128, gatv2_layers=3, n_heads=8,
        k_neighbors=8, pooler='induced', n_induced=8,
        pool_dim=0, gru_window=16, dropout=0.0,
        operators='k1', gate_type='surrogate',
        surrogate_M=50, surrogate_m_final=5,
        gate_node_feat=0,
        archive_capacity=archive_capacity,
        lb=lb, ub=ub,
    )


def _build_traj_record(*, gen, fid, seed, cumulative_fes, N, D,
                       parent_coords, parent_fitness,
                       selected_coords, selected_fitness,
                       extras, fn, strip: bool, lb: float, ub: float):
    """Per-gen trajectory snapshot. Mirrors L-SHADE full_trajectories layout.

    `_parent_*` are gen-start (pre-selection) populations; `_selected_*` are
    post-selection winners that become parents for gen+1. When `strip=True`,
    skips heavy diagnostic fields (M=20 proposals + their fn-evaluated
    fitness, donor-attention matrices, F/CR realized samples, h_global,
    donor index sweeps) — needed only for downstream attention/donor
    analyses, not for state-space metrics.
    """
    import torch

    _pc_f = parent_coords.float()
    rec = {
        'gen': int(gen),
        'fid': int(fid), 'seed': int(seed),
        'cumulative_fes': int(cumulative_fes),
        'best_fit': float(parent_fitness.min().item()),
        'mean_fit': float(parent_fitness.mean().item()),
        # std() with default unbiased=True yields NaN at N=1 (Bessel correction
        # divides by N-1). LPSR-N can shrink the parent count to 1 near the end
        # of a run; unbiased=False keeps the scalar curve finite (just zero).
        'f_std': float(parent_fitness.std(unbiased=False).item()),
        'gap': float((parent_fitness.min() - fn.f_optimal).item()),
        'N': int(N),
        'coord_spread': float(_pc_f.std(dim=1, unbiased=False).mean().item()),
        'diameter': float((_pc_f.max(dim=1).values - _pc_f.min(dim=1).values).mean().item()),
    }
    if strip:
        # Lite/figure mode: keep only the scalar gap/spread curve. The raw
        # per-individual coord tensors are O(N*D) per gen and at N=1800/D=100
        # blow up to ~10 GB/chunk (blew the 2 TiB quota on 2026-05-27). The
        # convergence figure needs only `gap` vs `cumulative_fes`.
        return rec
    rec['_parent_coords'] = parent_coords.detach().to('cpu', torch.float32)
    rec['_parent_fitness'] = parent_fitness.detach().to('cpu', torch.float32)
    rec['_selected_coords'] = selected_coords.detach().to('cpu', torch.float32)
    rec['_selected_fitness'] = selected_fitness.detach().to('cpu', torch.float32)

    dk = extras.get('deltas_k_live')
    if dk is not None:
        dk_s = dk.squeeze(3)
        proposals = (parent_coords.unsqueeze(0) + dk_s).clamp(lb, ub)
        M = proposals.shape[0]
        prop_flat = proposals.reshape(M * N, D).to(torch.float64)
        prop_fit = fn(prop_flat).reshape(M, 1, N).float()
        rec['_proposals'] = proposals.detach().to('cpu', torch.float32)
        rec['_proposal_fitness'] = prop_fit.detach().to('cpu', torch.float32)
    saf = extras.get('surr_all_fit')
    if saf is not None:
        rec['_surr_all_fit'] = saf.detach().to('cpu', torch.float32)
    ss = extras.get('surr_scores')
    if ss is not None:
        rec['_surr_scores'] = ss.detach().to('cpu', torch.float32)
    for k in ('_realized_F', '_realized_CR', '_A_pbest', '_A_r1', '_A_r2'):
        v = extras.get(k)
        if v is not None:
            rec[k] = v.detach().to('cpu', torch.float32)
    for k in ('_pbest_idx_m', '_r1_idx_m', '_r2_idx_m'):
        v = extras.get(k)
        if v is not None:
            rec[k] = v.detach().to('cpu', torch.int32)
    hg = extras.get('h_global')
    if hg is not None:
        rec['_h_global'] = hg.detach().to('cpu', torch.float32)
    ha = extras.get('h_aug_live')
    if ha is not None:
        rec['_h_aug'] = ha.detach().to('cpu', torch.float32)
    return rec


def _load_disen_heads(ckpt_path, device):
    """Load disen MLP heads (h_explor, h_exploit) from a ckpt's
    `disentangle_heads_state_dict`. Returns dict {h_explor, h_exploit}."""
    import torch
    import torch.nn as nn
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    head_sd = sd.get('disentangle_heads_state_dict')
    if head_sd is None:
        raise ValueError(f'No disentangle_heads_state_dict in {ckpt_path}')
    def make_head():
        return nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1)).to(device).eval()
    def _strip(prefix):
        return {k[len(prefix):]: v for k, v in head_sd.items() if k.startswith(prefix)}
    h_explor = make_head()
    h_exploit = make_head()
    h_explor.load_state_dict(_strip('h_explor.'))
    h_exploit.load_state_dict(_strip('h_exploit.'))
    h_explor.requires_grad_(False)
    h_exploit.requires_grad_(False)
    return {'h_explor': h_explor, 'h_exploit': h_exploit}


def _load_jepa_predictor(ckpt_path, device):
    """Load a trained JEPA predictor from analysis/jepa_predictor_2026_05_05.py.
    Returns the network in eval mode + (h_dim, action_dim) for input prep."""
    import torch
    import sys
    from pathlib import Path
    _analysis_dir = str(Path(__file__).resolve().parent / 'analysis')
    if _analysis_dir not in sys.path:
        sys.path.insert(0, _analysis_dir)
    from jepa_predictor_2026_05_05 import JEPAPredictor
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    h_dim = ck.get('h_dim', 128)
    action_dim = ck.get('action_dim', 12)
    net = JEPAPredictor(h_dim=h_dim, action_dim=action_dim).to(device).eval()
    net.load_state_dict(ck['predictor_state_dict'])
    for p in net.parameters(): p.requires_grad_(False)
    return {'predictor': net, 'h_dim': h_dim, 'action_dim': action_dim}


def _worker_init(ckpt_path, device, archive_capacity=0, archive_evict='fifo',
                 disen_heads_ckpt=None, jepa_ckpt=None, donor_chunk=None):
    """Build the model ONCE per worker. Reused across all jobs on this worker."""
    import torch
    import torch.multiprocessing as _tmp
    # Avoid FD-exhaustion EMFILE under heavy traj logging — switch shared-tensor
    # IPC to file-backed pages.
    try:
        _tmp.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    from l2o.model_factory import build_model

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = _build_args(archive_capacity=archive_capacity)
    # Match the eval architecture to the trained checkpoint. `_build_args`
    # hardcodes the deployed config (sparse_gatv2, k=8); the set-attention
    # ablation and the k-sweep differ in exactly these two fields, so a
    # hardcoded build would silently load the wrong architecture under
    # strict=False (random message-passing layers / mismatched graph degree).
    _cfg = ck.get('config')
    if _cfg is not None:
        _cfgd = vars(_cfg) if hasattr(_cfg, '__dict__') else _cfg
        for _f in ('backbone_type', 'k_neighbors'):
            if _cfgd.get(_f) is not None:
                setattr(args, _f, _cfgd[_f])
    backbone, variant, gen_step = build_model(args, device)
    # Archive eviction policy (only meaningful if archive_capacity > 0). Set
    # AFTER build because GenerationStep reads it via getattr fallback to 'fifo'.
    if archive_capacity > 0:
        gen_step.archive_evict = archive_evict

    _bb_load = backbone.load_state_dict(ck['backbone_state_dict'], strict=False)
    # Guard against silent architecture mismatch (Bug Prevention #5): the only
    # acceptable "missing" keys are buffers/None; a message-passing layer left
    # at init would mean the wrong backbone_type was built.
    _suspect = [k for k in _bb_load.missing_keys if 'layers.' in k]
    if _suspect:
        raise RuntimeError(
            f"backbone load left {len(_suspect)} message-passing keys "
            f"uninitialized (e.g. {_suspect[:3]}); backbone_type mismatch vs "
            f"checkpoint config {getattr(args, 'backbone_type', '?')!r}")
    variant.load_state_dict(ck['variant_state_dict'], strict=False)
    if gen_step.surrogate is not None and 'surrogate_state_dict' in ck:
        gen_step.surrogate.load_state_dict(ck['surrogate_state_dict'], strict=False)

    backbone.eval(); variant.eval(); gen_step.eval()
    if hasattr(variant, 'eval_all_select_active'):
        variant.eval_all_select_active = False
    if hasattr(gen_step, 'pool_selection'):
        gen_step.pool_selection = False
    # Force deltas_k_live retention so trajectory logging can recover the
    # M=20 proposals per parent. No-op if --log-trajectories is off.
    gen_step.keep_deltas_k_live = True

    # Inject disen heads from a separate ckpt for q_*_1pp selectors.
    if disen_heads_ckpt is not None:
        gen_step.disen_heads = _load_disen_heads(disen_heads_ckpt, device)

    # Inject JEPA predictor for jepa_*_1pp selectors. Predicts h_proposal from
    # (h_parent, action). Used in conjunction with disen_heads for value scoring
    # on PREDICTED h instead of REAL h_aug.
    if jepa_ckpt is not None:
        gen_step.jepa_predictor = _load_jepa_predictor(jepa_ckpt, device)

    # donor_selector chunked forward — bounds peak memory at large N (D≥40 LPSR-matched).
    if donor_chunk is not None:
        from encoder.operators.donor_selection import DonorSelectionGATv2
        for m in backbone.modules():
            if isinstance(m, DonorSelectionGATv2):
                m.query_chunk_size = donor_chunk

    _WORKER_STATE['backbone'] = backbone
    _WORKER_STATE['variant'] = variant
    _WORKER_STATE['gen_step'] = gen_step
    _WORKER_STATE['device'] = device


def _apply_force_fcr(variant, f_now, cr_now):
    """Set the eval-time F/CR injection attrs on every head, unconditionally.

    BatchedDiffAttDE only READS _force_F_attr/_force_CR_attr via getattr
    (de_heads.py ~809) and never initializes them, so the historical
    `if hasattr(...)` gate made --force-f-schedule a silent no-op (discovered
    2026-06-12: fcr_static produced 1479/1479 pairs bit-identical to control).
    """
    if not hasattr(variant, 'heads'):
        return
    for h in variant.heads:
        h._force_F_attr = f_now
        h._force_CR_attr = cr_now


def _apply_fcr_shade(variant, static=False, memory=None):
    """Set the eval-time SHADE F/CR lesion attrs on every head (mirrors
    _apply_force_fcr). `static` enables the no-adaptation Cauchy/Normal draw;
    `memory` is a shared LShadeMemory for the adaptive draw. Both default off so
    a no-arg call resets. For the adaptive lesion the SAME memory instance must
    also be set on gen_step (the post-selection update reads it there) — see the
    dispatch call site.
    """
    if not hasattr(variant, 'heads'):
        return
    for h in variant.heads:
        h._fcr_shade_static_attr = static
        h._fcr_shade_memory_attr = memory


def _apply_per_m_donors(variant, on: bool):
    """Toggle Option A on all DE-like heads that support it."""
    if not hasattr(variant, 'heads'):
        return
    for h in variant.heads:
        if hasattr(h, 'per_m_donors'):
            h.per_m_donors = on


def _apply_use_ste(variant, on: bool):
    """Toggle STE crossover on all DE-like heads that support it.
    Default in operators is True (post-d2bb237). Setting False reverts to
    pre-d2bb237 soft sigmoid forward — eval-time A/B for STE attribution.
    """
    if not hasattr(variant, 'heads'):
        return
    for h in variant.heads:
        if hasattr(h, 'use_ste'):
            h.use_ste = on


def eval_single(fid, seed, D, N, budget, max_gens,
                selection_spec='topk', M_var=1, per_m_donors=False,
                use_ste=True, log_trajectory=False, log_traj_strip=False,
                greedy_1to1=False, lpsr_n=False, lpsr_n_min=4,
                force_f_schedule='none', log_timing=False, lesion='none'):
    import torch
    _t_wall_start = time.perf_counter() if log_timing else None
    from l2o.schedules import (PopulationGenState, compute_lpsr_n_target,
                               compute_surrogate_m, gather_pop,
                               lpsr_keep_indices)
    from encoder.opt_variant import _clamp_fitness
    from encoder.cec2017_torch import CEC2017Torch
    from encoder.similarity_graph_gpu import build_sparse_graphs_gpu

    backbone = _WORKER_STATE['backbone']
    variant = _WORKER_STATE['variant']
    gen_step = _WORKER_STATE['gen_step']
    device = _WORKER_STATE['device']
    args = _build_args()

    # Reset any per-fid stateful caches on variant.
    if hasattr(variant, '_oracle_best_k'):
        variant._oracle_best_k = None

    _apply_per_m_donors(variant, per_m_donors)
    _apply_use_ste(variant, use_ste)

    # Inference-time lesion (paper §4.4). donor_uniform* patches the donor head
    # once per job; it persists across all generations because gen_step holds a
    # single backbone instance. edge_* / temporal_* / nodefeat_* are applied
    # per-gen in the loop below. The patch is on this worker's private model copy.
    _DONOR_LESION_ROLES = {
        'donor_uniform': None,          # all roles
        'donor_uniform_pbest': (0,),    # pbest only
        'donor_uniform_r1r2': (1, 2),   # difference-vector roles only
        # Candidate improved deployment recipe: the two uniformly non-negative
        # lesions combined (pbest uniform + phase conditioning frozen).
        'combo_pbest_global': (0,),
    }
    if lesion in _DONOR_LESION_ROLES:
        from analysis.lesion_ops import wrap_uniform_donor
        sel = gen_step.backbone.backbone.donor_selector
        if sel is None:
            raise RuntimeError('donor lesion requires a donor head')
        sel.forward = wrap_uniform_donor(sel.forward,
                                         roles=_DONOR_LESION_ROLES[lesion])

    # P5 (Non-Learned Donor): replace the learned donor logits by a zero-param
    # fitness-rank heuristic (r1/pbest->best region, r2->worst region, query-
    # independent). Tests whether the learned donor head adds value beyond argsort.
    if lesion == 'nld':
        from analysis.lesion_ops import wrap_nld_donor
        sel = gen_step.backbone.backbone.donor_selector
        if sel is None:
            raise RuntimeError('nld lesion requires a donor head')
        sel.forward = wrap_nld_donor(sel.forward)

    # Action 2: F-schedule override. Stages from LSHADE measured in this
    # session (FES_frac -> F): see analysis report. We test two variants:
    #   'lshade_clamped' : within Beta head bound [0.1, 0.9]
    #   'lshade_full'    : full LSHADE range incl. F=1.35 mid-stage
    # Each schedule is a dict with optional F and CR curves. None = leave to head.
    F_SCHEDULES = {
        'none':           {'F': None, 'CR': None},
        'lshade_clamped': {'F': [(0.10,0.52),(0.20,0.87),(0.40,0.90),(0.60,0.70),(0.85,0.57),(1.01,0.61)], 'CR': None},
        'lshade_full':    {'F': [(0.10,0.52),(0.20,0.87),(0.40,1.35),(0.60,0.70),(0.85,0.57),(1.01,0.61)], 'CR': None},
        # Action 2.6: F clamped to 0.95 (slightly above bound, well above used range)
        'lshade_F095':    {'F': [(0.10,0.52),(0.20,0.87),(0.40,0.95),(0.60,0.70),(0.85,0.57),(1.01,0.61)], 'CR': None},
        # Action 2.7: CR LSHADE schedule, F free
        'lshade_CR_only': {'F': None, 'CR': [(0.10,0.49),(0.20,0.55),(0.40,0.63),(0.60,0.75),(0.85,0.91),(1.01,0.73)]},
        # Lesion fcr_static: freeze F/CR at the deployed head's measured flat
        # output (F2 trace 2026-06-12: F~0.70, CR~0.45). Tests whether the Beta
        # head's state-dependence carries any load at inference.
        'const_deployed': {'F': [(1.01, 0.70)], 'CR': [(1.01, 0.45)]},
    }
    # Lesion fcr_static rides the existing F/CR-injection mechanism.
    if lesion == 'fcr_static':
        force_f_schedule = 'const_deployed'

    if force_f_schedule not in F_SCHEDULES:
        raise ValueError(f'unknown force_f_schedule: {force_f_schedule}')
    sched = F_SCHEDULES[force_f_schedule]
    F_schedule = sched['F']
    CR_schedule = sched['CR']

    def _val_at(curve, fes_frac):
        if curve is None: return None
        for upper, val in curve:
            if fes_frac < upper: return val
        return curve[-1][1]

    # Reset head attributes so a previous run doesn't leak.
    _apply_force_fcr(variant, None, None)
    # fcr_shade lesions (eval-only): keep the learned donor selection intact and
    # swap ONLY the F/CR draw for a canonical SHADE one (static = Cauchy/Normal
    # around 0.5, no adaptation; adaptive = full LShadeMemory success-history).
    # Replaces the fcr_static strawman, whose constant F/CR also kills all spread.
    _apply_fcr_shade(variant)            # reset (persistent-worker hygiene)
    gen_step._fcr_shade_memory = None
    if lesion in ('fcr_shade_static', 'fcr_shade_adaptive') and force_f_schedule != 'none':
        raise ValueError(f'{lesion} is mutually exclusive with --force-f-schedule='
                         f'{force_f_schedule} (both override F/CR; the force path '
                         'runs after the shade draw and would silently win)')
    if lesion == 'fcr_shade_static':
        _apply_fcr_shade(variant, static=True)
    elif lesion == 'fcr_shade_adaptive':
        from encoder.lshade_memory import LShadeMemory
        _shade_mem = LShadeMemory(B=1, H=6, device=device)
        _apply_fcr_shade(variant, memory=_shade_mem)   # same instance set below
        gen_step._fcr_shade_memory = _shade_mem

    from cec2017.reference.torch_cec2017 import make_benchmark_fn
    fn = make_benchmark_fn(fid, D, device)  # TERSQ_BENCHMARK env: legacy|official
    gen_step.eval_fn = fn

    gru_W = 16
    torch.manual_seed(seed)

    N_init = N  # palanca 1: original N for the LPSR linear schedule
    coords = (torch.rand(1, N, D, device=device) * 200 - 100).to(torch.float64)
    fitness = _clamp_fitness(fn(coords.reshape(-1, D)).reshape(1, N))
    gap_init = max((fitness.min(dim=1).values - fn.f_optimal).item(), 1.0)

    coords_ring = torch.zeros(1, gru_W, N, D, dtype=torch.float32, device=device)
    fitness_ring = torch.zeros(1, gru_W, N, dtype=torch.float32, device=device)

    pop_state = PopulationGenState(B=1, device=device)

    cumulative_fes = N
    best_gap = (fitness.min() - fn.f_optimal).item()
    gen_last = 0

    traj = [] if log_trajectory else None

    with torch.no_grad():
        for gen in range(max_gens):
            if cumulative_fes >= budget:
                break
            gen_last = gen

            # Palanca 1 (LPSR over N). Shared helper with train and
            # canonical_eval — same per-batch gather so this path is correct
            # for any B (the prior fancy-index `coords[:, keep_idx]` was only
            # safe under B=1).
            if lpsr_n:
                N_target = compute_lpsr_n_target(
                    N_init, lpsr_n_min, cumulative_fes / budget)
                if N_target < coords.shape[1]:
                    keep_idx = lpsr_keep_indices(fitness, N_target)
                    coords = gather_pop(coords, keep_idx, dim=1)
                    fitness = gather_pop(fitness, keep_idx, dim=1)
                    coords_ring = gather_pop(coords_ring, keep_idx, dim=2)
                    fitness_ring = gather_pop(fitness_ring, keep_idx, dim=2)
                    pop_state.reset_baseline()
                    N = N_target

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

            cache = build_sparse_graphs_gpu(
                coords.float(), fitness.float(),
                step_num=cumulative_fes, max_steps=budget, ndim=D,
                k_neighbors=8,
                stagnation_counters=pop_state.stagnation_counters,
                delta_fitnesses=pop_state.delta_fitnesses,
                contraction_rates=pop_state.contraction_rates,
                prev_coords=prev_c, prev_fitnesses=prev_f)

            # Edge-feature lesion (paper §4.4): replace the k-NN edge features
            # by their per-feature mean (edge_mean) or by zeros (edge_zero).
            if lesion in ('edge_mean', 'edge_zero', 'combo_edge_temporal'):
                from analysis.lesion_ops import lesion_edge_feat
                cache.edge_feat = lesion_edge_feat(
                    cache.edge_feat,
                    'edge_mean' if lesion == 'combo_edge_temporal' else lesion)

            # Node-feature lineage lesion: reset history channels 5-7 to the
            # gen-0 in-distribution defaults; current-state channels intact.
            if lesion == 'nodefeat_history':
                from analysis.lesion_ops import lesion_node_feat_history
                cache.node_feat = lesion_node_feat_history(cache.node_feat)

            # Node-feature spatial-state lesion: mean-impute channels 1-4.
            if lesion == 'nodefeat_state':
                from analysis.lesion_ops import lesion_node_feat_state
                cache.node_feat = lesion_node_feat_state(cache.node_feat)

            # Node-feature rank lesion: zero channel 0 (couples to donor bias,
            # documented in lesion_ops).
            if lesion == 'nodefeat_rank':
                from analysis.lesion_ops import lesion_node_feat_rank
                cache.node_feat = lesion_node_feat_rank(cache.node_feat)

            # Global-feature phase lesion: freeze global_feat at its gen-0
            # vector so the network's phase/budget conditioning never advances.
            if lesion in ('global_static', 'combo_pbest_global'):
                if gen == 0:
                    _global_frozen = cache.global_feat.clone()
                cache.global_feat = _global_frozen

            surr_m_now = compute_surrogate_m(
                args.surrogate_M, args.surrogate_m_final,
                cumulative_fes / budget, N)
            # Under LPSR-N, the parent count can drop below the LPSR-M_sel
            # schedule. Selector topk requires M_sel <= current N.
            surr_m_now = min(surr_m_now, N)

            # Capture parent coords BEFORE selection so trajectory logging can
            # reconstruct proposals as parents + deltas_k_live (the deltas in
            # extras are relative to the parents at gen-start, not the post-
            # selection coords). Under lite/strip we only read scalar summaries
            # off these tensors and never store them, so a zero-copy detach()
            # view is enough (gen_step.run returns a fresh tensor, not in-place).
            if log_trajectory:
                if log_traj_strip:
                    parent_coords  = coords.detach()
                    parent_fitness = fitness.detach()
                else:
                    parent_coords  = coords.detach().clone()
                    parent_fitness = fitness.detach().clone()
            else:
                parent_coords = None
                parent_fitness = None

            # Action 2: forced F/CR schedule injection per-gen. Unconditional
            # set — the old hasattr gate was a silent no-op (see _apply_force_fcr).
            fes_frac_now = cumulative_fes / budget
            _apply_force_fcr(variant,
                             _val_at(F_schedule, fes_frac_now),
                             _val_at(CR_schedule, fes_frac_now))

            # Temporal-window lesions (paper §4.4). NOTE the semantics:
            #  - temporal_single: n_valid=1 over the FULL window. Because the
            #    encoder slices features[:n_valid] of an oldest-first window,
            #    this feeds the OLDEST in-window snapshot (up to W-1 gens
            #    stale) — a stale-snapshot lesion. Kept for provenance.
            #  - temporal_current: slices the window to the last timestep so
            #    the encoder sees the true CURRENT snapshot (clean no-history
            #    lesion).
            n_valid_eff = (1 if lesion in ('temporal_single',
                                           'combo_edge_temporal')
                           else n_valid)
            if lesion == 'temporal_current':
                from analysis.lesion_ops import lesion_hist_current
                coords_hist, fitness_hist = lesion_hist_current(
                    coords_hist, fitness_hist)
                n_valid_eff = 1

            # Temporal fitness-static lesion: freeze fitness history at the
            # current snapshot (coords trajectory intact) — isolates the fitness
            # pathway of the temporal encoder from its coordinate pathway.
            if lesion == 'temporal_fitness_static':
                from analysis.lesion_ops import lesion_fitness_hist_static
                fitness_hist = lesion_fitness_hist_static(fitness_hist)

            result = gen_step.run(
                coords=coords, fitness=fitness, cache=cache,
                f_optimal=fn.f_optimal, M=M_var, gumbel_tau=1.0,
                node_feat=cache.node_feat, global_feat=cache.global_feat,
                coords_hist=coords_hist, fitness_hist=fitness_hist,
                n_valid=n_valid_eff, fes_frac=cumulative_fes / budget,
                surrogate_M=surr_m_now,
                selection_spec=selection_spec,
                greedy_1to1=greedy_1to1)

            coords = result['new_coords'].detach()
            fitness = result['new_fitness'].detach()
            extras = result.get('extras', {})
            _fes = extras.get('fes_used', float(N))
            cumulative_fes += (_fes.item() if hasattr(_fes, 'item') else _fes)

            gap = (fitness.min() - fn.f_optimal).item()
            if gap < best_gap:
                best_gap = gap

            if log_trajectory:
                traj.append(_build_traj_record(
                    gen=gen, fid=fid, seed=seed, cumulative_fes=cumulative_fes,
                    N=N, D=D,
                    parent_coords=parent_coords, parent_fitness=parent_fitness,
                    selected_coords=coords, selected_fitness=fitness,
                    extras=extras, fn=fn, strip=log_traj_strip,
                    lb=gen_step.lb, ub=gen_step.ub,
                ))

    out = {
        'fid': fid, 'seed': seed,
        'gap_init': gap_init, 'gap_final': best_gap,
        'gap_ratio': best_gap / gap_init,
        'fes': cumulative_fes, 'gens': gen_last + 1,
        'f_optimal': float(fn.f_optimal),
    }
    if log_timing:
        # Wall time spans the full per-(fid,seed) eval (init pop fitness +
        # generation loop). Excludes process-launch / worker-init costs (those
        # are amortized across all (fid,seed) jobs on a worker).
        wall_seconds = time.perf_counter() - _t_wall_start
        out['wall_seconds'] = wall_seconds
        # FES-normalized cost. Use max(cumulative_fes,1) to avoid div-by-zero
        # if a run aborts before even the init eval is charged.
        out['seconds_per_fes'] = wall_seconds / max(cumulative_fes, 1)
    if log_trajectory:
        out['_traj'] = traj
    import torch as _t, gc as _gc
    try:
        del coords, fitness, parent_coords, parent_fitness, result, extras
    except NameError:
        pass
    _gc.collect()
    if _t.cuda.is_available():
        _t.cuda.empty_cache()
        _t.cuda.ipc_collect()
    return out


def _build_jobs(fids, n_seeds, seed_offset=0):
    """Generate (fid, seed) tuples.

    seed_offset shifts the seed index by `seed_offset * n_seeds` so distinct
    offsets produce disjoint seed sets per fid. seed_offset=0 reproduces the
    historical formula `42 + si*1000 + fid` exactly (back-compat with all
    previously saved eval JSONs).
    """
    return [(fid, 42 + (si + seed_offset * n_seeds) * 1000 + fid)
            for fid in fids for si in range(n_seeds)]


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--seed-offset', type=int, default=0,
                   help='Shift seed indices by offset*seeds for disjoint '
                        'parallel runs. Default 0 reproduces historical '
                        'seeds. Used by sbatch arrays to scale seeds/fid.')
    p.add_argument('--D', type=int, default=10)
    p.add_argument('--N', type=int, default=50)
    p.add_argument('--budget-mult', type=int, default=1000)
    p.add_argument('--max-gens', type=int, default=2000)
    p.add_argument('--device', default='cuda')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', default='eval_results/E7d_full.json')
    p.add_argument('--fids', default='all')
    p.add_argument('--selection', default='topk',
                   help='selection spec: topk | uniform | exp:LAM | weibull:K:LAM | '
                        'power:ALPHA | random_1pp | oracle_1pp')
    p.add_argument('--M-var', type=int, default=1,
                   help='number of displacement samples per generation (M in gen_step.run)')
    p.add_argument('--per-m-donors', action='store_true',
                   help='Option A: resample donor triple (pbest/r1/r2) per M sample '
                        'instead of sharing one triple per parent')
    p.add_argument('--ste-crossover', default=True,
                   action=argparse.BooleanOptionalAction,
                   help='Use STE crossover (forward hard {0,1}, backward soft '
                        'sigmoid grad). Introduced d2bb237. --no-ste-crossover '
                        'reverts to pre-d2bb237 soft sigmoid forward, used to '
                        'evaluate pre-STE checkpoints under their training regime.')
    p.add_argument('--log-trajectories', type=str, default='',
                   help='If set, path to .pt file where per-gen trajectory '
                        'records are dumped (mirrors L-SHADE full_trajectories '
                        'layout). Captures pop coords/fitness, M=20 proposals '
                        'with their TRUE fitness (extra fn evals NOT charged '
                        'to budget — diagnostic only), surrogate scores. Slow.')
    p.add_argument('--log-trajectories-lite', action='store_true',
                   help='Strip heavy diagnostic fields (M=20 proposals + their '
                        'fn-evaluated fitness, donor-attention matrices, F/CR '
                        'realized samples, h_global, donor index sweeps) from '
                        'per-gen records. Keeps only parent/selected coords + '
                        'fitness + scalar metrics. ~40× smaller dump and skips '
                        'the diagnostic prop_fit fn-eval. Use for state-space '
                        'metrics (Phase 0.5 battery) where heavy fields are '
                        'not consumed.')
    p.add_argument('--greedy-1to1', action='store_true',
                   help='Palanca 2: bypass selector + pool topk; each parent gets '
                        '1 random proposal and competes slot-wise (LSHADE-style '
                        'monotone). Charges N FES per gen.')
    p.add_argument('--lpsr-N', dest='lpsr_n', action='store_true',
                   help='Palanca 1: Linear Population Size Reduction over N. '
                        'N(t) = round(N_init - (N_init - N_min)*fes_frac). '
                        'Default N_min=4 (LSHADE parity).')
    p.add_argument('--lpsr-N-min', dest='lpsr_n_min', type=int, default=4,
                   help='Minimum N at end of budget under --lpsr-N. Default 4.')
    p.add_argument('--force-f-schedule', dest='force_f_schedule', default='none',
                   choices=['none', 'lshade_clamped', 'lshade_full',
                            'lshade_F095', 'lshade_CR_only'],
                   help='Action 2: bypass Beta head and inject F/CR per fes_frac.')
    p.add_argument('--lesion', default='none',
                   choices=['none', 'edge_mean', 'edge_zero',
                            'temporal_single', 'temporal_current',
                            'donor_uniform',
                            'combo_edge_temporal', 'temporal_fitness_static',
                            'nodefeat_history', 'nodefeat_state',
                            'nodefeat_rank', 'global_static',
                            'donor_uniform_pbest', 'donor_uniform_r1r2',
                            'fcr_static', 'fcr_shade_static',
                            'fcr_shade_adaptive', 'combo_pbest_global',
                            'nld'],
                   help='Inference-time component lesion on the deployed ckpt '
                        '(paper §4.4): edge_mean/edge_zero knock out the k-NN '
                        'edge features, temporal_single collapses the trajectory '
                        'window to the current snapshot (n_valid=1), '
                        'donor_uniform replaces donor logits by uniform '
                        'selection. Second wave: combo_edge_temporal = edge_mean '
                        '+ temporal_single (distributed-redundancy test), '
                        'temporal_fitness_static freezes only the fitness '
                        'history (coords trajectory intact), nodefeat_history '
                        'resets lineage channels 5-7 to gen-0 defaults, '
                        'donor_uniform_pbest / _r1r2 uniformize only those '
                        'roles. No retraining: zero training-seed confound.')
    p.add_argument('--archive-capacity', type=int, default=0,
                   help='External per-batch FIFO archive of discarded parents '
                        'for graph-native augmented donor pool. Default 0 = '
                        'disabled. Match training-time value for in-distribution '
                        'eval (e.g. arms trained with --archive-capacity 78 '
                        'should be evaluated with archive 78).')
    p.add_argument('--archive-evict', choices=['fifo', 'random'], default='fifo',
                   help='Archive eviction policy when full. Match training.')
    p.add_argument('--disen-heads-ckpt', default=None,
                   help='Path to ckpt containing disentangle_heads_state_dict. '
                        'When provided, loads h_explor and h_exploit heads, '
                        'attaches them to gen_step, and enables q_*_1pp selectors.')
    p.add_argument('--jepa-ckpt', default=None,
                   help='Path to JEPA predictor ckpt (predictor_state_dict + '
                        'h_dim + action_dim). When provided alongside '
                        '--disen-heads-ckpt, enables jepa_*_1pp selectors that '
                        'predict h via JEPA from (h_parent, action) and score '
                        'with disen heads. Test 3 of γ rollout.')
    p.add_argument('--donor-chunk', type=int, default=None,
                   help='Chunk size for donor_selector forward; bounds peak '
                        'memory at large N (D≥40 LPSR-matched). None = monolithic.')
    p.add_argument('--log-timing', action='store_true',
                   help='Emit per-(fid,seed) wall-clock timings into each '
                        'result JSON')
    return p


def main():
    p = _build_parser()
    args = p.parse_args()

    import torch.multiprocessing as _tmp
    try:
        _tmp.set_sharing_strategy('file_system')
    except RuntimeError:
        pass

    from encoder.cec2017_torch import get_all_func_ids
    if args.fids == 'all':
        fids = sorted(get_all_func_ids(args.D))
    else:
        fids = [int(x) for x in args.fids.split(',')]

    budget = args.budget_mult * args.D
    jobs = _build_jobs(fids, args.seeds, seed_offset=args.seed_offset)

    print(f'Eval {args.ckpt}')
    print(f'  {len(fids)} funcs × {args.seeds} seeds (offset={args.seed_offset}) '
          f'= {len(jobs)} trajectories')
    print(f'  budget={budget} FES, N={args.N}, D={args.D}, max_gens={args.max_gens}')
    print(f'  selection={args.selection!r}  M_var={args.M_var}  per_m_donors={args.per_m_donors}  ste_crossover={args.ste_crossover}  lesion={args.lesion!r}')
    print(f'  greedy_1to1={args.greedy_1to1}  lpsr_n={args.lpsr_n}  lpsr_n_min={args.lpsr_n_min}')
    print(f'  archive_capacity={args.archive_capacity}  archive_evict={args.archive_evict!r}')
    print(f'  workers={args.workers}, device={args.device}')
    print()

    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(args.ckpt, args.device, args.archive_capacity, args.archive_evict,
                  args.disen_heads_ckpt, args.jepa_ckpt, args.donor_chunk),
    ) as pool:
        futures = {
            pool.submit(eval_single, fid, seed,
                        args.D, args.N, budget, args.max_gens,
                        args.selection, args.M_var,
                        args.per_m_donors,
                        args.ste_crossover,
                        bool(args.log_trajectories),
                        args.log_trajectories_lite,
                        args.greedy_1to1, args.lpsr_n, args.lpsr_n_min,
                        args.force_f_schedule,
                        args.log_timing, args.lesion): (fid, seed)
            for fid, seed in jobs
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
                results.append(r)
                el = time.perf_counter() - t0
                print(f'  [{done:3d}/{len(jobs)}] F{r["fid"]:02d} s{r["seed"]:>5}: '
                      f'gap={r["gap_final"]:.3e}  ratio={r["gap_ratio"]:.4f}  '
                      f'gens={r["gens"]}  ({el:.0f}s)', flush=True)
            except Exception as e:
                fid, seed = futures[fut]
                print(f'  [{done:3d}/{len(jobs)}] F{fid:02d} s{seed}: ERROR {e}',
                      flush=True)

    total = time.perf_counter() - t0
    print(f'\nTotal {total:.0f}s ({total/len(jobs):.1f}s/traj effective)')

    # Per-fid summary. Both ratio and gap_abs are reported because in
    # rotated CEC2017 hybrids, gap_init is 1e10-1e11 — gap_ratio alone
    # hides 4-6 OOM differences in absolute solution quality.
    print(f'\n{"FID":>5} {"gap_ratio_med":>14} {"gap_ratio_mean":>16} '
          f'{"gap_med":>14} {"gap_mean":>14} {"n_seeds":>8}')
    print('-' * 80)
    per_fid = {}
    for fid in fids:
        rs = [r for r in results if r['fid'] == fid]
        if not rs: continue
        ratios = sorted(r['gap_ratio'] for r in rs)
        gaps = sorted(r['gap_final'] for r in rs)
        med = ratios[len(ratios)//2]
        mean = sum(ratios)/len(ratios)
        gap_med = gaps[len(gaps)//2]
        gap_mean = sum(gaps)/len(gaps)
        per_fid[fid] = {'median': med, 'mean': mean,
                        'gap_median': gap_med, 'gap_mean': gap_mean,
                        'n': len(rs), 'ratios': ratios, 'gaps': gaps}
        print(f'F{fid:02d}   {med:>14.4f} {mean:>16.4f} '
              f'{gap_med:>14.3e} {gap_mean:>14.3e} {len(rs):>8}')

    all_ratios = [r['gap_ratio'] for r in results]
    if all_ratios:
        print(f'\nOverall: mean={sum(all_ratios)/len(all_ratios):.4f}  '
              f'median={sorted(all_ratios)[len(all_ratios)//2]:.4f}  '
              f'n={len(all_ratios)}')
    else:
        # No valid results (e.g. a chunk killed mid-run). Don't crash before the
        # JSON save below — emit an empty-but-valid file so the merge can skip it.
        print('\nOverall: n=0 (no valid results)')

    # Strip _traj from results before JSON serialization (tensors aren't JSON-friendly).
    results_json = []
    traj_records = []
    for r in results:
        traj = r.pop('_traj', None)
        results_json.append(r)
        if traj is not None:
            traj_records.extend(traj)

    # Atomic writes: dump to a .tmp sibling then os.replace into place. A
    # process killed mid-write would otherwise leave a non-empty truncated
    # JSON that the sbatch idempotency check `[[ -s "$CJ" ]]` happily accepts
    # as done, causing the next merge to crash on the partial file.
    import os as _os
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + '.tmp')
    with open(tmp_out, 'w') as f:
        json.dump({'config': vars(args), 'results': results_json,
                   'per_fid': per_fid}, f, indent=2)
    _os.replace(tmp_out, out_path)
    print(f'\nSaved {out_path}')

    if args.log_trajectories:
        import torch as _torch
        traj_path = Path(args.log_trajectories)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_traj = traj_path.with_suffix(traj_path.suffix + '.tmp')
        _torch.save({'records': traj_records, 'config': vars(args)}, tmp_traj)
        _os.replace(tmp_traj, traj_path)
        print(f'Saved {len(traj_records)} trajectory records to {traj_path}')


if __name__ == '__main__':
    main()
