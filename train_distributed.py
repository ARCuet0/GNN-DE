"""Function-parallel distributed L2O training — universal entry point.

Each GPU runs a DIFFERENT CEC2017 function per step. Backbone gradients are
all-reduced across GPUs. Supports all architecture x loss x benchmark axes
via CLI flags.

Launch:
    torchrun --nproc_per_node=4 train_distributed.py --steps 5000
    python train_distributed.py --steps 2000
"""
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from encoder.opt_variant import _clamp_fitness
from l2o.dist_utils import (
    is_distributed, get_rank, get_world_size, is_main,
    setup_distributed, cleanup_distributed,
    broadcast_params, allreduce_grads, broadcast_es_stopped,
    check_nan_any_rank, get_grad_norm, build_named_groups,
    compute_clip_metrics,
)
from l2o.model_factory import build_model
from l2o.loss_fns import (
    compute_hitting_loss, compute_geo_losses,
    compute_lupi_loss, compute_gate_bce, compute_oracle_router_loss,
    compute_fcr_grid_loss, compute_fcr_oracle_from_m_loss,
    compute_donor_oracle_loss, compute_attn_diag,
    pairwise_ranking_loss, compute_gate_auc, build_gate_diag,
    compute_cf_improvement_loss,
    build_step_diagnostics,
)
from l2o.task_pool import (
    Task, spawn_task, compute_pool_weights, make_augmented_fn,
    update_task, get_all_fn_ids, simple_mix_selection,
)
from l2o.canonical_eval import run_canonical_eval, load_backbone_checkpoint
from l2o.early_stop import should_early_stop
from l2o.schedules import (PopulationGenState, compute_lpsr_n_target,
                           compute_surrogate_m, gather_pop, lpsr_keep_indices)
from l2o.cli import parse_args

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


_log = logging.getLogger(__name__)


def apply_vram_cap(args, vram_gb=None):
    """Cap bptt_chunk (graph depth per backward) based on available VRAM.

    VRAM scales with bptt_chunk (activations held until backward + detach),
    NOT with bptt_w (total gens with gradient).  Between chunks, activations
    are freed, so bptt_w can be arbitrarily large.
    """
    _GB_PER_POP_GEN = 0.037
    if getattr(args, 'gate_type', '') == 'surrogate':
        _M = max(args.m_samples, 1)
        _GB_PER_POP_GEN = 0.025 * (1 + _M)
    _HEADROOM = 0.80
    _BPTT_MIN_GENS = 4

    if vram_gb is None:
        if not torch.cuda.is_available():
            return
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    vram_usable = vram_gb * _HEADROOM

    if args.b_per_gpu <= 0:
        args.b_per_gpu = max(1, int(vram_usable / (_BPTT_MIN_GENS * _GB_PER_POP_GEN)))

    max_chunk_from_vram = max(_BPTT_MIN_GENS,
                              int(vram_usable / (args.b_per_gpu * _GB_PER_POP_GEN)))
    if args.bptt_chunk > max_chunk_from_vram:
        _log.info("VRAM CAP: bptt_chunk %d -> %d (%.1fGB / %d pops / %.3f GB/pop/gen)",
                  args.bptt_chunk, max_chunk_from_vram, vram_usable,
                  args.b_per_gpu, _GB_PER_POP_GEN)
        args.bptt_chunk = max_chunk_from_vram


def _get_x_star(fn, device):
    """Extract optimum location for LUPI, handling plain and augmented fns."""
    # Augmented fns have Q (rotation) and s (shift) from random affine transform
    if hasattr(fn, 'Q') and hasattr(fn, 's'):
        Q, s = fn.Q, fn.s
        base = getattr(fn, 'base_fn', None)
        z_star = getattr(base, 'shift', None)
        if z_star is None:
            sm = getattr(base, 'shift_mat', None)
            z_star = sm[0] if sm is not None else None
        if z_star is not None:
            return (Q @ z_star.to(Q.dtype) + s).to(device).unsqueeze(0).float()
        return None
    _shift = getattr(fn, 'shift', None)
    if _shift is None:
        sm = getattr(fn, 'shift_mat', None)
        _shift = sm[0] if sm is not None else None
    return _shift.to(device).unsqueeze(0).float() if _shift is not None else None


def _per_component_backward(components, all_params):
    """Backward each (loss, max_norm) component independently, clip per-component,
    sum the clipped gradients into p.grad. Mitigates explosion amplification
    from one loss component dominating the total gradient norm post-clip
    (per A.0 finding: disen aux loss amplifies explosion rate 2.4x vs null).

    Behavior contract:
      - For each (loss_i, max_norm_i): zero grads, backward (retain_graph for
        all but last), clip the just-computed gradient to max_norm_i, accumulate
        into a buffer.
      - At end, copy buffer → p.grad. Caller's downstream pipeline (allreduce +
        post-allreduce clip) operates on the accumulated, per-component-clipped
        gradient.
      - Non-finite or no-grad components are silently skipped.

    Coste: ~N_components × backward time per chunk (was 1× before).
    Memory: retain_graph=True keeps activations until last component.
    """
    valid = [(loss, mn) for (loss, mn) in components
             if torch.is_tensor(loss) and loss.requires_grad
             and loss.grad_fn is not None and torch.isfinite(loss)]
    if not valid:
        return
    # Initialize buffer
    grad_buffer = {}
    for p in all_params:
        if p.requires_grad:
            grad_buffer[id(p)] = torch.zeros_like(p)
    # Process each component
    for i, (loss_i, max_norm_i) in enumerate(valid):
        # Zero current grads
        for p in all_params:
            if p.grad is not None:
                p.grad.zero_()
        retain = i < len(valid) - 1
        try:
            loss_i.backward(retain_graph=retain)
        except RuntimeError:
            continue
        # Clip THIS component's gradient
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=max_norm_i)
        # Accumulate
        for p in all_params:
            if p.grad is not None and id(p) in grad_buffer:
                grad_buffer[id(p)].add_(p.grad)
    # Restore accumulated grad
    for p in all_params:
        if id(p) in grad_buffer:
            p.grad = grad_buffer[id(p)]


def _decompose_chunk_backward(hit_loss, chunk_geo, chunk_geo_names,
                              all_params, save_dir, step, chunk_idx, rank,
                              named_groups=None):
    """Per-term gradient decomposition for the current chunk.

    For each named loss term (hit + chunk_geo entries), measure its gradient
    norm in isolation and the cosine alignment between every pair of terms.

    Activated by env var LOG_GRAD_DECOMPOSE; off by default. Writes one
    JSONL line per chunk to ``save_dir/grad_decomp_rank{rank}.jsonl``.
    Side-effect: zeros all parameter gradients (caller must call backward
    on total_loss again to accumulate the real signal).
    """
    out_path = save_dir / f'grad_decomp_rank{rank}.jsonl'

    def _flat_grad():
        return torch.cat([p.grad.detach().reshape(-1) if p.grad is not None
                          else torch.zeros_like(p).reshape(-1)
                          for p in all_params])

    def _zero():
        for p in all_params:
            if p.grad is not None:
                p.grad.zero_()

    # Each chunk_geo entry is a per-gen loss; same name appears once per gen.
    # Backward each entry separately, then aggregate gradients by name AFTER
    # summing in gradient-space. Total per-name gradient =
    #     (1 / len(chunk_geo)) * Σ ∇(named entry)   (matches geo_loss = mean).
    # hit_loss enters total_loss directly, so no scaling.
    geo_n = max(1, len(chunk_geo))

    from collections import defaultdict
    grouped_grads = defaultdict(list)
    errors = {}

    # hit
    _zero()
    if torch.isfinite(hit_loss):
        try:
            hit_loss.backward(retain_graph=True)
            grouped_grads['hit'].append(_flat_grad())
        except RuntimeError as e:
            errors['hit'] = str(e)[:120]
    else:
        errors['hit'] = 'non-finite'

    # chunk_geo entries
    for name, term in zip(chunk_geo_names, chunk_geo):
        _zero()
        if not torch.isfinite(term):
            errors[name] = errors.get(name) or 'non-finite'
            continue
        if not term.requires_grad or term.grad_fn is None:
            errors[name] = (errors.get(name) or
                f'no-grad val={float(term):.3e} rg={term.requires_grad}')
            continue
        try:
            term.backward(retain_graph=True)
        except RuntimeError as e:
            errors[name] = str(e)[:120]
            continue
        grouped_grads[name].append(_flat_grad())

    # Per-name total gradient (with mean(chunk_geo) scaling for geo terms)
    grads = {}
    for name, gs in grouped_grads.items():
        total = torch.stack(gs).sum(dim=0)
        if name != 'hit':
            total = total / geo_n
        grads[name] = total

    # Per-param-group decomposition (which subsystem each loss term updates)
    per_group = {}
    if named_groups:
        # Build offsets so we can slice the flat gradient by group
        offsets = {}
        cur = 0
        param_to_idx = {id(p): i for i, p in enumerate(all_params)}
        for gname, params in named_groups.items():
            slices = [param_to_idx[id(p)] for p in params if id(p) in param_to_idx]
            if not slices:
                continue
            # Compute the slice index ranges
            group_idxs = []
            cur = 0
            for i, p in enumerate(all_params):
                n = p.numel()
                if id(p) in {id(q) for q in params}:
                    group_idxs.append((cur, cur + n))
                cur += n
            offsets[gname] = group_idxs
        for term_name, g in grads.items():
            per_group[term_name] = {}
            for gname, ranges in offsets.items():
                gnorm_sq = 0.0
                for s, e in ranges:
                    gnorm_sq += float(g[s:e].pow(2).sum().item())
                per_group[term_name][gname] = gnorm_sq ** 0.5

    norms = {n: float(g.norm().item()) for n, g in grads.items()}
    cosines = {}
    names = list(grads.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ga, gb = grads[a], grads[b]
            denom = (ga.norm() * gb.norm()).clamp(min=1e-30)
            cosines[f'{a}__{b}'] = float((ga @ gb / denom).item())

    _zero()
    record = {
        'step': step, 'chunk': chunk_idx,
        'norms': norms, 'cosines': cosines,
        'per_group': per_group,
        'order': ['hit'] + list(chunk_geo_names),
        'n_chunk_geo': len(chunk_geo),
        'errors': errors,
    }
    with open(out_path, 'a') as f:
        f.write(json.dumps(record) + '\n')


def main():
    args = parse_args()
    if args.task_mix_vanilla_prob is not None:
        if not (0.0 <= args.task_mix_vanilla_prob <= 1.0):
            raise ValueError(
                f'--task-mix-vanilla-prob must be in [0,1], got '
                f'{args.task_mix_vanilla_prob}')
        args.no_curriculum = True
    # Apply global RNG seed BEFORE any module construction so weight init,
    # task sampling, and curriculum draws are reproducible across reps.
    # Per-rank offset so DDP workers don't all sample identical tasks.
    if getattr(args, 'seed', None) is not None:
        _seed = int(args.seed)
        _rank_offset = int(os.environ.get('RANK', '0'))
        random.seed(_seed + _rank_offset)
        np.random.seed(_seed + _rank_offset)
        torch.manual_seed(_seed + _rank_offset)
        torch.cuda.manual_seed_all(_seed + _rank_offset)
    local_rank = setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

    save_dir = Path(args.save_dir)
    if is_main():
        save_dir.mkdir(parents=True, exist_ok=True)
    if is_distributed():
        dist.barrier()

    log = logging.getLogger(f'rank{rank}')
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler())
    if is_main():
        log.addHandler(logging.FileHandler(save_dir / 'train.log', mode='a'))

    # Auto-detect B and cap bptt_chunk from VRAM.
    if torch.cuda.is_available():
        apply_vram_cap(args)

    log.info("rank=%d/%d device=%s steps=%d B=%d M=%d D=%s N=%d",
             rank, world_size, device, args.steps, args.b_per_gpu,
             args.m_samples, args.D, args.N)

    from encoder.similarity_graph_gpu import build_sparse_graphs_gpu
    from encoder.grad_stabilizers import scale_gradient, make_gen_clip_hook

    backbone, variant, gen_step = build_model(args, device)
    # Gradient checkpointing: recompute backbone forward during backward
    # to save ~75 MB/gen of activation storage. Critical for surrogate mode
    # (2 backbone forwards per gen) but beneficial in all modes.
    if getattr(args, 'gate_type', '') == 'surrogate':
        backbone.use_checkpoint = True

    # Donor oracle loss needs per-m donor diversity.
    if getattr(args, 'donor_oracle_weight', 0.0) > 0 and not getattr(args, 'per_m_donors', False):
        raise ValueError(
            "--donor-oracle-weight > 0 requires --per-m-donors. Without per-m "
            "donor resampling all M proposals share the same (pbest, r1, r2) "
            "triple, making the oracle targets trivial.")

    # Option A — per-M donor resampling in BatchedDiffAttDE (and any head that
    # exposes `per_m_donors`). Off by default; opt-in via --per-m-donors.
    if getattr(args, 'per_m_donors', False) and hasattr(variant, 'heads'):
        for _h in variant.heads:
            if hasattr(_h, 'per_m_donors'):
                _h.per_m_donors = True
    # Donor mode — 'lshade' bypasses the neural donor_selector with hand-
    # crafted L-SHADE rules (per-individual top-p_i pbest, uniform r1 over
    # active pop, uniform r2 over pop ∪ archive). Default 'neural' preserves
    # current behavior.
    _donor_mode = getattr(args, 'donor_mode', 'neural')
    _pbest_max = getattr(args, 'lshade_pbest_max', 0.11)
    if hasattr(variant, 'heads'):
        for _h in variant.heads:
            if hasattr(_h, 'donor_mode'):
                _h.donor_mode = _donor_mode
            if hasattr(_h, 'lshade_pbest_max'):
                _h.lshade_pbest_max = _pbest_max
    if _donor_mode == 'lshade' and getattr(args, 'donor_oracle_weight', 0.0) > 0:
        log.info("E13 distillation: --donor-mode lshade with --donor-oracle-weight=%s; "
                 "the neural donor selector is supervised on the L-SHADE-driven "
                 "Gumbel samples (uniform within allowed set).",
                 args.donor_oracle_weight)
    # Archive eviction policy — 'fifo' (E9 default) or 'random' (L-SHADE faithful).
    _archive_evict = getattr(args, 'archive_evict', 'fifo')
    if hasattr(gen_step, 'archive_capacity'):
        gen_step.archive_evict = _archive_evict
    # E13: instantiate L-SHADE F/CR memory teacher + attach to head.
    _fcr_mode = getattr(args, 'fcr_mode', 'beta')
    _lshade_mem = None
    if _fcr_mode == 'lshade' and hasattr(variant, 'heads'):
        from encoder.lshade_memory import LShadeMemory
        _lshade_mem = LShadeMemory(B=args.b_per_gpu,
                                    H=getattr(args, 'lshade_memory_H', 6),
                                    device=device)
        for _h in variant.heads:
            if hasattr(_h, 'fcr_mode') and _h.fcr_mode == 'lshade':
                _h._lshade_teacher = _lshade_mem
        log.info("L-SHADE memory teacher attached: B=%d H=%d (fcr_mode=lshade, "
                 "fcr_distill_weight=%s)",
                 args.b_per_gpu, getattr(args, 'lshade_memory_H', 6),
                 getattr(args, 'fcr_distill_weight', 0.0))
    # Architectural probe: bias-only A_pbest (drops h @ h^T term).
    if getattr(args, 'bias_only_pbest', False) and hasattr(variant, 'heads'):
        for _h in variant.heads:
            if hasattr(_h, 'bias_only_pbest'):
                _h.bias_only_pbest = True
                log.info("bias_only_pbest enabled on head %s", type(_h).__name__)
    gen_step.compute_contrafactual_grad = False
    gen_step.keep_deltas_k_live = args.contrafactual
    # Eval all offspring (including gate-inactive) so BCE gets true
    # contrafactual labels, not tautological zeros for inactive individuals.
    variant.eval_all_select_active = True
    gen_step.pool_selection = True

    start_step = 0
    fn_curriculum = {}
    fn_level_hits = {}
    bptt_w = {}
    _resumed_task_pool = None

    _ck = None
    if args.resume and os.path.exists(args.resume):
        if is_main():
            _ck = torch.load(args.resume, map_location=device, weights_only=False)

            def _filter_sd(target_module, ckpt_sd, name):
                """Keep only keys that exist in target with matching shape.
                Log dropped keys so silent truncation is visible."""
                mod_sd = target_module.state_dict()
                kept = {k: v for k, v in ckpt_sd.items()
                        if k in mod_sd and v.shape == mod_sd[k].shape}
                missing = set(mod_sd) - set(kept)
                extra = set(ckpt_sd) - set(mod_sd)
                shape_mismatch = [k for k in ckpt_sd
                                  if k in mod_sd and ckpt_sd[k].shape != mod_sd[k].shape]
                log.info("%s resume: kept=%d/%d, missing=%d, extra=%d, shape_mismatch=%d",
                         name, len(kept), len(mod_sd), len(missing),
                         len(extra), len(shape_mismatch))
                if shape_mismatch:
                    log.info("  shape_mismatch sample: %s", shape_mismatch[:5])
                if missing:
                    log.info("  missing sample: %s", sorted(missing)[:5])
                return kept

            backbone.load_state_dict(
                _filter_sd(backbone, _ck['backbone_state_dict'], 'Backbone'),
                strict=False)
            variant.load_state_dict(
                _filter_sd(variant, _ck['variant_state_dict'], 'Variant'),
                strict=False)
            start_step = _ck.get('step', 0) + 1
            fn_curriculum = _ck.get('fn_curriculum', {})
            fn_level_hits = _ck.get('fn_level_hits', {})
            raw_bptt = _ck.get('bptt_w', {})
            if isinstance(raw_bptt, dict):
                bptt_w = raw_bptt
            _resumed_task_pool = _ck.get('task_pool', None)
            log.info("Resumed from %s (step=%d)", args.resume, start_step)
    elif is_main():
        n_loaded, skipped = load_backbone_checkpoint(backbone, args.backbone_ckpt, device)
        log.info("Loaded backbone: %d params, skipped=%s", n_loaded, skipped or '{}')

    # Post-resume head expansions (e.g., AdaptiveFCRCauchy 2→3-output for
    # --fcr-learn-sigma). Must run on ALL ranks (not gated by is_main) so
    # broadcast_params below sees identical parameter shapes; must run
    # AFTER load_state_dict on main so the trained μ_F/μ_CR rows survive
    # and only the σ_F row gets fresh init. Idempotent and a no-op when
    # the flag is off, so it's safe to call unconditionally.
    from l2o.model_factory import apply_post_resume_head_fixes
    apply_post_resume_head_fixes(variant, args)

    broadcast_params(backbone)
    broadcast_params(variant)
    broadcast_params(gen_step)

    if args.reset_step_counter and is_main():
        start_step = 0
        log.info("--reset-step-counter: forcing start_step=0 "
                 "(weights resumed, step counter reset)")

    _start_step_t = torch.tensor([start_step], dtype=torch.long, device=device)
    if is_distributed():
        dist.broadcast(_start_step_t, src=0)
    start_step = int(_start_step_t.item())

    if start_step >= args.steps:
        raise SystemExit(
            f"Configured run would do 0 steps: start_step={start_step} "
            f">= args.steps={args.steps}. Pass --steps {start_step + 100} "
            f"or use --reset-step-counter to restart from step 0.")

    # ── Split donor_selector out of backbone for separate LR warmstart ──
    # Stateless DE refactor: the donor_selector is initialized from scratch
    # and converges fastest with a dedicated, higher LR for its warmup phase.
    # Using the backbone's SSL-trained base LR (3e-5) would starve it.
    # See adelante-clever-wall.md §Warm-start mitigations.
    _donor_sel_params = []
    _donor_sel_ids = set()
    _inner_bb = getattr(backbone, 'backbone', backbone)  # TemporalSparse wraps inner
    _donor_sel = getattr(_inner_bb, 'donor_selector', None)
    if _donor_sel is not None:
        _donor_sel_params = list(_donor_sel.parameters())
        _donor_sel_ids = set(id(p) for p in _donor_sel_params)

    bb_params = [p for p in backbone.parameters()
                 if id(p) not in _donor_sel_ids]
    var_params = list(variant.parameters())
    all_params = bb_params + var_params + _donor_sel_params

    _wd = getattr(args, 'weight_decay', 0.0)
    _lr_donor = args.lr_variant * 3  # 3× warmstart; decays via scheduler
    if _wd > 0:
        # Split: apply WD to weight matrices (ndim >= 2), skip scalars/biases/norms
        def _split_wd(params, lr):
            decay = [p for p in params if p.ndim >= 2]
            no_decay = [p for p in params if p.ndim < 2]
            groups = []
            if decay:
                groups.append({'params': decay, 'lr': lr, 'weight_decay': _wd})
            if no_decay:
                groups.append({'params': no_decay, 'lr': lr, 'weight_decay': 0.0})
            return groups
        param_groups = _split_wd(bb_params, args.lr_backbone) + \
                       _split_wd(var_params, args.lr_variant)
        if _donor_sel_params:
            param_groups += _split_wd(_donor_sel_params, _lr_donor)
        log.info("Weight decay %.1e on %d params (ndim>=2), 0 on %d params",
                 _wd,
                 sum(p.numel() for p in all_params if p.ndim >= 2),
                 sum(p.numel() for p in all_params if p.ndim < 2))
    else:
        param_groups = [
            {'params': bb_params, 'lr': args.lr_backbone},
            {'params': var_params, 'lr': args.lr_variant},
        ]
        if _donor_sel_params:
            param_groups.append(
                {'params': _donor_sel_params, 'lr': _lr_donor,
                 'name': 'bb.donor_selector'})
    if _donor_sel_params:
        log.info("donor_selector param group: %d params @ lr=%.2e (3× var LR)",
                 sum(p.numel() for p in _donor_sel_params), _lr_donor)
    # Surrogate params (if present)
    if gen_step.surrogate is not None:
        surr_params = list(gen_step.surrogate.parameters())
        all_params += surr_params
        param_groups.append({'params': surr_params, 'lr': args.lr_variant,
                             'weight_decay': _wd})

    # [2026-05-04 disentangle] DisentangleHeads — instantiated when
    # any --disentangle-lambda-* > 0. Heads sit on top of h_aug from the
    # surrogate forward; trained jointly with backbone.
    disentangle_heads = None
    _dis_total_w = (getattr(args, 'disentangle_lambda_e', 0.0)
                    + getattr(args, 'disentangle_lambda_x', 0.0)
                    + getattr(args, 'disentangle_lambda_h', 0.0))
    if _dis_total_w > 0:
        if getattr(args, 'gate_type', '') != 'surrogate':
            raise ValueError(
                "--disentangle-lambda-* > 0 requires --gate-type surrogate "
                "(needs h_aug from augmented-pop forward).")
        from l2o.disentangle_loss import DisentangleHeads
        disentangle_heads = DisentangleHeads(hidden_dim=128).to(device)
        dis_params = list(disentangle_heads.parameters())
        all_params += dis_params
        param_groups.append({'params': dis_params, 'lr': args.lr_variant,
                             'weight_decay': _wd, 'name': 'disentangle_heads'})
        log.info("DisentangleHeads instantiated (%d params, λ_e=%.2f λ_x=%.2f λ_h=%.2f)",
                 sum(p.numel() for p in dis_params),
                 args.disentangle_lambda_e, args.disentangle_lambda_x, args.disentangle_lambda_h)
        # Restore heads from checkpoint if present. _ck was loaded above for
        # backbone/variant resume; reuse it if available.
        if args.resume and os.path.exists(args.resume):
            try:
                _ck_dis = _ck if '_ck' in dir() and _ck is not None else \
                          torch.load(args.resume, map_location=device, weights_only=False)
                if 'disentangle_heads_state_dict' in _ck_dis:
                    disentangle_heads.load_state_dict(_ck_dis['disentangle_heads_state_dict'])
                    log.info("DisentangleHeads resumed from %s", args.resume)
                else:
                    log.info("Checkpoint has no disentangle_heads_state_dict — heads start from init")
            except Exception as _e:
                log.info("DisentangleHeads resume failed (%s) — heads start from init", _e)

    optimizer = torch.optim.Adam(param_groups)

    if args.resume and os.path.exists(args.resume):
        if _ck is None:
            _ck = torch.load(args.resume, map_location=device, weights_only=False)
        if 'optimizer_state_dict' in _ck and os.environ.get('SKIP_OPTIM') != '1':
            try:
                optimizer.load_state_dict(_ck['optimizer_state_dict'])
            except (ValueError, KeyError, RuntimeError) as _e:
                # Model structure changed (e.g., different K) — optimizer state
                # incompatible. Start optimizer fresh.
                log.info("Optimizer resume skipped (structure changed): %s", _e)
        del _ck

    log.info("Params: total=%d", sum(p.numel() for p in all_params))

    # Precompute gate/rest param sets for per-group clipping
    if variant.activity_gate is not None:
        _gate_params = list(variant.activity_gate.parameters())
        _gate_set = set(id(p) for p in _gate_params)
        _rest_params = [p for p in all_params if id(p) not in _gate_set]
    else:
        _gate_params = []
        _rest_params = all_params

    _lf_levels = None
    if args.env == 'linked-flame':
        _lf_levels = [int(s) for s in args.lf_levels.split(',') if s.strip()]
    all_fn_ids = get_all_fn_ids(args.single_fn, env=args.env, lf_levels=_lf_levels)
    # Canonical eval always uses CEC2017 (held-out), regardless of training env.
    all_fn_ids_eval = (get_all_fn_ids(args.single_fn, env='cec2017')
                       if args.env == 'linked-flame' else all_fn_ids)
    # Parse --fn-downweight "13:0.5,19:0.5" → {13: 0.5, 19: 0.5}
    _fn_wt_overrides = {}
    if getattr(args, 'fn_downweight', ''):
        for pair in args.fn_downweight.split(','):
            pair = pair.strip()
            if ':' in pair:
                fid, mult = pair.split(':')
                _fn_wt_overrides[int(fid)] = float(mult)
        if _fn_wt_overrides:
            log.info("Fn weight overrides: %s", _fn_wt_overrides)
    dims_list = args.D
    # N scales with D: base N at base D, capped by VRAM.
    # Default mapping: D=10→N=50, D=30→N=100 (i.e. N = min(5*D, 100))
    _N_for_D = {d: min(5 * d, 100) for d in dims_list}
    if len(dims_list) == 1:
        _N_for_D[dims_list[0]] = args.N  # respect explicit --N for single-D
    N = _N_for_D[dims_list[0]]  # default for init/eval
    B, M = args.b_per_gpu, args.m_samples
    MAX_GENS = 2000
    gru_W = args.gru_window
    FIT_DAMP = args.fit_damp
    HITTING_BB_SCALE = args.hitting_bb_scale
    MIN_FES = args.min_fes

    ref_gap_init = {}
    target_gaps = {}
    n_targets = args.n_targets
    target_fractions = [10 ** (-i * 10.0 / (n_targets - 1)) for i in range(n_targets)]

    if args.env == 'linked-flame':
        from encoder.linked_flame import LinkedFlameEnv
        aug_cache = LinkedFlameEnv(device=device, dims=tuple(dims_list))

        with torch.no_grad():
            for D in dims_list:
                _N_d = _N_for_D[D]
                for fid in all_fn_ids:   # fid here = Level (1..5)
                    rng_tmp = torch.Generator(device='cpu')
                    rng_tmp.manual_seed(0xCAFE + fid * 31 + D)
                    fn_tmp = aug_cache.sample(fid=fid, D=D, rng=rng_tmp)
                    gaps = []
                    for _ in range(5):
                        c = (torch.rand(B, _N_d, D, device=device) * 200 - 100).to(torch.float64)
                        f = fn_tmp(c.reshape(-1, D)).reshape(B, _N_d)
                        gaps.append((f.min(dim=-1).values - fn_tmp.f_optimal).mean().item())
                    ref_gap_init[(fid, D)] = max(sum(gaps) / len(gaps), 1.0)
                    target_gaps[(fid, D)] = [ref_gap_init[(fid, D)] * f for f in target_fractions]
    elif args.env == 'bbob':
        # BBOB: differentiable torch port, domain [-5,5]; variety via native
        # instances. ref-gap init drawn from the actual search domain (args.lb/ub).
        from encoder.bbob_torch import BBOBTorch
        _span = float(args.ub - args.lb)
        with torch.no_grad():
            for D in dims_list:
                _N_d = _N_for_D[D]
                for fid in all_fn_ids:
                    fn_tmp = BBOBTorch(fid, D, device)
                    gaps = []
                    for _ in range(5):
                        c = (torch.rand(B, _N_d, D, device=device) * _span + args.lb).to(torch.float64)
                        f = fn_tmp(c.reshape(-1, D)).reshape(B, _N_d)
                        gaps.append((f.min(dim=-1).values - fn_tmp.f_optimal).mean().item())
                    ref_gap_init[(fid, D)] = max(sum(gaps) / len(gaps), 1.0)
                    target_gaps[(fid, D)] = [ref_gap_init[(fid, D)] * f for f in target_fractions]

        from encoder.augmented_bbob import AugmentedBBOB
        aug_cache = AugmentedBBOB(device=device, dims=tuple(dims_list))
    else:
        from encoder.cec2017_torch import CEC2017Torch

        with torch.no_grad():
            for D in dims_list:
                _N_d = _N_for_D[D]
                for fid in all_fn_ids:
                    fn_tmp = CEC2017Torch(fid, D, device)
                    gaps = []
                    for _ in range(5):
                        c = (torch.rand(B, _N_d, D, device=device) * 200 - 100).to(torch.float64)
                        f = fn_tmp(c.reshape(-1, D)).reshape(B, _N_d)
                        gaps.append((f.min(dim=-1).values - fn_tmp.f_optimal).mean().item())
                    ref_gap_init[(fid, D)] = max(sum(gaps) / len(gaps), 1.0)
                    target_gaps[(fid, D)] = [ref_gap_init[(fid, D)] * f for f in target_fractions]

        from encoder.augmented_cec2017 import AugmentedCEC2017
        aug_cache = AugmentedCEC2017(device=device, dims=tuple(dims_list))

    # Warm start pool (optional)
    _ws_pool = None
    if args.warm_start_prob > 0:
        from l2o.warm_start import WarmStartPool
        _ws_dir = Path(args.warm_start_dir) / f'D{dims_list[0]}'
        _ws_pool = WarmStartPool(_ws_dir, gru_window=gru_W)
        log.info("Warm start pool: prob=%.2f, dir=%s, fids=%s",
                 args.warm_start_prob, _ws_dir, _ws_pool.available_fids)

    task_pool = {}
    next_task_id = 0
    if not fn_curriculum:
        for _ in range(args.pool_size):
            task_pool[next_task_id] = spawn_task(next_task_id, all_fn_ids, args.bptt_w_init)
            next_task_id += 1
    else:
        if _resumed_task_pool:
            for td in _resumed_task_pool:
                t = Task.from_dict(td)
                t.curriculum_idx = min(t.curriculum_idx, n_targets - 1)
                t.max_level_ever = min(t.max_level_ever, n_targets - 1)
                task_pool[t.task_id] = t
                next_task_id = max(next_task_id, t.task_id + 1)
        else:
            for fid in all_fn_ids:
                t = spawn_task(next_task_id, all_fn_ids, args.bptt_w_init)
                t.fid = fid
                t.aug_seed = 0
                t.curriculum_idx = fn_curriculum.get(fid, 0)
                t.max_level_ever = t.curriculum_idx
                t.hits = fn_level_hits.get(fid, [])
                t.bptt_w = bptt_w.get(fid, args.bptt_w_init)
                task_pool[next_task_id] = t
                next_task_id += 1

    diag_path = save_dir / f'diagnostics_rank{rank}.jsonl'
    diag_file = None
    fn_rng = random.Random(42)

    # Full trajectory logging: save 1 in every traj_every steps
    # Set to 0 to disable (avoids OOM on 12GB GPU with M=20)
    traj_dir = save_dir / 'trajectories'
    traj_every = int(os.environ.get('TRAJ_EVERY', 0))

    named_groups = build_named_groups(backbone, variant, gen_step)

    if is_main():
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)

    _es_stopped = False
    _best_eval_gr = float('inf')
    _evals_without_improvement = 0
    _prev_param_snapshot = None
    _param_drift_history = []
    gate_bce_scale_ema = torch.tensor(1.0, device=device)

    for step in range(start_step, args.steps):
      if _es_stopped:
        break
      try:
        t0 = time.perf_counter()
        optimizer.zero_grad()

        fn_tensor = torch.zeros(world_size, dtype=torch.long, device=device)
        seed_tensor = torch.zeros(world_size, dtype=torch.long, device=device)
        tid_tensor = torch.zeros(world_size, dtype=torch.long, device=device)
        tgt_tensor = torch.zeros(world_size, dtype=torch.float32, device=device)
        bptt_w_tensor = torch.zeros(world_size, dtype=torch.long, device=device)
        fes_tensor = torch.zeros(1, dtype=torch.long, device=device)
        dim_tensor = torch.zeros(1, dtype=torch.long, device=device)
        if is_main():
            if args.task_mix_vanilla_prob is not None:
                # Probe mode: per-step Bernoulli vanilla/aug, no pool, no curriculum.
                picks = simple_mix_selection(world_size, all_fn_ids,
                                             args.task_mix_vanilla_prob, fn_rng)
                D = random.choice(dims_list)
                dim_tensor[0] = D
                MAX_FES = args.budget_mult * D
                step_fes = random.randint(MIN_FES, MAX_FES)
                fes_tensor[0] = step_fes
                fn_tensor[:] = torch.tensor([p[0] for p in picks], device=device)
                seed_tensor[:] = torch.tensor([p[1] for p in picks], device=device)
                tid_tensor[:] = -1  # sentinel: not in pool
                # Use middle target so hitting loss has signal both sides.
                _mid = n_targets // 2
                tgt_tensor[:] = torch.tensor(
                    [target_gaps[(fid, D)][_mid] for (fid, _) in picks],
                    device=device, dtype=torch.float32)
                bptt_w_tensor[:] = args.bptt_w_init
            else:
                task_ids, task_weights = compute_pool_weights(task_pool,
                                                              fn_weight_overrides=_fn_wt_overrides)
                selected_tids = fn_rng.choices(task_ids, weights=task_weights, k=world_size)
                selected_tasks = [task_pool[tid] for tid in selected_tids]
                D = random.choice(dims_list)
                dim_tensor[0] = D
                MAX_FES = args.budget_mult * D
                step_fes = random.randint(MIN_FES, MAX_FES)
                fes_tensor[0] = step_fes
                fn_tensor[:] = torch.tensor([t.fid for t in selected_tasks], device=device)
                seed_tensor[:] = torch.tensor([t.aug_seed for t in selected_tasks], device=device)
                tid_tensor[:] = torch.tensor(selected_tids, device=device)
                tgt_tensor[:] = torch.tensor(
                    [target_gaps[(t.fid, D)][min(t.curriculum_idx, n_targets - 1)] for t in selected_tasks],
                    device=device, dtype=torch.float32)
                bptt_w_tensor[:] = torch.tensor([t.bptt_w for t in selected_tasks], device=device)
        if is_distributed():
            for t in [fn_tensor, seed_tensor, tid_tensor, tgt_tensor, bptt_w_tensor, fes_tensor, dim_tensor]:
                dist.broadcast(t, src=0)

        D = dim_tensor[0].item()
        N = _N_for_D[D]
        MAX_FES = args.budget_mult * D
        my_fn_id = fn_tensor[rank].item()
        my_aug_seed = seed_tensor[rank].item()
        my_task_id = tid_tensor[rank].item()
        my_target = tgt_tensor[rank].item()
        step_fes = fes_tensor[0].item()
        my_bptt_w = min(int(bptt_w_tensor[rank].item()), max(1, step_fes // N))
        log_tgt = math.log1p(my_target)
        # Per-step gen-loop cut threshold: under --task-mix-vanilla-prob, the
        # curriculum target is fixed mid (poor calibration → cut fires too
        # early on easy fns). Replace with absolute 1e-5: only stop the loop
        # once the population has effectively converged.
        _hit_cut_thr = (1e-5 if args.task_mix_vanilla_prob is not None
                        else my_target)

        if my_aug_seed == 0:
            # vanilla (un-augmented) base function — env-aware
            if args.env == 'bbob':
                from encoder.bbob_torch import BBOBTorch
                fn = BBOBTorch(my_fn_id, D, device)
            else:
                fn = CEC2017Torch(my_fn_id, D, device)
        else:
            fn = make_augmented_fn(my_fn_id, D, device, my_aug_seed, aug_cache)
        gen_step.eval_fn = fn
        variant._oracle_best_k = None

        x_star = _get_x_star(fn, device)

        # ── Population initialization: warm start or random ──
        _used_warm_start = False
        _ws_fes_frac_start = 0.0
        if (_ws_pool is not None
                # warm start uses vanilla CEC17 populations regardless of aug_seed
                and fn_rng.random() < args.warm_start_prob
                and _ws_pool.has_fid(my_fn_id)):
            _ws = _ws_pool.sample(my_fn_id, B, device, rng=fn_rng)
            if _ws is not None:
                coords = _ws['coords']
                # Re-evaluate fitness with our CEC2017Torch (not stored fitness)
                # to ensure consistency with the function instance used for training
                fitness = _clamp_fitness(fn(coords.reshape(-1, D)).reshape(B, N))
                coords_ring = _ws['coords_ring']
                fitness_ring = _ws['fitness_ring']
                cumulative_fes = _ws['cumulative_fes']
                step_fes = MAX_FES  # full budget, absolute
                _used_warm_start = True
                _ws_fes_frac_start = _ws['fes_frac_start']

        if not _used_warm_start:
            coords = (torch.rand(B, N, D, device=device) * 200 - 100).to(torch.float64)
            fitness = _clamp_fitness(fn(coords.reshape(-1, D)).reshape(B, N))
            coords_ring = torch.zeros(B, gru_W, N, D, dtype=torch.float32, device=device)
            fitness_ring = torch.zeros(B, gru_W, N, dtype=torch.float32, device=device)
            cumulative_fes = 0

        # E13: reset L-SHADE memory at the start of each training step
        # (each step is a fresh CEC2017 instance, so the F/CR memory must
        # restart from neutral 0.5).
        if _lshade_mem is not None:
            _lshade_mem.resize(B)  # defensive: B may change between steps
            _lshade_mem.reset()

        live_coords, live_fitness = [], []
        chunk_best, chunk_fes, chunk_geo = [], [], []
        chunk_cf = []
        chunk_geo_names = []  # parallel names for grad-decomposition probe
        _default_fes_batch = torch.full((B,), float(N), device=device)
        n_bptt_gens = n_chunks_done = total_gens = 0
        gc_hook, _gc_stats = make_gen_clip_hook(max_norm=100.0)
        acc_entropy, acc_active_frac = [], []
        acc_lupi_dist_k, acc_lupi_align_k = [], []
        acc_hit_loss, acc_geo_loss, acc_gate_bce, acc_gate_auc = [], [], [], []
        # `acc_surrogate_pw` is the PairwiseSurrogate ranking loss accumulator,
        # decoupled from `acc_gate_bce` so the surrogate path (gated by
        # --surrogate-loss-weight) does not collide with legacy ActivityGate /
        # RankerGate diagnostics.
        acc_surrogate_pw = []
        acc_fcr_diag = []
        acc_donor_diag = []
        acc_attn_diag = []
        acc_surr_sel_imp = []
        acc_oracle_agreement = []
        _K = variant.K
        _log_traj = traj_every > 0 and is_main() and (step % traj_every == 0)
        _traj_gens = [] if _log_traj else None
        acc_winner_counts = torch.zeros(_K, device=device)
        acc_surr_parent_count = 0  # parents selected by surrogate
        acc_surr_total_count = 0   # total selected by surrogate
        acc_improved_count = torch.tensor(0.0, device=device)
        acc_improved_total = 0
        acc_improved_active = torch.tensor(0.0, device=device)
        acc_active_total = torch.tensor(0.0, device=device)
        pop_state = PopulationGenState(B, device)
        gap_init = (fitness.min(dim=1).values - fn.f_optimal).clamp(min=1e-8).mean().item()

        gen = 0
        while cumulative_fes < step_fes and gen < MAX_GENS:
            # LPSR-N: shrink active population from --N down to --lpsr-N-min
            # linearly with cumulative_fes / step_fes. Reindex coords, fitness,
            # ring buffers, and live BPTT buffers via the shared helper so the
            # train and eval idioms stay bit-identical.
            if getattr(args, 'lpsr_N', False):
                _N_target = compute_lpsr_n_target(
                    args.N, args.lpsr_N_min, cumulative_fes / step_fes)
                if _N_target < coords.shape[1]:
                    _keep_idx = lpsr_keep_indices(fitness, _N_target)
                    coords = gather_pop(coords, _keep_idx, dim=1)
                    fitness = gather_pop(fitness, _keep_idx, dim=1)
                    coords_ring = gather_pop(coords_ring, _keep_idx, dim=2)
                    fitness_ring = gather_pop(fitness_ring, _keep_idx, dim=2)
                    if live_coords:
                        live_coords = [gather_pop(c, _keep_idx, dim=1)
                                       for c in live_coords]
                        live_fitness = [gather_pop(f, _keep_idx, dim=1)
                                        for f in live_fitness]
                    pop_state.reset_baseline()
                    N = _N_target

            ri = gen % gru_W
            coords_ring[:, ri] = coords.detach().float()
            fitness_ring[:, ri] = fitness.detach().float()
            in_bptt = n_bptt_gens < my_bptt_w

            if in_bptt:
                live_coords.append(coords.float())
                live_fitness.append(fitness.float())
                if len(live_coords) > gru_W:
                    live_coords = live_coords[-gru_W:]
                    live_fitness = live_fitness[-gru_W:]
            n_valid = min(gen + 1, gru_W)

            if in_bptt and live_coords:
                coords_hist = torch.stack(live_coords, dim=1)
                fitness_hist = torch.stack(live_fitness, dim=1)
                n_valid = len(live_coords)
            else:
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

            # E13: capture pre-gen fitness for L-SHADE memory update post-gen.
            # In the eval branch, `fitness` is rebound to result['new_fitness']
            # before the unified extras block, so we need to snapshot now.
            _pre_gen_fitness = fitness.detach() if _lshade_mem is not None else None

            if in_bptt:
                c_graph = coords.detach().float().requires_grad_(True)
                f_graph = fitness.detach().float().requires_grad_(True)
                cache = build_sparse_graphs_gpu(
                    c_graph, f_graph, step_num=cumulative_fes, max_steps=step_fes, ndim=D,
                    k_neighbors=args.k_neighbors,
                    stagnation_counters=pop_state.stagnation_counters,
                    delta_fitnesses=pop_state.delta_fitnesses,
                    contraction_rates=pop_state.contraction_rates,
                    prev_coords=prev_c, prev_fitnesses=prev_f)

                old_coords = coords
                _surr_m_now = compute_surrogate_m(
                    args.surrogate_M, args.surrogate_m_final,
                    cumulative_fes / step_fes, N)
                # Under LPSR-N, current N can drop below LPSR-M_sel schedule.
                _surr_m_now = min(_surr_m_now, N)
                result = gen_step.run(
                    coords=coords, fitness=fitness, cache=cache,
                    f_optimal=fn.f_optimal, M=M, gumbel_tau=args.gumbel_tau,
                    node_feat=cache.node_feat, global_feat=cache.global_feat,
                    coords_hist=coords_hist, fitness_hist=fitness_hist,
                    n_valid=n_valid, fes_frac=cumulative_fes / step_fes,
                    gate_target_frac=args.gate_target_frac,
                    k_neighbors=args.k_neighbors,
                    step_num=cumulative_fes, max_steps=step_fes,
                    surrogate_M=_surr_m_now,
                    selection_spec=args.train_selection)
                extras = result.get('extras', {})
                n_bptt_gens += 1

                acc_entropy.append(extras.get('entropy', torch.tensor(0.0, device=device)))
                acc_active_frac.append(extras.get('active_fraction', torch.tensor(1.0, device=device)))
                _surr_top = extras.get('surr_top_idx')
                if _surr_top is not None:
                    # Surrogate mode: count which heads' proposals are in top-M
                    # Indices: 0..N-1 = parents, N..2N-1 = head0, 2N..3N-1 = head1, ...
                    _flat = _surr_top.reshape(-1)
                    _is_parent = (_flat < N)
                    acc_surr_parent_count += _is_parent.sum().item()
                    acc_surr_total_count += _flat.numel()
                    # Proposal layout: permute(1,0,2,3,4).reshape(B, M*N*K, D)
                    # so proposal index p maps to head k = (p % (N*K)) % K
                    _prop_idx = (_flat[~_is_parent] - N)
                    _head_id = (_prop_idx % (N * _K)) % _K
                    for k in range(_K):
                        acc_winner_counts[k] += (_head_id == k).sum().float()
                    # Selection quality: compare selected vs rejected proposals
                    _saf = extras.get('surr_all_fit')
                    if _saf is not None:
                        with torch.no_grad():
                            _N_aug = _saf.shape[1]
                            _N_prop = _N_aug - N
                            _pf_rep = fitness.detach().float().repeat(
                                1, (_N_prop + N - 1) // N)[:, :_N_prop]
                            _prop_imp = _pf_rep - _saf[:, N:].float()  # (B, N_prop)
                            _sel_mask = torch.zeros(B, _N_prop, device=device, dtype=torch.bool)
                            for _b in range(B):
                                _sel_props = _surr_top[_b][_surr_top[_b] >= N] - N
                                if len(_sel_props) > 0:
                                    _sel_mask[_b, _sel_props] = True
                            _sel_imp = _prop_imp[_sel_mask]
                            _rej_imp = _prop_imp[~_sel_mask]
                            acc_surr_sel_imp.append({
                                'sel_imp_rate': (_sel_imp > 0).float().mean().item() if len(_sel_imp) > 0 else 0.0,
                                'rej_imp_rate': (_rej_imp > 0).float().mean().item() if len(_rej_imp) > 0 else 0.0,
                                'sel_med_imp': _sel_imp.median().item() if len(_sel_imp) > 0 else 0.0,
                                'rej_med_imp': _rej_imp.median().item() if len(_rej_imp) > 0 else 0.0,
                            })
                else:
                    _winner = extras.get('winner')
                    if _winner is not None:
                        acc_winner_counts += _winner.reshape(-1).bincount(minlength=_K).float()
                _imp = result.get('improved')
                if _imp is not None:
                    acc_improved_count += _imp.sum()
                    acc_improved_total += _imp.numel()
                    _amask = extras.get('active_mask')
                    if _amask is not None:
                        _active = (_amask > 0.5)
                        acc_improved_active += (_imp & _active).sum()
                        acc_active_total += _active.sum()

                chunk_best.append(result['best_fit'])
                chunk_fes.append(extras.get('fes_per_batch', _default_fes_batch))

                if args.cf_loss_weight > 0.0:
                    _bf_live = result.get('best_fit_live')
                    _pf_in = result.get('parent_fit_in')
                    if _bf_live is not None and _pf_in is not None:
                        cf_step = compute_cf_improvement_loss(
                            _pf_in, _bf_live,
                            weight=args.cf_loss_weight,
                            normalize=args.cf_loss_normalize)
                        chunk_cf.append(cf_step)

                geo = compute_geo_losses(extras, old_coords, fitness, fn, args, device)
                chunk_geo.extend(geo)
                if len(geo) > 0:
                    chunk_geo_names.extend([f'geo_{i}' for i in range(len(geo))])

                if args.contrafactual and x_star is not None:
                    dk_live = extras.get('deltas_k_live')
                    if dk_live is not None:
                        lupi_loss, di, al = compute_lupi_loss(
                            dk_live, c_graph, x_star, args.contra_weight)
                        if lupi_loss is not None:
                            chunk_geo.append(lupi_loss)
                            chunk_geo_names.append('lupi')
                        acc_lupi_dist_k.append(di)
                        acc_lupi_align_k.append(al)

                # PairwiseSurrogate ranking loss. Gated by
                # --surrogate-loss-weight (NOT --gate-bce-weight). This is the
                # only loss path that trains gen_step.surrogate; selection
                # itself (top_idx) is non-differentiable. Default weight is
                # 0.0 so legacy slurm scripts that do not opt in keep their
                # surrogate at init (matching prior behavior).
                if args.gate_type == 'surrogate' and args.surrogate_loss_weight > 0:
                    _surr_scores = extras.get('surr_scores')
                    _surr_all_fit = extras.get('surr_all_fit')
                    if _surr_scores is not None and _surr_all_fit is not None:
                        _N_aug = _surr_scores.shape[1]
                        _N_prop = _N_aug - N
                        _improvements = torch.zeros_like(_surr_scores)
                        _parent_rep = fitness.detach().float().repeat(1, _N_prop // N)
                        if _parent_rep.shape[1] < _N_prop:
                            _parent_rep = fitness.detach().float().repeat(
                                1, (_N_prop + N - 1) // N)[:, :_N_prop]
                        _improvements[:, N:] = _parent_rep[:, :_N_prop] - _surr_all_fit[:, N:].float()
                        _surr_loss = pairwise_ranking_loss(
                            _surr_scores, _improvements.detach(),
                            n_pairs=args.gate_n_pairs,
                            threshold_quantile=args.threshold_quantile)
                        if torch.isfinite(_surr_loss):
                            chunk_geo.append(args.surrogate_loss_weight * _surr_loss)
                            chunk_geo_names.append('surrogate_pw')
                            acc_surrogate_pw.append(_surr_loss.detach())
                        with torch.no_grad():
                            _labels = (_improvements > 0).float().reshape(-1)
                            _scores_flat = _surr_scores.detach().reshape(-1)
                            acc_gate_auc.append(build_gate_diag(
                                _scores_flat, _labels))

                elif args.gate_bce_weight > 0 and variant.activity_gate is not None:
                    if args.gate_type == 'ranker':
                        # Pairwise ranking loss for RankerGate
                        _gate_scores = extras.get('gate_scores')
                        _off_all = extras.get('off_fitness_all')
                        if _gate_scores is not None and _off_all is not None:
                            _off_red = _off_all.min(dim=0).values if _off_all.dim() == 3 else _off_all
                            _improvements = (fitness.detach() - _off_red).detach()
                            _pw_loss = pairwise_ranking_loss(
                                _gate_scores, _improvements,
                                n_pairs=args.gate_n_pairs,
                                threshold_quantile=args.threshold_quantile)
                            if torch.isfinite(_pw_loss):
                                chunk_geo.append(args.gate_bce_weight * _pw_loss)
                                chunk_geo_names.append('gate_bce_ranker')
                                acc_gate_bce.append(_pw_loss.detach())
                            with torch.no_grad():
                                _labels = (_improvements > 0).float().reshape(-1)
                                _scores_flat = _gate_scores.detach().reshape(-1)
                                acc_gate_auc.append(build_gate_diag(
                                    _scores_flat, _labels, extras.get('active_mask')))
                    else:
                        # Original contrafactual BCE for ActivityGate
                        bce_loss, gate_bce_scale_ema, bce_det, gate_diag = compute_gate_bce(
                            variant, extras, fitness, args.gate_bce_weight, gate_bce_scale_ema)
                        if bce_loss is not None:
                            chunk_geo.append(bce_loss)
                            chunk_geo_names.append('gate_bce_legacy')
                        if bce_det is not None:
                            acc_gate_bce.append(bce_det)
                        if gate_diag is not None:
                            acc_gate_auc.append(gate_diag)

                # Oracle CE loss for router
                if args.oracle_router_weight > 0:
                    oracle_loss, oracle_agree = compute_oracle_router_loss(
                        extras, fitness)
                    if oracle_loss is not None and torch.isfinite(oracle_loss):
                        chunk_geo.append(args.oracle_router_weight * oracle_loss)
                        chunk_geo_names.append('oracle_router')
                    if oracle_agree is not None:
                        acc_oracle_agreement.append(oracle_agree)

                # Beta F/CR supervised loss: dispatch to grid (legacy, extra FES)
                # or from_m (FES-free, uses realized F/CR of best-m).
                if args.fcr_beta_weight > 0 and '_F_mean' in extras:
                    if getattr(args, 'fcr_oracle_mode', 'from_m') == 'from_m':
                        fcr_loss, fcr_diag = compute_fcr_oracle_from_m_loss(
                            extras, fitness, weight=args.fcr_beta_weight)
                    else:
                        fcr_loss, fcr_diag = compute_fcr_grid_loss(
                            extras, old_coords, fitness, fn,
                            weight=args.fcr_beta_weight,
                            lb=args.lb, ub=args.ub)
                    if fcr_loss is not None:
                        chunk_geo.append(fcr_loss)
                        chunk_geo_names.append('fcr_beta')
                    if fcr_diag:
                        acc_fcr_diag.append(fcr_diag)

                # Donor oracle CE across M proposals. Dispatched when weight>0
                # OR when --donor-diag-always is set (probe/diag-only mode).
                _donor_w = getattr(args, 'donor_oracle_weight', 0.0)
                _donor_diag_always = (getattr(args, 'donor_diag_always', False)
                                      and getattr(args, 'per_m_donors', False))
                if _donor_w > 0 or _donor_diag_always:
                    donor_loss, donor_diag = compute_donor_oracle_loss(
                        extras, fitness, weight=_donor_w,
                        w_pbest=getattr(args, 'donor_w_pbest', 1.0),
                        w_r1=getattr(args, 'donor_w_r1', 1.0),
                        w_r2=getattr(args, 'donor_w_r2', 1.0),
                        r2_mode=getattr(args, 'donor_r2_mode', 'ce'),
                        r2_soft_frac=getattr(args, 'donor_r2_soft_frac', 0.3))
                    if donor_loss is not None:
                        chunk_geo.append(donor_loss)
                        chunk_geo_names.append('donor_oracle')
                    if donor_diag:
                        acc_donor_diag.append(donor_diag)

                # E13: F/CR online distillation loss (MSE on μ_F_pred vs realized F).
                # Active under --fcr-mode lshade with --fcr-distill-weight > 0.
                _fcr_distill_w = getattr(args, 'fcr_distill_weight', 0.0)
                if _fcr_distill_w > 0:
                    from l2o.loss_fns import compute_fcr_distill_loss
                    distill_loss, distill_diag = compute_fcr_distill_loss(
                        extras, weight=_fcr_distill_w,
                        mode=getattr(args, 'fcr_distill_mode', 'mse'))
                    if distill_loss is not None:
                        chunk_geo.append(distill_loss)
                        chunk_geo_names.append('fcr_distill')
                    if distill_diag:
                        acc_fcr_diag.append(distill_diag)

                # KL distillation: donor_selector logits → L-SHADE atomic soft
                # target (FES-aware adaptive p). Active under --kl-distill-weight>0.
                _kl_distill_w = getattr(args, 'kl_distill_weight', 0.0)
                if _kl_distill_w > 0:
                    from l2o.loss_fns import compute_kl_lshade_distill_loss
                    _fes_progress = float(cumulative_fes) / float(max(step_fes, 1))
                    kl_loss, kl_diag = compute_kl_lshade_distill_loss(
                        extras, fitness,
                        fes_progress=_fes_progress,
                        weight=_kl_distill_w,
                        p_max=getattr(args, 'kl_p_max', 0.2),
                        p_min=getattr(args, 'kl_p_min', 0.05),
                        w_pbest=getattr(args, 'kl_w_pbest', 1.0),
                        w_r1=getattr(args, 'kl_w_r1', 1.0),
                        w_r2=getattr(args, 'kl_w_r2', 1.0))
                    if kl_loss is not None:
                        chunk_geo.append(kl_loss)
                        chunk_geo_names.append('kl_distill')
                    if kl_diag:
                        acc_donor_diag.append(kl_diag)

                # [2026-05-04 disentangle] 2D structured supervision
                # (q_explor=acercamiento x*_global, q_exploit=acercamiento x*_local)
                # with HSIC orthogonality penalty. Per user proposal +
                # feedback_disentangle_explor_exploit_design_2026_05_04.
                if disentangle_heads is not None and x_star is not None:
                    _h_aug_live = extras.get('h_aug_live')
                    _coords_aug_live = extras.get('coords_aug_live')
                    _N_parents = extras.get('N_parents', N)
                    _M_var = extras.get('M_var', M)
                    _K_heads = extras.get('K_heads', 1)
                    if _h_aug_live is None or _coords_aug_live is None:
                        if step < 3:
                            log.info("disentangle skipped (h_aug_live=%s, coords_aug_live=%s)",
                                     _h_aug_live is not None, _coords_aug_live is not None)
                    if _h_aug_live is not None and _coords_aug_live is not None:
                        from l2o.disentangle_loss import compute_disentangle_loss
                        _M_proposals_total = _M_var * _N_parents * _K_heads
                        try:
                            dis_loss, dis_diag = compute_disentangle_loss(
                                _h_aug_live, _coords_aug_live,
                                old_coords, fitness,
                                x_global=x_star, ndim=D,
                                N=_N_parents, M_proposals=_M_proposals_total,
                                heads=disentangle_heads,
                                lambda_e=args.disentangle_lambda_e,
                                lambda_x=args.disentangle_lambda_x,
                                lambda_h=args.disentangle_lambda_h,
                                k_pop=getattr(args, 'disentangle_k_pop', 5),
                                random_target=getattr(args, 'disen_random_target', False))
                            if torch.isfinite(dis_loss):
                                chunk_geo.append(dis_loss)
                                chunk_geo_names.append('disentangle')
                                acc_attn_diag.append(dis_diag)
                                if step < 2 and gen == 0:
                                    log.info("disentangle gen0 step%d: L_e=%.4f L_x=%.4f HSIC=%.4f R²_e=%.3f R²_x=%.3f",
                                             step, dis_diag['disentangle_L_e'],
                                             dis_diag['disentangle_L_x'], dis_diag['disentangle_L_hsic'],
                                             dis_diag['disentangle_R2_explor'], dis_diag['disentangle_R2_exploit'])
                                    if 'antileak_cor_explor' in dis_diag:
                                        log.info("disentangle ANTILEAK gen0 step%d: cor_explor=%.4f cor_exploit=%.4f (expect |cor|<0.05 for random_target)",
                                                 step, dis_diag['antileak_cor_explor'], dis_diag['antileak_cor_exploit'])
                            else:
                                if step < 2:
                                    log.warning("disentangle non-finite at step %d: %s", step, dis_loss)
                        except Exception as _e:
                            log.warning("disentangle loss failed: %s", _e)

                # Structural attn diagnostics (entropy, Pearson(A_pbest, f_j),
                # Pearson(F_mean, fitness_rank)). Zero-weight — always on.
                _attn_diag = compute_attn_diag(extras, fitness)
                if _attn_diag:
                    acc_attn_diag.append(_attn_diag)

                coords = scale_gradient(result['new_coords'], HITTING_BB_SCALE)
                fitness = scale_gradient(result['new_fitness'], HITTING_BB_SCALE)
                coords = FIT_DAMP * coords + (1 - FIT_DAMP) * coords.detach()
                coords.register_hook(gc_hook)
                fitness = FIT_DAMP * fitness + (1 - FIT_DAMP) * fitness.detach()
                fitness.register_hook(gc_hook)
            else:
                with torch.no_grad():
                    cache = build_sparse_graphs_gpu(
                        coords.float(), fitness.float(),
                        step_num=cumulative_fes, max_steps=step_fes, ndim=D,
                        k_neighbors=args.k_neighbors,
                        stagnation_counters=pop_state.stagnation_counters,
                        delta_fitnesses=pop_state.delta_fitnesses,
                        contraction_rates=pop_state.contraction_rates,
                        prev_coords=prev_c, prev_fitnesses=prev_f)
                    _surr_m_now = compute_surrogate_m(
                        args.surrogate_M, args.surrogate_m_final,
                        cumulative_fes / step_fes, N)
                    _surr_m_now = min(_surr_m_now, N)
                    result = gen_step.run(
                        coords=coords, fitness=fitness, cache=cache,
                        f_optimal=fn.f_optimal, M=M, gumbel_tau=args.gumbel_tau,
                        node_feat=cache.node_feat, global_feat=cache.global_feat,
                        coords_hist=coords_hist, fitness_hist=fitness_hist,
                        n_valid=n_valid, fes_frac=cumulative_fes / step_fes,
                        gate_target_frac=args.gate_target_frac,
                        k_neighbors=args.k_neighbors,
                        step_num=cumulative_fes, max_steps=step_fes,
                        surrogate_M=_surr_m_now,
                        selection_spec=args.train_selection)
                    coords = result['new_coords']
                    fitness = result['new_fitness']

            extras = result.get('extras', {})

            # E13: update L-SHADE F/CR memory with this gen's successful trials.
            # Per-(b, n): pick the best-m trial; if it improved over the parent,
            # log its (F, CR, Δ) into the memory's success buffer.
            if _lshade_mem is not None and _pre_gen_fitness is not None:
                _F_real = extras.get('_realized_F')
                _CR_real = extras.get('_realized_CR')
                _off_all = extras.get('off_fitness_all')
                if _F_real is not None and _CR_real is not None and _off_all is not None:
                    with torch.no_grad():
                        # off_all (M, B, N): trial fitness per (m, b, n).
                        # best_m_idx (B, N): m-index of best trial per (b, n).
                        best_off, best_m = _off_all.min(dim=0)  # (B, N), (B, N)
                        # Improvement Δ = parent_fit - best_off (positive = improved).
                        delta = (_pre_gen_fitness.float() - best_off.float())
                        success = delta > 0.0
                        # Gather F, CR at best-m per (b, n).
                        Mdim, Bdim, Ndim = _F_real.shape
                        # best_m has shape (B, N); we want F_real[best_m, b, n].
                        b_idx = torch.arange(Bdim, device=_F_real.device).view(-1, 1).expand(-1, Ndim)
                        n_idx = torch.arange(Ndim, device=_F_real.device).view(1, -1).expand(Bdim, -1)
                        F_succ = _F_real[best_m, b_idx, n_idx].float()  # (B, N)
                        CR_succ = _CR_real[best_m, b_idx, n_idx].float()
                        _lshade_mem.update(F_succ, CR_succ, delta, success)

            # Full trajectory capture (1 in traj_every steps)
            if _traj_gens is not None:
                _tg = {
                    'gen': gen,
                    'coords': coords.detach().cpu().numpy().astype('float32'),
                    'fitness': fitness.detach().float().cpu().numpy(),
                    'new_coords': result['new_coords'].detach().cpu().numpy().astype('float32'),
                    'new_fitness': result['new_fitness'].detach().float().cpu().numpy(),
                }
                _sa = extras.get('surr_all_fit')
                if _sa is not None:
                    _tg['surr_all_fit'] = _sa.detach().float().cpu().numpy()
                _ss = extras.get('surr_scores')
                if _ss is not None:
                    _tg['surr_scores'] = _ss.detach().float().cpu().numpy()
                _st = extras.get('surr_top_idx')
                if _st is not None:
                    _tg['surr_top_idx'] = _st.detach().cpu().numpy()
                _dk = extras.get('deltas_k')
                if _dk is not None:
                    _tg['deltas_k'] = _dk.detach().cpu().numpy().astype('float32')
                _traj_gens.append(_tg)

            _fes = extras.get('fes_used', N)
            cumulative_fes += (_fes.detach().item() if torch.is_tensor(_fes) else _fes)
            total_gens += 1
            gen += 1

            if in_bptt and n_bptt_gens % args.bptt_chunk == 0 and chunk_best:
                # Backward unscaled; uniform scaling applied after loop
                hit_loss = compute_hitting_loss(
                    chunk_best, fn.f_optimal, log_tgt, args,
                    gap_init=gap_init, chunk_fes=chunk_fes)
                geo_loss = (torch.stack(chunk_geo).mean()
                            if chunk_geo else torch.tensor(0.0, device=device))
                cf_loss = (torch.stack(chunk_cf).mean()
                           if chunk_cf else torch.tensor(0.0, device=device))
                total_loss = hit_loss + geo_loss + cf_loss
                if os.environ.get('LOG_GRAD_DECOMPOSE'):
                    _decompose_chunk_backward(
                        hit_loss, chunk_geo, chunk_geo_names,
                        all_params, save_dir, step, n_chunks_done, rank,
                        named_groups=named_groups)
                _disen_clip = getattr(args, 'disen_grad_clip', 0.0)
                if _disen_clip > 0 and chunk_geo:
                    # A.1c: split disen aux from rest, backward + clip independently
                    _disen_terms = [t for t, n in zip(chunk_geo, chunk_geo_names)
                                    if 'disentangle' in n]
                    _other_terms = [t for t, n in zip(chunk_geo, chunk_geo_names)
                                    if 'disentangle' not in n]
                    _disen_loss = (torch.stack(_disen_terms).mean()
                                   if _disen_terms else None)
                    _other_geo = (torch.stack(_other_terms).mean()
                                  if _other_terms else torch.tensor(0.0, device=device))
                    _main_loss = hit_loss + _other_geo + cf_loss
                    components = []
                    if torch.isfinite(_main_loss):
                        components.append((_main_loss, args.grad_clip))
                    if _disen_loss is not None and torch.isfinite(_disen_loss):
                        components.append((_disen_loss, _disen_clip))
                    _per_component_backward(components, all_params)
                elif torch.isfinite(total_loss):
                    total_loss.backward()
                acc_hit_loss.append(hit_loss.detach())
                acc_geo_loss.append(geo_loss.detach())
                chunk_best, chunk_fes, chunk_geo = [], [], []
                chunk_cf = []
                chunk_geo_names = []
                n_chunks_done += 1
                coords = coords.detach().requires_grad_(True)
                fitness = fitness.detach().requires_grad_(True)
                live_coords = [c.detach() for c in live_coords]
                live_fitness = [f.detach() for f in live_fitness]

            if gen % 5 == 4 and ((fitness.min(dim=1).values.detach() - fn.f_optimal) < _hit_cut_thr).sum().item() > B / 2:
                break

        if chunk_best:
            hit_loss = compute_hitting_loss(
                chunk_best, fn.f_optimal, log_tgt, args,
                gap_init=gap_init, chunk_fes=chunk_fes)
            geo_loss = (torch.stack(chunk_geo).mean()
                        if chunk_geo else torch.tensor(0.0, device=device))
            cf_loss = (torch.stack(chunk_cf).mean()
                       if chunk_cf else torch.tensor(0.0, device=device))
            total_loss = hit_loss + geo_loss + cf_loss
            if os.environ.get('LOG_GRAD_DECOMPOSE'):
                _decompose_chunk_backward(
                    hit_loss, chunk_geo, chunk_geo_names,
                    all_params, save_dir, step, n_chunks_done, rank)
            _disen_clip = getattr(args, 'disen_grad_clip', 0.0)
            if _disen_clip > 0 and chunk_geo:
                _disen_terms = [t for t, n in zip(chunk_geo, chunk_geo_names)
                                if 'disentangle' in n]
                _other_terms = [t for t, n in zip(chunk_geo, chunk_geo_names)
                                if 'disentangle' not in n]
                _disen_loss = (torch.stack(_disen_terms).mean()
                               if _disen_terms else None)
                _other_geo = (torch.stack(_other_terms).mean()
                              if _other_terms else torch.tensor(0.0, device=device))
                _main_loss = hit_loss + _other_geo + cf_loss
                components = []
                if torch.isfinite(_main_loss):
                    components.append((_main_loss, args.grad_clip))
                if _disen_loss is not None and torch.isfinite(_disen_loss):
                    components.append((_disen_loss, _disen_clip))
                _per_component_backward(components, all_params)
            elif torch.isfinite(total_loss):
                total_loss.backward()
            acc_hit_loss.append(hit_loss.detach())
            acc_geo_loss.append(geo_loss.detach())
            n_chunks_done += 1

        # Uniform chunk scaling: divide accumulated gradients by actual chunk count
        if n_chunks_done > 1:
            _chunk_scale = 1.0 / n_chunks_done
            for p in all_params:
                if p.grad is not None:
                    p.grad.mul_(_chunk_scale)

        # Allreduce first (average across ranks), then clip per-group.
        allreduce_grads(all_params, world_size)
        # Per-group clip: gate BCE gradients are much larger than other modules.
        if _gate_params:
            torch.nn.utils.clip_grad_norm_(_gate_params, max_norm=50.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(_rest_params, max_norm=args.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=args.grad_clip)
        has_nan = check_nan_any_rank(grad_norm, device)
        if has_nan:
            optimizer.zero_grad()
            if is_distributed():
                dist.barrier()
        if not has_nan:
            optimizer.step()

        dt = time.perf_counter() - t0
        gap_per_pop = fitness.min(dim=1).values - fn.f_optimal
        target_hit = (gap_per_pop < my_target).sum().item() > B / 2
        gap_final = gap_per_pop.median().item()
        gap_ratio = gap_final / max(gap_init, 1e-12)

        if not args.no_curriculum:
            hit_local = torch.tensor(1.0 if target_hit else 0.0, device=device)
            tid_local = torch.tensor(my_task_id, dtype=torch.long, device=device)
            gnorm_local = torch.tensor(
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm, device=device)
            gens_local = torch.tensor(total_gens, dtype=torch.long, device=device)

            if is_distributed():
                all_tids = torch.zeros(world_size, dtype=torch.long, device=device)
                all_hits = torch.zeros(world_size, device=device)
                all_gnorms = torch.zeros(world_size, device=device)
                all_gens = torch.zeros(world_size, dtype=torch.long, device=device)
                dist.all_gather_into_tensor(all_tids, tid_local)
                dist.all_gather_into_tensor(all_hits, hit_local)
                dist.all_gather_into_tensor(all_gnorms, gnorm_local)
                dist.all_gather_into_tensor(all_gens, gens_local)
                if is_main():
                    _seen_tids = set()
                    for i in range(world_size):
                        _tid = all_tids[i].item()
                        if _tid not in task_pool or _tid in _seen_tids:
                            continue
                        _seen_tids.add(_tid)
                        next_task_id = update_task(
                            task_pool[_tid], hit=all_hits[i].item() > 0.5,
                            gn_val=all_gnorms[i].item(), total_gens=all_gens[i].item(),
                            args=args, task_pool=task_pool, next_task_id=next_task_id,
                            all_fn_ids=all_fn_ids, n_targets=n_targets, step=step, logger=log)
            else:
                if my_task_id in task_pool:
                    gn_val = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    next_task_id = update_task(
                        task_pool[my_task_id], hit=target_hit, gn_val=gn_val,
                        total_gens=total_gens, args=args, task_pool=task_pool,
                        next_task_id=next_task_id, all_fn_ids=all_fn_ids,
                        n_targets=n_targets, step=step, logger=log)

        if has_nan:
            continue

        # Save full trajectory (1 in traj_every)
        if _traj_gens is not None and _traj_gens:
            traj_dir.mkdir(parents=True, exist_ok=True)
            _traj_meta = {'step': step, 'fn': my_fn_id, 'D': D, 'N': N,
                          'task_id': my_task_id, 'gap_init': gap_init,
                          'gap_final': gap_final, 'f_optimal': float(fn.f_optimal)}
            _traj_arrays = {}
            for i, tg in enumerate(_traj_gens):
                for k, v in tg.items():
                    if isinstance(v, np.ndarray):
                        _traj_arrays[f'g{i:03d}/{k}'] = v
                    else:
                        _traj_meta[f'g{i:03d}/{k}'] = v
            _traj_arrays['metadata'] = np.array(_traj_meta)
            np.savez_compressed(traj_dir / f'step_{step:06d}.npz', **_traj_arrays)

        diag = build_step_diagnostics(
            step=step, rank=rank, fn_id=my_fn_id, D=D, N=N,
            task_id=my_task_id, aug_seed=my_aug_seed,
            target_hit=target_hit,
            gap_init=gap_init, gap_final=gap_final, gap_ratio=gap_ratio,
            total_gens=total_gens, n_bptt_gens=n_bptt_gens,
            bptt_w=my_bptt_w, step_fes=step_fes,
            cumulative_fes=cumulative_fes, n_chunks=n_chunks_done,
            dt=dt, grad_norm=grad_norm, diag_every=args.diag_every,
            named_groups=named_groups, get_grad_norm_fn=get_grad_norm,
            acc={
                'entropy': acc_entropy, 'active_frac': acc_active_frac,
                'lupi_dist_k': acc_lupi_dist_k, 'lupi_align_k': acc_lupi_align_k,
                'hit_loss': acc_hit_loss, 'geo_loss': acc_geo_loss,
                'gate_bce': acc_gate_bce,
                'surrogate_pw': acc_surrogate_pw,
                'gate_auc': acc_gate_auc,
                'oracle_agreement': acc_oracle_agreement,
                'improved_count': acc_improved_count,
                'improved_total': acc_improved_total,
                'winner_counts': acc_winner_counts,
                'surr_parent_count': acc_surr_parent_count,
                'surr_total_count': acc_surr_total_count,
                'surr_sel_imp': acc_surr_sel_imp,
                'fcr_diag': acc_fcr_diag,
                'donor_diag': acc_donor_diag,
                'attn_diag': acc_attn_diag,
            })

        if _used_warm_start:
            diag['warm_start'] = True
            diag['ws_fes_frac_start'] = round(_ws_fes_frac_start, 4)

        # Clip-activation metrics on the existing --grad-clip mechanism.
        # Combined with diag['fn_id'], these jsonl entries support fid-composition
        # + severity analysis of clip activations (Phase 0 finding 2026-04-30).
        diag.update(compute_clip_metrics(grad_norm, args.grad_clip))

        if diag_file is None:
            diag_file = open(diag_path, 'a' if (args.resume and start_step > 0) else 'w')
        diag_file.write(json.dumps(diag) + '\n')
        diag_file.flush()

        if step % 10 == 0:
            _t_idx = task_pool[my_task_id].curriculum_idx if my_task_id in task_pool else -1
            _rp = diag.get('route_pct', [0, 0, 0, 0])
            _vram = ""
            if torch.cuda.is_available():
                _alloc = torch.cuda.memory_allocated() / 1e9
                _peak = torch.cuda.max_memory_allocated() / 1e9
                _vram = f"  vram={_alloc:.1f}/{_peak:.1f}GB"
                torch.cuda.reset_peak_memory_stats()
            log.info("step %d  t%d:F%02d D%d  T%d %s  gap %.1e->%.1e (%.3f)"
                     "  gens=%d(%d)  gnorm=%.3f  dt=%.1fs%s",
                     step, my_task_id, my_fn_id, D, _t_idx,
                     "HIT" if target_hit else "miss",
                     gap_init, gap_final, gap_ratio,
                     total_gens, n_bptt_gens, diag['grad_norm'], dt, _vram)

        if is_distributed():
            stop_tensor = torch.tensor(1.0 if _es_stopped else 0.0, device=device)
            dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item() > 0.5:
                _es_stopped = True

        if _es_stopped or (step + 1) % args.ckpt_every == 0:
            if is_main():
                _ckpt = {
                    'step': step, 'backbone_state_dict': backbone.state_dict(),
                    'variant_state_dict': variant.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'task_pool': [t.to_dict() for t in task_pool.values()],
                    'next_task_id': next_task_id, 'config': vars(args),
                }
                if gen_step.surrogate is not None:
                    _ckpt['surrogate_state_dict'] = gen_step.surrogate.state_dict()
                if disentangle_heads is not None:
                    _ckpt['disentangle_heads_state_dict'] = disentangle_heads.state_dict()
                torch.save(_ckpt, save_dir / f'step_{step+1}.pth')
            if is_distributed():
                dist.barrier()

        is_eval_step = (
            args.eval_every > 0 and step > 0 and step % args.eval_every == 0)
        if is_eval_step and is_main():
            backbone.eval(); variant.eval(); gen_step.eval()
            _eval_D = dims_list[0]
            eval_results = run_canonical_eval(
                gen_step, all_fn_ids_eval, _eval_D, args.N, B,
                args.budget_mult * _eval_D, gru_W, device, build_sparse_graphs_gpu,
                gumbel_tau=args.gumbel_tau,
                m_samples=args.m_samples,
                k_neighbors=args.k_neighbors,
                variant=variant,
                selection_spec=args.train_selection,
                surrogate_M_init=args.surrogate_M,
                surrogate_m_final=args.surrogate_m_final,
                gate_target_frac=args.gate_target_frac,
                lpsr_n=args.lpsr_N,
                lpsr_n_min=args.lpsr_N_min)
            backbone.train(); variant.train(); gen_step.train()
            mean_gr = sum(eval_results.values()) / max(len(eval_results), 1)

            current_params = torch.cat([p.detach().float().flatten() for p in all_params])
            if _prev_param_snapshot is not None:
                rel_drift = (current_params - _prev_param_snapshot).norm().item() / max(_prev_param_snapshot.norm().item(), 1e-8)
            else:
                rel_drift = 1.0
            _param_drift_history.append(rel_drift)
            _prev_param_snapshot = current_params

            log.info("EVAL step %d: mean_gap_ratio=%.4f drift=%.6f", step, mean_gr, rel_drift)
            eval_diag = {'step': step, 'type': 'canonical_eval',
                         'mean_gap_ratio': round(mean_gr, 6), 'per_fn': eval_results}
            if diag_file is None:
                diag_file = open(diag_path, 'a')
            diag_file.write(json.dumps(eval_diag) + '\n')
            diag_file.flush()

            if mean_gr < _best_eval_gr:
                _best_eval_gr = mean_gr
                _evals_without_improvement = 0
                _best_ckpt = {'step': step, 'backbone_state_dict': backbone.state_dict(),
                              'variant_state_dict': variant.state_dict(),
                              'mean_gap_ratio': mean_gr}
                if gen_step.surrogate is not None:
                    _best_ckpt['surrogate_state_dict'] = gen_step.surrogate.state_dict()
                if disentangle_heads is not None:
                    _best_ckpt['disentangle_heads_state_dict'] = disentangle_heads.state_dict()
                torch.save(_best_ckpt, save_dir / 'best_eval.pth')
            else:
                _evals_without_improvement += 1

            if should_early_stop(
                patience_steps=args.patience,
                eval_every_steps=args.eval_every,
                evals_without_improvement=_evals_without_improvement,
                drift_history=_param_drift_history,
            ):
                log.info("EARLY STOP: %d evals without improvement + frozen params",
                         _evals_without_improvement)
                _es_stopped = True

        # Eval runs only on rank 0, but the early-stop flag must reach every
        # rank before the next iteration's `if _es_stopped: break` check —
        # otherwise rank 0 exits the loop and the rest deadlock at the next
        # allreduce. Broadcast outside the is_main() guard.
        if is_eval_step:
            _es_stopped = broadcast_es_stopped(_es_stopped, device)

      except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and 'CUDA' not in str(e):
            raise
        log.warning("step %d rank %d: %s: %s", step, rank, type(e).__name__, str(e)[:200])
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        if is_distributed():
            # Match the EXACT collective sequence of healthy ranks:
            # 1. allreduce_grads (with zero grads after zero_grad)
            # 2. check_nan_any_rank (will detect inf → skip optimizer)
            # 3. curriculum all_gathers (with zero data)
            allreduce_grads(all_params, world_size)
            _oom_norm = torch.tensor(float('inf'), device=device)
            check_nan_any_rank(_oom_norm, device)
            # optimizer.step() skipped on all ranks (has_nan=True)
            if not args.no_curriculum:
                _z_l = torch.zeros(1, dtype=torch.long, device=device)
                _z_f = torch.zeros(1, dtype=torch.float32, device=device)
                _d_tids = torch.zeros(world_size, dtype=torch.long, device=device)
                _d_hits = torch.zeros(world_size, dtype=torch.float32, device=device)
                _d_gnorms = torch.zeros(world_size, dtype=torch.float32, device=device)
                _d_gens = torch.zeros(world_size, dtype=torch.long, device=device)
                dist.all_gather_into_tensor(_d_tids, _z_l)
                dist.all_gather_into_tensor(_d_hits, _z_f)
                dist.all_gather_into_tensor(_d_gnorms, _z_f)
                dist.all_gather_into_tensor(_d_gens, _z_l)
            # Match early-stop broadcast from healthy path
            _stop = torch.tensor(1.0 if _es_stopped else 0.0, device=device)
            dist.broadcast(_stop, src=0)
            if _stop.item() > 0.5:
                _es_stopped = True

    if diag_file:
        diag_file.close()
    if is_main():
        _final = {
            'step': args.steps - 1, 'backbone_state_dict': backbone.state_dict(),
            'variant_state_dict': variant.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'task_pool': [t.to_dict() for t in task_pool.values()],
            'next_task_id': next_task_id, 'config': vars(args),
        }
        if gen_step.surrogate is not None:
            _final['surrogate_state_dict'] = gen_step.surrogate.state_dict()
        if disentangle_heads is not None:
            _final['disentangle_heads_state_dict'] = disentangle_heads.state_dict()
        torch.save(_final, save_dir / 'final.pth')
    log.info("Training complete: %d steps (start_step=%d, args.steps=%d), rank %d",
             args.steps - start_step, start_step, args.steps, rank)
    cleanup_distributed()


if __name__ == '__main__':
    main()
