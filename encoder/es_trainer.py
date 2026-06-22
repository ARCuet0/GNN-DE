"""
es_trainer.py — Unified ES (Evolution Strategy) training loop.

Head-agnostic: accepts any model with es_param_groups() and any rollout
function that returns RolloutResult. Extracted from GNN_MOS_Classic/train_l2o_es.py
to share all battle-tested improvements across K=2, K=4, and future systems.

Features inherited from GNN_MOS_Classic:
  - AdaptiveSigma (SNR-based step-size adaptation)
  - Antithetic epsilon sampling (40% variance reduction)
  - Rank normalization (Salimans 2017)
  - Adam optimizer on ES gradient
  - L2 regularization on perturbed params
  - Phase 2 parameter expansion (heads → all)
  - Generic multiprocess workers
  - Early stopping with patience
  - Rich 16-metric diagnostics (JSONL)
  - Held-out evaluation via callback
  - Unified checkpoint format
"""
import collections
import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
import torch.multiprocessing as mp

from encoder.adaptive_sigma import AdaptiveSigma
from encoder.es_utils import (
    RolloutResult, collect_params, write_params,
    FnWrapper, fn_to_info, make_fn, sample_function,
)
from encoder.ring_buffer import PopulationRingBuffer

log = logging.getLogger(__name__)


# ======================================================================
# Generic multiprocess worker
# ======================================================================

def _generic_worker(worker_id, base_flat_shared, n_params, perturb_mode,
                    input_q, output_q, device,
                    model_factory, rollout_fn, rollout_kwargs,
                    gru_window, gumbel_tau, span,
                    gamma, reward_p, entropy_bonus):
    """Persistent worker: builds model once, processes perturbation tasks."""
    try:
        if device != 'cpu':
            dev_idx = 0 if device == 'cuda' else int(str(device).split(':')[-1])
            torch.cuda.set_device(dev_idx)

        model = model_factory(device)
        model.eval()
        _, meta = collect_params(model, perturb_mode)

        ring_bufs = None
        last_B = last_N = last_D = 0

        while True:
            task = input_q.get()
            if task is None:
                break

            m_idx, eps_m, fn_infos, D, N, B, es_sigma = task

            flat = base_flat_shared.to(device) + es_sigma * eps_m.to(device)
            write_params(flat, meta)

            fns = [make_fn(info, device) for info in fn_infos]

            if B != last_B or N != last_N or D != last_D:
                ring_bufs = [PopulationRingBuffer(gru_window, N, D, device)
                             for _ in range(B)]
                last_B, last_N, last_D = B, N, D

            try:
                result = rollout_fn(
                    model, D, N, fns, device, span,
                    gru_window, gumbel_tau, ring_bufs,
                    gamma=gamma, reward_p=reward_p,
                    entropy_bonus=entropy_bonus, **rollout_kwargs)
            except Exception as e:
                import traceback
                log.warning("Worker %d rollout failed: %s\n%s",
                            worker_id, e, traceback.format_exc())
                result = RolloutResult(0.0, 0.0, 0.0)

            output_q.put((m_idx, result.neg_return, result.entropy,
                         result.gap_closure))

    except Exception:
        import traceback
        traceback.print_exc()
        try:
            output_q.put(('DEAD', worker_id, 0.0, 0.0))
        except Exception:
            pass


# ======================================================================
# Unified ES training loop
# ======================================================================

def train_es(
    # Model
    model,
    model_factory: Callable,
    # Rollout
    rollout_fn: Callable[..., RolloutResult],
    rollout_kwargs: Optional[Dict] = None,
    *,
    # ES core
    n_steps: int = 5000,
    M: int = 32,
    sigma_init: float = 0.05,
    lr: float = 1e-3,
    sigma_mode: str = 'adaptive',
    sigma_schedule: Optional[List[float]] = None,
    sigma_patience: int = 150,
    # Population
    pop_per_dim: int = 5,
    dims=(10, 30, 50),
    batch_fns: int = 4,
    no_level3: bool = False,
    # Function sampling
    allowed_fids=None,
    no_augment: bool = False,
    fn_sampler: Optional[Callable] = None,
    # Perturbation
    perturb_mode: str = 'gat+heads',
    phase2_step: int = 0,
    l2_coeff: float = 0.0,
    # Return shaping
    gamma: float = 0.99,
    reward_p: float = 1.0,
    entropy_bonus: float = 0.0,
    # Workers
    n_workers: int = 0,
    # Checkpointing
    save_dir: str = 'checkpoints',
    save_every: int = 200,
    log_every: int = 10,
    diag_every: int = 50,
    run_id: str = '',
    # Eval
    eval_fn: Optional[Callable] = None,
    eval_every: int = 200,
    # Resume
    resume_ckpt: Optional[str] = None,
    resume_hook: Optional[Callable] = None,
    # Early stopping
    patience: int = 0,
    # Shared
    device: str = 'cuda',
    gru_window: int = 8,
    gumbel_tau: float = 0.5,
    span: float = 200.0,
) -> float:
    """Head-agnostic ES training loop.

    Args:
        model: nn.Module with es_param_groups() method.
        model_factory: Callable(device) -> model, for worker spawning.
        rollout_fn: Callable returning RolloutResult.
        rollout_kwargs: Extra kwargs forwarded to rollout_fn.
        resume_hook: Callable(model, ckpt_dict) for model-specific resume logic.
        eval_fn: Callable(model, device, dims, step) -> dict for held-out eval.

    Returns:
        Best avg gap closure achieved.
    """
    from encoder.augmented_cec2017 import AugmentedCEC2017

    if rollout_kwargs is None:
        rollout_kwargs = {}

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Resume from checkpoint
    if resume_ckpt is not None:
        log.info("Resuming from %s", resume_ckpt)
        ckpt = torch.load(resume_ckpt, map_location=device, weights_only=False)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'], strict=False)
        elif 'router_state_dict' in ckpt:
            model.load_state_dict(ckpt['router_state_dict'], strict=False)
        if resume_hook is not None:
            resume_hook(model, ckpt)
        log.info("Checkpoint loaded (step %d, avg_gc=%.4f)",
                 ckpt.get('step', 0), ckpt.get('avg_gc', 0.0))

    # Collect params for ES
    current_perturb_mode = perturb_mode
    base_flat, meta = collect_params(model, current_perturb_mode)
    n_params = base_flat.numel()
    total_params = sum(p.numel() for p in model.parameters())
    log.info("ES: %d/%d params perturbed (mode=%s), M=%d, sigma=%.3f, lr=%.1e",
             n_params, total_params, current_perturb_mode, M, sigma_init, lr)
    if phase2_step > 0:
        log.info("Phase 2 at step %d: expand to 'all' (%d params)",
                 phase2_step, total_params)

    # Adam state
    m_est = torch.zeros(n_params, device=device)
    v_est = torch.zeros(n_params, device=device)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8
    n_adam_updates = 0

    # Sigma adaptation
    if sigma_schedule is None:
        sigma_schedule = [sigma_init, sigma_init * 0.4,
                          sigma_init * 0.2, sigma_init * 0.1]
    es_sigma = sigma_init
    adaptive = AdaptiveSigma(
        sigma_init=es_sigma,
        sigma_min=sigma_schedule[-1],
        sigma_max=sigma_schedule[0] * 2,
    )

    # Function augmentation
    aug = None if no_augment else AugmentedCEC2017(device=device, dims=dims)

    # Tracking
    history = []
    WINDOW = 50
    recent_gcs = collections.deque(maxlen=WINDOW)
    best_avg_gc = -1.0
    steps_since_improvement = 0

    t0_total = time.time()
    B = batch_fns
    dims_list = list(dims)

    # --- Launch worker pool ---
    workers = []
    input_q = output_q = None
    base_flat_shared = None

    if n_workers > 0:
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        base_flat_shared = base_flat.cpu().share_memory_()
        input_q = mp.Queue()
        output_q = mp.Queue()
        for w in range(n_workers):
            p = mp.Process(target=_generic_worker, args=(
                w, base_flat_shared, n_params, current_perturb_mode,
                input_q, output_q, device,
                model_factory, rollout_fn, rollout_kwargs,
                gru_window, gumbel_tau, span,
                gamma, reward_p, entropy_bonus))
            p.daemon = True
            p.start()
            workers.append(p)
        log.info("Launched %d worker processes", n_workers)

    # =================================================================
    # Main training loop
    # =================================================================
    step = -1
    for step in range(n_steps):
        t0 = time.time()

        # Cycle dims
        if step % len(dims_list) == 0:
            random.shuffle(dims_list)
        D = dims_list[step % len(dims_list)]
        ppd = pop_per_dim if no_level3 else random.choice([3, 5, 7])
        N = ppd * D

        # Sample B functions
        fns = []
        _sampler = fn_sampler or sample_function
        for _ in range(B):
            fn = _sampler(aug, device, D, dims, allowed_fids,
                          no_augment, random.randint(0, 2**31))
            fns.append(fn)

        # Antithetic perturbations
        eps_device = 'cpu' if n_workers > 0 else device
        M_half = M // 2
        eps_half = torch.randn(M_half, n_params, device=eps_device)
        eps_all = torch.cat([eps_half, -eps_half], dim=0)

        losses = torch.zeros(M, device=device)
        entropies = torch.zeros(M, device=device)
        gcs = torch.zeros(M, device=device)

        if n_workers > 0:
            # --- Multiprocess path ---
            base_flat_shared.copy_(base_flat.cpu())
            fn_infos = [fn_to_info(fn) for fn in fns]

            for m in range(M):
                input_q.put((m, eps_all[m], fn_infos, D, N, B, es_sigma))

            n_dead = 0
            for _ in range(M):
                result = output_q.get(timeout=300)
                if result[0] == 'DEAD':
                    n_dead += 1
                    log.error("Worker %d died!", result[1])
                    continue
                m_idx, loss, ent, gc = result
                losses[m_idx] = loss
                entropies[m_idx] = ent
                gcs[m_idx] = gc
            if n_dead > 0:
                log.error("%d/%d workers died — aborting", n_dead, n_workers)
                break
        else:
            # --- Sequential path ---
            if eps_all.device != base_flat.device:
                eps_all = eps_all.to(device)

            ring_bufs = [PopulationRingBuffer(gru_window, N, D, device)
                         for _ in range(B)]

            for m in range(M):
                write_params(base_flat + es_sigma * eps_all[m], meta)

                result = rollout_fn(
                    model, D, N, fns, device, span,
                    gru_window, gumbel_tau, ring_bufs,
                    gamma=gamma, reward_p=reward_p,
                    entropy_bonus=entropy_bonus, **rollout_kwargs)

                losses[m] = result.neg_return
                entropies[m] = result.entropy
                gcs[m] = result.gap_closure

            write_params(base_flat, meta)

        # ── ES gradient with rank normalization ──
        ranks = losses.argsort().argsort().float()
        shaped = (ranks / max(M - 1, 1)) - 0.5
        eps_gpu = eps_all.to(device) if eps_all.device != losses.device else eps_all
        g_est = (eps_gpu * shaped.unsqueeze(1)).mean(dim=0) / es_sigma

        # AdaptiveSigma: observe gradient signal quality
        improv_frac = (losses < losses.median()).float().mean().item()
        if sigma_mode == 'adaptive':
            es_sigma = adaptive.observe(eps_gpu, shaped, improv_frac)

        # ── Adam update ──
        n_adam_updates += 1
        if l2_coeff > 0:
            g_est = g_est + l2_coeff * base_flat
        m_est = beta1 * m_est + (1 - beta1) * g_est
        v_est = beta2 * v_est + (1 - beta2) * g_est ** 2
        m_hat = m_est / (1 - beta1 ** n_adam_updates)
        v_hat = v_est / (1 - beta2 ** n_adam_updates)
        base_flat = base_flat - lr * m_hat / (v_hat.sqrt() + eps_adam)

        write_params(base_flat, meta)

        # ── Logging ──
        mean_gc = gcs.mean().item()
        best_gc = gcs.max().item()
        mean_ent = entropies.mean().item()
        mean_loss = losses.mean().item()
        recent_gcs.append(mean_gc)
        avg_gc = sum(recent_gcs) / len(recent_gcs)

        # ── Early stopping ──
        if avg_gc > best_avg_gc + 1e-4:
            best_avg_gc = avg_gc
            steps_since_improvement = 0
        else:
            steps_since_improvement += 1
        if patience > 0 and steps_since_improvement >= patience:
            log.info("Early stopping at step %d (no improvement for %d steps, "
                     "best avg_gc=%.4f)", step, patience, best_avg_gc)
            break

        # ── Legacy sigma modes ──
        if sigma_mode == 'entropy':
            S_HIGH, S_LOW = sigma_schedule[0], sigma_schedule[-1]
            ENT_HIGH, ENT_LOW = 1.0, 0.3
            if mean_ent >= ENT_HIGH:
                target = S_HIGH
            elif mean_ent <= ENT_LOW:
                target = S_LOW
            else:
                t = (mean_ent - ENT_LOW) / (ENT_HIGH - ENT_LOW)
                target = S_LOW + t * (S_HIGH - S_LOW)
            es_sigma = 0.9 * es_sigma + 0.1 * target
            es_sigma = max(es_sigma, S_LOW)
        # 'fixed' and 'adaptive' modes: no legacy sigma update needed

        # ── Phase 2 expansion ──
        if (phase2_step > 0 and step == phase2_step
                and current_perturb_mode != 'all'):
            write_params(base_flat, meta)
            current_perturb_mode = 'all'
            base_flat, meta = collect_params(model, 'all')
            n_params = base_flat.numel()
            m_est = torch.zeros(n_params, device=device)
            v_est = torch.zeros(n_params, device=device)
            n_adam_updates = 0
            adaptive.reset(sigma_init=sigma_init)
            es_sigma = adaptive.sigma
            steps_since_improvement = 0
            log.info("Phase 2: expanded to 'all' (%d params), reset sigma=%.4f",
                     n_params, es_sigma)

        dt = time.time() - t0

        # ── Rich diagnostics (JSONL) ──
        if diag_every > 0 and step % diag_every == 0:
            bfn = base_flat.norm().item()
            update_norm = (lr * m_hat / (v_hat.sqrt() + eps_adam)).norm().item()
            diag = {
                'step': step, 'D': D, 'N': N,
                'g_norm': g_est.norm().item(),
                'g_max': g_est.abs().max().item(),
                'w_norm': bfn,
                'dw_norm': update_norm,
                'snr': update_norm * M / es_sigma if es_sigma > 0 else 0,
                'loss_std': losses.std().item(),
                'loss_min': losses.min().item(),
                'loss_max': losses.max().item(),
                'm_hat_norm': m_hat.norm().item(),
                'v_hat_mean': v_hat.sqrt().mean().item(),
                'sigma_to_w': es_sigma / max(bfn, 1e-10),
                'ent_std': entropies.std().item(),
                'gc_std': gcs.std().item(),
                'improv_frac': improv_frac,
            }
            diag_path = save_path / f"{run_id}_diag.jsonl" if run_id else save_path / "es_diag.jsonl"
            with open(diag_path, 'a') as f:
                f.write(json.dumps(diag) + '\n')

            log.info("  diag: ‖g‖=%.4f ‖w‖=%.1f ‖Δw‖=%.5f snr=%.2f "
                     "loss_std=%.3f gc_std=%.3f",
                     diag['g_norm'], bfn, update_norm, diag['snr'],
                     diag['loss_std'], diag['gc_std'])

        # ── Step log ──
        rec = {
            'step': step, 'D': D, 'N': N,
            'mean_loss': mean_loss, 'mean_gc': mean_gc,
            'best_gc': best_gc, 'avg_gc': avg_gc,
            'mean_entropy': mean_ent, 'sigma': es_sigma,
            'perturb_mode': current_perturb_mode,
            'n_perturbed': n_params, 'step_time': dt,
        }
        history.append(rec)

        if step % log_every == 0:
            elapsed = time.time() - t0_total
            eta_h = elapsed / (step + 1) * (n_steps - step - 1) / 3600
            log.info(
                "step %4d | D=%d N=%d | loss=%.3f gc=%.3f avg=%.3f "
                "best=%.3f | ent=%.3f σ=%.4f | %.1fs (ETA %.1fh)",
                step, D, N, mean_loss, mean_gc, avg_gc, best_gc,
                mean_ent, es_sigma, dt, eta_h)

        # ── Checkpoint ──
        if (step + 1) % save_every == 0 or step == n_steps - 1:
            ckpt = {
                'model_state_dict': model.state_dict(),
                'base_flat': base_flat.cpu(),
                'perturb_mode': current_perturb_mode,
                'es_state': {
                    'm_est': m_est.cpu(), 'v_est': v_est.cpu(),
                    'n_updates': n_adam_updates,
                    'adaptive_sigma': adaptive.state_dict(),
                },
                'step': step + 1,
                'avg_gc': avg_gc,
                'config': {
                    'M': M, 'sigma_init': sigma_init, 'lr': lr,
                    'sigma_mode': sigma_mode, 'perturb_mode': perturb_mode,
                    'dims': list(dims), 'pop_per_dim': pop_per_dim,
                    'batch_fns': batch_fns, 'gamma': gamma,
                    'n_params': n_params,
                },
                'history': history[-save_every:],
            }
            prefix = run_id or 'es'
            torch.save(ckpt, save_path / f"{prefix}_step{step + 1}.pth")
            if avg_gc >= best_avg_gc - 1e-6:
                torch.save(ckpt, save_path / f"{prefix}_best.pth")
                log.info("New best avg_gc=%.4f saved", avg_gc)

        # ── Full history ──
        if (step + 1) % (save_every * 5) == 0 or step == n_steps - 1:
            hist_name = f"{run_id}_history.json" if run_id else "es_history.json"
            with open(save_path / hist_name, 'w') as f:
                json.dump(history, f)

        # ── Held-out evaluation callback ──
        if (eval_fn is not None and eval_every > 0
                and step % eval_every == 0 and step > 0):
            eval_res = eval_fn(model, device, dims, step)
            if eval_res:
                eval_name = f"{run_id}_eval.jsonl" if run_id else "es_eval.jsonl"
                with open(save_path / eval_name, 'a') as f:
                    f.write(json.dumps({'step': step, **eval_res}) + '\n')
                log.info("EVAL step %d | %s", step,
                         ' | '.join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                    for k, v in eval_res.items()))

    # ── Shutdown workers ──
    if workers:
        for _ in workers:
            input_q.put(None)
        for p in workers:
            p.join(timeout=10)

    total_time = time.time() - t0_total
    log.info("ES complete. %d steps in %.1fs (%.2fs/step). Best avg gc: %.4f",
             step + 1, total_time,
             total_time / max(step + 1, 1), best_avg_gc)
    return best_avg_gc
