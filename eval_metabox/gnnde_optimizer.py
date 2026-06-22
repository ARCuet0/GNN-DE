"""GNN-DE (TersQ deployed checkpoint) wrapped as a MetaBox ``Basic_Optimizer``.

Lets the deployed TersQ k1+surrogate model be evaluated inside the MetaBox
platform (e.g. on the Protein-Docking suite) alongside the RLDE-AFL baselines.
It mirrors ``eval_e7d_parallel.eval_single`` for the loop structure but applies
the non-CEC bounds overrides that ``eval_bbob_smoke.py`` uses (eval_single omits
them ONLY because the CEC2017 box IS ``[-100, 100]``; protein is ``[-1.5, 1.5]``):

  * population initialised in the problem's ``[lb, ub]`` box (not ``[-100, 100]``),
  * ``gen_step.lb`` / ``gen_step.ub`` set to the problem box (proposals are
    clamped to these inside ``_run_surrogate``),
  * ``backbone.temporal.coord_range = (ub - lb) / 2`` (the temporal encoder
    divides coords by this; default 100 would compress protein coords ~66x),
  * ``build_sparse_graphs_gpu(..., lb=lb, ub=ub)`` so the graph normalisation box
    matches the problem (default ``[-100, 100]`` would collapse coords_norm ~66x
    and saturate the diversity/ruggedness global features), and
  * an ``eval_fn`` adapter wrapping ``problem.eval`` (numpy ``[K, D] -> [K]``)
    into the ``(K, D) torch -> (K,) torch`` shape the generation step expects,
  * best-so-far cost-curve logging following the MetaBox ``Random_search``
    convention (so the curve has ``n_logpoint + 1`` entries), and
  * a FES cap so cumulative FES never exceeds ``config.maxFEs`` (the paper's
    protein budget is 500; MetaBox's ``config.py`` would otherwise force 2000).

Recipe is pinned to the deployed inference recipe: ``selection_spec='random_1pp'``,
``M_var=20``, ``per_m_donors=True`` (see CLAUDE.md and the F5 finding).

This is a SMOKE-grade wrapper. For the full 280x51 MetaBox run it still needs to
be registered in the MetaBox baseline registry / Tester. See
finding_rldeafl_protein_docking_metabox_2026_06_10.
"""
import os
import sys

import numpy as np
import torch

# Repo root = parent of this file's directory (eval_metabox/).
TERSQ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_METABOX_CLONE = os.path.join(TERSQ_ROOT, 'literature', 'rl_metabbo', 'MetaBox')

DEPLOYED_CKPT = os.path.join(
    TERSQ_ROOT, 'checkpoint', 'published_checkpoint.pth')

# Deployed inference recipe.
_N = 50
_M_VAR = 20
_SELECTION = 'random_1pp'
_SURROGATE_M = 50       # LPSR-M schedule start (matches eval_e7d_parallel._build_args)
_SURROGATE_M_FINAL = 5  # LPSR-M schedule end
_GRU_W = 16


from eval_metabox._metabox_compat import get_basic_optimizer

Basic_Optimizer = get_basic_optimizer()


def make_metabox_eval_fn(problem, device='cpu', f_optimal=0.0):
    """Wrap a MetaBox ``problem.eval`` into the tensor eval_fn GNN-DE expects.

    GNN-DE calls ``eval_fn(coords)`` with ``coords`` a ``(K, D)`` torch tensor
    (any dtype/device) and requires a finite ``(K,)`` torch tensor back, on the
    same device, minimization-consistent. Protein energy is already a minimized
    positive scalar, so no sign flip is needed. ``.f_optimal`` is attached to
    mirror the CEC2017 benchmark fn that ``eval_single`` consumes.
    """
    def eval_fn(coords):
        x_np = coords.detach().to('cpu', torch.float64).numpy()
        y = problem.eval(x_np)
        if torch.is_tensor(y):
            y = y.detach().cpu().numpy()
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        return torch.from_numpy(y).to(device=coords.device, dtype=torch.float64)

    eval_fn.f_optimal = float(f_optimal)
    return eval_fn


# Module-level cache: build the model ONCE (a per-episode rebuild would re-init
# random message-passing layers and dominate wall-clock).
_TERSQ = {}


def _ensure_tersq(device, ckpt=DEPLOYED_CKPT):
    """Build backbone+variant+gen_step once via eval_e7d_parallel._worker_init
    (which carries the strict architecture-mismatch guard, Bug Prevention #5)."""
    if _TERSQ.get('ready') and _TERSQ.get('ckpt') == ckpt and _TERSQ.get('device') == device:
        return _TERSQ
    if TERSQ_ROOT not in sys.path:
        sys.path.insert(0, TERSQ_ROOT)
    import eval_e7d_parallel as E

    # Reuse the worker init so model loading / arch guard / deploy knobs match
    # the canonical eval path exactly.
    E._WORKER_STATE.clear()
    E._worker_init(ckpt, device)

    _TERSQ['state'] = E._WORKER_STATE
    _TERSQ['apply_per_m_donors'] = E._apply_per_m_donors
    _TERSQ['apply_use_ste'] = E._apply_use_ste
    _TERSQ['ckpt'] = ckpt
    _TERSQ['device'] = device
    _TERSQ['ready'] = True
    return _TERSQ


class GNN_DE(Basic_Optimizer):
    """Deployed TersQ optimizer as a MetaBox Basic_Optimizer.

    Config attributes used: ``maxFEs``, ``n_logpoint`` (default 50),
    ``log_interval`` (default ``maxFEs // n_logpoint``), ``device`` (default
    'cpu'), ``full_meta_data`` (default False), and optional ``gnnde_ckpt``.
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._max_fes = int(config.maxFEs)
        self._n_logpoint = int(getattr(config, 'n_logpoint', 50))
        self.log_interval = int(getattr(config, 'log_interval',
                                        max(1, self._max_fes // self._n_logpoint)))
        self.device = getattr(config, 'device', 'cpu')
        self.full_meta_data = bool(getattr(config, 'full_meta_data', False))
        self.ckpt = getattr(config, 'gnnde_ckpt', DEPLOYED_CKPT)
        self._seed = None

    def __str__(self):
        return 'GNN_DE'

    def seed(self, seed=None):
        super().seed(seed)
        self._seed = self.rng_seed

    def run_episode(self, problem):
        from l2o.schedules import PopulationGenState, compute_surrogate_m
        from encoder.opt_variant import _clamp_fitness
        from encoder.similarity_graph_gpu import build_sparse_graphs_gpu

        st = _ensure_tersq(self.device, self.ckpt)
        state = st['state']
        gen_step = state['gen_step']
        variant = state['variant']
        device = state['device']

        st['apply_per_m_donors'](variant, True)   # --per-m-donors
        st['apply_use_ste'](variant, True)         # deployed STE crossover

        # Per-episode state reset (mirror eval_single). Inert under the deployed
        # recipe but guards against leakage if this wrapper is reused for an
        # ablation (oracle / F-schedule selector) across the shared cached model.
        if hasattr(variant, '_oracle_best_k'):
            variant._oracle_best_k = None
        if hasattr(variant, 'heads'):
            for h in variant.heads:
                if hasattr(h, '_force_F_attr'):
                    h._force_F_attr = None
                if hasattr(h, '_force_CR_attr'):
                    h._force_CR_attr = None

        D = int(problem.dim)
        N = _N
        budget = self._max_fes
        lb = float(np.min(problem.lb)) if hasattr(problem.lb, '__len__') else float(problem.lb)
        ub = float(np.max(problem.ub)) if hasattr(problem.ub, '__len__') else float(problem.ub)
        gen_step.lb, gen_step.ub = lb, ub          # proposals are clamped to this box
        # Temporal encoder normalisation: default coord_range=100 (CEC) would
        # compress the [-1.5,1.5] protein coords ~66x. Mirror eval_bbob_smoke.py.
        state['backbone'].temporal.coord_range = (ub - lb) / 2.0

        f_optimal = 0.0 if getattr(problem, 'optimum', None) is None else float(problem.optimum)
        eval_fn = make_metabox_eval_fn(problem, device, f_optimal)
        gen_step.eval_fn = eval_fn

        problem.reset()  # MetaBox: zeroes T1 timing before eval (Random_search does this)
        if not hasattr(self, 'rng_cpu'):
            self.seed(self._seed if self._seed is not None else 0)
        _seed = self._seed if self._seed is not None else 0
        # The DE head's gumbel-softmax donor sampling + Beta (F,CR) sampling draw
        # from the GLOBAL torch stream (no generator hook exists), so seed it per
        # episode for reproducibility, exactly as eval_single does. This requires
        # episodes to run SEQUENTIALLY per process (the Magerit shard runner does);
        # it is the head's only RNG and cannot be isolated without a code change.
        torch.manual_seed(_seed)
        # sel_rng additionally isolates the two components that CAN be isolated
        # (init population + random_1pp selection) from global-stream contamination
        # and aligns with MetaBox's per-optimizer generator convention.
        sel_rng = (self.rng_gpu if (self.device != 'cpu' and getattr(self, 'rng_gpu', None) is not None)
                   else self.rng_cpu)

        meta_X = [] if self.full_meta_data else None
        meta_C = [] if self.full_meta_data else None

        # ---- init population in [lb, ub]^D ------------------------------
        coords = (torch.rand(1, N, D, device=device, generator=sel_rng) * (ub - lb) + lb).to(torch.float64)
        fitness = _clamp_fitness(eval_fn(coords.reshape(-1, D)).reshape(1, N))
        cumulative_fes = N
        gbest = float(fitness.min().item())
        if self.full_meta_data:
            meta_X.append(coords.squeeze(0).cpu().numpy())
            meta_C.append(fitness.squeeze(0).cpu().numpy())

        # cost curve, MetaBox Random_search convention
        cost = [gbest]
        log_index = 1

        coords_ring = torch.zeros(1, _GRU_W, N, D, dtype=torch.float32, device=device)
        fitness_ring = torch.zeros(1, _GRU_W, N, dtype=torch.float32, device=device)
        pop_state = PopulationGenState(B=1, device=device)

        with torch.no_grad():
            gen = 0
            while cumulative_fes < budget:
                ri = gen % _GRU_W
                coords_ring[:, ri] = coords.float()
                fitness_ring[:, ri] = fitness.float()
                n_valid = min(gen + 1, _GRU_W)
                if gen < _GRU_W:
                    idx = list(range(gen + 1))
                else:
                    start = (gen + 1) % _GRU_W
                    idx = [(start + i) % _GRU_W for i in range(_GRU_W)]
                coords_hist = coords_ring[:, idx]
                fitness_hist = fitness_ring[:, idx]
                prev_c = coords_ring[:, (ri - 1) % _GRU_W].float() if gen > 0 else None
                prev_f = fitness_ring[:, (ri - 1) % _GRU_W].float() if gen > 0 else None

                pop_state.update(coords, fitness)

                cache = build_sparse_graphs_gpu(
                    coords.float(), fitness.float(),
                    step_num=cumulative_fes, max_steps=budget, ndim=D,
                    k_neighbors=8, lb=lb, ub=ub,
                    stagnation_counters=pop_state.stagnation_counters,
                    delta_fitnesses=pop_state.delta_fitnesses,
                    contraction_rates=pop_state.contraction_rates,
                    prev_coords=prev_c, prev_fitnesses=prev_f)

                surr_m_now = compute_surrogate_m(
                    _SURROGATE_M, _SURROGATE_M_FINAL, cumulative_fes / budget, N)
                # FES cap: never overshoot maxFEs. fes_used == M_sel under
                # random_1pp (all selected indices are proposals), so capping
                # M_sel caps the charge for this generation. int() because the
                # selector's topk wants an int k (cumulative_fes is float).
                surr_m_now = int(min(surr_m_now, N, budget - cumulative_fes))
                if surr_m_now <= 0:
                    break

                result = gen_step.run(
                    coords=coords, fitness=fitness, cache=cache,
                    f_optimal=f_optimal, M=_M_VAR, gumbel_tau=1.0,
                    node_feat=cache.node_feat, global_feat=cache.global_feat,
                    coords_hist=coords_hist, fitness_hist=fitness_hist,
                    n_valid=n_valid, fes_frac=cumulative_fes / budget,
                    surrogate_M=surr_m_now,
                    selection_spec=_SELECTION,
                    selection_generator=sel_rng,
                    greedy_1to1=False)

                coords = result['new_coords'].detach()
                fitness = result['new_fitness'].detach()
                extras = result.get('extras', {})
                _fes = extras.get('fes_used', float(N))
                cumulative_fes += (_fes.item() if hasattr(_fes, 'item') else _fes)

                gen_min = float(fitness.min().item())
                if gen_min < gbest:
                    gbest = gen_min

                while cumulative_fes >= log_index * self.log_interval:
                    log_index += 1
                    cost.append(gbest)

                if self.full_meta_data:
                    meta_X.append(coords.squeeze(0).cpu().numpy())
                    meta_C.append(fitness.squeeze(0).cpu().numpy())
                gen += 1

        # pad / truncate to n_logpoint + 1 (MetaBox convention)
        if len(cost) >= self._n_logpoint + 1:
            cost = cost[:self._n_logpoint + 1]
            cost[-1] = gbest
        else:
            while len(cost) < self._n_logpoint + 1:
                cost.append(gbest)

        results = {'cost': cost, 'fes': int(cumulative_fes)}
        if self.full_meta_data:
            results['metadata'] = {'X': meta_X, 'Cost': meta_C}
        return results
