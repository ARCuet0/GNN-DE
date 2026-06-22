"""
opt_variant.py — Unified interface for L2O operator variants.

All metaheuristic systems (K=2 Classic, K=4 NeuralMeta, K=6 HyperOPT, etc.)
implement OptVariant.step() to receive backbone embeddings and produce
offspring displacements. The GenerationStep class orchestrates:

    graph build → backbone forward → variant.step() → eval → selection → loss

This decouples the backbone (shared perception) from the decision policy
(variant-specific routing + operators).
"""
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .unified_loss import adaptive_log1p_loss
from .grad_stabilizers import soft_min_scalar, soft_min, hard_min_ste
from .graph_builder_sparse_delta import augment_sparse_cache


def _clamp_fitness(f: torch.Tensor) -> torch.Tensor:
    """Replace non-finite fitness values with dtype max to prevent NaN propagation."""
    if f.is_floating_point():
        fmax = torch.finfo(f.dtype).max
        return f.clamp(min=-fmax, max=fmax)
    return f


def _shade_successes_from_selection(top_idx, all_fit, parent_fit,
                                    realized_F, realized_CR, N):
    """Per-selected-proposal (F, CR, Δfit, success) for the SHADE memory update
    (fcr_shade_adaptive lesion).

    Augmented-pop index layout (per_m_donors, K=1): parents occupy [0, N);
    proposal (m, n) sits at N + m*N + n. A selected entry is a successful trial
    when it is a proposal (idx >= N) whose realized fitness beats its parent n.

    Args:
        top_idx:       (B, S) selected augmented indices.
        all_fit:       (B, N_aug) fitness with realized values at selected slots.
        parent_fit:    (B, N) gen-start parent fitness.
        realized_F/CR: (M, B, N) F/CR actually used per proposal.
        N:             parent-population size this generation.

    Returns F_sel, CR_sel, delta, success_mask, each (B, S).
    """
    B, S = top_idx.shape
    is_prop = top_idx >= N
    p = (top_idx - N).clamp(min=0)
    m = p // N
    n = p % N
    child = all_fit.gather(1, top_idx)
    parent = parent_fit.gather(1, n)
    delta = parent - child
    success_mask = is_prop & (child < parent)
    b_idx = torch.arange(B, device=top_idx.device).view(B, 1).expand(B, S)
    F_sel = realized_F[m, b_idx, n]
    CR_sel = realized_CR[m, b_idx, n]
    return F_sel, CR_sel, delta, success_mask


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias, no mean shift)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.scale


class OptVariant(ABC, nn.Module):
    """Abstract base for L2O operator variants.

    Subclasses implement step() to transform backbone embeddings into
    offspring displacements. Everything else (backbone forward, eval,
    selection, loss) is handled by GenerationStep.

    Existing systems map to:
        K=2 Classic:  ClassicVariant (SHADE + LS1 Gumbel mask)
        K=4 Neural:   NeuralMetaVariant (4 diff operators + expert-choice)
        K=6 HyperOPT: HyperOPTVariant (6 kernel heads + cross-dim attn)
    """

    @abstractmethod
    def step(
        self,
        h: torch.Tensor,           # (B, N, gatv2_hidden) node embeddings
        h_per_head: torch.Tensor,   # (B, N, n_heads, head_dim) per-head
        h_global: torch.Tensor,     # (B, global_out) graph-level
        coords: torch.Tensor,       # (B, N, D) float64 population
        fitness: torch.Tensor,      # (B, N) float64 fitness
        cache,                      # TopologyCache
        D: int,                     # problem dimensionality
        M: int = 1,                 # number of displacement samples
        gumbel_tau: float = 1.0,    # routing temperature
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        """Produce M displacement samples per individual.

        Args:
            h, h_per_head, h_global: backbone outputs
            coords, fitness: current population state
            cache: TopologyCache (for edge_index access if needed)
            D: problem dimensionality
            M: number of samples (for M-sampling trick or Gumbel)
            gumbel_tau: routing temperature

        Returns:
            delta: (M, B, N, D) displacement vectors (in coordinate space,
                   NOT normalized by span — GenerationStep handles clamping)
            extras: dict with variant-specific diagnostics
                    (entropy, routing_probs, per_expert_stats, etc.)
        """
        ...


class GenerationStep(nn.Module):
    """Orchestrates one generation: backbone → variant → eval → select → loss.

    Decouples perception (backbone) from decision (variant). Handles:
    - Backbone forward pass (produces embeddings)
    - Variant step (produces displacements)
    - Offspring evaluation (CEC2017 or custom fn)
    - Greedy selection (keep if improved)
    - Loss computation (adaptive_log1p_loss)
    """

    def __init__(self, backbone: nn.Module, variant: OptVariant,
                 eval_fn: Optional[Callable] = None,
                 soft_min_beta: Optional[float] = None,
                 surrogate=None, build_graph_fn=None,
                 archive_capacity: int = 0,
                 surrogate_augment_strategy: str = 'rebuild',
                 lb: Optional[float] = None, ub: Optional[float] = None):
        super().__init__()
        self.backbone = backbone
        self.variant = variant
        self.eval_fn = eval_fn  # CEC2017 function: (N, D) -> (N,)
        self.soft_min_beta = soft_min_beta
        self.surrogate = surrogate
        self.build_graph_fn = build_graph_fn
        # Search-domain bounds. Required explicit: span-relative coord
        # normalization (graph builder) and offspring clamp sites both depend
        # on the actual box, not a default.
        assert lb is not None, "GenerationStep requires explicit lb"
        assert ub is not None, "GenerationStep requires explicit ub"
        self.lb = float(lb)
        self.ub = float(ub)
        # D1000 line: how to build the (parents+proposals) graph in surrogate.
        #   'rebuild' (legacy default): _build_graph_cache(coords_aug, ...)
        #             — re-runs sparse kNN over N_aug nodes (still O(N_aug^2)
        #             via cdist or NN-Descent fallback).
        #   'delta'   (D1000 lema): augment_sparse_cache inherits parent kNN
        #             per proposal — strict O(N_aug * k).
        assert surrogate_augment_strategy in ('rebuild', 'delta')
        self.surrogate_augment_strategy = surrogate_augment_strategy
        # RMSNorm before variant heads — prevents h_out scale drift
        _hdim = getattr(backbone, 'hidden_dim', None) or getattr(backbone, 'gatv2_hidden', 128)
        self.h_norm = RMSNorm(_hdim)
        # Graph-native archive (E9_archive_K50). State allocated lazily on
        # first surrogate forward when capacity > 0; see archive_design.md
        # decisions D1, D8, D9, D10. K=0 disables completely (T1 bit-exact).
        self.archive_capacity = archive_capacity
        self.archive_coords = None
        self.archive_fitness = None
        self.archive_mask = None
        self.archive_age = None
        self._archive_gen_counter = 0
        # CPU-side mirror of `archive_mask.any()`. Updated only when an
        # enqueue first writes to a previously-empty archive (or an
        # _ensure_archive resets state). Lets the per-gen `augment_archive`
        # gate avoid a GPU→CPU sync (CLAUDE.md: NEVER .item() on hot path).
        self._archive_has_entries = False

    def _get_default_fes(self, B, N, device):
        """Cached (B,) tensor of float(N) for when gate is inactive."""
        c = getattr(self, '_default_fes_cache', None)
        if c is None or c.shape[0] != B or c.device != device:
            c = torch.full((B,), float(N), device=device)
            self._default_fes_cache = c
        return c

    @staticmethod
    def _unpack_backbone(result):
        """Return (h, e, h_per_head, h_global, h_pool, donor_logits, donor_cand_idx).

        BackboneOutput NamedTuple is the canonical return; legacy 4/5-tuple
        fallback kept for tests that predate the stateless-DE refactor.
        donor_cand_idx is non-None only when SparseGATv2Backbone was built
        with `donor_kind='knn'` (D1000 line).
        """
        if hasattr(result, 'h_pooled') and hasattr(result, 'donor_logits'):
            cand_idx = getattr(result, 'donor_cand_idx', None)
            return (result.h, result.e, result.h_per_head, result.h_global,
                    result.h_pooled, result.donor_logits, cand_idx)
        if len(result) == 5:
            h, e, h_per_head, h_global, h_pool = result
        else:
            h, e, h_per_head, h_global = result
            h_pool = None
        return h, e, h_per_head, h_global, h_pool, None, None

    def _get_zeros_cache(self, B, device):
        """Cached (B,) zero tensor for surrogate graph building."""
        c = getattr(self, '_zeros_cache', None)
        if c is None or c.shape[0] != B or c.device != device:
            c = torch.zeros(B, device=device)
            self._zeros_cache = c
        return c

    def _build_node_features(self, coords, fitness, B, N, D):
        """Emergency fallback: minimal node features when no graph builder is available.

        In production, pass node_feat from build_similarity_graph_gpu() instead.
        This only fires when node_feat=None in run() — a testing-only path.
        """
        f = fitness.float()   # (B, N)
        ranks = f.argsort(dim=-1).argsort(dim=-1).float()
        fit_rank = ranks / max(N - 1, 1) * 2 - 1  # (B, N)

        # (B, N, 8) — only fit_rank is meaningful, rest are zeros
        node_f = torch.zeros(B, N, 8, device=coords.device)
        node_f[..., 0] = fit_rank
        return node_f

    def _build_global_features(self, coords, fitness, B, N, D):
        """Emergency fallback: minimal global features when no graph builder is available.

        In production, pass global_feat from build_similarity_graph_gpu() instead.
        """
        return torch.zeros(B, 13, device=coords.device)

    def _build_graph_cache(self, coords_f, fit_f, D, kwargs, B, alive=None):
        """Invoke build_graph_fn with the standard temporal/topology kwargs.

        Centralizes the boilerplate shared by `_run_surrogate` (proposal pool)
        and `run` (archive-augmented donor pool). Returns whatever the builder
        returns (typically `SparseTopologyCache`).
        """
        zeros = self._get_zeros_cache(B, coords_f.device)
        return self.build_graph_fn(
            coords_f, fit_f,
            step_num=kwargs.get('step_num', 0),
            max_steps=kwargs.get('max_steps', 10000),
            ndim=D, k_neighbors=kwargs.get('k_neighbors', 8),
            stagnation_counters=zeros,
            delta_fitnesses=zeros,
            contraction_rates=zeros,
            alive=alive,
            lb=self.lb, ub=self.ub)

    def _ensure_archive(self, B: int, D: int, device, dtype):
        """Lazily allocate per-batch archive state (D8: per-batch buffer).

        Allocates once on first surrogate call when archive_capacity > 0.
        Re-allocates if (B, D, device, dtype) changes (e.g. between train and
        eval batches with different shapes). All slots initialized invalid
        (D9: progressive warmup; archive_mask all False).

        The coords tensor uses requires_grad=False unconditionally (D7: gradient
        blocked at archive — archive is stored data, not parameters).
        """
        if self.archive_capacity <= 0:
            return
        K = self.archive_capacity
        existing = self.archive_coords
        if (existing is not None
                and existing.shape == (B, K, D)
                and existing.device == device
                and existing.dtype == dtype):
            return
        self.archive_coords = torch.zeros(B, K, D, device=device, dtype=dtype)
        # archive_fitness uses float32 for fitness comparison space
        self.archive_fitness = torch.zeros(B, K, device=device,
                                           dtype=torch.float32)
        self.archive_mask = torch.zeros(B, K, device=device, dtype=torch.bool)
        self.archive_age = torch.zeros(B, K, device=device, dtype=torch.int64)
        self._archive_gen_counter = 0
        self._archive_has_entries = False

    def _enqueue_archive(self, coords: torch.Tensor, fitness: torch.Tensor,
                         active: torch.Tensor):
        """Enqueue active rows into per-batch FIFO archive (D1, D7, D9, D10).

        Args:
            coords:  (B, N, D) — candidate rows. Detached internally (D7).
            fitness: (B, N)    — corresponding fitness values.
            active:  (B, N)    — bool mask; True rows are enqueued.

        Behavior per batch element b (D9: progressive, D10: FIFO):
          1. Increment shared gen counter (one per call → entries from this
             call share an age, FIFO-stable across calls).
          2. For each active row j in batch b (in row order):
             - If any slot has mask=False, fill the lowest-index empty slot.
             - Else, overwrite the slot with smallest archive_age (oldest).
             - Mark mask=True; set age = gen_counter.

        No-op if archive_capacity == 0 (defensive; caller should also skip).
        """
        if self.archive_capacity <= 0 or self.archive_coords is None:
            return
        B, N = active.shape
        K = self.archive_capacity
        # One shared gen counter increment per call. All entries from this
        # call get the same age value, which preserves FIFO across gens.
        self._archive_gen_counter += 1
        gen = self._archive_gen_counter
        # Archive must be a frozen snapshot — even if coords/fitness carry
        # grad in the BPTT chain.
        coords_d = coords.detach()
        fitness_d = fitness.detach().to(self.archive_fitness.dtype)
        evict_mode = getattr(self, 'archive_evict', 'fifo')
        for b in range(B):
            active_b = active[b]
            if not active_b.any():
                continue
            rows = active_b.nonzero(as_tuple=True)[0]  # (n_active_b,)
            for r in rows.tolist():
                # Pick destination slot: empty first, else evict per policy.
                if not self.archive_mask[b].all():
                    empties = (~self.archive_mask[b]).nonzero(as_tuple=True)[0]
                    slot = empties[0].item()
                elif evict_mode == 'random':
                    slot = int(torch.randint(0, K, (1,)).item())
                else:  # 'fifo' (default, E9)
                    slot = self.archive_age[b].argmin().item()
                self.archive_coords[b, slot] = coords_d[b, r]
                self.archive_fitness[b, slot] = fitness_d[b, r]
                self.archive_mask[b, slot] = True
                self.archive_age[b, slot] = gen
        # We just wrote at least one row; any subsequent forward can rely on
        # the archive having content. CPU-side flag avoids `.any().item()`
        # checks in the hot path of GenerationStep.run.
        self._archive_has_entries = True

    def _run_surrogate(self, coords, fitness, D, extras, f_optimal,
                       h, h_global, node_feat, parent_cache=None, **kwargs):
        """Surrogate path: score augmented population, select top-M, evaluate.

        Uses ALL M samples from variant.step(), giving M*N*K proposals.
        The surrogate scores N + M*N*K candidates and selects top-M_sel.

        parent_cache: SparseTopologyCache from the parent forward. Required by
            surrogate_augment_strategy='delta' to inherit parent kNN per
            proposal without rebuilding the graph. Ignored by 'rebuild'.
        """
        B, N = coords.shape[:2]
        K = self.variant.K
        deltas_k = extras['deltas_k_live']  # (M_var, B, N, K, D)
        M_var = deltas_k.shape[0]

        # No-op when archive_capacity == 0 (K=0 path stays bit-exact).
        self._ensure_archive(B, D, coords.device, coords.dtype)

        # All M samples × K heads → M*N*K proposals
        # deltas_k: (M_var, B, N, K, D) → proposals: (M_var, B, N, K, D)
        proposals = (coords.unsqueeze(0).unsqueeze(3) + deltas_k.double()).clamp(self.lb, self.ub)
        # Reshape to (B, M_var*N*K, D)
        prop_flat = proposals.permute(1, 0, 2, 3, 4).reshape(B, M_var * N * K, D)
        N_prop = M_var * N * K
        coords_aug = torch.cat([coords, prop_flat], dim=1)  # (B, N + N_prop, D)
        N_aug = N + N_prop

        # Impute fitness: each proposal gets its parent's fitness
        fit_parents = fitness.float()
        # proposals[m, b, n, k] comes from parent n → repeat M_var*K times
        fit_imputed = fit_parents.repeat(1, M_var * K)  # (B, M_var*N*K)
        fit_aug = torch.cat([fit_parents, fit_imputed], dim=1)

        # Inference fast path: when the selector ignores surr_scores (random_1pp,
        # oracle_1pp, oracle_kpp consume only scores.shape/dtype/device — see
        # surrogate_selection.py:173/231/248) and no disen/jepa head reads h_aug,
        # skip the augmented-pop backbone forward entirely. Saves ~22 GiB at
        # N_aug=11340 D=30 N=540 (the single allocation that OOMs L40S 48GB).
        # Training always materializes scores (surrogate loss + disen loss).
        _spec = kwargs.get('selection_spec', 'topk')
        _sel_mode = _spec.split(':')[0]
        _disen_heads = getattr(self, 'disen_heads', None)
        _disen_path = (_disen_heads is not None
                       and (_spec.startswith('q_exploit')
                            or _spec.startswith('q_explor')
                            or _spec.startswith('jepa_')))
        _score_free = _sel_mode in ('random_1pp', 'oracle_1pp', 'oracle_kpp')
        _skip_aug_backbone = (not self.training
                              and _score_free
                              and not _disen_path)

        if _skip_aug_backbone:
            # No backbone forward; dummy scores tensor (only shape/dtype/device
            # consumed by select() for these modes).
            h_aug = None
            surr_scores = torch.zeros(
                B, N_aug, device=coords.device, dtype=coords.dtype)
        else:
            # Build graph + backbone forward on augmented pop.
            # surrogate_augment_strategy='delta' inherits parent kNN per proposal
            # — strict O(N_aug * k), no cdist over augmented coords. Requires
            # parent_cache from the caller's parent forward.
            if (self.surrogate_augment_strategy == 'delta'
                    and parent_cache is not None):
                cache_aug = augment_sparse_cache(
                    parent_cache, deltas_k, coords, fitness)
            else:
                cache_aug = self._build_graph_cache(
                    coords_aug.float(), fit_aug, D, kwargs, B)

            # Inference-only bf16 autocast halves peak VRAM on the augmented-pop
            # forward. Training stays fp32 for gradient stability.
            _use_bf16 = (not self.training
                         and coords.device.type == 'cuda'
                         and torch.cuda.is_bf16_supported())
            with torch.amp.autocast('cuda', enabled=_use_bf16, dtype=torch.bfloat16):
                result_aug = self.backbone.encode(
                    cache_aug.node_feat, cache_aug.global_feat, cache_aug,
                    coords_hist=coords_aug.unsqueeze(1),
                    fitness_hist=fit_aug.unsqueeze(1),
                    n_valid=1,
                    # B2 (set_attention_edge) builds its dense edge bias
                    # from the live coords/fitness of the augmented pool.
                    coords=coords_aug, fitness=fit_aug)
            h_aug = result_aug[0].float() if _use_bf16 else result_aug[0]
            # Score all candidates (h_global from original pop provides function context)
            surr_scores = self.surrogate(h_aug, h_global=h_global)  # (B, N_aug)

        # Disen-head scores (q_explor / q_exploit): computed if disen_heads attr
        # is attached to gen_step (eval-time injection for q_*_1pp selectors).
        # For jepa_*_1pp selectors: predict h via JEPA from (h_parent, action)
        # and score the predicted h with disen heads. Test 3 of γ rollout.
        disen_scores_eval = None
        disen_heads = getattr(self, 'disen_heads', None)
        jepa_predictor = getattr(self, 'jepa_predictor', None)
        spec = kwargs.get('selection_spec', 'topk')
        if disen_heads is not None and (spec.startswith('q_exploit')
                                         or spec.startswith('q_explor')
                                         or spec.startswith('jepa_')):
            # For jepa_* modes, replace h_aug with JEPA-predicted h built from
            # (h_parent, action). action = (delta [D], F [1], CR [1]).
            if spec.startswith('jepa_') and jepa_predictor is not None:
                pred_net = jepa_predictor['predictor']
                action_dim = jepa_predictor['action_dim']
                # Action vector per (m, b, n): cat(delta, F, CR)
                # deltas_k: (M_var, B, N, K=1, D) → squeeze K → (M_var, B, N, D)
                delta_mbnd = deltas_k.squeeze(3) if deltas_k.shape[3] == 1 \
                              else deltas_k[:, :, :, 0, :]   # (M_var, B, N, D)
                # F, CR realized: (M_var, B, N) (set in extras by neural_k4)
                F_vals = extras.get('_realized_F')
                CR_vals = extras.get('_realized_CR')
                if F_vals is None or CR_vals is None:
                    # Fallback: zero F/CR (action_dim mismatch will surface).
                    F_vals = torch.zeros(M_var, B, N, device=coords.device)
                    CR_vals = torch.zeros(M_var, B, N, device=coords.device)
                # Build action vectors (M_var, B, N, D+2). Cast to float32 to match predictor.
                action_mbn = torch.cat(
                    [delta_mbnd.float(),
                     F_vals.float().unsqueeze(-1),
                     CR_vals.float().unsqueeze(-1)],
                    dim=-1)
                # Reshape to (B, M_var*N, D+2) matching h_aug proposal layout
                # (per-m-donors: index N + m*N + i for proposal m at parent i).
                action_bmn = action_mbn.permute(1, 0, 2, 3).reshape(B, M_var * N, -1)
                # h_parent for each proposal: parent at index i, repeated M_var times.
                # h_aug[:, :N, :] is parents.
                h_parents = h_aug[:, :N, :].float()                      # (B, N, h_dim)
                h_parent_rep = h_parents.unsqueeze(1).expand(B, M_var, N, -1) \
                                       .reshape(B, M_var * N, -1)        # (B, M_var*N, h_dim)
                # Predict h for proposals
                Bf, Pf, Hf = h_parent_rep.shape
                h_pred_flat = pred_net(h_parent_rep.reshape(Bf * Pf, Hf),
                                        action_bmn.reshape(Bf * Pf, action_dim))
                h_pred_props = h_pred_flat.reshape(Bf, Pf, Hf)            # (B, M_var*N, h_dim)
                # Build h_for_scoring: parents from real h_aug, proposals from JEPA pred
                h_for_scoring = torch.cat([h_parents, h_pred_props], dim=1)  # (B, N + M_var*N, h_dim)
            else:
                h_for_scoring = h_aug
            # Score with disen head (exploit or explor)
            head_key = 'h_exploit' if 'exploit' in spec else 'h_explor'
            if head_key in disen_heads:
                disen_scores_eval = disen_heads[head_key](h_for_scoring).squeeze(-1)

        # Palanca 2 (greedy 1:1, eval-only). Bypass selector + pool topk: each
        # parent gets exactly ONE random proposal and competes slot-wise. Charges
        # N FES per gen. Restores LSHADE-style monotonicity on every slot.
        if kwargs.get('greedy_1to1', False) and self.eval_fn is not None:
            sel_rng = kwargs.get('selection_generator')
            offsets = torch.randint(
                0, M_var * K, (B, N), device=coords.device,
                generator=sel_rng if sel_rng is not None else None)
            m_pick = offsets // K
            k_pick = offsets % K
            n_idx = torch.arange(N, device=coords.device).unsqueeze(0).expand(B, -1)
            flat_in_prop = m_pick * N * K + n_idx * K + k_pick   # (B, N) in [0, N_prop)
            aug_idx = flat_in_prop + N                            # (B, N) in [N, N_aug)
            chosen_coords = coords_aug.gather(
                1, aug_idx.unsqueeze(-1).expand(-1, -1, D))       # (B, N, D)
            chosen_fit = _clamp_fitness(
                self.eval_fn(chosen_coords.reshape(-1, D))).reshape(B, N)
            keep_parent = fitness <= chosen_fit                   # (B, N)
            new_coords = torch.where(keep_parent.unsqueeze(-1), coords, chosen_coords)
            new_fitness = torch.where(keep_parent, fitness, chosen_fit.to(fitness.dtype))
            # FES = N (one eval per parent)
            extras['fes_used'] = float(N)
            extras['fes_per_batch'] = torch.full((B,), float(N), device=coords.device)
            extras['surr_n_proposals'] = torch.full(
                (B,), float(N), device=coords.device, dtype=torch.long)
            extras['surr_top_idx'] = aug_idx.detach()
            extras['surr_sel_mask'] = torch.zeros(B, N_aug, device=coords.device)
            extras['h_global'] = h_global.detach()
            extras['h'] = h.detach()
            extras['h_live'] = h
            if node_feat is not None:
                extras['node_feat'] = node_feat.detach()
            # off_fitness_all is consumed by training losses only; eval path
            # leaves imputed (parent fitness) — same convention as the
            # non-greedy non-oracle branch above (line ~347).
            _imputed = fit_aug[:, N:].reshape(B, M_var, N, K)
            extras['off_fitness_all'] = _imputed[..., 0].transpose(0, 1).detach()
            # Archive: enqueue parents that were displaced by their proposal
            if self.archive_capacity > 0:
                displaced_mask = ~keep_parent
                if displaced_mask.any():
                    self._enqueue_archive(coords, fitness, displaced_mask)
            return {
                'new_coords': new_coords,
                'new_fitness': new_fitness,
                'extras': extras,
                'loss': None,
                'best_fit': new_fitness.min(dim=1).values,
            }

        # Parse selection spec (eval-time override; train always uses topk).
        from encoder.operators.surrogate_selection import parse_spec, select
        spec = kwargs.get('selection_spec', 'topk')
        sel_mode, sel_params = parse_spec(spec)
        M_sel = kwargs.get('surrogate_M', N)
        sel_rng = kwargs.get('selection_generator')

        # Oracle needs real proposal fitness AT selection time. Training also
        # evaluates all proposals (for pairwise labels).
        need_prop_fit = (sel_mode in ('oracle_1pp', 'oracle_kpp')) or self.training

        if self.eval_fn is not None:
            if need_prop_fit:
                prop_coords = coords_aug[:, N:]  # (B, N_prop, D)
                prop_fit = _clamp_fitness(
                    self.eval_fn(prop_coords.reshape(-1, D))).reshape(B, N_prop)
                all_fit = torch.cat([fitness.detach().to(prop_fit.dtype),
                                     prop_fit], dim=1)  # (B, N_aug)
            else:
                all_fit = fit_aug.clone()

            if self.training:
                extras['surr_scores'] = surr_scores
                extras['surr_all_fit'] = all_fit.detach()
                extras['surr_parent_fit'] = fitness.detach()

            # Selection.
            top_idx, sel_mask = select(
                sel_mode, sel_params,
                scores=surr_scores, fit_aug=all_fit,
                N=N, M_sel=M_sel, M_var=M_var, K=K,
                generator=sel_rng,
                disen_scores=disen_scores_eval)

            # Evaluate selected candidates if we didn't pre-eval all proposals.
            if not need_prop_fit:
                sel_coords = coords_aug.gather(
                    1, top_idx.unsqueeze(-1).expand(-1, -1, D))
                sel_fit = _clamp_fitness(
                    self.eval_fn(sel_coords.reshape(-1, D))).reshape(B, M_sel)
                all_fit.scatter_(1, top_idx, sel_fit.to(all_fit.dtype))

            # FES accounting. For oracle_1pp: count M_sel (the extra evals that
            # gave us oracle info are "free" by the upper-bound convention).
            # For all other modes: count proposals selected (parents are 0 FES).
            is_proposal = top_idx >= N
            n_proposals = is_proposal.sum(dim=1)
            if sel_mode == 'oracle_1pp':
                extras['fes_used'] = float(M_sel)
                extras['fes_per_batch'] = torch.full(
                    (B,), float(M_sel), device=coords.device)
            else:
                # Tensor form (consistent with the activity_gate / noop_mask
                # branches at lines ~792/799). Train loop syncs once at
                # train_distributed.py:1313 via the `torch.is_tensor` guard,
                # so emitting a scalar here would just add a redundant sync.
                _np_float = n_proposals.float()
                extras['fes_used'] = _np_float.mean()
                extras['fes_per_batch'] = _np_float

            # fcr_shade_adaptive lesion: adapt the SHADE memory from the
            # realized F/CR of the selected proposals that beat their parent.
            shade_mem = getattr(self, '_fcr_shade_memory', None)
            if shade_mem is not None and not self.training:
                # _shade_successes_from_selection assumes the deployed K=1 proposal
                # stride (N + m*N + n); K>1 would change it to m*N*K + n*K + k.
                assert K == 1, 'fcr_shade_adaptive lesion assumes K=1 (deployed path)'
                rF = extras.get('_realized_F')
                rCR = extras.get('_realized_CR')
                if rF is not None and rCR is not None:
                    F_sel, CR_sel, delta, smask = _shade_successes_from_selection(
                        top_idx, all_fit, fitness, rF, rCR, N)
                    shade_mem.update(F_sel, CR_sel, delta, smask)
        else:
            top_idx, sel_mask = select(
                sel_mode, sel_params,
                scores=surr_scores, fit_aug=fit_aug,
                N=N, M_sel=M_sel, M_var=M_var, K=K,
                generator=sel_rng,
                disen_scores=disen_scores_eval)
            is_proposal = top_idx >= N
            n_proposals = is_proposal.sum(dim=1)
            all_fit = (coords_aug ** 2).sum(dim=-1) + f_optimal
            extras['fes_used'] = float(N_aug)
            extras['fes_per_batch'] = torch.full(
                (B,), float(N_aug), device=coords.device)

        extras['surr_n_proposals'] = n_proposals

        # Expose embeddings for diagnostics
        extras['h_global'] = h_global.detach()
        extras['h'] = h.detach()
        extras['h_live'] = h
        if node_feat is not None:
            extras['node_feat'] = node_feat.detach()
        # Expose augmented-pop h_aug + coords_aug for disentangle loss (live for backprop).
        # Consumed by train_distributed.py when --disentangle-lambda-e > 0.
        # h_aug may be None in the inference fast path (score-free selectors).
        if h_aug is not None:
            extras['h_aug_live'] = h_aug
        extras['coords_aug_live'] = coords_aug
        extras['N_parents'] = N
        extras['M_var'] = M_var
        extras['K_heads'] = K
        # off_fitness_all shape MUST mirror the non-surrogate path at :419 —
        # (M, B, N) so M-axis oracle losses (donor_oracle, fcr_oracle_from_m)
        # can consume it. Proposal flat layout is m*N*K + n*K + k
        # (see prop_flat permute-reshape in this function); for K=1 that's
        # m*N + n. For K>1 we take head-0's proposals, which are the ones
        # the DE head's _A_pbest / _pbest_idx_m / _realized_F refer to.
        K = self.variant.K
        _prop_fit = all_fit[:, N:].reshape(B, M_var, N, K)    # (B, M, N, K)
        extras['off_fitness_all'] = _prop_fit[..., 0].transpose(0, 1).detach()  # (M, B, N)

        # Pool selection: parents + ONLY selected proposals (no parent duplication).
        # Selected parents are already in coords/fitness — adding them again would
        # create duplicates in the pool. Only proposals are new candidates.
        sel_coords = coords_aug.gather(
            1, top_idx.unsqueeze(-1).expand(-1, -1, D))  # (B, M_sel, D)
        sel_fit = all_fit.gather(1, top_idx)  # (B, M_sel)
        # Parent slots get worst-possible fitness so they can't win
        # (they already compete as parents in the first N slots)
        _fmax = torch.finfo(sel_fit.dtype).max / 2
        sel_fit = torch.where(is_proposal, sel_fit, _fmax)
        pool_coords = torch.cat([coords, sel_coords], dim=1)  # (B, N+M_sel, D)
        pool_fitness = torch.cat([fitness, sel_fit], dim=1)  # (B, N+M_sel)
        _, top_pool = pool_fitness.topk(N, dim=1, largest=False)
        new_coords = pool_coords.gather(
            1, top_pool.unsqueeze(-1).expand(-1, -1, D))
        new_fitness = pool_fitness.gather(1, top_pool)

        # Enqueue displaced parents (those whose index in [0, N) does not
        # appear in top_pool[b]). Vectorized membership avoids a per-batch
        # python loop here.
        if self.archive_capacity > 0:
            parent_range = torch.arange(N, device=coords.device).view(1, N)
            in_top = (top_pool.unsqueeze(2)
                      == parent_range.unsqueeze(1)).any(dim=1)  # (B, N)
            displaced_mask = ~in_top  # (B, N) — True where parent was discarded
            if displaced_mask.any():
                self._enqueue_archive(coords, fitness, displaced_mask)

        # Track which candidates entered final population
        extras['surr_top_idx'] = top_idx.detach()
        extras['surr_sel_mask'] = sel_mask.detach()

        return {
            'new_coords': new_coords,
            'new_fitness': new_fitness,
            'extras': extras,
            'loss': None,
            'best_fit': new_fitness.min(dim=1).values,
        }

    def run(
        self,
        coords: torch.Tensor,        # (B, N, D) float64
        fitness: torch.Tensor,        # (B, N) float64
        cache,                        # TopologyCache
        f_optimal: float,             # CEC2017 optimum
        M: int = 1,
        gumbel_tau: float = 1.0,
        node_feat: Optional[torch.Tensor] = None,   # (B, N, node_in) override
        global_feat: Optional[torch.Tensor] = None,  # (B, global_in) override
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run one generation step.

        Args:
            coords: population coordinates
            fitness: population fitness
            cache: TopologyCache
            f_optimal: known optimum
            M: number of displacement samples
            node_feat, global_feat: override auto-built features

        Returns:
            dict with:
                loss: scalar (differentiable)
                new_coords: (B, N, D) updated population
                new_fitness: (B, N) updated fitness
                extras: variant-specific diagnostics
        """
        B, N, D = coords.shape

        # Build or use provided features
        if node_feat is None:
            node_feat = self._build_node_features(coords, fitness, B, N, D)
        if global_feat is None:
            global_feat = self._build_global_features(coords, fitness, B, N, D)

        node_feat = node_feat.to(coords.device)
        global_feat = global_feat.to(coords.device)

        # Extract temporal kwargs if provided
        coords_hist = kwargs.get('coords_hist')
        fitness_hist = kwargs.get('fitness_hist')
        n_valid = kwargs.get('n_valid')
        temporal_kw = {}
        if coords_hist is not None:
            temporal_kw = dict(coords_hist=coords_hist,
                               fitness_hist=fitness_hist, n_valid=n_valid)
        # Live (coords, fitness) — used by TemporalSetAttentionEdgeBackbone
        # (B2 arm of the 2026-05-29 ablation) to build the dense edge
        # bias. Other backbones accept these via **_ignored.
        temporal_kw['coords'] = coords
        temporal_kw['fitness'] = fitness

        # ── Graph-native archive (E9_archive_K50): augment first backbone fwd
        # Active only in the surrogate path with archive populated. When
        # gated off (capacity=0, mask all-False, or non-surrogate variant),
        # this branch is a no-op and the path is bit-exact to the K=0
        # baseline (T1, T2). All assignments here are conditional on
        # `augment_archive`, so the standard path is untouched.
        augment_archive = (
            self.archive_capacity > 0
            and self._archive_has_entries
            and self.surrogate is not None
            and getattr(self.variant, 'gate_type', None) == 'surrogate'
        )
        donor_coords_for_op = None
        n_active_arg = None
        donor_mask_arg = None
        if augment_archive:
            arc_coords = self.archive_coords
            arc_fit = self.archive_fitness
            arc_mask = self.archive_mask  # (B, K) bool
            K_cap = self.archive_capacity
            # Augmented donor pool (B, N+K, D). Preserve coords.dtype so the
            # operator's einsum (gather over donor pool) matches the rest of
            # the BPTT chain (float64 in deployed config).
            donor_coords_for_op = torch.cat([
                coords, arc_coords.to(coords.dtype)], dim=1)
            donor_fit_full = torch.cat([
                fitness.float().to(arc_fit.dtype), arc_fit], dim=1)
            donor_mask_arg = torch.cat([
                torch.ones(B, N, dtype=torch.bool, device=coords.device),
                arc_mask], dim=1)
            n_active_arg = N
            # Augment temporal window (replicate archive across time so the
            # temporal encoder sees a constant trajectory for archive nodes).
            if coords_hist is not None and coords_hist.dim() == 4:
                W_t = coords_hist.shape[1]
                arc_h = arc_coords.unsqueeze(1).expand(
                    -1, W_t, -1, -1).to(coords_hist.dtype)
                coords_hist_aug = torch.cat([coords_hist, arc_h], dim=2)
                arc_fh = arc_fit.unsqueeze(1).expand(
                    -1, W_t, -1).to(fitness_hist.dtype)
                fitness_hist_aug = torch.cat([fitness_hist, arc_fh], dim=2)
                temporal_kw = dict(coords_hist=coords_hist_aug,
                                   fitness_hist=fitness_hist_aug,
                                   n_valid=n_valid,
                                   coords=donor_coords_for_op,
                                   fitness=donor_fit_full)
            # Rebuild graph cache over the augmented pool. alive=donor_mask
            # propagates the warmup mask so masked slots are flagged on the
            # cache.
            if self.build_graph_fn is not None:
                cache = self._build_graph_cache(
                    donor_coords_for_op, donor_fit_full, D, kwargs, B,
                    alive=donor_mask_arg)
                node_feat = cache.node_feat
                global_feat = cache.global_feat

        # Backbone forward (bf16 autocast for VRAM savings on CUDA)
        encode_extra = {}
        if augment_archive:
            encode_extra = dict(n_active=n_active_arg,
                                donor_mask=donor_mask_arg)
        if coords.is_cuda:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                result = self.backbone.encode(
                    node_feat, global_feat, cache,
                    **temporal_kw, **encode_extra)
            (h, e, h_per_head, h_global, h_pool,
             donor_logits, donor_cand_idx) = self._unpack_backbone(result)
            h = h.float()
            h_per_head = h_per_head.float()
            h_global = h_global.float()
            if h_pool is not None:
                h_pool = h_pool.float()
            if donor_logits is not None:
                donor_logits = donor_logits.float()
        else:
            result = self.backbone.encode(
                node_feat, global_feat, cache,
                **temporal_kw, **encode_extra)
            (h, e, h_per_head, h_global, h_pool,
             donor_logits, donor_cand_idx) = self._unpack_backbone(result)

        # Slice augmented embeddings down to active [:N] for the variant —
        # variant code expects coords-aligned tensors. donor_logits already
        # has shape (B, N, N+K, 3) from forward_asym.
        def _to_active(t):
            if t is None or not augment_archive or t.shape[1] == N:
                return t
            return t[:, :N]
        h = _to_active(h)
        h_per_head = _to_active(h_per_head)
        h_pool = _to_active(h_pool)
        node_feat = _to_active(node_feat)

        # RMSNorm: stabilize embedding scale before variant heads
        h = self.h_norm(h)

        # Variant step → displacements (strip temporal + internal kwargs)
        variant_kw = {k: v for k, v in kwargs.items()
                      if k not in ('coords_hist', 'fitness_hist', 'n_valid',
                                   'tau_ema')}
        variant_kw['h_pool'] = h_pool
        variant_kw['node_feat'] = node_feat
        # Forward augmented donor pool to the operator (used by
        # BatchedDiffAttDE.compute_params for r1/r2 gather).
        if donor_coords_for_op is not None:
            variant_kw['donor_coords'] = donor_coords_for_op
        # D1000 line: when backbone uses kNN-restricted donor head, forward
        # the per-parent candidate index map so operators can gather donor
        # coords without materializing the dense (B, N, N_pool) one-hot.
        if donor_cand_idx is not None:
            variant_kw['donor_cand_idx'] = donor_cand_idx
        delta, extras = self.variant.step(
            h, h_per_head, h_global, coords, fitness,
            cache, D, M=M, gumbel_tau=gumbel_tau,
            donor_logits=donor_logits, **variant_kw)

        # Expose h_per_head for diagnostics — defined here as a local from
        # the backbone encode; opt_variant.run() is the only place it's
        # in scope. Both surrogate and non-surrogate paths benefit.
        if h_per_head is not None:
            extras['h_per_head'] = h_per_head.detach()

        # ── Surrogate path: score all N×K proposals, select top-M, evaluate ──
        if self.surrogate is not None and extras.get('surrogate_mode'):
            return self._run_surrogate(
                coords, fitness, D, extras, f_optimal, h, h_global,
                node_feat, parent_cache=cache, **kwargs)

        # delta carries straight-through gradient: hard selection forward,
        # soft routing backward. Same delta for population AND loss.
        offspring = (coords.unsqueeze(0) + delta).clamp(self.lb, self.ub)  # (M, B, N, D)

        # Evaluate offspring
        winner = extras.get('winner')
        active_mask = extras.get('active_mask')  # (B, N) from activity gate, or None
        has_noop = getattr(self.variant, '_has_noop', False)
        NOOP_IDX = getattr(self.variant, 'K', 4) - 1
        noop_mask = (winner == NOOP_IDX) if (winner is not None and has_noop) else None

        if self.eval_fn is not None:
            off_flat = offspring.reshape(-1, D)
            fit_flat = _clamp_fitness(self.eval_fn(off_flat))
            off_fitness = fit_flat.reshape(M, B, N)
            _eval_all = getattr(self.variant, 'eval_all_select_active', False)
            if active_mask is not None:
                # All offspring are evaluated (real fitness preserved).
                # Gate exclusion is applied in the selection step, not here.
                # off_fitness stays real for: gradient, BCE labels, loss.
                extras['off_fitness_all'] = off_fitness.detach()
                _fes_batch = active_mask.sum(dim=-1).float()  # (B,)
                extras['fes_used'] = _fes_batch.mean()
                extras['fes_per_batch'] = _fes_batch  # (B,) for FES-weighted loss
            elif noop_mask is not None:
                parent_fit = fitness.unsqueeze(0).expand_as(off_fitness)
                off_fitness = torch.where(noop_mask, parent_fit, off_fitness)
                # noop_mask is (M, B, N); average over M then sum over N → (B,)
                _fes_batch = (~noop_mask).float().mean(dim=0).sum(dim=-1)  # (B,)
                extras['fes_used'] = _fes_batch.mean()
                extras['fes_per_batch'] = _fes_batch
            else:
                extras['fes_used'] = float(N)
                extras['fes_per_batch'] = self._get_default_fes(B, N, offspring.device)
        else:
            off_fitness = (offspring ** 2).sum(dim=-1) + f_optimal
            extras['fes_used'] = float(N)
            extras['fes_per_batch'] = self._get_default_fes(B, N, offspring.device)

        # Expose detached embeddings for external warmup/router loss
        if hasattr(self.variant, '_last_h_expert'):
            extras['h_expert'] = self.variant._last_h_expert
        extras['h_global'] = h_global.detach()
        extras['h'] = h.detach()
        extras['h_live'] = h  # with gradient for LUPI aux heads
        if node_feat is not None:
            extras['node_feat'] = node_feat.detach()

        # Evaluate ALL K operators (detached deltas — no grad to heads, only for diagnostics/router)
        deltas_k_det = extras.get('deltas_k')  # (M, B, N, K, D) already detached
        if deltas_k_det is not None and self.eval_fn is not None:
            with torch.no_grad():
                K_ops = deltas_k_det.shape[3]
                off_k = (coords.unsqueeze(0).unsqueeze(3) + deltas_k_det).clamp(self.lb, self.ub)
                fk = _clamp_fitness(self.eval_fn(off_k.reshape(-1, D))).reshape(M, B, N, K_ops)
                if has_noop:
                    fk[..., NOOP_IDX] = fitness.unsqueeze(0).expand(M, B, N)
                extras['fit_per_k'] = fk
                # Feed oracle to variant for next gen's routing
                if hasattr(self.variant, '_oracle_best_k'):
                    self.variant._oracle_best_k = fk.mean(dim=0).argmin(dim=-1)

        # LUPI: differentiable contrafactual eval (grad flows to heads)
        if getattr(self, 'compute_contrafactual_grad', False) and self.eval_fn is not None:
            deltas_k_live = extras.get('deltas_k_live')
            if deltas_k_live is not None:
                K_ops = deltas_k_live.shape[3]
                off_k_live = (coords.unsqueeze(0).unsqueeze(3) + deltas_k_live).clamp(self.lb, self.ub)
                fk_live = _clamp_fitness(self.eval_fn(off_k_live.reshape(-1, D))).reshape(M, B, N, K_ops)
                extras['fit_per_k_grad'] = fk_live

        # Drop deltas_k_live when not needed to free the K-head computation graph
        # keep_deltas_k_live: distance-based LUPI needs it even without fitness ghost evals
        if (not getattr(self, 'compute_contrafactual_grad', False)
                and not getattr(self, 'keep_deltas_k_live', False)):
            extras.pop('deltas_k_live', None)

        # Best-of-M selection
        best_m = off_fitness.argmin(dim=0)
        best_off = offspring[best_m, torch.arange(B).unsqueeze(1), torch.arange(N).unsqueeze(0)]
        best_fit = off_fitness[best_m, torch.arange(B).unsqueeze(1), torch.arange(N).unsqueeze(0)]
        best_fit_traj = best_fit  # default; overridden by pool+eval_all path
        top_idx = None

        # Selection
        _pool_sel = getattr(self, 'pool_selection', False)
        if _pool_sel and _eval_all and active_mask is not None:
            # (μ+λ) pool selection on ACTIVE offspring.
            # Trajectory fitness: active → offspring fitness, inactive → parent fitness.
            # Differentiable through active_mask (ST estimator) → gate gets BPTT gradient.
            parent_fit_expand = fitness.unsqueeze(0).expand_as(off_fitness)
            mask_expand = active_mask.unsqueeze(0).expand_as(off_fitness)
            traj_fitness = mask_expand * off_fitness + (1.0 - mask_expand) * parent_fit_expand.detach()
            # Best-of-M on trajectory fitness
            best_m_traj = traj_fitness.argmin(dim=0)
            best_fit_traj = traj_fitness[best_m_traj, torch.arange(B).unsqueeze(1), torch.arange(N).unsqueeze(0)]
            best_off_traj = offspring[best_m_traj, torch.arange(B).unsqueeze(1), torch.arange(N).unsqueeze(0)]
            # Pool: parents + active offspring (inactive have parent fitness → can't win)
            pool_coords = torch.cat([coords, best_off_traj], dim=1)       # (B, 2N, D)
            pool_fitness = torch.cat([fitness.detach(), best_fit_traj], dim=1)  # (B, 2N)
            _, top_idx = pool_fitness.topk(N, dim=1, largest=False)  # (B, N)
            new_coords = pool_coords.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, D))
            new_fitness = pool_fitness.gather(1, top_idx)
        elif _pool_sel:
            # Pool without gate
            pool_coords = torch.cat([coords, best_off], dim=1)
            pool_fitness = torch.cat([fitness.detach(), best_fit], dim=1)
            _, top_idx = pool_fitness.topk(N, dim=1, largest=False)
            new_coords = pool_coords.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, D))
            new_fitness = pool_fitness.gather(1, top_idx)
        else:
            # Greedy 1:1 selection
            improved = best_fit < fitness
            if active_mask is not None:
                can_replace = improved & (active_mask > 0.5)
            else:
                can_replace = improved
            new_coords = torch.where(can_replace.unsqueeze(-1), best_off, coords)
            new_fitness = torch.where(can_replace, best_fit, fitness)

        # Loss on actual offspring — adaptive_log1p unconditional
        if active_mask is not None:
            eval_mask = active_mask.unsqueeze(0).expand(M, B, N).bool()
        elif noop_mask is not None:
            eval_mask = ~noop_mask
        else:
            eval_mask = None
        tau_ema_in = kwargs.get('tau_ema')
        if tau_ema_in is None:
            with torch.no_grad():
                if eval_mask is not None and eval_mask.any():
                    tau_ema_in = (off_fitness[eval_mask] - f_optimal).clamp(min=0).mean().item()
                else:
                    tau_ema_in = (off_fitness - f_optimal).clamp(min=0).mean().item()
                tau_ema_in = max(tau_ema_in, 1.0)
        loss, tau_ema_out = adaptive_log1p_loss(
            off_fitness, f_optimal, tau_ema=tau_ema_in,
            eval_mask=eval_mask)

        # For pool selection, 'improved' is derived from who got selected
        if _pool_sel and top_idx is not None:
            improved = (top_idx >= N)
        else:
            top_idx = None

        # best_fit for hitting loss: use trajectory fitness (has gate gradient) when available
        _bf = best_fit_traj if (_pool_sel and _eval_all and active_mask is not None) else best_fit
        result = {
            'loss': loss,
            'new_coords': new_coords,
            'new_fitness': new_fitness,
            'best_fit': (soft_min(_bf, self.soft_min_beta, dim=-1,
                                 normalize=False)
                        if self.soft_min_beta is not None
                        else _bf.min(dim=-1).values),  # (B,) WITH gradient, per-batch + gate
            'extras': extras,
            'off_fitness': off_fitness.detach(),
            'improved': improved.detach(),
            'best_fit_live': best_fit,
            'parent_fit_in': fitness.detach(),
        }
        if tau_ema_out is not None:
            result['tau_ema'] = tau_ema_out
        return result
