"""
neural_k4.py — K=4 Neural Meta variant (batched operators, no Python loops).

Uses BatchedDiffLSHADE/MTS/CMA/SBX that operate natively on (B, N, ...).
Node-choice soft routing: each node distributes over K operators via softmax(dim=K).
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.opt_variant import OptVariant
from encoder.batched_operators import (
    BatchedDiffDE, BatchedDiffAttDE, BatchedNoOp,
    NeuralCoordLS, NeuralCMAES,
)
from encoder.gumbel import antithetic_gumbel_noise

# Re-export activity gate hierarchy (backward compat)
from encoder.variants.activity_gate import (  # noqa: F401
    _adaptive_threshold_gate,
    _ActivityGateBase,
    ActivityGate,
    ActivityGateMLP,
    ActivityGateAttn,
    RankerGate,
)

# Re-export operator set configurations (backward compat)
from encoder.variants.operator_sets import (  # noqa: F401
    BATCHED_OPERATOR_CLASSES,
    BATCHED_OPERATOR_CLASSES_K5,
    BATCHED_OPERATOR_CLASSES_DIRECT,
    BATCHED_OPERATOR_CLASSES_K5_ATT,
    BATCHED_OPERATOR_CLASSES_NEURAL,
    BATCHED_OPERATOR_CLASSES_NEURAL_ATT,
    BATCHED_OPERATOR_CLASSES_GATED,
)

log = logging.getLogger(__name__)


class NeuralK4Variant(OptVariant):
    """K=4 differentiable operators with node-choice soft Gumbel routing.

    Fully batched over B populations — no Python loops.
    LayerNorm before scorers prevents logit drift.
    """

    DE_HEAD_IDX = 0
    _NO_POOL_TYPES = (BatchedDiffDE, BatchedDiffAttDE, BatchedNoOp)

    def __init__(self, K=4, head_dim=16, gatv2_hidden=64,
                 operator_classes=None, pool_dim=0,
                 gate_node_feat_dim=0, gate_type='adaptive',
                 fcr_mode='beta'):
        super().__init__()
        self.K = K
        self.head_dim = head_dim
        self.gatv2_hidden = gatv2_hidden
        self.pool_dim = pool_dim
        self.gate_type = gate_type
        self._oracle_best_k = None  # (B, N) GPU tensor, set by opt_variant
        self.oracle_epsilon = 0.04  # fraction of pop to force to oracle-best

        if operator_classes is None:
            operator_classes = BATCHED_OPERATOR_CLASSES

        # Derive which heads receive pool skip from operator types
        self._pool_skip_heads = frozenset(
            k for k, cls in enumerate(operator_classes)
            if pool_dim > 0 and cls not in self._NO_POOL_TYPES
        )

        self.heads = nn.ModuleList()
        for k, cls in enumerate(operator_classes):
            bd = gatv2_hidden + pool_dim if k in self._pool_skip_heads else gatv2_hidden
            if cls == BatchedNoOp:
                self.heads.append(cls(embed_dim=head_dim, head_idx=k))
            else:
                # E13: BatchedDiffAttDE accepts fcr_mode for Cauchy/Beta dispatch.
                head_kwargs = {'embed_dim': head_dim, 'head_idx': k, 'backbone_dim': bd}
                if cls is BatchedDiffAttDE and fcr_mode is not None:
                    head_kwargs['fcr_mode'] = fcr_mode
                try:
                    self.heads.append(cls(**head_kwargs))
                except TypeError:
                    head_kwargs.pop('fcr_mode', None)
                    head_kwargs.pop('backbone_dim', None)
                    self.heads.append(cls(**head_kwargs))

        # Detect which operators are present (for diagnostic extras)
        self._has_noop = any(isinstance(h, BatchedNoOp) for h in self.heads)
        self._de_idx = next((k for k, h in enumerate(self.heads)
                             if isinstance(h, (BatchedDiffDE, BatchedDiffAttDE))), None)
        self._coordls_idx = next((k for k, h in enumerate(self.heads)
                                  if isinstance(h, NeuralCoordLS)), None)
        self._cmaes_idx = next((k for k, h in enumerate(self.heads)
                                if isinstance(h, NeuralCMAES)), None)

        # Projections live inside each head (head.proj + head.proj_norm).
        # Variant only owns the routing scorer.

        # LayerNorm before scoring — prevents logit drift
        self.score_norm = nn.LayerNorm(head_dim)

        # Single scorer on normalized embeddings: (B, N, K, head_dim) -> (B, N, K)
        self.scorer = nn.Linear(head_dim, 1)

        # Global context -> multiplicative gate on per-node logits
        # Modulates (scales) per-node routing without overriding per-node ranking
        self.global_route_gate = nn.Linear(gatv2_hidden, K)

        # Activity gate: per-individual binary decision (operate or preserve)
        # Replaces NoOp as an operator — gate decides before routing.
        # use_global: gate sees cat(h_i, h_global) for population-relative decisions.
        if gate_type == 'ranker':
            self.activity_gate = RankerGate(gatv2_hidden, use_global=True,
                                            node_feat_dim=gate_node_feat_dim)
        elif gate_type in ('none', 'surrogate'):
            self.activity_gate = None
        else:  # 'adaptive'
            self.activity_gate = ActivityGate(gatv2_hidden, use_global=True,
                                             node_feat_dim=gate_node_feat_dim)

    # ── Backward-compat properties for diagnostics/probes ──

    @property
    def head_proj(self):
        """Returns first head's projection for diagnostics."""
        return self.heads[0].proj

    @property
    def head_projs(self):
        """Collect projections from all heads (backward compat)."""
        return nn.ModuleList([h.proj for h in self.heads if h.proj is not None])

    @property
    def head_norms(self):
        """Collect proj_norms from all heads (backward compat)."""
        return nn.ModuleList([h.proj_norm for h in self.heads if h.proj_norm is not None])

    def _get_expert_embeddings(self, h, h_per_head, h_aug=None):
        """Get (B, N, K, head_dim) from each head's internal projection."""
        K = self.K
        expert_slices = []
        for k in range(K):
            h_in = h_aug if (h_aug is not None and k in self._pool_skip_heads) else h
            expert_slices.append(self.heads[k].get_embedding(h_in))
        return torch.stack(expert_slices, dim=2)  # (B, N, K, head_dim)

    def _expose_de_extras(self, params_list, extras):
        """Extract DE head diagnostics (F/CR Beta, realized, diff_vector) into extras."""
        if self._de_idx is None or self._de_idx >= len(params_list):
            return
        de_p = params_list[self._de_idx]
        if '_realized_F' in de_p:
            extras['_realized_F'] = de_p['_realized_F']
            extras['_realized_CR'] = de_p['_realized_CR']
        # Donor selection diagnostics for distillation analysis (Q5).
        # _A_*_neural carry the pre-override neural logits with grad — needed
        # by compute_donor_oracle_loss under donor_mode='lshade'.
        for k in ('_pbest_idx_m', '_r1_idx_m', '_r2_idx_m',
                  '_A_pbest', '_A_r1', '_A_r2',
                  '_A_pbest_neural', '_A_r1_neural', '_A_r2_neural'):
            if k in de_p:
                extras[k] = de_p[k]
        if '_F_mean' in de_p:
            extras['_F_mean'] = de_p['_F_mean']
            extras['_CR_mean'] = de_p['_CR_mean']
            # Beta-only keys (absent under fcr_mode='lshade'/'cauchy_neural').
            if '_f_alpha' in de_p and de_p['_f_alpha'] is not None:
                extras['_f_alpha'] = de_p['_f_alpha'].detach()
                extras['_f_beta'] = de_p['_f_beta'].detach()
        # E13: Cauchy head outputs (with grad — needed for distill loss).
        if '_mu_F_pred' in de_p:
            extras['_mu_F_pred'] = de_p['_mu_F_pred']
            extras['_mu_CR_pred'] = de_p['_mu_CR_pred']
        # 2026-04-28 falsification arm C: learned σ_F (with grad — required
        # by compute_fcr_distill_loss(mode='cauchy_nll') and by cauchy_neural
        # inference sampling with non-default sigma).
        if '_sigma_F_pred' in de_p:
            extras['_sigma_F_pred'] = de_p['_sigma_F_pred']
        if '_diff_vector' in de_p:
            extras['_diff_vector'] = de_p['_diff_vector']
        # Per-m donor indices for supervised donor-oracle loss
        # (only populated when head.per_m_donors=True).
        if '_pbest_idx_m' in de_p:
            extras['_pbest_idx_m'] = de_p['_pbest_idx_m']
            extras['_r1_idx_m'] = de_p['_r1_idx_m']
            extras['_r2_idx_m'] = de_p['_r2_idx_m']
        # Attention logits (with grad) for compute_attn_diag and
        # compute_donor_oracle_loss. Previously only surfaced in the standard
        # routing path (via `extras.update(_de_attn)`); missing in surrogate
        # path made pbest diagnostics blind under --gate-type surrogate.
        if '_A_pbest' in de_p:
            extras['_A_pbest'] = de_p['_A_pbest']
            extras['_A_r1'] = de_p['_A_r1']
            extras['_A_r2'] = de_p['_A_r2']
            for k in ('_A_pbest_neural', '_A_r1_neural', '_A_r2_neural'):
                if k in de_p:
                    extras[k] = de_p[k]

    def _compute_all_deltas(self, h_expert, h, h_backbone_aug, coords, fitness,
                            cache, soft_probs, bounds_span, M, kwargs,
                            h_global=None, donor_logits=None):
        """Compute displacements from all K heads.

        Args:
            donor_logits: optional (B, N, N_pool, n_roles) from
                          backbone.donor_selector. Forwarded to every head's
                          compute_params; stateless DE heads require it,
                          other heads ignore it via **_ignored. When the
                          backbone uses donor_kind='knn' (D1000 line) the
                          shape is (B, N, k_donor, n_roles) and a paired
                          donor_cand_idx (B, N, k_donor) is also forwarded
                          via kwargs to remap local→global donor indices.

        Returns:
            (deltas_k, params_list): deltas_k is (M, B, N, K, D),
            params_list is list of K param dicts (needed for DE attention extraction).
        """
        adj = getattr(cache, 'adj', None)
        knn_idx = getattr(cache, 'knn_idx', None)
        K = self.K
        # Augmented donor pool (B, N+K_archive, D) when graph-native archive
        # is integrated. Forwarded to operators that consume it
        # (BatchedDiffAttDE); other heads ignore via **_ignored.
        donor_coords = kwargs.get('donor_coords')
        donor_cand_idx = kwargs.get('donor_cand_idx')
        params_list = [
            self.heads[k].compute_params(
                h_expert[:, :, k], coords, fitness, adj=adj,
                route_probs=soft_probs, bounds_span=bounds_span,
                knn_idx=knn_idx,
                h_backbone=h_backbone_aug if (h_backbone_aug is not None and k in self._pool_skip_heads) else h,
                fes_frac=kwargs.get('fes_frac', 0.0),
                h_global=h_global,
                donor_logits=donor_logits,
                donor_coords=donor_coords,
                donor_cand_idx=donor_cand_idx)
            for k in range(K)
        ]
        deltas_k = torch.stack([
            self.heads[k].sample_batch(params_list[k], coords, bounds_span, M)
            for k in range(K)
        ], dim=3)  # (M, B, N, K, D)
        return deltas_k, params_list

    def _inject_oracle_floor(self, winner, route_weights_soft):
        """Flip lowest-margin nodes to oracle-best operator (all GPU, O(1)).

        For each (batch, m-sample), if the oracle-best operator has fewer
        than eps*N nodes, flip the lowest-margin non-oracle nodes to it.
        """
        M, B, N = winner.shape
        oracle = self._oracle_best_k  # (B, N)
        eps_count = max(int(self.oracle_epsilon * N), 1)  # e.g. 2 out of 50

        oracle_exp = oracle.unsqueeze(0).expand(M, B, N)  # (M, B, N)
        already_oracle = (winner == oracle_exp)  # (M, B, N)

        # Per (m,b): how many more nodes does oracle need?
        need_more = (eps_count - already_oracle.sum(dim=-1)).clamp(min=0)  # (M, B)

        # Margin: router confidence in current choice over oracle
        # Low margin = best candidate to flip
        oracle_prob = route_weights_soft.gather(
            -1, oracle_exp.unsqueeze(-1)).squeeze(-1)  # (M, B, N)
        winner_prob = route_weights_soft.gather(
            -1, winner.unsqueeze(-1)).squeeze(-1)  # (M, B, N)
        margin = (winner_prob - oracle_prob).masked_fill(
            already_oracle, float('inf'))  # (M, B, N)

        # Sort by margin, flip the first need_more[m,b] nodes
        _, flip_order = margin.sort(dim=-1)  # (M, B, N) ascending
        rank = torch.arange(N, device=winner.device).view(1, 1, N)
        flip_mask_sorted = rank < need_more.unsqueeze(-1)  # (M, B, N)

        # Scatter back to original node positions
        flip_mask = torch.zeros_like(already_oracle)
        flip_mask.scatter_(-1, flip_order, flip_mask_sorted)

        return torch.where(flip_mask, oracle_exp, winner)

    def step(self, h, h_per_head, h_global, coords, fitness,
             cache, D, M=1, gumbel_tau=1.0, bounds_span=200.0,
             donor_logits=None, **kwargs):
        """Produce M displacements via K=4 node-choice soft Gumbel routing.

        Args:
            donor_logits: (B, N, N, n_roles) from `backbone.donor_selector`,
                          required by stateless DE head (`BatchedDiffAttDE`).
                          Other heads ignore it. If None AND any head is
                          stateless, falls back to zero-valued logits
                          (diag-masked) so the call does not crash — this
                          preserves `variant.step` for isolation tests where
                          no backbone produces donor_logits.
        """
        B, N = h.shape[:2]
        K = self.K

        # Fallback: isolation tests may call step() without a backbone-produced
        # donor_logits. Zero-logit + diag mask → approximately-uniform random
        # selection. Production always supplies real donor_logits.
        if (donor_logits is None and self._de_idx is not None
                and isinstance(self.heads[self._de_idx], BatchedDiffAttDE)):
            donor_logits = torch.zeros(B, N, N, 3, device=h.device,
                                       dtype=h.dtype)
            eye = torch.eye(N, dtype=torch.bool,
                            device=h.device).unsqueeze(0).unsqueeze(-1)
            donor_logits = donor_logits.masked_fill(eye, -1e9)
        h_pool = kwargs.get('h_pool', None)

        # Precompute augmented backbone once (avoids double cat in expert + params)
        if h_pool is not None and self.pool_dim > 0:
            h_backbone_aug = torch.cat([h, h_pool], dim=-1)
        else:
            h_backbone_aug = None

        # Get (B, N, K, head_dim) expert embeddings — works for any n_heads
        h_expert = self._get_expert_embeddings(h, h_per_head, h_aug=h_backbone_aug)
        self._last_h_expert = h_expert.detach()  # saved for external head warmup
        self._last_h = h.detach()  # saved for head warmup h_backbone

        # ── Surrogate mode: compute all K deltas, skip routing ──
        if self.gate_type == 'surrogate':
            soft_probs = torch.ones(B, N, K, device=h.device) / K
            deltas_k, params_list = self._compute_all_deltas(
                h_expert, h, h_backbone_aug, coords, fitness, cache,
                soft_probs, bounds_span, M, kwargs, h_global=h_global,
                donor_logits=donor_logits)

            delta = torch.zeros(M, B, N, coords.shape[-1], device=h.device)
            extras = {
                'surrogate_mode': True,
                'active_mask': torch.ones(B, N, device=h.device),
                'active_fraction': torch.tensor(1.0),
                'entropy': torch.tensor(0.0),
                'deltas_k': deltas_k.detach(),
                'deltas_k_live': deltas_k,
                'routing_probs': soft_probs.detach(),
                'winner': torch.zeros(M, B, N, dtype=torch.long, device=h.device),
            }
            self._expose_de_extras(params_list, extras)
            return delta, extras

        # ── Standard routing path (adaptive/ranker/none gates) ──
        node_feat = kwargs.get('node_feat')
        fes_frac = kwargs.get('fes_frac', 0.0)
        if self.activity_gate is None:
            active_mask = torch.ones(B, N, device=h.device)
        elif self.gate_type == 'ranker':
            gate_scores = self.activity_gate(h, h_global=h_global,
                                             node_feat=node_feat)
            target_frac = kwargs.get('gate_target_frac', 0.5)
            active_mask = RankerGate.select_topk(gate_scores, target_frac)
        else:
            active_mask = self.activity_gate(h, h_global=h_global,
                                             node_feat=node_feat,
                                             fes_frac=fes_frac)

        h_normed = self.score_norm(h_expert)
        logits = self.scorer(h_normed).squeeze(-1)
        gate = torch.sigmoid(self.global_route_gate(h_global))
        logits = logits * (0.5 + gate.unsqueeze(1))
        logits = 2.3 * torch.tanh(logits / 2.3)

        soft_probs = F.softmax(logits, dim=-1)

        gumbel_noise = antithetic_gumbel_noise(M, B, N, K, device=h.device)
        y = (logits.unsqueeze(0) + gumbel_noise) / gumbel_tau
        route_weights_soft = F.softmax(y, dim=-1)
        winner = route_weights_soft.argmax(dim=-1)

        if self._oracle_best_k is not None and self.oracle_epsilon > 0:
            winner = self._inject_oracle_floor(winner, route_weights_soft)

        deltas_k, params_list = self._compute_all_deltas(
            h_expert, h, h_backbone_aug, coords, fitness, cache,
            soft_probs, bounds_span, M, kwargs, h_global=h_global,
            donor_logits=donor_logits)

        # DE attention matrices for geometric auxiliary losses
        _de_attn = {}
        if self._de_idx is not None:
            de_params = params_list[self._de_idx]
            _a = de_params.get('_A_pbest')
            if _a is not None:
                _de_attn = {'_A_pbest': _a, '_A_r1': de_params.get('_A_r1'), '_A_r2': de_params.get('_A_r2')}

        # Split ST
        D_coord = coords.shape[-1]
        winner_idx = winner.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, D_coord)
        delta_for_heads = deltas_k.gather(3, winner_idx).squeeze(3)
        delta_for_router = (route_weights_soft.unsqueeze(-1) * deltas_k.detach()).sum(dim=3)
        delta = delta_for_heads + (delta_for_router - delta_for_router.detach())

        # Post-hoc masking: inactive individuals get zero delta.
        _skip_mask = self.training and getattr(self, 'eval_all_select_active', False)
        if not _skip_mask:
            delta = delta * active_mask.unsqueeze(0).unsqueeze(-1)  # (M, B, N, D) * (1, B, N, 1)

        # Diagnostics
        entropy = -(soft_probs * soft_probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
        active_fraction = active_mask.mean()

        extras = {
            'entropy': entropy.detach(),
            'active_mask': active_mask,  # NOT detached for adaptive gate (ST)
            'active_fraction': active_fraction,
            'routing_probs': soft_probs.detach(),
            'logits': logits.detach(),
            'logits_live': logits,  # with gradient for oracle router loss
            'winner': winner.detach(),
            'deltas_k': deltas_k.detach(),
            'deltas_k_live': deltas_k,  # with gradient for LUPI contrafactual
        }
        # Ranker gate: expose live scores for pairwise loss
        if self.gate_type == 'ranker':
            extras['gate_scores'] = gate_scores  # (B, N) with gradient

        # Aux tensors for geometric losses (WITH grad where needed)
        if self._coordls_idx is not None:
            cls_p = params_list[self._coordls_idx]
            if 'dim_bias' in cls_p:
                extras['dim_bias_coordls'] = cls_p['dim_bias']
                extras['sensitivity_target_coordls'] = cls_p.get('_sensitivity_target')
        if self._cmaes_idx is not None:
            cma_p = params_list[self._cmaes_idx]
            if '_attn_logits' in cma_p:
                extras['attn_logits_cmaes'] = cma_p['_attn_logits']

        # Insert DE attention matrices (with grad) for geometric auxiliary losses
        extras.update(_de_attn)

        self._expose_de_extras(params_list, extras)

        return delta, extras

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Load with automatic migration of old checkpoint keys.

        Handles four eras:
        1. Ancient:  shared `head_proj.{0,2}.*` -> split per head
        2. Previous: `head_projs.{k}.*` in variant -> `heads.{k}.proj.*`
        2b.           `head_norms.{k}.*` -> `heads.{k}.proj_norm.*`
        3. CURRENT (stateless DE refactor): drops operator-internal keys that
           no longer exist in the new BatchedDiffAttDE:
              heads.{k}.proj.*, heads.{k}.proj_norm.*
              heads.{k}.diff_{query,key}_r[12].*
              heads.{k}.global_cond.*
              heads.{k}.alpha_{pbest,r1,r2}, heads.{k}.beta_progress
           AND drops variant-level router keys whose dims changed
           (head_dim 16 -> gatv2_hidden 128 when stateless DE is used):
              score_norm.*, scorer.*
           The new backbone.donor_selector.* keys are initialized fresh.
        """
        import re

        migrated = dict(state_dict)
        K, hd = self.K, self.head_dim

        # ── Era 3: drop operator-internal keys for stateless DE heads ──
        # These keys are present in E7d-era checkpoints but absent in the
        # stateless `BatchedDiffAttDE`. We drop unconditionally for any head
        # whose current instance has `proj is None` (marker of stateless).
        _stateless_head_ks = {
            k for k, h in enumerate(self.heads)
            if isinstance(h, BatchedDiffAttDE) and getattr(h, 'proj', None) is None
        }
        # If ANY head is stateless, we consider the whole variant on Era 3
        # (router score_norm/scorer dims follow from the max operator head_dim).
        if _stateless_head_ks:
            era3_patterns = [
                re.compile(r'heads\.(\d+)\.proj\.'),
                re.compile(r'heads\.(\d+)\.proj_norm\.'),
                re.compile(r'heads\.(\d+)\.diff_(?:query|key)_r[12]'),
                re.compile(r'heads\.(\d+)\.global_cond'),
                re.compile(r'heads\.(\d+)\.alpha_(?:pbest|r1|r2)$'),
                re.compile(r'heads\.(\d+)\.beta_progress$'),
            ]
            for key in list(migrated.keys()):
                for pat in era3_patterns:
                    m = pat.match(key)
                    if m and int(m.group(1)) in _stateless_head_ks:
                        migrated.pop(key)
                        break

            # Variant-level router keys: drop if shapes don't match current.
            # Current `score_norm` / `scorer` are sized at head_dim (=128 when
            # stateless). Old checkpoints stored them at 16-dim. Dropping here
            # lets super().load_state_dict init from the fresh module params.
            _current_shapes = {n: p.shape for n, p in self.named_parameters()}
            _current_shapes.update({n: b.shape for n, b in self.named_buffers()})
            for router_key in ('score_norm.weight', 'score_norm.bias',
                               'scorer.weight', 'scorer.bias'):
                if router_key in migrated:
                    old_shape = tuple(migrated[router_key].shape)
                    new_shape = _current_shapes.get(router_key)
                    if new_shape is None or old_shape != tuple(new_shape):
                        migrated.pop(router_key)

        # Era 1: shared head_proj -> per-head proj (skip heads without proj, e.g. NoOp)
        if 'head_proj.0.weight' in migrated:
            for k in range(K):
                if self.heads[k].proj is None:
                    continue
                migrated[f'heads.{k}.proj.0.weight'] = migrated['head_proj.0.weight'].clone()
                migrated[f'heads.{k}.proj.0.bias'] = migrated['head_proj.0.bias'].clone()
                migrated[f'heads.{k}.proj.2.weight'] = migrated['head_proj.2.weight'][k * hd:(k + 1) * hd].clone()
                migrated[f'heads.{k}.proj.2.bias'] = migrated['head_proj.2.bias'][k * hd:(k + 1) * hd].clone()
                migrated[f'heads.{k}.proj_norm.weight'] = torch.ones(hd)
                migrated[f'heads.{k}.proj_norm.bias'] = torch.zeros(hd)
            for old_key in ['head_proj.0.weight', 'head_proj.0.bias',
                            'head_proj.2.weight', 'head_proj.2.bias']:
                migrated.pop(old_key, None)

        # Era 2: head_projs.{k}.* in variant -> heads.{k}.proj.*
        old_proj_keys = [k for k in migrated if k.startswith('head_projs.')]
        if old_proj_keys and f'heads.0.proj.0.weight' not in migrated:
            for old_key in list(old_proj_keys):
                parts = old_key.split('.', 2)  # ['head_projs', '0', '0.weight']
                k_idx = int(parts[1])
                rest = parts[2]
                if k_idx < len(self.heads) and self.heads[k_idx].proj is not None:
                    new_key = f'heads.{k_idx}.proj.{rest}'
                    migrated[new_key] = migrated.pop(old_key)
                else:
                    migrated.pop(old_key)  # drop proj for heads without proj

        # Era 2b: head_norms.{k}.* -> heads.{k}.proj_norm.*
        old_norm_keys = [k for k in migrated if k.startswith('head_norms.')]
        if old_norm_keys:
            for old_key in list(old_norm_keys):
                parts = old_key.split('.', 2)  # ['head_norms', '0', 'weight']
                k_idx = int(parts[1])
                rest = parts[2]
                if k_idx < len(self.heads) and self.heads[k_idx].proj_norm is not None:
                    new_key = f'heads.{k_idx}.proj_norm.{rest}'
                    migrated[new_key] = migrated.pop(old_key)
                else:
                    migrated.pop(old_key)

        # Inject defaults for DE-head params that may be absent in old checkpoints
        for k in range(K):
            if isinstance(self.heads[k], BatchedDiffDE):
                migrated.setdefault(f'heads.{k}.memory_gate', torch.tensor(0.0))
                migrated.setdefault(f'heads.{k}.subpop_scale', torch.tensor(0.0))
            # Only inject global_cond default for heads that actually have it
            # (stateless BatchedDiffAttDE removed global_cond in Era 3).
            if (hasattr(self.heads[k], 'global_cond')
                    and self.heads[k].global_cond is not None):
                migrated.setdefault(
                    f'heads.{k}.global_cond.weight',
                    torch.zeros_like(self.heads[k].global_cond.weight))

        return super().load_state_dict(migrated, strict=strict, **kwargs)
