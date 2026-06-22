"""Task pool management and curriculum logic for L2O training."""
import logging
import math
import random

import torch

log = logging.getLogger(__name__)


# ── Function management constants ──
# Functions excluded from training. Two principled reasons:
#   1. Empirical pathology: structural gradient issues that produce 10^5x median
#      backbone gradient (e.g. F19 katsuura-D=1 — Phase 0 2026-04-30).
#   2. BLACKLIST sync: cec2017.benchmark.BLACKLIST flags numerical-correctness
#      divergence vs opfunu reference; we don't train on values we don't trust.
#
# F17, F19 added 2026-04-30 (Commit B+):
#   F19: Hybrid 10 with 6-partition layout puts katsuura on D=1; the 10/D^1.2
#        exponent collapses, producing gradient ~10^5x median. BLACKLIST-flagged.
#   F17: Hybrid 8 hitting-path dominance 493x; BLACKLIST-flagged for NaN in
#        opfunu at D=10. Phase 0 (analysis/phase0_a3_step7000.json) confirms.
EXCLUDE_FIDS = {2, 16, 17, 18, 19, 28, 29}

# BLACKLIST entries with no measurable training pathology. Each entry is a
# deliberate, dated tolerance backed by Phase 0 evidence. Review trigger is the
# first measurement after Phase 1 retrain completes — calendar-free, milestone-tied.
ACCEPTED_DIVERGENT = {
    5: ("FIXME(review at Phase 1 retrain completion + 1 week): schaffer_f7 scale issue "
        "per cec2017.BLACKLIST. Phase 0 2026-04-30 (analysis/phase0_a3_step7000.json): "
        "benign in pairwise (0/19 anti-corr) and hitting (5/19 moderate). "
        "Re-evaluate with fresh post-Phase-1 measurement."),
    21: ("FIXME(review at Phase 1 retrain completion + 1 week): rel_error > 1e-8 vs opfunu "
         "per cec2017.BLACKLIST. Phase 0 2026-04-30: benign across all losses (max ratio 1.1x, "
         "anti-corr 0-3/19). Re-evaluate post-Phase-1."),
    23: ("FIXME(review at Phase 1 retrain completion + 1 week): rel_error > 1e-8 vs opfunu "
         "per cec2017.BLACKLIST. Phase 0 2026-04-30: benign across all losses (max ratio 1.0x, "
         "anti-corr 1-4/19). Re-evaluate post-Phase-1."),
}


def _validate_blacklist_exclude_consistency():
    """BLACKLIST ⊆ EXCLUDE_FIDS ∪ ACCEPTED_DIVERGENT. Fires at module import.

    cec2017.BLACKLIST flags numerical-correctness divergence vs opfunu reference.
    A function so flagged must explicitly declare training intent: either excluded
    (EXCLUDE_FIDS) or empirically-tolerated (ACCEPTED_DIVERGENT, with FIXME and
    review trigger). Silent inclusion is a process bug — see Phase 0 2026-04-30
    F19 finding for the cost of letting these desync.
    """
    from cec2017.benchmark import BLACKLIST
    bl_fids = {fid for (fid, _ndim) in BLACKLIST}
    missing = bl_fids - EXCLUDE_FIDS - set(ACCEPTED_DIVERGENT.keys())
    assert not missing, (
        f"Functions {sorted(missing)} are in cec2017.BLACKLIST but not in "
        f"EXCLUDE_FIDS nor ACCEPTED_DIVERGENT. New BLACKLIST entries must "
        f"declare intent: add to EXCLUDE_FIDS (don't train) or to "
        f"ACCEPTED_DIVERGENT with FIXME and Phase 0 evidence."
    )


_validate_blacklist_exclude_consistency()

HIT_RATE_WINDOW = 15
HIT_RATE_THRESHOLD = 0.8
STALL_WINDOW = 30
STALL_THRESHOLD = 0.10

# Fraction of newly spawned tasks that are vanilla (aug_seed=0).
# Canonical eval runs on vanilla CEC17, so the train distribution MUST include
# the vanilla subspace or the model never sees it. 0.33 = ~1/3 vanilla, 2/3 augmented.
VANILLA_TASK_PROB = 0.33


def get_all_fn_ids(single_fn=0, env='cec2017', lf_levels=None):
    """Return the list of fid values active for the chosen training env.

    For env='cec2017' (default): CEC2017 fids 1..29 minus EXCLUDE_FIDS.
    For env='linked-flame': linked-flame Levels (1..5), filtered by
    lf_levels (default all five).
    """
    if single_fn > 0:
        return [single_fn]
    if env == 'linked-flame':
        levels = lf_levels if lf_levels is not None else [1, 2, 3, 4, 5]
        for L in levels:
            if L not in (1, 2, 3, 4, 5):
                raise ValueError(f"linked-flame Level {L} is not in 1..5")
        return list(levels)
    if env == 'bbob':
        # BBOB has 24 functions (1..24); the differentiable torch port
        # (encoder/bbob_torch.py) validates all of them vs cocoex — no blacklist.
        return list(range(1, 25))
    return [i for i in range(1, 30) if i not in EXCLUDE_FIDS]


class Task:
    """A single augmented training task in the pool."""
    __slots__ = ('task_id', 'fid', 'aug_seed', 'curriculum_idx',
                 'max_level_ever', 'hits', 'bptt_w',
                 'steps_since_record', 'age')

    def __init__(self, task_id, fid, aug_seed, bptt_w_init=20):
        self.task_id = task_id
        self.fid = fid
        self.aug_seed = aug_seed
        self.curriculum_idx = 0
        self.max_level_ever = 0
        self.hits = []
        self.bptt_w = bptt_w_init
        self.steps_since_record = 0
        self.age = 0

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}

    @classmethod
    def from_dict(cls, d):
        t = cls.__new__(cls)
        for k in cls.__slots__:
            setattr(t, k, d[k])
        return t


def simple_mix_selection(world_size, all_fn_ids, vanilla_prob, fn_rng):
    """Curriculum-bypass per-step task selection.

    Each rank gets a fresh (fid, aug_seed) pair: a random fid from
    `all_fn_ids`, and aug_seed=0 (vanilla CEC17) with probability
    `vanilla_prob` else a fresh random seed in [1, 2**31-1] (augmented).

    No persistent state, no curriculum-weighted sampling, no advance/retire.
    Used by --task-mix-vanilla-prob to remove the abrupt curriculum
    transitions that drown small effects in probe iterations.
    """
    out = []
    for _ in range(world_size):
        fid = fn_rng.choice(all_fn_ids)
        if fn_rng.random() < vanilla_prob:
            seed = 0
        else:
            seed = fn_rng.randint(1, 2**31 - 1)
        out.append((fid, seed))
    return out


def spawn_task(task_id, all_fn_ids, bptt_w_init=20, vanilla_prob=VANILLA_TASK_PROB):
    """Create a new task with random fid (category-balanced) + aug_seed.

    With probability `vanilla_prob`, aug_seed=0 (vanilla CEC17, matches eval
    distribution). Otherwise random seed (augmented rotation + shift).
    """
    fid = random.choice(all_fn_ids)
    if random.random() < vanilla_prob:
        aug_seed = 0
    else:
        aug_seed = random.randint(1, 2**31 - 1)
    return Task(task_id, fid, aug_seed, bptt_w_init)


def compute_pool_weights(task_pool, fn_weight_overrides=None):
    """Curriculum-weighted sampling over task pool.

    Args:
        fn_weight_overrides: optional dict {fid: multiplier} to scale
            sampling weight for specific functions (e.g. {13: 0.5, 19: 0.5}).
    """
    task_ids = []
    weights = []
    for tid, task in task_pool.items():
        task_ids.append(tid)
        recent = task.hits[-HIT_RATE_WINDOW:] if task.hits else []
        n_samples = len(task.hits)
        hr = sum(recent) / max(len(recent), 1) if recent else 0.0
        if n_samples < 20:
            w = 0.5
        elif hr >= HIT_RATE_THRESHOLD:
            w = 0.15
        elif hr < STALL_THRESHOLD:
            w = 0.15
        else:
            w = 1.0
        if fn_weight_overrides and task.fid in fn_weight_overrides:
            w *= fn_weight_overrides[task.fid]
        weights.append(w)
    return task_ids, weights


def make_augmented_fn(fid, D, device, aug_seed, aug_cache):
    """Create a deterministic augmented function from (fid, aug_seed)."""
    rng = torch.Generator(device='cpu')
    rng.manual_seed(aug_seed)
    return aug_cache.sample(fid=fid, D=D, rng=rng)


def update_task(task, hit, gn_val, total_gens, args, task_pool,
                next_task_id, all_fn_ids, n_targets, step, logger):
    """Unified curriculum update for a single task (deduplicates dist/single-GPU).

    Returns updated next_task_id.
    """
    task.hits.append(hit)
    if len(task.hits) > 2 * HIT_RATE_WINDOW:
        task.hits = task.hits[-HIT_RATE_WINDOW:]
    task.age += 1
    task.steps_since_record += 1

    # BPTT window: snap to actual gens on HIT, grow on miss
    if not math.isfinite(gn_val) or gn_val > 10.0:
        task.bptt_w = max(task.bptt_w - 10, args.bptt_w_min)
    elif hit:
        task.bptt_w = max(min(task.bptt_w, total_gens + 20), args.bptt_w_min)
    else:
        task.bptt_w = min(task.bptt_w + 10, args.bptt_w_max)

    # Advance / demote
    _retire = False
    if len(task.hits) >= HIT_RATE_WINDOW:
        recent = task.hits[-HIT_RATE_WINDOW:]
        hr = sum(recent) / len(recent)
        if hr >= HIT_RATE_THRESHOLD and task.curriculum_idx < n_targets - 1:
            task.curriculum_idx += 1
            task.hits = []
            if task.curriculum_idx > task.max_level_ever:
                task.max_level_ever = task.curriculum_idx
                task.steps_since_record = 0
            logger.info("task %d F%02d ADVANCE -> T%d (max=%d) step %d",
                        task.task_id, task.fid, task.curriculum_idx,
                        task.max_level_ever, step)
        elif hr >= HIT_RATE_THRESHOLD and task.curriculum_idx == n_targets - 1:
            # Mastered the final level — retire as SOLVED
            logger.info("task %d F%02d SOLVED (mastered T%d) -> RETIRE step %d",
                        task.task_id, task.fid, task.curriculum_idx, step)
            _retire = True
        elif len(task.hits) >= STALL_WINDOW and hr < STALL_THRESHOLD:
            if task.curriculum_idx > 0:
                task.curriculum_idx -= 1
                task.hits = []
                logger.info("task %d F%02d DEMOTE -> T%d step %d",
                            task.task_id, task.fid, task.curriculum_idx, step)
            else:
                task.hits = task.hits[-5:]

    # Retire: ceiling reached
    if not _retire and (task.age >= args.min_task_age and
          task.steps_since_record >= args.ceiling_window):
        logger.info("task %d F%02d CEILING (max=T%d, %d steps) -> RETIRE step %d",
                    task.task_id, task.fid, task.max_level_ever,
                    task.steps_since_record, step)
        _retire = True

    if _retire:
        del task_pool[task.task_id]
        new_task = spawn_task(next_task_id, all_fn_ids, args.bptt_w_init)
        task_pool[next_task_id] = new_task
        _tag = "VANILLA" if new_task.aug_seed == 0 else f"seed={new_task.aug_seed}"
        logger.info("task %d F%02d(%s) SPAWNED step %d",
                    next_task_id, new_task.fid, _tag, step)
        next_task_id += 1

    return next_task_id
