"""
techniques_v2.py — SOTA technique dispatch for MOS v2.

8 techniques organized into 7 operator families:

  Coordinate-wise local search:
    0. MTS-LS1: budget-limited coordinate-wise search (Tseng & Chen 2008)
  Crossover (pair-based):
    1. SBX + polynomial mutation (Simulated Binary Crossover)
  Adaptive DE (SHADE):
    2. Mini-SHADE: current-to-pbest/1, success-history F/CR, jSO Fw schedule
  Gaussian perturbation:
    3. Gaussian-LS with 1/5 success rule (Rechenberg adaptive sigma)
  Drift-to-best (DE):
    4. DE/current-to-best/1 with Cauchy F
  EDA sampling (diagonal):
    5. CEM/EDA: sample from elite N(mean, diag_std) (Cross-Entropy Method)
  Heavy-tailed exploration:
    6. Cauchy mutation: heavy-tailed perturbation for escaping local optima
  EDA sampling (full covariance):
    7. CovEMNA: full-covariance Gaussian with momentum (shrinkage estimator)

Total offspring = N_survive (one child per assigned survivor).
Technique 0 (MTS-LS1) costs ~max_evals_per_ind evaluations per individual;
all others cost 0 extra evaluations (1 mandatory post-dispatch eval each).

Changes from v1 (March 6, 2026):
  - T2: DE/rand/1 (fixed F=0.8) → Mini-SHADE (adaptive F/CR, pbest mutation)
  - T3: Linear sigma decay → 1/5 success rule adaptive sigma
  - T4: GWO-drift (stateless, weak) → DE/current-to-best/1 (proven, elite-guided)
  - T6: BBO migration (redundant) → Cauchy mutation (heavy-tailed exploration)
  - T7: CMA-ES ask() without tell() (BROKEN) → CovEMNA with momentum
  - Added TechniqueState for persistent state across generations
"""
import numpy as np

from .techniques import sbx_crossover

# --- 8 techniques ---
N_TECHNIQUES = 8
TECHNIQUE_NAMES = [
    "MTS-LS1", "SBX", "SHADE", "Gaussian-LS",
    "DE/best", "CEM/EDA", "Cauchy", "CovEMNA",
]


# ======================================================================
# Persistent state for stateful techniques
# ======================================================================

class SHADEMemory:
    """Success-history adaptive F, CR (Tanabe & Fukunaga 2013).

    Maintains a circular buffer of successful F and CR values.
    Updated via deferred reporting: caller invokes report_child_fitness()
    after evaluating children.
    """

    def __init__(self, H=10):
        self.H = H
        self.M_F = np.full(H, 0.5)
        self.M_CR = np.full(H, 0.5)
        self.k = 0
        # Pending trials for deferred reporting
        self._pending_F = None
        self._pending_CR = None
        self._pending_parent_fit = None

    def sample(self, M, rng):
        """Sample F and CR values for M individuals."""
        F = np.empty(M)
        CR = np.empty(M)
        for i in range(M):
            r = int(rng.integers(0, self.H))
            # F ~ Cauchy(M_F[r], 0.1), truncated to (0, 1]
            fi = 0.0
            for _ in range(20):  # safety limit; clamp below is intentional fallback
                fi = float(rng.standard_cauchy() * 0.1 + self.M_F[r])
                if fi > 0:
                    break
            F[i] = min(max(fi, 0.01), 1.0)  # clamp ensures valid F even if loop exhausts
            # CR ~ Normal(M_CR[r], 0.1), clipped to [0, 1]
            CR[i] = float(np.clip(rng.normal(self.M_CR[r], 0.1), 0.0, 1.0))
        return F, CR

    def record_trials(self, F_vals, CR_vals, parent_fitness):
        """Store F, CR, parent fitness for deferred update."""
        self._pending_F = np.asarray(F_vals, dtype=np.float64)
        self._pending_CR = np.asarray(CR_vals, dtype=np.float64)
        self._pending_parent_fit = np.asarray(parent_fitness, dtype=np.float64)

    def report_child_fitness(self, child_fitness):
        """Update memory with success info (call after evaluation)."""
        if self._pending_F is None or len(self._pending_F) == 0:
            return
        child_fitness = np.asarray(child_fitness, dtype=np.float64)
        n = min(len(self._pending_F), len(child_fitness))
        parent_fit = self._pending_parent_fit[:n]
        c_fit = child_fitness[:n]

        success = c_fit < parent_fit
        if success.any():
            S_F = self._pending_F[:n][success]
            S_CR = self._pending_CR[:n][success]
            delta = np.abs(parent_fit[success] - c_fit[success])
            w = delta / (delta.sum() + 1e-15)
            # Weighted Lehmer mean for F
            self.M_F[self.k] = float(
                (w * S_F ** 2).sum() / ((w * S_F).sum() + 1e-15))
            # Weighted arithmetic mean for CR
            self.M_CR[self.k] = float((w * S_CR).sum())
            self.k = (self.k + 1) % self.H

        self._pending_F = None
        self._pending_CR = None
        self._pending_parent_fit = None


class GaussAdaptiveState:
    """1/5 success rule for Gaussian-LS sigma adaptation (Rechenberg 1973)."""

    def __init__(self):
        self.sigma_scale = 1.0
        self.prev_median = None

    def adapt(self, subpop_fitness):
        """Adapt sigma_scale based on fitness improvement trend."""
        curr_median = float(np.median(subpop_fitness))
        if self.prev_median is not None:
            if curr_median < self.prev_median:  # population improved
                self.sigma_scale = min(self.sigma_scale * 1.2, 3.0)
            else:
                self.sigma_scale = max(self.sigma_scale * 0.82, 0.1)
        self.prev_median = curr_median
        return self.sigma_scale


class CovarianceState:
    """Momentum-based full-covariance estimation (EMNA-style)."""

    def __init__(self, D):
        self.D = D
        self.mean = None
        self.cov = None
        self.alpha = 0.5  # momentum: 0.5 = equal weight past/present

    def update_and_sample(self, subpop_coords, subpop_fitness, bounds, rng):
        M, D = subpop_coords.shape
        assert D == self.D, f"CovarianceState dimension mismatch: expected {self.D}, got {D}"
        n_elite = max(2, int(0.3 * M))
        elite_idx = np.argsort(subpop_fitness)[:n_elite]
        elite = subpop_coords[elite_idx]

        new_mean = elite.mean(axis=0)

        # Covariance estimation with shrinkage
        if n_elite >= 3:
            sample_cov = np.cov(elite.T) if n_elite > 1 else np.eye(D)
            if sample_cov.ndim == 0:  # scalar case (D=1)
                sample_cov = np.array([[float(sample_cov)]])
            diag_target = np.diag(np.diag(sample_cov) + 1e-10)
            # Ledoit-Wolf-style shrinkage: more shrinkage with fewer samples
            shrink = min(1.0, max(0.1, D / (n_elite + D)))
            new_cov = (1 - shrink) * sample_cov + shrink * diag_target
        else:
            new_cov = np.diag(np.var(elite, axis=0) + 1e-10)

        # For D > 200, force diagonal to avoid O(D^3)
        if D > 200:
            new_cov = np.diag(np.diag(new_cov))

        # Momentum blending
        if self.mean is None:
            self.mean = new_mean
            self.cov = new_cov
        else:
            self.mean = self.alpha * self.mean + (1 - self.alpha) * new_mean
            self.cov = self.alpha * self.cov + (1 - self.alpha) * new_cov

        # Sample children
        try:
            if D > 200:
                # Diagonal sampling (fast)
                std = np.sqrt(np.diag(self.cov))
                children = rng.normal(self.mean, std, size=(M, D))
            else:
                children = rng.multivariate_normal(self.mean, self.cov, size=M)
        except np.linalg.LinAlgError:
            std = np.sqrt(np.abs(np.diag(self.cov)) + 1e-10)
            children = rng.normal(self.mean, std, size=(M, D))

        return np.clip(children, bounds[:, 0], bounds[:, 1]).astype(np.float64)


class TechniqueState:
    """Persistent state container for all stateful techniques.

    Create one per optimization run, pass to apply_technique_to_subpop().
    Call report_children() after evaluating children to update SHADE memory.
    """

    def __init__(self, D=30):
        self.shade = SHADEMemory(H=10)
        self.gauss = GaussAdaptiveState()
        self.cov = CovarianceState(D)

    def report_children(self, assignments, parent_fitness, child_fitness):
        """Report evaluation results for deferred state updates.

        Call this after evaluating ALL children from a generation step.

        Args:
            assignments: (N,) int technique IDs per individual
            parent_fitness: (N,) float parent fitness before technique
            child_fitness: (N,) float child fitness after evaluation
        """
        mask2 = assignments == 2
        if mask2.any():
            self.shade.report_child_fitness(child_fitness[mask2])

        # Update Gaussian 1/5 rule with actual child success
        if self.gauss is not None:
            mask3 = assignments == 3
            if mask3.any():
                n_success = int((child_fitness[mask3] < parent_fitness[mask3]).sum())
                n_total = int(mask3.sum())
                success_rate = n_success / n_total
                # 1/5 rule: expand sigma if success > 20%, shrink otherwise
                if success_rate > 0.2:
                    self.gauss.sigma_scale = min(self.gauss.sigma_scale * 1.2, 3.0)
                else:
                    self.gauss.sigma_scale = max(self.gauss.sigma_scale * 0.82, 0.1)


# ======================================================================
# Technique operators
# ======================================================================

def _shade_de(subpop_coords, subpop_fitness, all_survivor_coords,
              bounds, rng, gen_frac=0.5, technique_state=None, **kwargs):
    """
    Mini-SHADE: current-to-pbest/1 with success-history F/CR and jSO Fw.

    Tanabe & Fukunaga 2013 (SHADE) + Brest et al. 2017 (jSO Fw schedule).
    Adapted for MOS subpopulation slot: no LPSR (population externally managed).

    Mutation: v = x_i + Fw*(x_pbest - x_i) + F*(x_r1 - x_r2)
    """
    M, D = subpop_coords.shape

    # Sample adaptive F, CR
    if technique_state is not None:
        F_vals, CR_vals = technique_state.shade.sample(M, rng)
    else:
        # Stateless fallback: Cauchy(0.5, 0.1) / Normal(0.5, 0.1)
        F_vals = np.clip(rng.standard_cauchy(M) * 0.1 + 0.5, 0.01, 1.0)
        CR_vals = np.clip(rng.normal(0.5, 0.1, M), 0.0, 1.0)

    # jSO Fw schedule (Brest et al. 2017)
    if gen_frac < 0.2:
        Fw_scale = 0.7
    elif gen_frac < 0.4:
        Fw_scale = 0.8
    else:
        Fw_scale = 1.2

    # Select pbest: top p% of subpopulation (p ∈ [0.05, 0.2])
    p = max(0.05, 0.2 - 0.15 * gen_frac)  # p decreases over time (more greedy)
    n_pbest = max(1, min(int(p * M) + 1, M))  # clamp to [1, M]
    pbest_pool = np.argsort(subpop_fitness)[:n_pbest]

    N_all = len(all_survivor_coords)
    children = np.empty((M, D), dtype=np.float64)

    for i in range(M):
        x_i = subpop_coords[i]
        F_i = F_vals[i]
        CR_i = CR_vals[i]
        Fw_i = Fw_scale * F_i

        # x_pbest: random from top p%
        x_pbest = subpop_coords[pbest_pool[rng.integers(0, n_pbest)]]
        # x_r1, x_r2: random from full survivor population
        r1 = int(rng.integers(0, N_all))
        if N_all > 1:
            r2 = int(rng.integers(0, N_all))
            while r2 == r1:
                r2 = int(rng.integers(0, N_all))
        else:
            r2 = r1  # degenerate: difference vector is zero

        # Mutation: current-to-pbest/1
        v = x_i + Fw_i * (x_pbest - x_i) + F_i * (
            all_survivor_coords[r1] - all_survivor_coords[r2])

        # Binomial crossover
        cross_mask = rng.random(D) < CR_i
        j_rand = int(rng.integers(0, D))
        cross_mask[j_rand] = True
        children[i] = np.where(cross_mask, v, x_i)

    children = np.clip(children, bounds[:, 0], bounds[:, 1])

    # Record trials for deferred SHADE memory update
    if technique_state is not None:
        technique_state.shade.record_trials(F_vals, CR_vals, subpop_fitness)

    return children.astype(np.float64)


def _gaussian_ls_adaptive(subpop_coords, subpop_fitness, bounds, rng,
                           gen_frac=0.5, technique_state=None, **kwargs):
    """
    Gaussian local search with 1/5 success rule (Rechenberg 1973).

    Isotropic Gaussian perturbation with sigma adapted via population
    improvement tracking. sigma_scale increases on improvement, decreases
    on stagnation.
    """
    M, D = subpop_coords.shape
    span = bounds[:, 1] - bounds[:, 0]

    # Read sigma_scale (adaptation happens in report_children via actual 1/5 rule)
    if technique_state is not None:
        sigma_scale = technique_state.gauss.sigma_scale
    else:
        sigma_scale = 1.0

    # Base sigma with progress decay + adaptive scale
    sigma = 0.05 * sigma_scale * (1.0 - 0.7 * gen_frac) * span

    children = subpop_coords + rng.standard_normal((M, D)) * sigma
    return np.clip(children, bounds[:, 0], bounds[:, 1]).astype(np.float64)


def _de_current_to_best(subpop_coords, subpop_fitness, all_survivor_coords,
                         bounds, rng, gen_frac=0.5, **kwargs):
    """
    DE/current-to-best/1: drift toward THE best individual.

    v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)

    More aggressive exploitation than SHADE (which uses pbest, not best).
    F sampled from Cauchy(0.8, 0.1) — biased toward exploitation.
    """
    M, D = subpop_coords.shape
    N_all = len(all_survivor_coords)

    # Best in subpopulation
    best_idx = np.argmin(subpop_fitness)
    x_best = subpop_coords[best_idx]

    children = np.empty((M, D), dtype=np.float64)
    for i in range(M):
        # F from Cauchy(0.8, 0.1), biased high for aggressive exploitation.
        # Loop rejects F<=0; clamp below is intentional safety net if all 20 draws are negative.
        F = 0.0
        for _ in range(20):
            F = float(rng.standard_cauchy() * 0.1 + 0.8)
            if F > 0:
                break
        F = min(max(F, 0.01), 1.0)

        # CR from Normal(0.9, 0.05), high crossover for exploitation
        CR = float(np.clip(rng.normal(0.9, 0.05), 0.0, 1.0))

        x_i = subpop_coords[i]
        r1 = int(rng.integers(0, N_all))
        if N_all > 1:
            r2 = int(rng.integers(0, N_all))
            while r2 == r1:
                r2 = int(rng.integers(0, N_all))
        else:
            r2 = r1  # degenerate: difference vector is zero

        v = x_i + F * (x_best - x_i) + F * (
            all_survivor_coords[r1] - all_survivor_coords[r2])

        # Binomial crossover
        cross_mask = rng.random(D) < CR
        cross_mask[int(rng.integers(0, D))] = True
        children[i] = np.where(cross_mask, v, x_i)

    return np.clip(children, bounds[:, 0], bounds[:, 1]).astype(np.float64)


def _cem_sampling(subpop_coords, subpop_fitness, bounds, rng, **kwargs):
    """
    Cross-Entropy Method / EDA (Rubinstein & Kroese 2004).

    Select top-30% elite, estimate mean+std, sample M children from
    N(mean, std). Diagonal covariance — complementary to T7 (CovEMNA).
    """
    M, D = subpop_coords.shape
    n_elite = max(1, int(0.3 * M))
    elite_idx = np.argsort(subpop_fitness)[:n_elite]
    elite = subpop_coords[elite_idx]

    mean = elite.mean(axis=0)
    # Floor at 1e-6 * span to prevent zero-std collapse when n_elite=1
    span = bounds[:, 1] - bounds[:, 0]
    std = np.maximum(elite.std(axis=0), 1e-6 * span)

    children = rng.normal(mean, std, size=(M, D))
    return np.clip(children, bounds[:, 0], bounds[:, 1]).astype(np.float64)


def _cauchy_mutation(subpop_coords, subpop_fitness, bounds, rng,
                     gen_frac=0.5, **kwargs):
    """
    Heavy-tailed Cauchy mutation for escaping local optima.

    Cauchy distribution: ~20% of perturbations exceed 3sigma (vs <0.3%
    for Gaussian). Provides large jumps that Gaussian-LS (T3) cannot.

    Scale: 5% of bounds span, decaying mildly with progress.
    """
    M, D = subpop_coords.shape
    span = bounds[:, 1] - bounds[:, 0]

    scale = 0.05 * (1.0 - 0.3 * gen_frac) * span  # 5% → 3.5% of span
    perturbation = scale * rng.standard_cauchy((M, D))

    children = subpop_coords + perturbation
    return np.clip(children, bounds[:, 0], bounds[:, 1]).astype(np.float64)


def _covariance_emna(subpop_coords, subpop_fitness, bounds, rng,
                     technique_state=None, **kwargs):
    """
    Full-covariance EMNA with momentum and shrinkage estimator.

    Replaces broken CMA-ES (which was ask() without tell()).
    Uses Ledoit-Wolf-style shrinkage for small-sample covariance estimation
    and exponential momentum to accumulate structure across generations.

    D<=200: full covariance (O(D^3) Cholesky, ~1ms at D=100).
    D>200: diagonal (equivalent to CEM but with momentum).
    """
    if technique_state is not None:
        return technique_state.cov.update_and_sample(
            subpop_coords, subpop_fitness, bounds, rng)
    else:
        # Stateless fallback: CEM with full covariance (single-shot)
        M, D = subpop_coords.shape
        n_elite = max(2, int(0.3 * M))
        elite_idx = np.argsort(subpop_fitness)[:n_elite]
        elite = subpop_coords[elite_idx]
        mean = elite.mean(axis=0)

        if D <= 200 and n_elite >= 3:
            cov = np.cov(elite.T)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            cov += 1e-6 * np.eye(D)
            try:
                children = rng.multivariate_normal(mean, cov, size=M)
            except np.linalg.LinAlgError:
                std = np.sqrt(np.abs(np.diag(cov)) + 1e-10)
                children = rng.normal(mean, std, size=(M, D))
        else:
            std = elite.std(axis=0) + 1e-10
            children = rng.normal(mean, std, size=(M, D))

        return np.clip(children, bounds[:, 0], bounds[:, 1]).astype(np.float64)


# ======================================================================
# Pairing helper (for SBX)
# ======================================================================

def _random_pairs_from_subpop(coords, fitness, rng):
    """
    Form random pairs within a subpopulation.

    If M is odd, last node pairs with a random other node.
    Returns p1, p2 arrays of shape (M, D) — one child per node.
    """
    M, D = coords.shape
    if M == 0:
        return (np.zeros((0, D), dtype=np.float64),
                np.zeros((0, D), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64))

    if M == 1:
        return coords.copy(), coords.copy(), fitness.copy(), fitness.copy()

    idx = np.arange(M)
    rng.shuffle(idx)

    p1 = np.empty((M, D), dtype=np.float64)
    p2 = np.empty((M, D), dtype=np.float64)
    p1_fit = np.empty(M, dtype=np.float64)
    p2_fit = np.empty(M, dtype=np.float64)

    for k in range(0, M - 1, 2):
        i, j = idx[k], idx[k + 1]
        p1[i] = coords[i];  p2[i] = coords[j]
        p1_fit[i] = fitness[i];  p2_fit[i] = fitness[j]
        p1[j] = coords[j];  p2[j] = coords[i]
        p1_fit[j] = fitness[j];  p2_fit[j] = fitness[i]

    if M % 2 == 1:
        last = idx[-1]
        partner = rng.integers(0, M - 1)
        if partner >= last:
            partner += 1
        partner = min(partner, M - 1)
        p1[last] = coords[last];  p2[last] = coords[partner]
        p1_fit[last] = fitness[last];  p2_fit[last] = fitness[partner]

    return p1, p2, p1_fit, p2_fit


# ======================================================================
# Main dispatch
# ======================================================================

def apply_technique_to_subpop(technique_id, subpop_coords, subpop_fitness,
                               all_survivor_coords, bounds, rng,
                               gen_frac=0.5, eval_fn=None,
                               ls1_refiner=None, subpop_indices=None,
                               technique_state=None):
    """
    Apply a single technique to its assigned subpopulation.

    Each technique produces exactly M children (one per assigned node).

    Args:
        technique_id: int (0-7)
        subpop_coords: (M, D) coordinates of nodes assigned to this technique
        subpop_fitness: (M,) fitness values
        all_survivor_coords: (N_surv, D) full survivor population (for DE)
        bounds: (D, 2) lower/upper bounds
        rng: numpy random generator
        gen_frac: float [0,1] search progress fraction
        eval_fn: callable(x)->float, objective function (for LS1)
        ls1_refiner: LS1Refiner instance (for technique 0)
        subpop_indices: (M,) int, original population indices (for LS1 state)
        technique_state: TechniqueState instance (optional, enables adaptation)

    Returns:
        children: (M, D) float64, clipped to bounds
        extra_evals: (M,) int, extra evaluations beyond the mandatory
                     post-dispatch evaluation (0 for techniques 1-7,
                     probing evals for technique 0 / MTS-LS1)
    """
    M, D = subpop_coords.shape
    if M == 0:
        return np.zeros((0, D), dtype=np.float64), np.zeros(0, dtype=np.int32)

    if technique_id == 0:
        # MTS-LS1: budget-limited coordinate-wise local search
        if ls1_refiner is not None and eval_fn is not None:
            children, children_fitness, probing_evals = ls1_refiner.refine_subpop(
                subpop_coords, subpop_fitness, subpop_indices, eval_fn, rng)
            return children, probing_evals
        else:
            return subpop_coords.copy(), np.zeros(M, dtype=np.int32)

    elif technique_id == 1:
        # SBX: random pairs within subpop
        p1, p2, _, _ = _random_pairs_from_subpop(
            subpop_coords, subpop_fitness, rng)
        children = sbx_crossover(p1, p2, bounds, rng=rng)

    elif technique_id == 2:
        # Mini-SHADE: adaptive DE with pbest mutation
        children = _shade_de(
            subpop_coords, subpop_fitness, all_survivor_coords,
            bounds, rng, gen_frac=gen_frac,
            technique_state=technique_state)

    elif technique_id == 3:
        # Gaussian-LS with 1/5 success rule
        children = _gaussian_ls_adaptive(
            subpop_coords, subpop_fitness, bounds, rng,
            gen_frac=gen_frac, technique_state=technique_state)

    elif technique_id == 4:
        # DE/current-to-best/1: aggressive drift toward best
        children = _de_current_to_best(
            subpop_coords, subpop_fitness, all_survivor_coords,
            bounds, rng, gen_frac=gen_frac)

    elif technique_id == 5:
        # CEM/EDA: sample from elite distribution
        children = _cem_sampling(subpop_coords, subpop_fitness, bounds, rng)

    elif technique_id == 6:
        # Cauchy mutation: heavy-tailed exploration
        children = _cauchy_mutation(
            subpop_coords, subpop_fitness, bounds, rng,
            gen_frac=gen_frac)

    elif technique_id == 7:
        # CovEMNA: full-covariance with momentum
        children = _covariance_emna(
            subpop_coords, subpop_fitness, bounds, rng,
            technique_state=technique_state)

    else:
        raise ValueError(f"Unknown technique_id: {technique_id}")

    return children, np.zeros(M, dtype=np.int32)


def apply_all_techniques_to_node(node_coords, node_fitness,
                                  all_survivor_coords, all_survivor_fitness,
                                  bounds, rng, gen_frac=0.5,
                                  eval_fn=None, ls1_refiner=None):
    """
    Apply ALL K techniques to a single node. For SSL data collection.

    Stateless: no TechniqueState used (oracle evaluation is single-shot).

    Args:
        node_coords: (D,) single node coordinates
        node_fitness: scalar fitness
        all_survivor_coords: (N_surv, D) full survivor coordinates
        all_survivor_fitness: (N_surv,) full survivor fitness
        bounds: (D, 2)
        rng: numpy RNG
        gen_frac: float
        eval_fn: callable(x)->float (needed for technique 0 / MTS-LS1)
        ls1_refiner: LS1Refiner instance (for technique 0)

    Returns:
        children: (K, D) one child per technique
    """
    D = len(node_coords)
    K = N_TECHNIQUES
    children = np.zeros((K, D), dtype=np.float64)

    N_surv = len(all_survivor_coords)
    node_2d = node_coords.reshape(1, D)
    node_fit_1d = np.array([node_fitness])

    # Build a small context subpop: node + random survivors
    n_ctx = min(5, N_surv)

    # 0: MTS-LS1
    if ls1_refiner is not None and eval_fn is not None:
        child, _child_fit, _evals = ls1_refiner.refine_subpop(
            node_2d, node_fit_1d, None, eval_fn, rng)
        children[0] = child[0]
    else:
        children[0] = node_coords.copy()

    # 1: SBX
    partner_idx = rng.integers(0, N_surv)
    partner_coords = all_survivor_coords[partner_idx:partner_idx + 1]
    children[1] = sbx_crossover(node_2d, partner_coords, bounds, rng=rng)[0]

    # 2: SHADE (stateless single-shot)
    children[2] = _shade_de(
        node_2d, node_fit_1d, all_survivor_coords,
        bounds, rng, gen_frac=gen_frac)[0]

    # 3: Gaussian-LS (stateless single-shot)
    children[3] = _gaussian_ls_adaptive(
        node_2d, node_fit_1d, bounds, rng, gen_frac=gen_frac)[0]

    # 4: DE/current-to-best/1 (node + context)
    ctx_idx = rng.choice(N_surv, n_ctx, replace=False)
    de_subpop = np.vstack([node_2d, all_survivor_coords[ctx_idx]])
    de_fit = np.concatenate([[node_fitness], all_survivor_fitness[ctx_idx]])
    children[4] = _de_current_to_best(
        de_subpop, de_fit, all_survivor_coords,
        bounds, rng, gen_frac=gen_frac)[0]

    # 5: CEM/EDA (node + context)
    ctx_idx2 = rng.choice(N_surv, n_ctx, replace=False)
    cem_subpop = np.vstack([node_2d, all_survivor_coords[ctx_idx2]])
    cem_fit = np.concatenate([[node_fitness], all_survivor_fitness[ctx_idx2]])
    children[5] = _cem_sampling(cem_subpop, cem_fit, bounds, rng)[0]

    # 6: Cauchy mutation
    children[6] = _cauchy_mutation(
        node_2d, node_fit_1d, bounds, rng, gen_frac=gen_frac)[0]

    # 7: CovEMNA (stateless single-shot, node + context)
    ctx_idx4 = rng.choice(N_surv, n_ctx, replace=False)
    cov_subpop = np.vstack([node_2d, all_survivor_coords[ctx_idx4]])
    cov_fit = np.concatenate([[node_fitness], all_survivor_fitness[ctx_idx4]])
    children[7] = _covariance_emna(cov_subpop, cov_fit, bounds, rng)[0]

    return children


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    D = 10
    bounds = np.column_stack([np.full(D, -100), np.full(D, 100)])
    rng = np.random.default_rng(42)

    def sphere(x):
        return float(np.sum(x ** 2))

    N_surv = 40
    all_coords = rng.uniform(-100, 100, (N_surv, D))
    all_fitness = np.array([sphere(all_coords[i]) for i in range(N_surv)])

    print("=" * 60)
    print("  techniques_v2.py — 8 SOTA operators smoke test")
    print("=" * 60)

    from ls1_refinement import LS1Refiner
    ls1_ref = LS1Refiner(bounds, max_evals_per_individual=10)

    # Create persistent state
    state = TechniqueState(D=D)

    # --- Test 1: Subpopulation dispatch (with state) ---
    print("\n=== Subpopulation dispatch (stateful) ===")
    for tid, name in enumerate(TECHNIQUE_NAMES):
        M = int(rng.integers(8, 16))
        idx = rng.choice(N_surv, M, replace=False)
        subpop = all_coords[idx]
        subfit = all_fitness[idx]

        children, extra_evals = apply_technique_to_subpop(
            tid, subpop, subfit, all_coords, bounds, rng,
            gen_frac=0.3, eval_fn=sphere,
            ls1_refiner=ls1_ref, subpop_indices=idx,
            technique_state=state)
        assert children.shape == (M, D), f"{name}: wrong shape {children.shape}"
        assert extra_evals.shape == (M,), f"{name}: wrong evals shape"
        assert children.dtype == np.float64, f"{name}: wrong dtype"
        assert np.all(children >= -100) and np.all(children <= 100), \
            f"{name}: out of bounds"
        ev_str = f"extra_evals={int(extra_evals.sum()):>4}" if tid == 0 else ""
        print(f"  [{tid}] {name:<14}: M={M:>2} -> children {children.shape}, "
              f"mean={children.mean():>8.2f}  {ev_str}")

    # --- Test 2: SHADE memory update (deferred reporting) ---
    print("\n=== SHADE deferred reporting ===")
    M = 10
    idx = rng.choice(N_surv, M, replace=False)
    subpop = all_coords[idx]
    subfit = all_fitness[idx]

    children, _ = apply_technique_to_subpop(
        2, subpop, subfit, all_coords, bounds, rng,
        gen_frac=0.5, technique_state=state)
    # Simulate evaluation
    child_fitness = np.array([sphere(c) for c in children])
    # Report success
    assignments = np.full(M, 2, dtype=np.int32)
    state.report_children(assignments, subfit, child_fitness)
    print(f"  M_F[:3] = {state.shade.M_F[:3]}")
    print(f"  M_CR[:3] = {state.shade.M_CR[:3]}")
    print(f"  Memory updated: k={state.shade.k}")

    # --- Test 3: Edge cases ---
    print("\n=== Edge cases (M=0,1,2,3) ===")
    for M in [0, 1, 2, 3]:
        subpop = all_coords[:M]
        subfit = all_fitness[:M]
        for tid in range(N_TECHNIQUES):
            c, ev = apply_technique_to_subpop(
                tid, subpop, subfit, all_coords, bounds, rng,
                eval_fn=sphere, ls1_refiner=ls1_ref,
                subpop_indices=np.arange(M),
                technique_state=state)
            assert c.shape == (M, D), f"M={M} tid={tid}: shape {c.shape}"
            assert ev.shape == (M,), f"M={M} tid={tid}: evals shape {ev.shape}"
        print(f"  M={M}: all {N_TECHNIQUES} techniques OK")

    # --- Test 4: Stateless dispatch (no state) ---
    print("\n=== Stateless dispatch (backward compat) ===")
    for tid, name in enumerate(TECHNIQUE_NAMES):
        M = 8
        idx = rng.choice(N_surv, M, replace=False)
        children, ev = apply_technique_to_subpop(
            tid, all_coords[idx], all_fitness[idx], all_coords, bounds, rng,
            eval_fn=sphere, ls1_refiner=ls1_ref,
            subpop_indices=idx, technique_state=None)
        assert children.shape == (M, D)
        print(f"  [{tid}] {name:<14}: OK (no state)")

    # --- Test 5: SSL - all techniques on single node ---
    print("\n=== All techniques on single node (SSL) ===")
    node = all_coords[0]
    node_fit = all_fitness[0]
    children = apply_all_techniques_to_node(
        node, node_fit, all_coords, all_fitness, bounds, rng)
    assert children.shape == (N_TECHNIQUES, D)
    for tid, name in enumerate(TECHNIQUE_NAMES):
        child_fit = sphere(children[tid])
        improved = "IMPROVED" if child_fit < node_fit else ""
        print(f"    [{tid}] {name:<14}: fitness={child_fit:>10.1f}  {improved}")

    print("\n" + "=" * 60)
    print("  All smoke tests passed!")
    print("=" * 60)
