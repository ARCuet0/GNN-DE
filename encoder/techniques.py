"""
techniques.py — Multiple Offspring Sampling (MOS) reproductive techniques.

Four fundamentally different search operators for continuous optimization:
  0. BLX-alpha crossover + Gaussian mutation (existing GA mechanism)
  1. DE/rand/1 with binomial crossover (Differential Evolution)
  2. SBX + polynomial mutation (Simulated Binary Crossover)
  3. Gaussian local search (exploitation around better parent)

Each technique operates on pairs of parents and produces one child per pair.
All return float64 arrays clipped to bounds (required for SOCO10 ctypes).
"""
import numpy as np

N_TECHNIQUES = 4
TECHNIQUE_NAMES = ["BLX-alpha", "DE/rand/1", "SBX", "Gaussian-LS"]

# Default parameters per technique (tuned for SOCO benchmarks)
TECHNIQUE_DEFAULTS = {
    0: {"b_alpha": 1.67, "b_beta": 2.23, "margin": 0.11,
        "mut_prob": 0.09, "mut_vigor": 0.18},
    1: {"F": 0.8, "CR": 0.9},
    2: {"eta_c": 20.0, "mut_prob": 0.1, "eta_m": 20.0},
    3: {"sigma_frac": 0.05},
}


def blx_alpha(p1, p2, bounds, rng=None, **kwargs):
    """
    BLX-alpha crossover with Gaussian mutation.

    child = p1 + beta * (p2 - p1)  where beta ~ Beta(b_alpha, b_beta)
    then add Gaussian noise with probability mut_prob.

    Args:
        p1: (P, D) parent 1 coordinates
        p2: (P, D) parent 2 coordinates
        bounds: (D, 2) lower/upper bounds
        rng: numpy random generator
    Returns:
        children: (P, D) float64, clipped to bounds
    """
    if rng is None:
        rng = np.random.default_rng()
    cfg = {**TECHNIQUE_DEFAULTS[0], **kwargs}

    P, D = p1.shape
    bounds_span = bounds[:, 1] - bounds[:, 0]

    # Per-pair beta sampling
    beta = np.array([
        rng.beta(cfg["b_alpha"], cfg["b_beta"], size=D) for _ in range(P)
    ])
    margin = cfg["margin"] * bounds_span  # (D,)
    interpolation = -margin + (1 + 2 * margin) * beta  # (P, D)
    children = p1 + interpolation * (p2 - p1)

    # Gaussian mutation
    mut_mask = rng.binomial(1, cfg["mut_prob"], size=(P, D))
    noise = cfg["mut_vigor"] * bounds_span * rng.standard_normal((P, D))
    children += mut_mask * noise

    return np.clip(children, bounds[:, 0], bounds[:, 1]).astype(np.float64)


def de_rand_1(p1, p2, bounds, rng=None, population=None, **kwargs):
    """
    DE/rand/1 with binomial crossover.

    v = x_base + F * (p1 - p2)   where x_base is random from population
    child = binomial_crossover(p1, v, CR)

    If population is not available, uses p1 as base (becomes DE/current/1).

    Args:
        p1: (P, D) parent 1 coordinates (target vector)
        p2: (P, D) parent 2 coordinates
        bounds: (D, 2) lower/upper bounds
        rng: numpy random generator
        population: (S, D) full survivor coordinates for base selection
    Returns:
        children: (P, D) float64, clipped to bounds
    """
    if rng is None:
        rng = np.random.default_rng()
    cfg = {**TECHNIQUE_DEFAULTS[1], **kwargs}

    P, D = p1.shape
    F = cfg["F"]
    CR = cfg["CR"]

    # Base vector: random from population (or p1 fallback)
    if population is not None and len(population) > 2:
        base_idx = rng.integers(0, len(population), size=P)
        x_base = population[base_idx]
    else:
        x_base = p1.copy()

    # Mutation: v = x_base + F * (p1 - p2)
    v = x_base + F * (p1 - p2)

    # Binomial crossover
    cross_mask = rng.random((P, D)) < CR
    # Ensure at least one dimension is from mutant
    j_rand = rng.integers(0, D, size=P)
    for i in range(P):
        cross_mask[i, j_rand[i]] = True

    children = np.where(cross_mask, v, p1)

    return np.clip(children, bounds[:, 0], bounds[:, 1]).astype(np.float64)


def sbx_crossover(p1, p2, bounds, rng=None, **kwargs):
    """
    Simulated Binary Crossover (SBX) + polynomial mutation.

    Deb & Agrawal (1995). Standard in NSGA-II.
    Produces children that maintain the spread of parents.

    Args:
        p1: (P, D) parent 1 coordinates
        p2: (P, D) parent 2 coordinates
        bounds: (D, 2) lower/upper bounds
        rng: numpy random generator
    Returns:
        children: (P, D) float64, clipped to bounds
    """
    if rng is None:
        rng = np.random.default_rng()
    cfg = {**TECHNIQUE_DEFAULTS[2], **kwargs}

    P, D = p1.shape
    eta_c = cfg["eta_c"]
    eta_m = cfg["eta_m"]
    mut_prob = cfg["mut_prob"]
    bounds_lo = bounds[:, 0]
    bounds_hi = bounds[:, 1]

    # SBX crossover
    u = rng.random((P, D))
    beta_q = np.where(
        u <= 0.5,
        (2.0 * u) ** (1.0 / (eta_c + 1.0)),
        (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)),
    )

    # Randomly choose child 1 or child 2 per dimension
    sign = rng.choice([-1, 1], size=(P, D))
    children = 0.5 * ((p1 + p2) + sign * beta_q * (p1 - p2))

    # Polynomial mutation
    mut_mask = rng.random((P, D)) < mut_prob
    delta = rng.random((P, D))
    delta_q = np.where(
        delta < 0.5,
        (2.0 * delta) ** (1.0 / (eta_m + 1.0)) - 1.0,
        1.0 - (2.0 * (1.0 - delta)) ** (1.0 / (eta_m + 1.0)),
    )
    perturbation = delta_q * (bounds_hi - bounds_lo)
    children = np.where(mut_mask, children + perturbation, children)

    return np.clip(children, bounds_lo, bounds_hi).astype(np.float64)


def gaussian_local_search(p1, p2, bounds, rng=None, **kwargs):
    """
    Gaussian local search: perturb the better parent.

    Ignores the worse parent. Pure exploitation step.
    sigma decreases with generation (adaptive if gen_frac provided).

    Args:
        p1: (P, D) parent 1 coordinates
        p2: (P, D) parent 2 coordinates
        bounds: (D, 2) lower/upper bounds
        rng: numpy random generator
        p1_fitness: (P,) fitness of parent 1 (optional)
        p2_fitness: (P,) fitness of parent 2 (optional)
        gen_frac: float in [0,1], fraction of search completed (optional)
    Returns:
        children: (P, D) float64, clipped to bounds
    """
    if rng is None:
        rng = np.random.default_rng()
    cfg = {**TECHNIQUE_DEFAULTS[3], **kwargs}

    P, D = p1.shape
    sigma_frac = cfg["sigma_frac"]
    bounds_span = bounds[:, 1] - bounds[:, 0]

    # Select better parent (lower fitness = better in minimization)
    p1_fit = kwargs.get("p1_fitness", None)
    p2_fit = kwargs.get("p2_fitness", None)
    if p1_fit is not None and p2_fit is not None:
        p1_fit = np.asarray(p1_fit)
        p2_fit = np.asarray(p2_fit)
        use_p1 = (p1_fit <= p2_fit)[:, np.newaxis]  # (P, 1)
        base = np.where(use_p1, p1, p2)
    else:
        # Default: use p1 (typically the fitter parent from pairing)
        base = p1.copy()

    # Adaptive sigma: decrease with search progress
    gen_frac = kwargs.get("gen_frac", 0.5)
    adaptive_sigma = sigma_frac * (1.0 - 0.7 * gen_frac)  # shrink over time

    noise = adaptive_sigma * bounds_span * rng.standard_normal((P, D))
    children = base + noise

    return np.clip(children, bounds[:, 0], bounds[:, 1]).astype(np.float64)


# Dispatch table: technique_id → function
TECHNIQUE_FNS = [blx_alpha, de_rand_1, sbx_crossover, gaussian_local_search]


def apply_techniques(p1_coords, p2_coords, technique_ids, bounds,
                     population=None, rng=None, **kwargs):
    """
    Apply selected techniques to each pair.

    Args:
        p1_coords: (P, D) parent 1 coordinates
        p2_coords: (P, D) parent 2 coordinates
        technique_ids: (P,) int array — technique index per pair
        bounds: (D, 2) lower/upper bounds
        population: (S, D) all survivor coordinates (for DE base selection)
        rng: numpy random generator
        **kwargs: extra args passed to techniques (p1_fitness, p2_fitness, gen_frac)

    Returns:
        children: (P, D) float64, clipped to bounds
        technique_ids_used: (P,) int array — same as input (for logging)
    """
    if rng is None:
        rng = np.random.default_rng()

    P, D = p1_coords.shape
    children = np.zeros((P, D), dtype=np.float64)
    technique_ids = np.asarray(technique_ids, dtype=np.int32)

    # Group pairs by technique and apply in batch
    for tid in range(N_TECHNIQUES):
        mask = technique_ids == tid
        if not mask.any():
            continue
        p1_t = p1_coords[mask]
        p2_t = p2_coords[mask]

        tech_kwargs = dict(kwargs)
        # Pass population for DE
        if tid == 1 and population is not None:
            tech_kwargs["population"] = population

        # Pass per-pair fitness for local search
        if tid == 3:
            p1_fit = kwargs.get("p1_fitness")
            p2_fit = kwargs.get("p2_fitness")
            if p1_fit is not None:
                tech_kwargs["p1_fitness"] = np.asarray(p1_fit)[mask]
            if p2_fit is not None:
                tech_kwargs["p2_fitness"] = np.asarray(p2_fit)[mask]

        children[mask] = TECHNIQUE_FNS[tid](p1_t, p2_t, bounds, rng=rng,
                                            **tech_kwargs)

    return children, technique_ids.copy()


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    D = 10
    P = 20
    bounds = np.column_stack([np.full(D, -100), np.full(D, 100)])
    rng = np.random.default_rng(42)

    p1 = rng.uniform(-50, 50, (P, D))
    p2 = rng.uniform(-50, 50, (P, D))
    population = rng.uniform(-100, 100, (50, D))

    for tid, name in enumerate(TECHNIQUE_NAMES):
        fn = TECHNIQUE_FNS[tid]
        kw = {"population": population} if tid == 1 else {}
        if tid == 3:
            kw["p1_fitness"] = rng.random(P)
            kw["p2_fitness"] = rng.random(P)
        c = fn(p1, p2, bounds, rng=rng, **kw)
        assert c.shape == (P, D), f"{name}: wrong shape {c.shape}"
        assert c.dtype == np.float64, f"{name}: wrong dtype {c.dtype}"
        assert np.all(c >= bounds[:, 0]) and np.all(c <= bounds[:, 1]), \
            f"{name}: out of bounds"
        print(f"  {name}: OK, mean={c.mean():.2f}, std={c.std():.2f}")

    # Test dispatch
    tech_ids = rng.integers(0, N_TECHNIQUES, size=P)
    children, ids_used = apply_techniques(
        p1, p2, tech_ids, bounds, population=population, rng=rng,
        p1_fitness=rng.random(P), p2_fitness=rng.random(P), gen_frac=0.3,
    )
    print(f"\n  Dispatch: OK, shape={children.shape}, "
          f"techniques used: {np.bincount(ids_used, minlength=N_TECHNIQUES)}")
