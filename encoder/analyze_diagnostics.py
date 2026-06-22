"""
analyze_diagnostics.py — Block 0 diagnostic analysis functions.

All functions operate on numpy arrays loaded from diagnostic .npz files.
Pure numpy + sklearn — no torch dependency.

Block 0.1: Entropy-autocorrelation map
Block 0.2: Markov transition matrix
Block 0.3: NoOp analysis
Block 0.4: BPTT boundary effect
Block 0.5: Hidden state memory (budget phase classifier)
"""
import numpy as np
from typing import Dict


# ======================================================================
# 0.1 — Entropy-autocorrelation map
# ======================================================================

def per_individual_entropy(routing_probs: np.ndarray) -> np.ndarray:
    """Shannon entropy per individual per generation.

    Args:
        routing_probs: (n_gens, N, K) softmax probabilities

    Returns:
        entropy: (n_gens, N) Shannon entropy in nats
    """
    rp = np.clip(routing_probs, 1e-8, 1.0)
    return -(rp * np.log(rp)).sum(axis=-1)


def expert_autocorrelation(chosen_expert: np.ndarray, max_lag: int = 5) -> np.ndarray:
    """Autocorrelation of expert choice per individual over time.

    Uses Pearson autocorrelation of the integer expert sequence.
    For constant sequences (std=0), returns 0.

    Args:
        chosen_expert: (n_gens, N) integer expert indices
        max_lag: number of lag steps to compute

    Returns:
        autocorr: (N, max_lag) autocorrelation at lags 1..max_lag
    """
    n_gens, N = chosen_expert.shape
    result = np.zeros((N, max_lag), dtype=np.float64)

    for n in range(N):
        seq = chosen_expert[:, n].astype(np.float64)
        mu = seq.mean()
        std = seq.std()
        if std < 1e-10:
            # Constant sequence — no meaningful autocorrelation
            continue
        centered = seq - mu
        var = (centered ** 2).mean()
        for lag in range(1, max_lag + 1):
            if lag >= n_gens:
                break
            cov = (centered[:n_gens - lag] * centered[lag:]).mean()
            result[n, lag - 1] = cov / var

    return result


# ======================================================================
# 0.2 — Markov transition matrix
# ======================================================================

def markov_transition_matrix(chosen_expert: np.ndarray, K: int) -> np.ndarray:
    """Build K×K transition matrix from expert choices.

    Counts transitions across all individuals and generations.

    Args:
        chosen_expert: (n_gens, N) integer expert indices
        K: number of experts

    Returns:
        T: (K, K) row-stochastic transition matrix
    """
    counts = np.zeros((K, K), dtype=np.float64)

    # Vectorized: all transitions at once
    src = chosen_expert[:-1].ravel()  # (n_gens-1)*N
    dst = chosen_expert[1:].ravel()
    np.add.at(counts, (src, dst), 1)

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    return counts / row_sums


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Compute stationary distribution of transition matrix.

    Finds left eigenvector corresponding to eigenvalue 1.

    Args:
        T: (K, K) row-stochastic transition matrix

    Returns:
        pi: (K,) stationary distribution
    """
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    # Find eigenvector for eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)  # ensure non-negative
    pi /= pi.sum()
    return pi


# ======================================================================
# 0.3 — NoOp analysis
# ======================================================================

def nop_correlation(
    chosen_expert: np.ndarray,
    fitness_rank: np.ndarray,
    dist_to_best: np.ndarray,
    dist_to_nearest: np.ndarray,
    nop_idx: int = 3,
) -> Dict[str, float]:
    """Correlate NoOp assignment with population features.

    Args:
        chosen_expert: (n_gens, N)
        fitness_rank: (n_gens, N) ordinal rank (0=best)
        dist_to_best: (n_gens, N)
        dist_to_nearest: (n_gens, N)
        nop_idx: expert index for NoOp operator

    Returns:
        dict with point-biserial correlations:
          fitness_rank_corr: correlation(is_nop, fitness_rank)
          dist_to_best_corr: correlation(is_nop, dist_to_best)
          dist_to_nearest_corr: correlation(is_nop, dist_to_nearest)
          nop_fraction: overall fraction of nop assignments
    """
    is_nop = (chosen_expert == nop_idx).astype(np.float64).ravel()
    nop_frac = is_nop.mean()

    def _corr(a, b):
        a, b = a.ravel(), b.ravel()
        if a.std() < 1e-10 or b.std() < 1e-10:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return {
        'fitness_rank_corr': _corr(is_nop, fitness_rank.ravel()),
        'dist_to_best_corr': _corr(is_nop, dist_to_best.ravel()),
        'dist_to_nearest_corr': _corr(is_nop, dist_to_nearest.ravel()),
        'nop_fraction': float(nop_frac),
    }


# ======================================================================
# 0.4 — BPTT boundary effect
# ======================================================================

def bptt_boundary_effect(
    routing_probs: np.ndarray,
    bptt_segment_pos: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Mean entropy and routing concentration as function of BPTT segment position.

    Args:
        routing_probs: (n_gens, N, K)
        bptt_segment_pos: (n_gens,) position within segment (-1 = outside)

    Returns:
        dict with:
          position: (P,) unique positions
          mean_entropy: (P,) mean Shannon entropy at each position
          std_entropy: (P,) std of entropy at each position
    """
    entropy = per_individual_entropy(routing_probs)  # (n_gens, N)

    # Only consider generations inside BPTT
    mask = bptt_segment_pos >= 0
    positions = bptt_segment_pos[mask]

    unique_pos = np.sort(np.unique(positions))
    mean_ent = np.zeros(len(unique_pos))
    std_ent = np.zeros(len(unique_pos))

    for i, pos in enumerate(unique_pos):
        gen_mask = (bptt_segment_pos == pos)
        ent_at_pos = entropy[gen_mask]  # (n_matching_gens, N)
        mean_ent[i] = ent_at_pos.mean()
        std_ent[i] = ent_at_pos.std()

    return {
        'position': unique_pos,
        'mean_entropy': mean_ent,
        'std_entropy': std_ent,
    }


# ======================================================================
# 0.5 — Hidden state memory (budget phase classifier)
# ======================================================================

def budget_phase_classifier(
    fitness: np.ndarray,
    n_bins: int = 4,
) -> Dict[str, float]:
    """Train linear classifier to predict budget phase from population features.

    Uses instantaneous population statistics (mean, std, min, max, median
    fitness) to predict which fraction of the budget has been consumed.
    If accuracy >> chance, the population state encodes temporal information.

    Args:
        fitness: (n_gens, N) fitness values
        n_bins: number of budget phases (uniform split)

    Returns:
        dict with accuracy, chance_level, per_class_accuracy
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    n_gens, N = fitness.shape

    # Features: population-level statistics per generation
    features = np.column_stack([
        fitness.mean(axis=1),
        fitness.std(axis=1),
        fitness.min(axis=1),
        fitness.max(axis=1),
        np.median(fitness, axis=1),
        np.percentile(fitness, 25, axis=1),
        np.percentile(fitness, 75, axis=1),
    ])  # (n_gens, 7)

    # Labels: budget phase
    labels = np.minimum(
        (np.arange(n_gens) * n_bins) // n_gens,
        n_bins - 1
    )

    # Standardize
    mu = features.mean(axis=0)
    std = features.std(axis=0).clip(min=1e-8)
    features = (features - mu) / std

    clf = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(clf, features, labels, cv=min(5, n_bins), scoring='accuracy')

    return {
        'accuracy': float(scores.mean()),
        'accuracy_std': float(scores.std()),
        'chance_level': 1.0 / n_bins,
    }
