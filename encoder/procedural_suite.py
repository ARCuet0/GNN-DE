"""ProceduralSuite — infinite procedural function generator for zero-shot L2O training.

Generates optimization landscapes from ~20 atom functions composed via 4 structural
patterns (unimodal, multimodal, hybrid, composition). Covers every CEC2017 landscape
pattern without instantiating any CEC2017 function.

Design: all transforms (rotation, shift, anisotropy) are on the INPUT coordinates.
Only a constant bias offset is added to the OUTPUT. This makes f_optimal exact.

Usage:
    suite = ProceduralSuite()
    fn = suite.sample(D=10, device='cuda', category='hybrid')
    y = fn(x)  # (N, D) -> (N,)
    print(fn.f_optimal)  # known optimum (scalar, exact)
"""
import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor

from .cec2017_torch import (
    sphere, elliptic, bent_cigar, discus, zakharov, rosenbrock,
    rastrigin, schaffer_f7, expanded_schaffer_f6, lunacek_bi_rastrigin,
    non_continuous_rastrigin, levy, modified_schwefel, ackley, griewank,
    katsuura, weierstrass, happy_cat, hgbat, grie_rosen_cec,
)


# ── Atom Registry ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AtomSpec:
    fn: Callable       # (B, D) -> (B,)
    kwargs: dict       # shift kwargs so minimum is at origin
    rotatable: bool    # False only for modified_schwefel
    name: str


UNIMODAL_ATOMS = [
    AtomSpec(sphere, {}, True, 'sphere'),
    AtomSpec(elliptic, {}, True, 'elliptic'),
    AtomSpec(bent_cigar, {}, True, 'bent_cigar'),
    AtomSpec(discus, {}, True, 'discus'),
    AtomSpec(zakharov, {}, True, 'zakharov'),
    AtomSpec(rosenbrock, {'shift': 1.0}, True, 'rosenbrock'),
]

MULTIMODAL_ATOMS = [
    AtomSpec(rastrigin, {}, True, 'rastrigin'),
    AtomSpec(schaffer_f7, {}, True, 'schaffer_f7'),
    AtomSpec(expanded_schaffer_f6, {}, True, 'expanded_schaffer_f6'),
    AtomSpec(lunacek_bi_rastrigin, {'shift': 2.5}, True, 'lunacek_bi_rastrigin'),
    AtomSpec(non_continuous_rastrigin, {}, True, 'non_continuous_rastrigin'),
    AtomSpec(levy, {'shift': 1.0}, True, 'levy'),
    AtomSpec(modified_schwefel, {}, False, 'modified_schwefel'),
    AtomSpec(ackley, {}, True, 'ackley'),
    AtomSpec(griewank, {}, True, 'griewank'),
    AtomSpec(katsuura, {}, True, 'katsuura'),
    AtomSpec(weierstrass, {}, True, 'weierstrass'),
    AtomSpec(happy_cat, {'shift': -1.0}, True, 'happy_cat'),
    AtomSpec(hgbat, {'shift': -1.0}, True, 'hgbat'),
    AtomSpec(grie_rosen_cec, {}, True, 'grie_rosen_cec'),
]

ALL_ATOMS = UNIMODAL_ATOMS + MULTIMODAL_ATOMS

# Atoms safe for hybrid partitions (can receive D=1 slices).
# Excluded atoms need D>=2 for pairs, rolls, or math domain constraints.
_HYBRID_UNSAFE = {'non_continuous_rastrigin', 'lunacek_bi_rastrigin',
                  'schaffer_f7', 'expanded_schaffer_f6', 'grie_rosen_cec'}
_HYBRID_SAFE_ATOMS = [a for a in ALL_ATOMS if a.name not in _HYBRID_UNSAFE]


# ── Shared Helpers ─────────────────────────────────────────────────────────

def _random_rotation(D: int, kappa_range: tuple, rng: torch.Generator,
                     device) -> Tensor:
    """SVD-conditioned rotation matrix."""
    raw = torch.randn(D, D, generator=rng)
    U, _, Vt = torch.linalg.svd(raw, full_matrices=False)

    kappa_lo, kappa_hi = kappa_range
    if kappa_hi <= kappa_lo:
        log_kappa = math.log10(kappa_lo)
    else:
        log_kappa = (torch.rand(1, generator=rng).item()
                     * math.log10(kappa_hi / kappa_lo)
                     + math.log10(kappa_lo))
    S_new = torch.logspace(0, log_kappa, D, dtype=torch.float64)
    S_new = S_new / S_new.prod().pow(1.0 / D)

    Q = U.to(torch.float64) @ torch.diag(S_new) @ Vt.to(torch.float64)
    return Q.to(device=device)


def _random_shift(D: int, Q: Tensor, rng: torch.Generator, device,
                  opt_in_input_space: Optional[Tensor] = None) -> Tensor:
    """Random shift so optimum stays in [-100, 100]^D."""
    if opt_in_input_space is None:
        opt_in_input_space = torch.zeros(D, dtype=torch.float64, device=device)
    rotated_opt = opt_in_input_space @ Q
    s_lo = -100.0 - rotated_opt
    s_hi = 100.0 - rotated_opt
    u = torch.rand(D, generator=rng).to(dtype=torch.float64, device=device)
    return s_lo + u * (s_hi - s_lo)


def _random_bias(rng: torch.Generator) -> float:
    """Random fitness offset in [0, 3000], matching CEC2017's range."""
    return torch.rand(1, generator=rng).item() * 3000.0


def _random_center(D: int, rng: torch.Generator, device) -> Tensor:
    """Random center in [-80, 80]^D for composition components."""
    return (torch.rand(D, generator=rng).to(dtype=torch.float64, device=device)
            * 160.0 - 80.0)


def _choose(items: list, rng: torch.Generator):
    """Random choice from list using rng."""
    idx = torch.randint(len(items), (1,), generator=rng).item()
    return items[idx]


def _compute_blend_weights(x: Tensor, centers_t: Tensor, sigmas_t: Tensor,
                           blend_type: int, D: int) -> Tensor:
    """Compute blending weights for composition functions.

    Returns (N, k) normalized weights.
    """
    dx_sq = ((x.unsqueeze(1) - centers_t.unsqueeze(0)) ** 2).sum(dim=2)

    if blend_type == BLEND_GAUSSIAN_RBF:
        w = torch.exp(-dx_sq / (2.0 * D * sigmas_t ** 2))
    elif blend_type == BLEND_INVERSE_DISTANCE:
        w = 1.0 / (torch.sqrt(dx_sq.clamp(min=1e-30)) + 1e-10)
    else:
        tau = float(D) * 100.0
        return torch.softmax(-dx_sq / tau, dim=1)

    w_sum = w.sum(dim=1, keepdim=True).clamp(min=1e-30)
    return w / w_sum


def _safe_partitions(k: int, D: int, perm: Tensor,
                     rng: torch.Generator) -> list:
    """Partition D dimensions into k groups, each with at least 1 dimension."""
    if k >= D:
        return [perm[i:i + 1] for i in range(D)]

    breakpoints = torch.randperm(D - 1, generator=rng)[:k - 1].sort().values + 1
    breakpoints = breakpoints.tolist()

    partitions = []
    prev = 0
    for bp in breakpoints:
        partitions.append(perm[prev:bp])
        prev = bp
    partitions.append(perm[prev:])
    return partitions


# ── Base Class ─────────────────────────────────────────────────────────────

class ProceduralFunction:
    """Base class for procedurally generated optimization landscapes."""

    __slots__ = ('f_optimal', 'D', 'device', 'category')

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ── Unimodal / Multimodal ─────────────────────────────────────────────────

class UnimodalFunction(ProceduralFunction):
    """Single atom + SVD rotation + shift. No output scaling."""

    __slots__ = ('atom', 'Q', 's', 'f_bias', 'atom_name')

    def __init__(self, atom: AtomSpec, Q: Tensor, s: Tensor, f_bias: float,
                 D: int, device):
        self.atom = atom
        self.Q = Q
        self.s = s
        self.f_bias = f_bias
        self.atom_name = atom.name
        self.D = D
        self.device = device
        self.category = 'unimodal'
        # Exact f_optimal: evaluate atom at z=0 (the optimum in transformed space)
        with torch.no_grad():
            _z = torch.zeros(1, D, dtype=torch.float64, device=device)
            base = float(atom.fn(_z, **atom.kwargs).item())
        self.f_optimal = base + f_bias

    def __call__(self, x: Tensor) -> Tensor:
        z = (x - self.s) @ self.Q.T
        return self.atom.fn(z, **self.atom.kwargs) + self.f_bias


class MultimodalFunction(UnimodalFunction):
    """Multimodal atom + augmentation."""

    def __init__(self, atom: AtomSpec, Q: Tensor, s: Tensor, f_bias: float,
                 D: int, device):
        super().__init__(atom, Q, s, f_bias, D, device)
        self.category = 'multimodal'


# ── Hybrid ─────────────────────────────────────────────────────────────────

class HybridFunction(ProceduralFunction):
    """Random dimension partition with different atoms per group."""

    __slots__ = ('atoms', 'Q', 's', 'f_bias', 'perm', 'partitions')

    def __init__(self, atoms: list, Q: Tensor, s: Tensor,
                 perm: Tensor, partitions: list, f_bias: float,
                 D: int, device):
        self.atoms = atoms
        self.Q = Q
        self.s = s
        self.f_bias = f_bias
        self.perm = perm
        self.partitions = partitions
        self.D = D
        self.device = device
        self.category = 'hybrid'
        # Exact f_optimal: sum of atom values at z=0 per partition
        with torch.no_grad():
            _z = torch.zeros(1, D, dtype=torch.float64, device=device)
            _z_perm = _z[:, perm]
            base = sum(float(a.fn(_z_perm[:, p], **a.kwargs).item())
                       for a, p in zip(atoms, partitions))
        self.f_optimal = base + f_bias

    def __call__(self, x: Tensor) -> Tensor:
        z = (x - self.s) @ self.Q.T
        z_perm = z[:, self.perm]
        result = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for atom, part_idx in zip(self.atoms, self.partitions):
            result = result + atom.fn(z_perm[:, part_idx], **atom.kwargs)
        return result + self.f_bias


# ── Composition ────────────────────────────────────────────────────────────

BLEND_GAUSSIAN_RBF = 0
BLEND_INVERSE_DISTANCE = 1
BLEND_SOFTMAX_DISTANCE = 2


class CompositionFunction(ProceduralFunction):
    """Weighted sum of k atoms at different centers."""

    __slots__ = ('atoms', 'Q_list', 'centers_t', 'sigmas_t', 'lambdas_',
                 'biases', 'blend_type', 'k', 'f_bias')

    def __init__(self, atoms: list, Q_list: list, centers: list,
                 sigmas: list, lambdas_: list, biases: list,
                 blend_type: int, f_bias: float, D: int, device):
        self.atoms = atoms
        self.Q_list = Q_list
        self.centers_t = torch.stack(centers)
        self.sigmas_t = torch.tensor(sigmas, dtype=torch.float64, device=device)
        self.lambdas_ = lambdas_
        self.biases = biases
        self.blend_type = blend_type
        self.f_bias = f_bias
        self.k = len(atoms)
        self.D = D
        self.device = device
        self.category = 'composition'
        # f_optimal: evaluate at ALL centers, take minimum
        with torch.no_grad():
            min_f = float('inf')
            for center in centers:
                val = float(self._eval_base(center.unsqueeze(0)).item())
                min_f = min(min_f, val)
        self.f_optimal = min_f + f_bias

    def _eval_base(self, x: Tensor) -> Tensor:
        """Evaluate without f_bias."""
        w = _compute_blend_weights(x, self.centers_t, self.sigmas_t,
                                   self.blend_type, self.D)
        result = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for i in range(self.k):
            z = (x - self.centers_t[i]) @ self.Q_list[i].T
            g = self.lambdas_[i] * self.atoms[i].fn(z, **self.atoms[i].kwargs)
            result = result + w[:, i] * (g + self.biases[i])
        return result

    def __call__(self, x: Tensor) -> Tensor:
        return self._eval_base(x) + self.f_bias


# ── Component Composition (shared base) ───────────────────────────────────

class _ComponentComposition(ProceduralFunction):
    """Base for compositions that blend callable sub-components at different centers.

    Subclasses only need to set `self.components`, `self.category`, then call
    `super().__init__(...)`.  Each component must have `.s` and `.f_bias`.
    """

    __slots__ = ('components', 'centers_t', 'sigmas_t', 'lambdas_',
                 'biases', 'blend_type', 'k', 'f_bias')

    def __init__(self, components: list, centers: list,
                 sigmas: list, lambdas_: list, biases: list,
                 blend_type: int, f_bias: float, D: int, device,
                 category: str):
        self.components = components
        self.centers_t = torch.stack(centers)
        self.sigmas_t = torch.tensor(sigmas, dtype=torch.float64, device=device)
        self.lambdas_ = lambdas_
        self.biases = biases
        self.blend_type = blend_type
        self.f_bias = f_bias
        self.k = len(components)
        self.D = D
        self.device = device
        self.category = category
        with torch.no_grad():
            min_f = float('inf')
            for center in centers:
                val = float(self._eval_base(center.unsqueeze(0)).item())
                min_f = min(min_f, val)
        self.f_optimal = min_f + f_bias

    def _eval_base(self, x: Tensor) -> Tensor:
        w = _compute_blend_weights(x, self.centers_t, self.sigmas_t,
                                   self.blend_type, self.D)
        result = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        for i in range(self.k):
            comp = self.components[i]
            x_local = (x - self.centers_t[i]) + comp.s
            g = self.lambdas_[i] * comp(x_local)
            result = result + w[:, i] * (g - comp.f_bias + self.biases[i])
        return result

    def __call__(self, x: Tensor) -> Tensor:
        return self._eval_base(x) + self.f_bias


class CompositionHybridFunction(_ComponentComposition):
    """Composition where each component is itself a HybridFunction."""

    def __init__(self, components: list, centers: list,
                 sigmas: list, lambdas_: list, biases: list,
                 blend_type: int, f_bias: float, D: int, device):
        super().__init__(components, centers, sigmas, lambdas_, biases,
                         blend_type, f_bias, D, device,
                         category='composition_hybrid')


# ── Fourier ─────��──────────────────────────────────────────────────────────

class FourierFunction(ProceduralFunction):
    """Sum of non-negative sinusoidal terms + optional quadratic bowl.

    f(z) = Σ_k A_k · 2·sin²(ω_k·z / 2) + α·‖z‖²
    All terms ≥ 0, all equal 0 at z=0 → f_optimal = f_bias (exact).
    """

    __slots__ = ('Q', 's', 'f_bias', 'amplitudes', 'frequencies', 'alpha_bowl')

    def __init__(self, Q: Tensor, s: Tensor, f_bias: float,
                 amplitudes: Tensor, frequencies: Tensor,
                 alpha_bowl: float, D: int, device):
        self.Q = Q
        self.s = s
        self.f_bias = f_bias
        self.amplitudes = amplitudes
        self.frequencies = frequencies
        self.alpha_bowl = alpha_bowl
        self.D = D
        self.device = device
        self.category = 'fourier'
        self.f_optimal = f_bias  # exact: f(z=0) = 0 by construction

    def __call__(self, x: Tensor) -> Tensor:
        z = (x - self.s) @ self.Q.T                                    # (N, D)
        dots = z @ self.frequencies.T                                   # (N, K)
        result = (self.amplitudes * 2.0 * torch.sin(dots / 2.0) ** 2).sum(dim=1)
        if self.alpha_bowl > 0:
            result = result + self.alpha_bowl * (z * z).sum(dim=1)
        return result + self.f_bias


class FourierCompositionFunction(_ComponentComposition):
    """Weighted blend of k FourierFunction components at different centers."""

    def __init__(self, components: list, centers: list,
                 sigmas: list, lambdas_: list, biases: list,
                 blend_type: int, f_bias: float, D: int, device):
        super().__init__(components, centers, sigmas, lambdas_, biases,
                         blend_type, f_bias, D, device,
                         category='fourier_composition')


# ── ProceduralSuite ────────────────────────────────────────────────────────

_CATEGORIES = ['unimodal', 'multimodal', 'hybrid', 'composition',
               'composition_hybrid', 'fourier', 'fourier_composition']

_DISPATCH = {
    'unimodal': '_sample_unimodal',
    'multimodal': '_sample_multimodal',
    'hybrid': '_sample_hybrid',
    'composition': '_sample_composition',
    'composition_hybrid': '_sample_composition_hybrid',
    'fourier': '_sample_fourier',
    'fourier_composition': '_sample_fourier_composition',
}


class ProceduralSuite:
    """Infinite procedural function generator for zero-shot L2O training."""

    def sample(self, D: int, device, category: Optional[str] = None,
               rng: Optional[torch.Generator] = None) -> ProceduralFunction:
        if rng is None:
            rng = torch.Generator()
        if category is None:
            category = _CATEGORIES[torch.randint(len(_CATEGORIES), (1,),
                                                  generator=rng).item()]
        method = _DISPATCH.get(category)
        if method is None:
            raise ValueError(f"Unknown category: {category}")
        return getattr(self, method)(D, device, rng)

    def sample_stratified(self, D: int, device,
                           rng: Optional[torch.Generator] = None) -> list:
        """Return one function per category (7 total)."""
        if rng is None:
            rng = torch.Generator()
        return [self.sample(D, device, category=cat, rng=rng)
                for cat in _CATEGORIES]

    def _sample_unimodal(self, D, device, rng):
        atom = _choose(UNIMODAL_ATOMS, rng)
        Q = _random_rotation(D, (1, 100), rng, device)
        s = _random_shift(D, Q, rng, device)
        f_bias = _random_bias(rng)
        return UnimodalFunction(atom, Q, s, f_bias, D, device)

    def _sample_multimodal(self, D, device, rng):
        atom = _choose(MULTIMODAL_ATOMS, rng)
        if atom.rotatable:
            Q = _random_rotation(D, (1, 20), rng, device)
        else:
            _ = torch.randn(D, D, generator=rng)
            _ = torch.rand(1, generator=rng)
            Q = torch.eye(D, dtype=torch.float64, device=device)
        s = _random_shift(D, Q, rng, device)
        f_bias = _random_bias(rng)
        return MultimodalFunction(atom, Q, s, f_bias, D, device)

    def _sample_hybrid(self, D, device, rng):
        k = min(torch.randint(2, 6, (1,), generator=rng).item(), D)
        perm = torch.randperm(D, generator=rng)
        partitions = _safe_partitions(k, D, perm, rng)
        atoms = [_choose(_HYBRID_SAFE_ATOMS, rng) for _ in range(k)]

        has_non_rotatable = any(not a.rotatable for a in atoms)
        if has_non_rotatable:
            _ = torch.randn(D, D, generator=rng)
            _ = torch.rand(1, generator=rng)
            Q = torch.eye(D, dtype=torch.float64, device=device)
        else:
            Q = _random_rotation(D, (1, 20), rng, device)

        s = _random_shift(D, Q, rng, device)
        f_bias = _random_bias(rng)
        return HybridFunction(atoms, Q, s, perm, partitions, f_bias, D, device)

    def _sample_composition(self, D, device, rng):
        k = torch.randint(2, 9, (1,), generator=rng).item()
        blend_type = torch.randint(0, 3, (1,), generator=rng).item()

        sigmas = [5.0 + torch.rand(1, generator=rng).item() * 55.0
                  for _ in range(k)]
        lambdas_ = [10.0 ** (torch.rand(1, generator=rng).item() * 7.0 - 6.0)
                     for _ in range(k)]
        biases = [i * 100.0 for i in range(k)]
        centers = [_random_center(D, rng, device) for _ in range(k)]
        atoms = [_choose(ALL_ATOMS, rng) for _ in range(k)]

        Q_list = []
        for atom in atoms:
            if atom.rotatable:
                Q_list.append(_random_rotation(D, (1, 20), rng, device))
            else:
                _ = torch.randn(D, D, generator=rng)
                _ = torch.rand(1, generator=rng)
                Q_list.append(torch.eye(D, dtype=torch.float64, device=device))

        f_bias = _random_bias(rng)
        return CompositionFunction(
            atoms, Q_list, centers, sigmas, lambdas_, biases,
            blend_type, f_bias, D, device)

    def _sample_composition_hybrid(self, D, device, rng):
        k = torch.randint(2, 5, (1,), generator=rng).item()
        blend_type = torch.randint(0, 3, (1,), generator=rng).item()

        sigmas = [10.0 + torch.rand(1, generator=rng).item() * 40.0
                  for _ in range(k)]
        lambdas_ = [1.0 for _ in range(k)]
        biases = [i * 100.0 for i in range(k)]
        centers = [_random_center(D, rng, device) for _ in range(k)]
        hybrids = [self._sample_hybrid(D, device, rng) for _ in range(k)]

        f_bias = _random_bias(rng)
        return CompositionHybridFunction(
            hybrids, centers, sigmas, lambdas_, biases,
            blend_type, f_bias, D, device)

    def _sample_fourier(self, D, device, rng):
        K = torch.randint(1, 21, (1,), generator=rng).item()

        # Amplitudes: log-uniform in [0.1, 100]
        log_A = torch.rand(K, generator=rng) * 3.0 - 1.0  # log10 in [-1, 2]
        amplitudes = (10.0 ** log_A).to(dtype=torch.float64, device=device)

        # Separable?
        separable = torch.rand(1, generator=rng).item() < 0.3

        if separable:
            # Per-dimension independent frequencies
            freq_mag = 0.05 + torch.rand(K, D, generator=rng) * 1.95
            frequencies = freq_mag.to(dtype=torch.float64, device=device)
        else:
            # Random direction + magnitude
            raw = torch.randn(K, D, generator=rng)
            norms = raw.norm(dim=1, keepdim=True).clamp(min=1e-8)
            directions = raw / norms
            magnitudes = 0.05 + torch.rand(K, 1, generator=rng) * 1.95
            frequencies = (directions * magnitudes).to(
                dtype=torch.float64, device=device)

        # 40% pure Fourier, 60% Fourier + quadratic bowl
        if torch.rand(1, generator=rng).item() < 0.4:
            alpha_bowl = 0.0
        else:
            alpha_bowl = 10.0 ** (torch.rand(1, generator=rng).item()
                                  * 2.0 - 2.0)  # [0.01, 1.0]

        Q = _random_rotation(D, (1, 50), rng, device)
        s = _random_shift(D, Q, rng, device)
        f_bias = _random_bias(rng)

        return FourierFunction(Q, s, f_bias, amplitudes, frequencies,
                               alpha_bowl, D, device)

    def _sample_fourier_composition(self, D, device, rng):
        k = torch.randint(2, 6, (1,), generator=rng).item()
        blend_type = torch.randint(0, 3, (1,), generator=rng).item()

        sigmas = [5.0 + torch.rand(1, generator=rng).item() * 55.0
                  for _ in range(k)]
        lambdas_ = [10.0 ** (torch.rand(1, generator=rng).item() * 4.0 - 2.0)
                     for _ in range(k)]
        biases = [i * 100.0 for i in range(k)]
        centers = [_random_center(D, rng, device) for _ in range(k)]
        components = [self._sample_fourier(D, device, rng) for _ in range(k)]

        f_bias = _random_bias(rng)
        return FourierCompositionFunction(
            components, centers, sigmas, lambdas_, biases,
            blend_type, f_bias, D, device)
