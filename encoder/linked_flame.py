"""linked_flame.py — RBF curriculum environment for L2O training.

Five-level RBF (Radial Basis Function) curriculum for parametrically-controlled
optimization landscapes:

  L1: single Gaussian (sphere-like; sanity / regression bar)
  L2: well-separated multimodal (3-5 modes, dominant + decoys)
  L3: dense multimodal (10-20 competing modes)
  L4: anisotropic mixture (rotated covariances, condition number sweep)
  L5: multi-scale composition (3 scale bands, ~9-15 modes total)

Each Level is a *distribution* over functions, sampled per minibatch via
LinkedFlameEnv.sample(fid=level, D=D, rng=g). Drop-in replacement for
AugmentedCEC2017.sample() — exposes identical interface (__call__,
f_optimal, shift, fid, D, device).

f_optimal is exact (L1, L2 by construction within numerical refinement
tolerance) or numerically refined (L3, L4, L5 via multi-start Adam descent
on CPU at construction time).
"""

import math

import torch


DOMAIN_HALF = 100.0   # search box [-DOMAIN_HALF, DOMAIN_HALF]^D
INNER_HALF = 80.0     # centroid-sampling sub-box


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _u(n, rng, dtype=torch.float64):
    return torch.rand(n, generator=rng, dtype=dtype)


def _sample_centroid(D, rng, half=INNER_HALF):
    """Uniform centroid in [-half, half]^D."""
    return (_u(D, rng) * 2.0 - 1.0) * half


# ----------------------------------------------------------------------
# Per-Level samplers — return params dict consumed by LinkedFlameInstance
# ----------------------------------------------------------------------

def _sample_l1(D, rng):
    """Level 1 — single Gaussian, isotropic.

    sigma scaled with sqrt(D) so signal is accessible from random init
    points across dimensions.
    """
    c = _sample_centroid(D, rng)
    u = _u(1, rng).item()
    sigma = (10.0 + 20.0 * u) * math.sqrt(D)   # [10*sqrt(D), 30*sqrt(D)]
    return {
        'centroids': c.unsqueeze(0),                                  # (1, D)
        'sigmas': torch.tensor([sigma], dtype=torch.float64),         # (1,)
        'amplitudes': torch.tensor([1.0], dtype=torch.float64),       # (1,)
        'precisions': None,
    }


def _sample_l2(D, rng, max_attempts=2000):
    """Level 2 — well-separated multimodal (3-5 modes).

    Pairwise centroid separation enforced > 4·max(sigma) + 1.0 so the
    dominant centroid is the global minimum within numerical tolerance.
    """
    K = int(torch.randint(3, 6, (1,), generator=rng).item())
    sigmas = 8.0 + 17.0 * _u(K, rng)   # [8, 25]

    amps = torch.zeros(K, dtype=torch.float64)
    amps[0] = 1.0
    if K > 1:
        amps[1:] = 0.3 + 0.55 * _u(K - 1, rng)   # [0.3, 0.85]

    sep = 4.0 * sigmas.max().item() + 1.0   # margin so test's strict > passes
    centroids = torch.zeros(K, D, dtype=torch.float64)

    for k in range(K):
        for _ in range(max_attempts):
            cand = _sample_centroid(D, rng)
            if k == 0 or all(
                (cand - centroids[j]).norm().item() > sep
                for j in range(k)
            ):
                centroids[k] = cand
                break
        else:
            raise RuntimeError(
                f"linked-flame L2: failed to place mode {k} after "
                f"{max_attempts} attempts (D={D}, sep={sep:.1f})."
            )

    return {
        'centroids': centroids,
        'sigmas': sigmas,
        'amplitudes': amps,
        'precisions': None,
    }


def _sample_l3(D, rng, max_attempts=2000):
    """Level 3 — dense multimodal (10-20 modes).

    Weaker separation (1.5·max(sigma)). When rejection fails, fall back to
    the candidate maximizing min-distance to existing centroids — modes can
    interfere, so f_optimal must be numerically refined regardless.
    """
    K_choices = (10, 12, 15, 18, 20)
    K = K_choices[int(torch.randint(0, len(K_choices), (1,),
                                    generator=rng).item())]
    sigmas = 5.0 + 10.0 * _u(K, rng)   # [5, 15]

    amps = 0.5 + 0.5 * _u(K, rng)
    amps[0] = 1.0   # force one dominant mode

    sep = 1.5 * sigmas.max().item()
    centroids = torch.zeros(K, D, dtype=torch.float64)

    for k in range(K):
        placed = False
        for _ in range(max_attempts):
            cand = _sample_centroid(D, rng)
            if k == 0 or all(
                (cand - centroids[j]).norm().item() > sep
                for j in range(k)
            ):
                centroids[k] = cand
                placed = True
                break
        if not placed:
            best_dist = -1.0
            best_cand = None
            for _ in range(500):
                cand = _sample_centroid(D, rng)
                if k == 0:
                    best_cand = cand
                    break
                min_d = min(
                    (cand - centroids[j]).norm().item() for j in range(k)
                )
                if min_d > best_dist:
                    best_dist = min_d
                    best_cand = cand
            centroids[k] = best_cand

    return {
        'centroids': centroids,
        'sigmas': sigmas,
        'amplitudes': amps,
        'precisions': None,
    }


def _sample_l4(D, rng):
    """Level 4 — anisotropic. Rotated covariance per mode, log-uniform spectrum.

    kappa_max ∈ {10, 100, 1000} selects difficulty per instance.
    Eigenvalues are volume-normalized so geom-mean(lambda) = sigma_base² = 100,
    keeping signal scale comparable to L3.
    """
    K_choices = (8, 10, 12)
    K = K_choices[int(torch.randint(0, len(K_choices), (1,),
                                    generator=rng).item())]
    kappa_choices = (10.0, 100.0, 1000.0)
    kappa_max = kappa_choices[int(torch.randint(0, 3, (1,), generator=rng).item())]
    log_kappa = math.log10(kappa_max)
    sigma_base_sq = 100.0   # geom mean target for lambda

    amps = 0.5 + 0.5 * _u(K, rng)
    amps[0] = 1.0

    centroids = torch.stack([_sample_centroid(D, rng) for _ in range(K)])

    precisions = torch.zeros(K, D, D, dtype=torch.float64)
    for k in range(K):
        raw = torch.randn(D, D, generator=rng, dtype=torch.float64)
        Q_k, _ = torch.linalg.qr(raw)
        log_lam_grid = torch.linspace(0, log_kappa, D, dtype=torch.float64)
        perm = torch.randperm(D, generator=rng)
        log_lam = log_lam_grid[perm]
        lambdas = 10.0 ** log_lam                                  # in [1, kappa_max]
        lambdas_norm = lambdas / lambdas.prod().pow(1.0 / D)       # geom mean = 1
        lambdas_final = sigma_base_sq * lambdas_norm               # geom mean = sigma_base_sq
        precisions[k] = Q_k @ torch.diag(1.0 / lambdas_final) @ Q_k.T

    return {
        'centroids': centroids,
        'sigmas': None,
        'amplitudes': amps,
        'precisions': precisions,
    }


def _sample_l5(D, rng):
    """Level 5 — multi-scale composition. 3 scale bands × 3-5 modes each."""
    sigma_ranges = [(20.0, 40.0), (8.0, 15.0), (2.0, 5.0)]   # wide / mid / narrow

    centroids_all, sigmas_all, amps_all = [], [], []
    for band_idx, (s_lo, s_hi) in enumerate(sigma_ranges):
        K_band = int(torch.randint(3, 6, (1,), generator=rng).item())
        sigmas_band = s_lo + (s_hi - s_lo) * _u(K_band, rng)
        amp_scale = (3 - band_idx) / 3.0    # wider scale -> larger amplitude
        amps_band = amp_scale * (0.5 + 0.5 * _u(K_band, rng))
        cents_band = torch.stack(
            [_sample_centroid(D, rng) for _ in range(K_band)]
        )
        centroids_all.append(cents_band)
        sigmas_all.append(sigmas_band)
        amps_all.append(amps_band)

    centroids = torch.cat(centroids_all, dim=0)
    sigmas = torch.cat(sigmas_all, dim=0)
    amps = torch.cat(amps_all, dim=0)
    # Renormalize so the global maximum amplitude is 1.0 (dominant signal)
    amps = amps / amps.max()

    return {
        'centroids': centroids,
        'sigmas': sigmas,
        'amplitudes': amps,
        'precisions': None,
    }


# ----------------------------------------------------------------------
# Instance class
# ----------------------------------------------------------------------

class LinkedFlameInstance:
    """A single instance of a linked-flame curriculum function.

    f(x) = -Σ_k a_k · exp(-½·quad_k(x))

      isotropic    : quad_k(x) = ||x - c_k||² / σ_k²
      anisotropic  : quad_k(x) = (x - c_k)ᵀ M_k (x - c_k),  M_k = Σ_k⁻¹

    Mirrors the AugmentedFunction interface (__call__, f_optimal, shift,
    fid, D, device) plus level/instance_seed for diagnostics.
    """

    __slots__ = (
        # Public CPU views (also the canonical buffers)
        'centroids', 'sigmas', 'amplitudes', 'precisions',
        # Identity / metadata
        'fid', 'level', 'D', 'device', 'instance_seed',
        # Outputs of construction
        'f_optimal', 'shift',
        # Device buffers (private)
        '_centroids_dev', '_amps_dev', '_sigmas_sq_dev', '_precisions_dev',
    )

    def __init__(self, params, fid, D, device, instance_seed):
        self.centroids = params['centroids']         # (K, D) cpu f64
        self.sigmas = params['sigmas']               # (K,) cpu f64 or None
        self.amplitudes = params['amplitudes']       # (K,) cpu f64
        self.precisions = params['precisions']       # (K, D, D) cpu f64 or None

        self.fid = fid
        self.level = fid
        self.D = D
        self.device = device
        self.instance_seed = instance_seed

        # Refine f_optimal numerically on CPU (deterministic)
        f_opt, shift_cpu = self._refine_optimum_cpu()
        self.f_optimal = float(f_opt)

        # Move buffers to evaluation device
        self._centroids_dev = self.centroids.to(device=device)
        self._amps_dev = self.amplitudes.to(device=device)
        if self.sigmas is not None:
            self._sigmas_sq_dev = (self.sigmas ** 2).to(device=device)
            self._precisions_dev = None
        else:
            self._sigmas_sq_dev = None
            self._precisions_dev = self.precisions.to(device=device)

        self.shift = shift_cpu.to(device=device)

    # ---- evaluation ----
    @staticmethod
    def _eval_with_buffers(x, centroids, sigmas_sq, precisions, amps):
        """Evaluate -Σ a_k exp(-½ quad_k). Tensors must share device.

        x: (N, D) → (N,)
        """
        diff = x.unsqueeze(1) - centroids.unsqueeze(0)        # (N, K, D)
        if sigmas_sq is not None:
            quad = (diff ** 2).sum(-1) / sigmas_sq.unsqueeze(0)   # (N, K)
        else:
            quad = torch.einsum('nkd,kde,nke->nk', diff, precisions, diff)
        kernel = torch.exp(-0.5 * quad)
        return -(amps.unsqueeze(0) * kernel).sum(-1)          # (N,)

    def __call__(self, x):
        return self._eval_with_buffers(
            x, self._centroids_dev,
            self._sigmas_sq_dev, self._precisions_dev, self._amps_dev,
        )

    # ---- refinement ----
    def _refine_optimum_cpu(self, n_steps=500, lr=0.5, n_random_starts=128):
        """Adam multi-start descent on CPU; returns (f_opt, shift) on CPU.

        Starts from K centroids + n_random_starts random uniform points in the
        inner box. Runs n_steps Adam steps in float64. Determinism is
        maintained by deriving the random-start RNG from instance_seed.
        """
        if self.sigmas is not None:
            sigmas_sq = self.sigmas ** 2
            precisions = None
        else:
            sigmas_sq = None
            precisions = self.precisions

        starts = [self.centroids]
        if n_random_starts > 0:
            cpu_rng = torch.Generator(device='cpu')
            cpu_rng.manual_seed((self.instance_seed * 7919 + 17) & 0x7FFFFFFF)
            rand = (torch.rand(n_random_starts, self.D, generator=cpu_rng,
                               dtype=torch.float64) * 2 - 1) * INNER_HALF
            starts.append(rand)
        # Refinement requires autograd even when called from a torch.no_grad()
        # context (e.g. train_distributed.py target-gap pre-compute).
        with torch.enable_grad():
            x = torch.cat(starts, dim=0).clone().requires_grad_(True)
            opt = torch.optim.Adam([x], lr=lr)
            for _ in range(n_steps):
                opt.zero_grad()
                f = self._eval_with_buffers(
                    x, self.centroids, sigmas_sq, precisions, self.amplitudes,
                )
                f.sum().backward()
                opt.step()
                with torch.no_grad():
                    x.clamp_(-DOMAIN_HALF, DOMAIN_HALF)

            with torch.no_grad():
                f_final = self._eval_with_buffers(
                    x, self.centroids, sigmas_sq, precisions, self.amplitudes,
                )
                best_idx = int(f_final.argmin().item())
                f_opt = f_final[best_idx].item()
                shift = x[best_idx].detach().clone()

        return f_opt, shift


# ----------------------------------------------------------------------
# Environment / factory
# ----------------------------------------------------------------------

class LinkedFlameEnv:
    """Sampler producing LinkedFlameInstance objects.

    Drop-in for AugmentedCEC2017: same .sample(fid=int|None, D=int|None,
    rng=Generator|None) signature, same returned interface (__call__,
    f_optimal, shift, fid, D, device).
    """

    LEVELS = (1, 2, 3, 4, 5)
    _SAMPLERS = {
        1: _sample_l1,
        2: _sample_l2,
        3: _sample_l3,
        4: _sample_l4,
        5: _sample_l5,
    }

    def __init__(self, device='cuda', dims=(10, 30)):
        self.device = device
        self.dims = tuple(dims)

    def sample(self, fid=None, D=None, rng=None):
        if rng is None:
            rng = torch.Generator(device='cpu')
            rng.manual_seed(int(torch.randint(0, 2**31, (1,)).item()))

        if fid is None:
            fid = int(torch.randint(1, 6, (1,), generator=rng).item())
        if fid not in self.LEVELS:
            raise ValueError(
                f"linked-flame fid (level) must be in {self.LEVELS}, got {fid}"
            )

        if D is None:
            dim_idx = int(torch.randint(0, len(self.dims), (1,),
                                        generator=rng).item())
            D = self.dims[dim_idx]

        instance_seed = int(torch.randint(
            0, 2**31 - 1, (1,), generator=rng,
        ).item())
        sub_rng = torch.Generator(device='cpu')
        sub_rng.manual_seed(instance_seed)

        params = self._SAMPLERS[fid](D, sub_rng)
        return LinkedFlameInstance(
            params=params, fid=fid, D=D,
            device=self.device, instance_seed=instance_seed,
        )
