"""
augmented_cec2017.py — Infinite differentiable optimization problems.

Wraps CEC2017Torch with random affine transforms (rotation, shift, scale)
to generate unlimited unique training functions. All transforms are
differentiable — gradients flow through f(x) back to model parameters.

Modified Schwefel is unbounded below in R^D, so any rotation/shift can
expose deeper minima and break f_optimal. Schwefel-containing functions
are split into two sets:
  - SCHWEFEL_TRAIN: returned WITHOUT augmentation (f_optimal exact)
  - SCHWEFEL_EXCLUDE: fully excluded (already in BLACKLIST or redundant)

This logic lives here so ALL callers (NEURAL_META, GNN_MOS_Classic,
NEURAL_ELA_MOS, HyperOPT) benefit automatically via aug.sample().

Usage:
    aug = AugmentedCEC2017(device='cuda')
    fn = aug.sample()          # random augmented OR pure function
    f = fn(x)                  # (N, D) -> (N,) differentiable
    f.mean().backward()        # gradients flow
"""

import math

import torch

from .cec2017_torch import CEC2017Torch


class AugmentedFunction:
    """A CEC2017 function with random affine transform applied."""

    __slots__ = ('base_fn', 'Q', 's', 'a', 'fid', 'D', 'device', 'f_optimal',
                 'shift')

    def __init__(self, base_fn, Q, s, a, fid, D, device):
        self.base_fn = base_fn
        self.Q = Q        # (D, D) rotation with SVD conditioning (kappa in [1, 20])
        self.s = s        # (D,) shift
        self.a = a        # scalar scale
        self.fid = fid
        self.D = D
        self.device = device
        self.f_optimal = a * base_fn.f_optimal
        # Optimum position in augmented space: x* = s + x_base* @ Q
        x_base_opt = base_fn.shift if hasattr(base_fn, 'shift') \
            else base_fn.shift_mat[0]
        self.shift = s + x_base_opt @ Q

    def __call__(self, x):
        """Evaluate augmented function. Fully differentiable.

        Args:
            x: (N, D) float64 tensor on device

        Returns:
            (N,) float64 fitness values with gradient
        """
        z = (x - self.s) @ self.Q.T
        return self.a * self.base_fn(z)


class PureFunction:
    """A CEC2017 function returned as-is (no augmentation).

    Same interface as AugmentedFunction so callers don't need to
    distinguish. Used for Schwefel-containing functions where
    augmentation would break f_optimal guarantees.
    """

    __slots__ = ('_fn', 'fid', 'D', 'device', 'f_optimal', 'shift')

    def __init__(self, fn_obj, fid, D, device):
        self._fn = fn_obj
        self.fid = fid
        self.D = D
        self.device = device
        self.f_optimal = fn_obj.f_optimal
        self.shift = fn_obj.shift if hasattr(fn_obj, 'shift') \
            else fn_obj.shift_mat[0]

    def __call__(self, x):
        return self._fn(x)


class AugmentedCEC2017:
    """Generate infinite unique optimization problems from 29 CEC2017 bases.

    Each call to sample() produces either:
      - AugmentedFunction (rotation + shift + scale) for safe functions
      - PureFunction (no transform) for Schwefel-containing functions

    All returned objects expose: fid, D, device, f_optimal, shift, __call__.
    """

    # Import the canonical blacklist from CEC2017_torch
    from .cec2017_torch import BLACKLIST

    # Modified Schwefel is unbounded below in R^D: its z*sin(sqrt(|z|)) term
    # grows as O(|z|). Any rotation or shift that changes the effective z-range
    # exposes deeper minima, making f_optimal incorrect. This affects:
    #   F9:  Modified Schwefel (direct)
    #   F11: Hybrid 2 (elliptic + schwefel + bent_cigar)
    #   F15: Hybrid 6 (schaffer + hgbat + rosenbrock + schwefel)
    #   F16: Hybrid 7 (elliptic + hgbat + schwefel + rastrigin + ...)
    #   F22: Composition 3 (rosenbrock + ackley + schwefel + rastrigin)
    #   F25: Composition 6 (schaffer + schwefel + griewank + rosenbrock + rastrigin)
    #   F26: Composition 7 (hgbat + rastrigin + schwefel + bent_cigar + ...)
    # F19, F21, F23 already in BLACKLIST; F29 uses F17 which contains schwefel.
    _SCHWEFEL_FIDS = frozenset({9, 11, 15, 16, 22, 25, 26})

    # Half included as non-augmented for category coverage:
    #   F9  (Multimodal), F16 (Hybrid), F22 (Composition), F25 (Composition)
    SCHWEFEL_TRAIN = frozenset({9, 16, 22, 25})

    # The rest are fully excluded from augmented sampling
    _SCHWEFEL_EXCLUDE = _SCHWEFEL_FIDS - SCHWEFEL_TRAIN
    AUG_BLACKLIST = {(fid, d) for fid in _SCHWEFEL_EXCLUDE
                     for d in (10, 30, 50)}

    def __init__(self, device='cuda', dims=(10, 30, 50)):
        self.device = device
        self.dims = dims
        self.fn_cache = {}  # (fid, D) -> CEC2017Torch

    def _get_base(self, fid, D):
        """Lazily create and cache CEC2017Torch instances."""
        key = (fid, D)
        if key not in self.fn_cache:
            self.fn_cache[key] = CEC2017Torch(fid, D, self.device)
        return self.fn_cache[key]

    def sample(self, fid=None, D=None, rng=None):
        """Sample a random function (augmented or pure).

        Args:
            fid: base function ID (1-29). Random if None.
            D: dimensionality. Random from self.dims if None.
            rng: torch.Generator for reproducibility. Random if None.

        Returns:
            AugmentedFunction or PureFunction with identical interface:
            fid, D, device, f_optimal, shift, __call__(x).
        """
        if rng is None:
            rng = torch.Generator(device='cpu')
            rng.manual_seed(torch.randint(0, 2**31, (1,)).item())

        if fid is None:
            fid = int(torch.randint(1, 30, (1,), generator=rng).item())
        if D is None:
            dim_idx = int(torch.randint(0, len(self.dims), (1,),
                                        generator=rng).item())
            D = self.dims[dim_idx]

        # Re-sample fid if blacklisted (keep D fixed — caller chose it)
        combined_bl = self.BLACKLIST | self.AUG_BLACKLIST
        for _ in range(10):
            if (fid, D) not in combined_bl:
                break
            fid = int(torch.randint(1, 30, (1,), generator=rng).item())

        if (fid, D) in combined_bl:
            raise RuntimeError(
                f"Could not find non-blacklisted (fid, D) after 10 attempts; "
                f"last attempt: ({fid}, {D})")

        base_fn = self._get_base(fid, D)

        # Schwefel-train functions: return without augmentation
        if fid in self.SCHWEFEL_TRAIN:
            # Consume rng state to keep reproducibility consistent
            _ = torch.randn(D, D, generator=rng)  # Q (SVD)
            _ = torch.rand(1, generator=rng)       # kappa
            _ = torch.rand(D, generator=rng)       # s
            _ = torch.rand(1, generator=rng)       # a
            return PureFunction(base_fn, fid, D, self.device)

        # Normal augmentation: SVD conditioning (anisotropic landscapes)
        raw = torch.randn(D, D, generator=rng)
        U, _, Vt = torch.linalg.svd(raw, full_matrices=False)

        # Condition number kappa ~ log-uniform [1, 20]
        log_kappa = torch.rand(1, generator=rng).item() * math.log10(20.0)
        kappa = 10.0 ** log_kappa

        # Singular values from 1 to kappa, volume-normalized (geom mean = 1)
        S_new = torch.logspace(0, log_kappa, D, dtype=torch.float64)
        S_new = S_new / S_new.prod().pow(1.0 / D)

        Q = (U.to(torch.float64) @ torch.diag(S_new) @ Vt.to(torch.float64))
        Q = Q.to(device=self.device)

        # Compute rotated optimum: x_aug* = s + x_base* @ Q
        # Choose s so x_aug* ∈ [-100, 100]^D.
        x_base_opt = base_fn.shift if hasattr(base_fn, 'shift') \
            else base_fn.shift_mat[0]
        rotated_opt = x_base_opt @ Q

        s_lo = -100.0 - rotated_opt
        s_hi = 100.0 - rotated_opt
        u = torch.rand(D, generator=rng).to(dtype=torch.float64,
                                            device=self.device)
        s = s_lo + u * (s_hi - s_lo)

        # Random log-uniform scale [0.1, 10]
        log_a = torch.rand(1, generator=rng).item() * 2.0 - 1.0  # [-1, 1]
        a = 10.0 ** log_a

        return AugmentedFunction(base_fn, Q, s, a, fid, D, self.device)


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")

    aug = AugmentedCEC2017(device=dev, dims=(10, 30))

    # Test 1: basic evaluation
    fn = aug.sample(fid=1, D=30)
    x = torch.rand(100, 30, device=dev, dtype=torch.float64) * 200 - 100
    f = fn(x)
    print(f"F{fn.fid} D={fn.D}: f.shape={f.shape}, "
          f"f.min={f.min().item():.2f}, f.max={f.max().item():.2f}")

    # Test 2: gradient flow
    x_grad = x.clone().requires_grad_(True)
    f_grad = fn(x_grad)
    loss = f_grad.mean()
    loss.backward()
    assert x_grad.grad is not None, "No gradient on input!"
    assert x_grad.grad.abs().sum() > 0, "Zero gradient!"
    print(f"Gradient flow OK: grad_norm={x_grad.grad.norm().item():.4f}")

    # Test 3: different augmentations produce different functions
    fn1 = aug.sample(fid=1, D=30)
    fn2 = aug.sample(fid=1, D=30)
    f1 = fn1(x.detach())
    f2 = fn2(x.detach())
    diff = (f1 - f2).abs().mean().item()
    print(f"Same base, different augmentation: mean_diff={diff:.2f} "
          f"(should be > 0)")
    assert diff > 0, "Two augmentations produced identical output!"

    # Test 4: random sampling across all fids and dims
    fids_seen = set()
    dims_seen = set()
    pure_count = 0
    for _ in range(500):
        fn = aug.sample()
        fids_seen.add(fn.fid)
        dims_seen.add(fn.D)
        if isinstance(fn, PureFunction):
            pure_count += 1
    print(f"Random sampling: {len(fids_seen)} fids, {len(dims_seen)} dims, "
          f"{pure_count}/500 pure (Schwefel)")

    # Test 5: reproducibility with rng
    rng1 = torch.Generator(device='cpu')
    rng1.manual_seed(42)
    fn_a = aug.sample(rng=rng1)

    rng2 = torch.Generator(device='cpu')
    rng2.manual_seed(42)
    fn_b = aug.sample(rng=rng2)

    x_test = torch.rand(10, fn_a.D, device=dev, dtype=torch.float64) * 200 - 100
    diff_repro = (fn_a(x_test) - fn_b(x_test)).abs().max().item()
    print(f"Reproducibility: max_diff={diff_repro:.2e} (should be ~0)")

    # Test 6: Schwefel functions are pure
    for fid in AugmentedCEC2017.SCHWEFEL_TRAIN:
        fn = aug.sample(fid=fid, D=10)
        assert isinstance(fn, PureFunction), f"F{fid} should be pure!"
        assert fn.f_optimal == fn._fn.f_optimal
    print(f"Schwefel-train functions returned as PureFunction: OK")

    # Test 7: all functions have shift attribute
    for _ in range(100):
        fn = aug.sample()
        assert hasattr(fn, 'shift'), f"F{fn.fid} missing shift!"
    print("All functions expose .shift: OK")

    # Test 8: SVD conditioning produces non-orthogonal Q
    non_ortho_count = 0
    for _ in range(100):
        fn = aug.sample()
        if isinstance(fn, AugmentedFunction):
            QtQ = fn.Q @ fn.Q.T
            eye = torch.eye(fn.D, device=dev, dtype=torch.float64)
            off_diag = (QtQ - eye).abs().max().item()
            if off_diag > 0.01:
                non_ortho_count += 1
    print(f"Non-orthogonal Q count: {non_ortho_count}/100 augmented "
          f"(should be >0 — kappa > 1 breaks orthogonality)")
    assert non_ortho_count > 0, "SVD conditioning never produced non-orthogonal Q!"

    # Test 9: conditioned Q produces finite, differentiable output
    fn = aug.sample(fid=1, D=30)
    x_cond = torch.rand(50, 30, device=dev, dtype=torch.float64) * 200 - 100
    x_cond.requires_grad_(True)
    f_cond = fn(x_cond)
    assert torch.isfinite(f_cond).all(), "Non-finite fitness with conditioned Q!"
    f_cond.mean().backward()
    assert x_cond.grad is not None and torch.isfinite(x_cond.grad).all(), \
        "Bad gradients with conditioned Q!"
    print("Conditioned Q: evaluation + gradient OK")

    print("\nAll smoke tests passed!")
