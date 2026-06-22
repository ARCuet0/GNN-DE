"""
batched_augmented_cec2017.py — B augmentations of the SAME (fid, D) evaluated in one kernel.

Given B augmentations sharing a base CEC2017 function, evaluates (B, N, D) → (B, N)
by stacking rotation matrices Q: (B, D, D), shifts s: (B, D), scales a: (B,).

The base function is called once on (B*N, D) flattened points — single kernel.

Usage:
    batched_aug = BatchedAugmentedCEC2017(device='cuda')
    fn = batched_aug.sample_batch(B=16, fid=3, D=30)
    x = torch.rand(16, 100, 30, device='cuda', dtype=torch.float64) * 200 - 100
    f = fn(x)        # (16, 100)
    f_b = fn.eval_single(3, x[3])  # (100,) — for LS1
"""

import math

import torch

from .augmented_cec2017 import AugmentedCEC2017


class BatchedAugmentedFunction:
    """B augmentations of the SAME (fid, D) base function."""

    __slots__ = ('base_fn', 'Q', 's', 'a', 'f_optimal', 'fid', 'D', 'B',
                 'device')

    def __init__(self, base_fn, Q, s, a, f_optimal, fid, D, B, device):
        self.base_fn = base_fn    # CEC2017Torch instance (shared)
        self.Q = Q                # (B, D, D)
        self.s = s                # (B, D)
        self.a = a                # (B,)
        self.f_optimal = f_optimal  # (B,)
        self.fid = fid
        self.D = D
        self.B = B
        self.device = device

    def __call__(self, x):
        """Evaluate B augmentations in one kernel.

        Args:
            x: (B, N, D) float64 tensor

        Returns:
            (B, N) float64 fitness
        """
        B, N, D = x.shape
        # Affine transform: z = (x - s) @ Q^T  per batch element
        z = torch.bmm(x - self.s.unsqueeze(1), self.Q.mT)  # (B, N, D)
        # Flatten, evaluate base fn once, reshape
        z_flat = z.reshape(B * N, D)
        f_flat = self.base_fn(z_flat)         # (B*N,)
        return self.a.unsqueeze(1) * f_flat.reshape(B, N)

    def eval_single(self, b, x):
        """Evaluate single augmentation b on (M, D) points.

        Used by batched_mtsls1_gpu which generates variable-size probes.

        Args:
            b: batch index
            x: (M, D) float64 tensor

        Returns:
            (M,) float64 fitness
        """
        z = (x - self.s[b]) @ self.Q[b].mT
        return self.a[b] * self.base_fn(z)


class BatchedPureFunction:
    """B copies of the SAME Schwefel-train function (no augmentation).

    All B batch elements evaluate identically — just reshapes to (B*N, D).
    """

    __slots__ = ('base_fn', 'f_optimal', 'fid', 'D', 'B', 'device')

    def __init__(self, base_fn, fid, D, B, device):
        self.base_fn = base_fn
        self.f_optimal = torch.full((B,), base_fn.f_optimal,
                                    dtype=torch.float64, device=device)
        self.fid = fid
        self.D = D
        self.B = B
        self.device = device

    def __call__(self, x):
        """(B, N, D) → (B, N)."""
        B, N, D = x.shape
        return self.base_fn(x.reshape(B * N, D)).reshape(B, N)

    def eval_single(self, b, x):
        """(M, D) → (M,)."""
        return self.base_fn(x)


class BatchedAugmentedCEC2017:
    """Generate B augmentations of the same (fid, D) for batched GPU collection.

    Reuses AugmentedCEC2017 for blacklist logic and CEC2017Torch caching.
    """

    def __init__(self, device='cuda', dims=(10, 30, 50)):
        self._aug = AugmentedCEC2017(device=device, dims=dims)
        self.device = device
        self.dims = dims

    def sample_batch(self, B, fid=None, D=None, seed=None):
        """Sample B augmentations of the same (fid, D).

        Args:
            B: batch size (number of parallel augmentations)
            fid: function ID (1-29). Random if None.
            D: dimensionality. Random if None.
            seed: base seed for reproducibility. Augmentation b uses seed+b.

        Returns:
            BatchedAugmentedFunction or BatchedPureFunction
        """
        if seed is None:
            seed = int(torch.randint(0, 2**31, (1,)).item())

        # Use first rng to pick fid/D if not specified
        rng0 = torch.Generator(device='cpu')
        rng0.manual_seed(seed)

        if fid is None:
            fid = int(torch.randint(1, 30, (1,), generator=rng0).item())
        if D is None:
            dim_idx = int(torch.randint(0, len(self.dims), (1,),
                                        generator=rng0).item())
            D = self.dims[dim_idx]

        # Re-sample fid if blacklisted
        combined_bl = AugmentedCEC2017.BLACKLIST | AugmentedCEC2017.AUG_BLACKLIST
        for _ in range(10):
            if (fid, D) not in combined_bl:
                break
            fid = int(torch.randint(1, 30, (1,), generator=rng0).item())

        if (fid, D) in combined_bl:
            raise RuntimeError(
                f"Could not find non-blacklisted (fid, D) after 10 attempts")

        base_fn = self._aug._get_base(fid, D)

        # Schwefel-train: return pure (no augmentation)
        if fid in AugmentedCEC2017.SCHWEFEL_TRAIN:
            return BatchedPureFunction(base_fn, fid, D, B, self.device)

        # Generate B independent augmentations
        Qs = []
        ss = []
        a_s = []
        f_opts = []

        for i in range(B):
            rng = torch.Generator(device='cpu')
            rng.manual_seed(seed + i)

            # SVD conditioning (same logic as AugmentedCEC2017.sample)
            raw = torch.randn(D, D, generator=rng)
            U, _, Vt = torch.linalg.svd(raw, full_matrices=False)

            log_kappa = torch.rand(1, generator=rng).item() * math.log10(20.0)
            kappa = 10.0 ** log_kappa
            S_new = torch.logspace(0, log_kappa, D, dtype=torch.float64)
            S_new = S_new / S_new.prod().pow(1.0 / D)

            Q = (U.to(torch.float64) @ torch.diag(S_new)
                 @ Vt.to(torch.float64)).to(device=self.device)

            # Shift: ensure augmented optimum in [-100, 100]^D
            x_base_opt = base_fn.shift if hasattr(base_fn, 'shift') \
                else base_fn.shift_mat[0]
            rotated_opt = x_base_opt @ Q

            s_lo = -100.0 - rotated_opt
            s_hi = 100.0 - rotated_opt
            u = torch.rand(D, generator=rng).to(dtype=torch.float64,
                                                 device=self.device)
            s = s_lo + u * (s_hi - s_lo)

            # Scale: log-uniform [0.1, 10]
            log_a = torch.rand(1, generator=rng).item() * 2.0 - 1.0
            a = 10.0 ** log_a

            Qs.append(Q)
            ss.append(s)
            a_s.append(a)
            f_opts.append(a * base_fn.f_optimal)

        Q_stacked = torch.stack(Qs)    # (B, D, D)
        s_stacked = torch.stack(ss)    # (B, D)
        a_stacked = torch.tensor(a_s, dtype=torch.float64,
                                 device=self.device)  # (B,)
        f_opt_stacked = torch.tensor(f_opts, dtype=torch.float64,
                                     device=self.device)  # (B,)

        return BatchedAugmentedFunction(
            base_fn, Q_stacked, s_stacked, a_stacked,
            f_opt_stacked, fid, D, B, self.device)
