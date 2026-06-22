"""AugmentedBBOB — infinite BBOB training problems via NATIVE instances.

Drop-in `aug_cache` for the task pool: mirrors the AugmentedCEC2017 /
LinkedFlameEnv `.sample(fid, D, rng)` interface. The returned object is a
`BBOBTorch` and exposes the same surface the trainer reads:
`fid, D, device, f_optimal, shift, __call__(x)` (differentiable).

DESIGN NOTE — why native instances, not an extra affine transform:
AugmentedCEC2017 generates variety by applying a random affine (rotation Q,
shift s, scale a) on top of each CEC2017 base. BBOB functions, however, already
embed their own shift / rotation / conditioning / boundary-penalty per instance;
stacking an extra affine on top would move the optimum off `x_opt` and break the
known `f_opt` and the `f_pen` boundary term for many functions (the same failure
mode AugmentedCEC2017 sidesteps with its Schwefel `PureFunction` special case).
BBOB's DESIGNED variety mechanism is its instance index, so each `sample()`
draws a fresh instance (different x_opt, rotations, f_opt) — infinite, correct,
and with the optimum value/location exactly known.
"""
import torch

from .bbob_torch import N_FUNCS, BBOBTorch


class AugmentedBBOB:
    """Generate infinite unique BBOB problems from the 24 base functions.

    Each sample() returns a BBOBTorch for a random (fid, D, instance).
    """

    def __init__(self, device='cuda', dims=(10,), instance_range=(1, 2000)):
        self.device = device
        self.dims = tuple(dims)
        self.instance_lo, self.instance_hi = instance_range

    def _get_base(self, fid, D, instance=1):
        """Parity with AugmentedCEC2017._get_base (instance-1 base)."""
        return BBOBTorch(fid, D, self.device, instance=instance)

    def sample(self, fid=None, D=None, rng=None):
        if rng is None:
            rng = torch.Generator(device='cpu')
            rng.manual_seed(int(torch.randint(0, 2 ** 31, (1,)).item()))
        if fid is None:
            fid = int(torch.randint(1, N_FUNCS + 1, (1,), generator=rng).item())
        if D is None:
            D = self.dims[int(torch.randint(0, len(self.dims), (1,),
                                            generator=rng).item())]
        instance = int(torch.randint(self.instance_lo, self.instance_hi + 1,
                                     (1,), generator=rng).item())
        return BBOBTorch(int(fid), int(D), self.device, instance=instance)
