"""CountSketch random projection for gradient direction tracking.

Projects high-dimensional gradient vectors (P~525K) to low-dimensional
sketches (d=128) preserving cosine similarity (Johnson-Lindenstrauss).
Used to detect gradient direction conflicts across CEC2017 functions.

Storage: two int tensors of length P (~4MB), not a dense P×d matrix (~269MB).
Runtime: single scatter_add kernel, <1ms on GPU for P=525K.
"""
import math
import torch


class GradientProjector:
    """CountSketch random projection for gradient direction tracking.

    Usage:
        params = list(model.parameters())
        proj = GradientProjector(params, d_proj=128, seed=42)
        # ... after loss.backward() ...
        sketch = proj.project(params)  # (d_proj,) float32
    """

    def __init__(self, params, d_proj=128, seed=42, device='cpu'):
        P = sum(p.numel() for p in params)
        gen = torch.Generator().manual_seed(seed)
        self.buckets = torch.randint(0, d_proj, (P,), generator=gen, device='cpu').to(device)
        self.signs = (torch.randint(0, 2, (P,), generator=gen, device='cpu').float() * 2 - 1).to(device)
        self.d_proj = d_proj
        self.P = P
        self._scale = math.sqrt(d_proj / P) if P > 0 else 1.0

    def project(self, params):
        """Project the .grad of each param into a (d_proj,) sketch.

        Args:
            params: list of nn.Parameter, same order as __init__.
                    Parameters with grad=None contribute zeros.

        Returns:
            Tensor of shape (d_proj,) on the same device as self.buckets.
        """
        pieces = []
        for p in params:
            if p.grad is not None:
                pieces.append(p.grad.detach().float().reshape(-1))
            else:
                pieces.append(torch.zeros(p.numel(), device=self.buckets.device))
        g = torch.cat(pieces)
        signed = g * self.signs
        proj = torch.zeros(self.d_proj, device=g.device)
        proj.scatter_add_(0, self.buckets, signed)
        proj *= self._scale
        return proj
