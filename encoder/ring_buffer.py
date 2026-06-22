"""
ring_buffer.py — GPU-resident circular buffer for population history.

Stores W generations of coordinates and per-individual fitness.
All tensors live on device; zero .item() calls except where
PyTorch API demands a Python int (GRU pack_padded_sequence length).
"""

import torch


class PopulationRingBuffer:
    """Fixed-capacity ring buffer for (x, fitness) history.

    Not an nn.Module — holds data, not learnable parameters.
    """

    def __init__(self, window: int, max_N: int, max_D: int,
                 device: torch.device):
        self.window = window
        self.max_N = max_N
        self.max_D = max_D
        self.device = device

        self.coords_buf = torch.zeros(window, max_N, max_D,
                                      device=device, dtype=torch.float32)
        self.fitness_buf = torch.zeros(window, max_N,
                                       device=device, dtype=torch.float32)
        self.valid_mask = torch.zeros(window,
                                      device=device, dtype=torch.bool)
        self._arange = torch.arange(window, device=device, dtype=torch.long)

        self.write_idx = torch.tensor(0, device=device, dtype=torch.long)
        self.f_init = torch.tensor(0.0, device=device, dtype=torch.float32)

    # ------------------------------------------------------------------
    def reset(self, f_init: torch.Tensor) -> None:
        """Clear buffer and store initial best fitness for normalisation."""
        self.coords_buf.zero_()
        self.fitness_buf.zero_()
        self.valid_mask.zero_()
        self.write_idx.zero_()
        self.f_init = f_init.detach().to(self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    def push(self, x: torch.Tensor, fitness: torch.Tensor) -> None:
        """Append one generation snapshot.

        Args:
            x:       (N, D) coordinates on device.
            fitness: (N,)   per-individual fitness on device.
        """
        idx = self.write_idx % self.window          # 0-dim tensor
        N, D = x.shape
        self.coords_buf[idx, :N, :D] = x.detach()
        self.fitness_buf[idx, :N] = fitness.detach()
        self.valid_mask[idx] = True
        self.write_idx += 1

    # ------------------------------------------------------------------
    def get_history(self):
        """Return chronologically ordered history.

        Returns:
            coords:     (W, max_N, max_D)  oldest-first
            fitness:    (W, max_N)          oldest-first
            valid_mask: (W,)                True for filled slots
            n_valid:    0-dim long tensor   number of valid timesteps
        """
        W = self.window
        n_valid = torch.clamp(self.write_idx, max=W)

        order = (self.write_idx - n_valid + self._arange) % W

        return (
            self.coords_buf[order],
            self.fitness_buf[order],
            self.valid_mask[order],
            n_valid,
        )
