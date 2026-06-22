"""Batched operator heads for neural L2O variant."""
from encoder.operators._base import _ParamMLP, _make_proj
from encoder.operators.de_heads import BatchedDiffDE, BatchedDiffAttDE, BatchedDiffLSHADE
from encoder.operators.coordls_heads import (
    BatchedDiffCoordLS, BatchedDiffMTSLS1, NeuralCoordLS,
)
from encoder.operators.cmaes_heads import BatchedDiffCMAES, NeuralCMAES
from encoder.operators.noop import BatchedNoOp, BatchedDiffSBX
from encoder.operators.donor_selection import DonorSelectionGATv2

__all__ = [
    '_ParamMLP', '_make_proj',
    'BatchedDiffDE', 'BatchedDiffAttDE', 'BatchedDiffLSHADE',
    'BatchedDiffCoordLS', 'BatchedDiffMTSLS1', 'NeuralCoordLS',
    'BatchedDiffCMAES', 'NeuralCMAES',
    'BatchedNoOp', 'BatchedDiffSBX',
    'DonorSelectionGATv2',
]
