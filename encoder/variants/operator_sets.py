"""Operator set configurations for NeuralK4Variant.

Each list defines which batched operator classes are instantiated as heads.
The list length determines K (number of operators).
"""
from encoder.batched_operators import (
    BatchedDiffDE, BatchedDiffAttDE, BatchedDiffCoordLS, BatchedDiffCMAES,
    BatchedNoOp, NeuralCoordLS, NeuralCMAES,
)
from encoder.direct_delta import BatchedDirectDelta

BATCHED_OPERATOR_CLASSES = [BatchedDiffDE, BatchedDiffCoordLS,
                            BatchedDiffCMAES, BatchedNoOp]

# K=5 hybrid: classical operators + learned direct delta (NoOp always last)
BATCHED_OPERATOR_CLASSES_K5 = [BatchedDiffDE, BatchedDiffCoordLS,
                               BatchedDiffCMAES, BatchedDirectDelta,
                               BatchedNoOp]

# K=4 with direct delta replacing NoOp
BATCHED_OPERATOR_CLASSES_DIRECT = [BatchedDiffDE, BatchedDiffCoordLS,
                                   BatchedDiffCMAES, BatchedDirectDelta]

# K=5 with attention-based DE (unified pbest/r1/r2 attention)
BATCHED_OPERATOR_CLASSES_K5_ATT = [BatchedDiffAttDE, BatchedDiffCoordLS,
                                   BatchedDiffCMAES, BatchedDirectDelta,
                                   BatchedNoOp]

# K=4 with neural-directed heads (learned dim scoring + mean shift)
BATCHED_OPERATOR_CLASSES_NEURAL = [BatchedDiffDE, NeuralCoordLS,
                                   NeuralCMAES, BatchedNoOp]

# K=5 neural + attention DE + direct delta
BATCHED_OPERATOR_CLASSES_NEURAL_ATT = [BatchedDiffAttDE, NeuralCoordLS,
                                       NeuralCMAES, BatchedDirectDelta,
                                       BatchedNoOp]

# K=4 with activity gate (no NoOp — gate handles elitism)
BATCHED_OPERATOR_CLASSES_GATED = [BatchedDiffAttDE, NeuralCoordLS,
                                  NeuralCMAES, BatchedDirectDelta]

# K=2: only the two functional heads (DE + Direct)
BATCHED_OPERATOR_CLASSES_K2 = [BatchedDiffAttDE, BatchedDirectDelta]

# K=1: DE only — multiple M-samples provide diversity
BATCHED_OPERATOR_CLASSES_K1 = [BatchedDiffAttDE]
