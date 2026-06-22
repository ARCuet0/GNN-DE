"""
similarity_graph_gpu.py — Re-export shim.

All graph builder implementations have been split into:
  - graph_utils.py                (shared helpers)
  - graph_builder_single.py       (single-pop + list-batched builders)
  - graph_builder_batched.py      (uniform-batched builder)
  - graph_builder_dense.py        (dense builder → TopologyCache)
  - graph_builder_sparse.py       (sparse builder → SparseTopologyCache)

This module re-exports every public name so existing imports keep working.
"""

from .graph_builder_single import (  # noqa: F401
    build_similarity_graph_gpu,
    build_batched_similarity_graphs_gpu,
)
from .graph_builder_batched import (  # noqa: F401
    build_batched_uniform_graphs_gpu,
)
from .graph_builder_dense import (  # noqa: F401
    build_dense_graphs_gpu,
)
from .graph_builder_sparse import (  # noqa: F401
    build_sparse_graphs_gpu,
)
from .dense_gatv2_backbone import TopologyCache  # noqa: F401
from .sparse_gatv2_backbone import SparseTopologyCache  # noqa: F401
