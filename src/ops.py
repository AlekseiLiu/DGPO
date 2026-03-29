"""Op primitives and constraints for NB101 and NB201."""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

# --- NB101 ---
NB101_OP_PRIMITIVES: List[str] = [
    "output",
    "input",
    "conv1x1-bn-relu",
    "conv3x3-bn-relu",
    "maxpool3x3",
]

NB101_NAME_TO_IDX = {name: idx for idx, name in enumerate(NB101_OP_PRIMITIVES)}
NB101_IDX_TO_NAME = {idx: name for name, idx in NB101_NAME_TO_IDX.items()}
NB101_NUM_VERTICES = 7
NB101_MAX_EDGES = 9

# --- NB201 ---
NB201_OP_PRIMITIVES: List[str] = [
    "output",
    "input",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
    "skip_connect",
    "none",
]

NB201_NAME_TO_IDX = {name: idx for idx, name in enumerate(NB201_OP_PRIMITIVES)}
NB201_IDX_TO_NAME = {idx: name for name, idx in NB201_NAME_TO_IDX.items()}
NB201_NUM_VERTICES = 8
NB201_MAX_EDGES = 28

# NB201 fixed adjacency matrix (DAG, upper triangular)
ADJACENCY_NB201 = np.array([
    [0, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.int32)


def upper_triangular(adjacency: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Extract upper triangle of adjacency matrix, zero diagonal."""
    matrix = np.asarray(adjacency).astype(int, copy=False)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"adjacency must be square, got shape {matrix.shape}")
    upper = np.triu(matrix, k=1)
    np.fill_diagonal(upper, 0)
    return upper


def indices_to_names(indices: Sequence[int], benchmark: str) -> List[str]:
    """Convert op indices to op names for the given benchmark."""
    idx_to_name = NB101_IDX_TO_NAME if benchmark == "nb101" else NB201_IDX_TO_NAME
    return [idx_to_name.get(int(idx), "unknown") for idx in indices]


def names_to_indices(names: Sequence[str], benchmark: str) -> List[int]:
    """Convert op names to op indices for the given benchmark."""
    name_to_idx = NB101_NAME_TO_IDX if benchmark == "nb101" else NB201_NAME_TO_IDX
    return [name_to_idx.get(name, -1) for name in names]


def ensure_io_ops(ops: List[str]) -> List[str]:
    """Force first node = input, last node = output."""
    if not ops:
        return ops
    ops = list(ops)
    ops[0] = "input"
    if len(ops) >= 2:
        ops[-1] = "output"
    return ops
