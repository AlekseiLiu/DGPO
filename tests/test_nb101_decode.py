"""Decoding of NB101 cells stored with trailing `output` padding."""
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import src
from src.rewards import _nb101_to_adj_ops
from vendored.nasbench.lib.model_spec import ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ["conv3x3-bn-relu", "conv1x1-bn-relu", "maxpool3x3"]
NUM_VERTICES = 7

# Op indices (src/ops.py): output=0, input=1, conv1x1=2, conv3x3=3, maxpool=4
OUT, INP, C1, C3, MP = 0, 1, 2, 3, 4
NAMES = ["output", "input", "conv1x1-bn-relu", "conv3x3-bn-relu", "maxpool3x3"]


def make_matrix(n, edges):
    m = np.zeros((n, n), dtype=int)
    for i, j in edges:
        m[i, j] = 1
    return m


def pad_like_loader(matrix, idx):
    """Independent copy of the dataset padding: `output` slots, zero matrix."""
    n = len(idx)
    idx = list(idx) + [OUT] * (NUM_VERTICES - n)
    padded = np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=int)
    padded[:n, :n] = matrix
    return padded, idx


def decode(matrix_upper, idx):
    sym = matrix_upper + matrix_upper.T
    return _nb101_to_adj_ops(torch.tensor(idx), torch.tensor(sym))


def spec_of(matrix, ops):
    return ModelSpec(matrix=matrix, ops=ops)


# Valid cells with n = 2..6 nodes: (indices, edges)
SMALL_CELLS = {
    2: ([INP, OUT], [(0, 1)]),
    3: ([INP, C3, OUT], [(0, 1), (1, 2)]),
    4: ([INP, C1, C3, OUT], [(0, 1), (1, 2), (2, 3), (0, 3)]),
    5: ([INP, C3, C1, MP, OUT], [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]),
    6: ([INP, C3, C1, MP, C3, OUT],
        [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]),
}


def test_imports_repo_under_test():
    assert Path(src.__file__).resolve().is_relative_to(REPO_ROOT)


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_padded_cell_round_trip(n):
    idx, edges = SMALL_CELLS[n]
    matrix = make_matrix(n, edges)
    ops = [NAMES[i] for i in idx]
    assert len(edges) <= 9 and ops[0] == "input" and ops[-1] == "output"
    original = spec_of(matrix, ops)
    assert original.valid_spec, "fixture must be a valid NB101 cell"

    padded, pidx = pad_like_loader(matrix, idx)
    assert padded.shape == (NUM_VERTICES, NUM_VERTICES) and len(pidx) == NUM_VERTICES

    dm, dops = decode(padded, pidx)
    spec = spec_of(dm, dops)
    assert spec.valid_spec
    assert spec.hash_spec(CANONICAL) == original.hash_spec(CANONICAL)


SEVEN_NODE = [
    ([INP, C1, C3, MP, C1, C3, OUT],
     [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (0, 6)]),
    ([INP, C3, C3, C1, MP, C1, OUT],
     [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)]),
]


@pytest.mark.parametrize("idx,edges", SEVEN_NODE)
def test_seven_node_cell_unchanged(idx, edges):
    matrix = make_matrix(NUM_VERTICES, edges)
    ops = [NAMES[i] for i in idx]
    assert spec_of(matrix, ops).valid_spec
    dm, dops = decode(matrix, idx)
    assert np.array_equal(dm, np.triu(matrix + matrix.T, 1))
    assert dops == ops


def test_decode_does_not_mutate_inputs():
    idx, edges = SMALL_CELLS[5]
    padded, pidx = pad_like_loader(make_matrix(5, edges), idx)
    nodes = torch.tensor(pidx)
    adj = torch.tensor(padded + padded.T)
    nodes_before, adj_before = nodes.clone(), adj.clone()
    _nb101_to_adj_ops(nodes, adj)
    assert torch.equal(nodes, nodes_before)
    assert torch.equal(adj, adj_before)


def test_trailing_output_with_edges_not_trimmed():
    # Slot 5 is an edgeless `output` label, but slot 6 has an incoming edge.
    idx = [INP, C1, C3, MP, C1, OUT, OUT]
    matrix = make_matrix(NUM_VERTICES, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 6)])
    dm, dops = decode(matrix, idx)
    assert dm.shape == (NUM_VERTICES, NUM_VERTICES)
    assert np.array_equal(dm, matrix)
    assert dops == [NAMES[i] for i in idx[:-1]] + ["output"]


def test_no_trim_when_remaining_last_is_not_connected_output():
    idx = [INP, C1, C1, C1, OUT, OUT, OUT]
    matrix = make_matrix(NUM_VERTICES, [(0, 1), (1, 2), (2, 3)])
    dm, dops = decode(matrix, idx)
    assert len(dops) == NUM_VERTICES and dm.shape == (NUM_VERTICES, NUM_VERTICES)
    assert dops[3] == "conv1x1-bn-relu"  # not relabelled as output
    assert not spec_of(dm, dops).valid_spec


def test_all_edgeless_stays_invalid():
    idx = [INP] + [OUT] * 6
    dm, dops = decode(np.zeros((NUM_VERTICES, NUM_VERTICES), dtype=int), idx)
    assert len(dops) == NUM_VERTICES
    assert not spec_of(dm, dops).valid_spec


def test_unused_middle_slot_still_pruned():
    idx = [INP, C1, C3, MP, C1, C3, OUT]
    matrix = make_matrix(NUM_VERTICES, [(0, 1), (1, 2), (2, 4), (4, 5), (5, 6)])
    dm, dops = decode(matrix, idx)
    assert len(dops) == NUM_VERTICES
    spec = spec_of(dm, dops)
    assert spec.valid_spec
    assert spec.matrix.shape[0] == 6


@pytest.mark.slow
def test_padded_cells_are_in_nb101():
    path = os.environ.get("NB101_TFRECORD")
    if not path:
        pytest.skip("NB101_TFRECORD not set")
    assert os.path.isfile(path), f"NB101_TFRECORD is not a file: {path}"
    from vendored.nasbench.api import NASBench

    api = NASBench(path)
    checked = 0
    for h in api.hash_iterator():
        fixed, _ = api.get_metrics_from_hash(h)
        matrix, ops = fixed["module_adjacency"], fixed["module_operations"]
        if len(ops) >= NUM_VERTICES:
            continue
        idx = [NAMES.index(o) for o in ops]
        padded, pidx = pad_like_loader(matrix, idx)
        dm, dops = decode(padded, pidx)
        spec = spec_of(dm, dops)
        assert api.is_valid(spec), (h, dops)
        assert api._hash_spec(spec) == h
        checked += 1
        if checked == 200:
            break
    assert checked == 200
