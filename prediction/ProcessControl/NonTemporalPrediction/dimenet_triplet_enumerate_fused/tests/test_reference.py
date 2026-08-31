# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REFERENCE_PATH = Path(__file__).parents[1] / "reference" / "reference.py"
SPEC = importlib.util.spec_from_file_location(
    "dimenet_triplet_reference", REFERENCE_PATH
)
REFERENCE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(REFERENCE)


def test_chain_excludes_backtracking_triplet():
    row_ptr = np.asarray([0, 0, 2, 3], dtype=np.int32)
    source_index = np.asarray([0, 2, 1], dtype=np.int32)
    *indices, counts = REFERENCE.dimenet_triplet_enumerate_fused(
        row_ptr, source_index, capacity=4
    )
    assert counts.tolist() == [1, 0]
    assert [value.tolist() for value in indices] == [[2], [1], [0], [0], [2]]
    assert REFERENCE.required_capacity(row_ptr, source_index) == 1


def test_capacity_reports_overflow_without_out_of_bounds_write():
    row_ptr = np.asarray([0, 0, 2, 4, 6], dtype=np.int32)
    source_index = np.asarray([2, 3, 0, 3, 0, 1], dtype=np.int32)
    *indices, counts = REFERENCE.dimenet_triplet_enumerate_fused(
        row_ptr, source_index, capacity=2
    )
    assert counts.tolist() == [2, 1]
    assert all(value.shape == (2,) for value in indices)


def test_random_graph_matches_direct_edge_pair_definition():
    rng = np.random.default_rng(20260817)
    nodes = 11
    target = rng.integers(nodes, size=47)
    source = rng.integers(nodes, size=47)
    order = np.lexsort((source, target))
    target, source = target[order], source[order]
    row_ptr = np.zeros(nodes + 1, dtype=np.int32)
    np.add.at(row_ptr, target + 1, 1)
    np.cumsum(row_ptr, out=row_ptr)
    source_index = source.astype(np.int32)
    capacity = 47 * 47
    *indices, counts = REFERENCE.dimenet_triplet_enumerate_fused(
        row_ptr, source_index, capacity
    )
    target_by_edge = np.repeat(np.arange(nodes, dtype=np.int32), np.diff(row_ptr))
    for i, j, k, edge_kj, edge_ji in zip(*indices):
        assert target_by_edge[edge_ji] == i
        assert source_index[edge_ji] == j
        assert target_by_edge[edge_kj] == j
        assert source_index[edge_kj] == k
        assert i != k
    assert counts[0] == REFERENCE.required_capacity(row_ptr, source_index)
    assert counts[1] == 0
