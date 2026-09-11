# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import importlib.util
from pathlib import Path

import numpy as np


def reference():
    spec = importlib.util.spec_from_file_location(
        "sage_reference", Path(__file__).parents[1] / "reference" / "reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mean_root_relu_deterministic():
    row = np.array([0, 2, 3], np.int32)
    column = np.array([0, 1, 0], np.int32)
    neighbor = np.array([[4, -1], [0, 3]], np.float32)
    root = np.array([[1, 2], [-2, -1]], np.float32)
    expected = np.array([[3, 3], [2, 0]], np.float32)
    actual = reference().csr_sage_mean_root_relu(row, column, neighbor, root)
    np.testing.assert_allclose(actual, expected)


def test_random_and_empty_neighbor_rows():
    generator = np.random.default_rng(20260801)
    row = np.array([0, 2, 2, 5, 6], np.int32)
    column = np.array([1, 3, 0, 1, 3, 2], np.int32)
    neighbor = generator.normal(size=(4, 8)).astype(np.float32)
    root = generator.normal(size=(4, 8)).astype(np.float32)

    def expected_node(node):
        start, end = row[node], row[node + 1]
        mean = neighbor[column[start:end]].mean(axis=0) if end > start else 0.0
        return root[node] + mean

    expected = np.maximum(
        np.stack([expected_node(node) for node in range(4)]),
        0.0,
    )
    actual = reference().csr_sage_mean_root_relu(row, column, neighbor, root)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def test_dispatch_boundary():
    dispatch = reference().use_custom_path
    assert dispatch(64, 256, 32, np.float32)
    assert not dispatch(63, 252, 32, np.float32)
    assert not dispatch(2708, 10556, 32, np.float16)


def test_host_api_contract():
    source = (
        Path(__file__).parents[1] / "op_host" / "csr_sage_mean_root_relu_fused_host.h"
    ).read_text()
    assert "aclnnCsrSageMeanRootReluFused" in source
    assert "neighbor_features" in source
    assert "root_features" in source
