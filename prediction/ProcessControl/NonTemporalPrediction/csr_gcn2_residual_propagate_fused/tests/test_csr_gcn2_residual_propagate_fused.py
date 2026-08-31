# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import sys
from pathlib import Path

import numpy as np
import pytest
from reference import csr_gcn2_residual_propagate

sys.path.insert(0, str(Path(__file__).parents[2]))
from spectral_test_helpers import check_residual_invalid


def test_two_node_residual_propagation() -> None:
    row_ptr = np.array([0, 1, 2], dtype=np.int32)
    source = np.array([1, 0], dtype=np.int32)
    weight = np.ones(2, dtype=np.float32)
    current = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    initial = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    actual = csr_gcn2_residual_propagate(
        row_ptr, source, weight, current, initial, 0.25
    )
    expected = 0.75 * current[::-1] + 0.25 * initial
    np.testing.assert_allclose(actual, expected)


def test_weighted_rows_and_empty_row() -> None:
    row_ptr = np.array([0, 2, 2, 3], dtype=np.int32)
    source = np.array([0, 1, 2], dtype=np.int32)
    weight = np.array([0.25, 0.75, -1.0], dtype=np.float32)
    current = np.arange(6, dtype=np.float32).reshape(3, 2)
    initial = np.ones_like(current)
    actual = csr_gcn2_residual_propagate(row_ptr, source, weight, current, initial, 0.1)
    np.testing.assert_allclose(actual[1], 0.1)
    assert np.isfinite(actual).all()


def test_alpha_boundaries() -> None:
    row_ptr = np.array([0, 1], dtype=np.int32)
    source = np.array([0], dtype=np.int32)
    weight = np.array([2.0], dtype=np.float32)
    current = np.array([[3.0]], dtype=np.float32)
    initial = np.array([[7.0]], dtype=np.float32)
    np.testing.assert_allclose(
        csr_gcn2_residual_propagate(row_ptr, source, weight, current, initial, 0.0),
        6.0,
    )
    np.testing.assert_allclose(
        csr_gcn2_residual_propagate(row_ptr, source, weight, current, initial, 1.0),
        7.0,
    )


def test_invalid_csr_is_rejected() -> None:
    check_residual_invalid(csr_gcn2_residual_propagate)


def test_non_finite_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        csr_gcn2_residual_propagate(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([np.inf], dtype=np.float32),
            np.ones((1, 1), np.float32),
            np.ones((1, 1), np.float32),
            0.1,
        )
    with pytest.raises(ValueError):
        csr_gcn2_residual_propagate(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones(1, np.float32),
            np.ones((1, 1), np.float32),
            np.ones((1, 1), np.float32),
            np.nan,
        )
