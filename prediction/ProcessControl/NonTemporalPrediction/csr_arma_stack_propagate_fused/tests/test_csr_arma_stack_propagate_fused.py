# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from typing import NamedTuple

import numpy as np
import pytest
from reference import csr_arma_stack_propagate


class ArmaInputs(NamedTuple):
    row_ptr: np.ndarray
    source: np.ndarray
    weight: np.ndarray
    projected: np.ndarray
    root: np.ndarray
    bias: np.ndarray


def sample_inputs():
    row_ptr = np.array([0, 1, 2], dtype=np.int32)
    source = np.array([1, 0], dtype=np.int32)
    weight = np.array([1.0, 0.5], dtype=np.float32)
    projected = np.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 2.0], [2.0, -3.0]]],
        dtype=np.float32,
    )
    root = np.array(
        [[[0.5, 0.5], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
        dtype=np.float32,
    )
    bias = np.array([[0.25, -0.25], [0.5, 0.5]], dtype=np.float32)
    return ArmaInputs(row_ptr, source, weight, projected, root, bias)


def test_weighted_stack_propagation_with_relu() -> None:
    actual = csr_arma_stack_propagate(*sample_inputs(), True)
    expected = np.array(
        [[[3.75, 4.25], [1.75, 1.75]], [[2.5, 0.0], [1.0, 1.5]]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(actual, expected)


def test_without_relu_preserves_negative_values() -> None:
    actual = csr_arma_stack_propagate(*sample_inputs(), False)
    np.testing.assert_allclose(actual[1, 0], [2.5, -1.5])


def test_empty_row_still_adds_root_and_bias() -> None:
    row_ptr = np.array([0, 0, 1], dtype=np.int32)
    source = np.array([0], dtype=np.int32)
    weight = np.ones(1, dtype=np.float32)
    projected = np.ones((1, 2, 2), dtype=np.float32)
    root = np.array([[[1.0, -2.0], [0.0, 0.0]]], dtype=np.float32)
    bias = np.array([[0.5, 0.5]], dtype=np.float32)
    actual = csr_arma_stack_propagate(
        row_ptr, source, weight, projected, root, bias, False
    )
    np.testing.assert_allclose(actual[0, 0], [1.5, -1.5])


@pytest.mark.parametrize(
    "row_ptr,source,weight",
    [
        (
            np.array([0, 2, 2], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones(1, dtype=np.float32),
        ),
        (
            np.array([0, 1, 1], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.ones(1, dtype=np.float32),
        ),
        (
            np.array([0, 1, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones(2, dtype=np.float32),
        ),
    ],
)
def test_invalid_csr_is_rejected(row_ptr, source, weight) -> None:
    projected = np.ones((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        csr_arma_stack_propagate(
            row_ptr,
            source,
            weight,
            projected,
            projected.copy(),
            np.zeros((1, 2), dtype=np.float32),
            False,
        )


def test_mismatched_stack_shapes_are_rejected() -> None:
    values = sample_inputs()
    with pytest.raises(ValueError):
        csr_arma_stack_propagate(
            values[0],
            values[1],
            values[2],
            values[3],
            values[4][:1],
            values[5],
            True,
        )


def test_non_finite_and_non_boolean_parameters_are_rejected() -> None:
    values = list(sample_inputs())
    values[2][0] = np.inf
    with pytest.raises(ValueError):
        csr_arma_stack_propagate(*values, True)
    values = sample_inputs()
    with pytest.raises(TypeError):
        csr_arma_stack_propagate(*values, 1)
