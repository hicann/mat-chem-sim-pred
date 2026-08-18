# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import numpy as np
import pytest

from reference import csr_film_modulated_mean_fused, is_supported


def inputs():
    return (
        np.array([0, 2, 3, 3], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([[1, -2], [3, 4], [5, 6]], dtype=np.float32),
        np.array([[0.5, 1], [-1, 0.5], [2, 2]], dtype=np.float32),
        np.array([[2, -1], [0.5, 2], [1, 1]], dtype=np.float32),
    )


def test_relu_modulation_and_mean():
    actual = csr_film_modulated_mean_fused(*inputs(), True)
    expected = np.array([[4.5, 1.5], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_activation_can_be_disabled():
    actual = csr_film_modulated_mean_fused(*inputs(), False)
    expected = np.array([[4.5, 0.0], [-0.5, -3.5], [0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)


def test_empty_row_returns_zero():
    actual = csr_film_modulated_mean_fused(*inputs())
    np.testing.assert_array_equal(actual[2], np.zeros(2, dtype=np.float32))


def test_random_matches_direct_loop():
    rng = np.random.default_rng(17)
    row_ptr = np.array([0, 2, 5, 6], dtype=np.int32)
    source = np.array([0, 2, 1, 2, 0, 1], dtype=np.int32)
    projected = rng.normal(size=(3, 8)).astype(np.float32)
    beta = rng.normal(size=(3, 8)).astype(np.float32)
    gamma = rng.normal(size=(3, 8)).astype(np.float32)
    actual = csr_film_modulated_mean_fused(
        row_ptr, source, projected, beta, gamma, True
    )
    expected = np.zeros_like(projected)
    for target in range(3):
        values = []
        for edge in range(row_ptr[target], row_ptr[target + 1]):
            values.append(
                np.maximum(gamma[target] * projected[source[edge]] + beta[target], 0)
            )
        expected[target] = np.mean(values, axis=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "position,replacement",
    [
        (0, np.array([1, 2, 3, 3], dtype=np.int32)),
        (0, np.array([0, 3, 2, 3], dtype=np.int32)),
        (1, np.array([0, 3, 0], dtype=np.int32)),
    ],
)
def test_malformed_graph_rejected(position, replacement):
    values = list(inputs())
    values[position] = replacement
    assert not is_supported(*values)
    with pytest.raises(ValueError):
        csr_film_modulated_mean_fused(*values)


def test_shape_dtype_and_activation_fall_back():
    values = list(inputs())
    values[2] = values[2].astype(np.float16)
    assert not is_supported(*values)
    values = list(inputs())
    values[3] = values[3][:-1]
    assert not is_supported(*values)
    assert not is_supported(*inputs(), apply_relu=1)


def test_training_and_non_contiguous_fall_back():
    assert not is_supported(*inputs(), requires_grad=True)
    values = list(inputs())
    values[2] = np.zeros((3, 4), dtype=np.float32)[:, ::2]
    assert not is_supported(*values)


def test_non_finite_rejected():
    values = list(inputs())
    values[4] = values[4].copy()
    values[4][0, 0] = np.inf
    assert not is_supported(*values)


def test_max_segment_and_channel_limits():
    edges = 2049
    assert not is_supported(
        np.array([0, edges], dtype=np.int32),
        np.zeros(edges, dtype=np.int32),
        np.ones((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
    )
    values = list(inputs())
    values[2] = np.ones((3, 513), dtype=np.float32)
    values[3] = np.ones((3, 513), dtype=np.float32)
    values[4] = np.ones((3, 513), dtype=np.float32)
    assert not is_supported(*values)
