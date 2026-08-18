# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import numpy as np
import pytest

from reference import csr_gat_attention_aggregate


def test_two_neighbor_softmax() -> None:
    actual = csr_gat_attention_aggregate(
        np.array([0, 2, 3], dtype=np.int32),
        np.array([0, 1, 0], dtype=np.int32),
        np.array([[[1.0]], [[3.0]]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
        np.array([[0.0]], dtype=np.float32),
        0.2,
    )
    np.testing.assert_allclose(actual[:, 0, 0], [2.761594, 1.0], rtol=1.0e-6)


def test_multi_head_multi_channel_shape() -> None:
    row_ptr = np.array([0, 1, 2], dtype=np.int32)
    source = np.array([1, 0], dtype=np.int32)
    projected = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    attention = np.ones((2, 2), dtype=np.float32)
    actual = csr_gat_attention_aggregate(
        row_ptr, source, projected, attention, attention, 0.2
    )
    np.testing.assert_allclose(actual[0], projected[1])
    np.testing.assert_allclose(actual[1], projected[0])


def test_empty_row_returns_zero() -> None:
    actual = csr_gat_attention_aggregate(
        np.array([0, 0, 1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.ones((2, 1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        np.ones((1, 1), dtype=np.float32),
        0.2,
    )
    np.testing.assert_allclose(actual[0], 0.0)


def test_negative_slope_changes_attention() -> None:
    row_ptr = np.array([0, 2, 2], dtype=np.int32)
    source = np.array([0, 1], dtype=np.int32)
    projected = np.array([[[-1.0]], [[-3.0]]], dtype=np.float32)
    attention_source = np.ones((1, 1), dtype=np.float32)
    attention_target = np.zeros((1, 1), dtype=np.float32)
    leaky = csr_gat_attention_aggregate(
        row_ptr, source, projected, attention_source, attention_target, 0.2
    )
    identity = csr_gat_attention_aggregate(
        row_ptr, source, projected, attention_source, attention_target, 1.0
    )
    assert not np.allclose(leaky, identity)


@pytest.mark.parametrize(
    "row_ptr,source",
    [
        (
            np.array([0, 2, 2], dtype=np.int32),
            np.array([0], dtype=np.int32),
        ),
        (
            np.array([0, 1, 1], dtype=np.int32),
            np.array([2], dtype=np.int32),
        ),
    ],
)
def test_invalid_csr_is_rejected(row_ptr, source) -> None:
    with pytest.raises(ValueError):
        csr_gat_attention_aggregate(
            row_ptr,
            source,
            np.ones((2, 1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            0.2,
        )


def test_attention_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        csr_gat_attention_aggregate(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones((1, 2, 2), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            0.2,
        )


def test_non_finite_and_invalid_slope_are_rejected() -> None:
    with pytest.raises(ValueError):
        csr_gat_attention_aggregate(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([[[np.inf]]], dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            0.2,
        )
    with pytest.raises(ValueError):
        csr_gat_attention_aggregate(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones((1, 1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            1.1,
        )
