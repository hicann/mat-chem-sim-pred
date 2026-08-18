# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Contract tests for CsrSignedCrossMeanPackFused."""

from dataclasses import dataclass

import numpy as np
import pytest

from reference import csr_signed_cross_mean_pack_fused as reference


@dataclass
class Inputs:
    positive_row: np.ndarray
    positive_source: np.ndarray
    negative_row: np.ndarray
    negative_source: np.ndarray
    features: np.ndarray
    positive_inverse: np.ndarray
    negative_inverse: np.ndarray

    def __iter__(self):
        return iter(vars(self).values())


def valid_inputs():
    positive_row = np.array([0, 2, 3], np.int32)
    positive_source = np.array([0, 1, 0], np.int32)
    negative_row = np.array([0, 1, 3], np.int32)
    negative_source = np.array([1, 0, 1], np.int32)
    features = np.concatenate(
        (
            np.array([[1.0] * 8, [2.0] * 8], np.float32),
            np.array([[10.0] * 8, [20.0] * 8], np.float32),
        ),
        axis=1,
    )
    positive_inverse = np.array([0.5, 1.0], np.float32)
    negative_inverse = np.array([1.0, 0.5], np.float32)
    return Inputs(
        positive_row,
        positive_source,
        negative_row,
        negative_source,
        features,
        positive_inverse,
        negative_inverse,
    )


def test_cross_sign_pack_order_and_means():
    output = reference(*valid_inputs())
    expected = np.array(
        [
            [1.5] * 8 + [20.0] * 8 + [15.0] * 8 + [2.0] * 8,
            [1.0] * 8 + [15.0] * 8 + [10.0] * 8 + [1.5] * 8,
        ],
        np.float32,
    )
    np.testing.assert_allclose(output, expected)


def _astype(value, dtype):
    return value.astype(dtype)


def _reshape(value, shape):
    return value.reshape(shape)


def _truncate(value, stop):
    return value[:stop]


def _ones(_value, shape):
    return np.ones(shape, np.float32)


def _set_item(value, index, replacement):
    value[index] = replacement
    return value


@pytest.mark.parametrize(
    "position,mutation,arguments",
    [
        (0, _astype, (np.int64,)),
        (4, _astype, (np.float64,)),
        (5, _astype, (np.float64,)),
        (4, _reshape, (-1,)),
        (4, _ones, ((2, 14),)),
        (0, _truncate, (-1,)),
        (0, _set_item, (-1, 2)),
        (0, _set_item, (1, 4)),
        (1, _set_item, (0, 3)),
        (5, _truncate, (-1,)),
        (5, _set_item, (0, np.nan)),
    ],
)
def test_invalid_inputs(position, mutation, arguments):
    values = list(valid_inputs())
    values[position] = mutation(values[position], *arguments)
    with pytest.raises((TypeError, ValueError)):
        reference(*values)
