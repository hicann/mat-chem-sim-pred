# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Reference tests for CsrGatv2DynamicAttentionAggregateFused."""

import numpy as np
import pytest

from reference import csr_gatv2_dynamic_attention_aggregate_fused as reference


def valid_inputs():
    row = np.array([0, 2, 3], np.int32)
    source = np.array([0, 1, 0], np.int32)
    left = np.array([[[1.0, -1.0]], [[2.0, 1.0]]], np.float32)
    right = np.array([[[0.5, 0.5]], [[-0.5, 1.0]]], np.float32)
    attention = np.array([[0.75, -0.25]], np.float32)
    return row, source, left, right, attention


def test_stable_multi_channel_softmax():
    output = reference(*valid_inputs(), 0.2)
    assert output.shape == (2, 1, 2)
    assert np.isfinite(output).all()
    assert np.all(output[0, 0] >= np.array([1.0, -1.0]))


@pytest.mark.parametrize(
    "field", ["csr_dtype", "value_dtype", "shape", "endpoint", "source", "slope"]
)
def test_invalid_inputs(field):
    row, source, left, right, attention = valid_inputs()
    slope = 0.2
    if field == "csr_dtype":
        row = row.astype(np.int64)
    elif field == "value_dtype":
        left = left.astype(np.float64)
    elif field == "shape":
        attention = attention[:, :1]
    elif field == "endpoint":
        row[-1] = 2
    elif field == "source":
        source[0] = 4
    else:
        slope = 1.5
    with pytest.raises((TypeError, ValueError)):
        reference(row, source, left, right, attention, slope)
