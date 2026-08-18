# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Reference tests for CsrTransformerDotAttentionAggregateFused."""

import numpy as np
import pytest

from reference import csr_transformer_dot_attention_aggregate_fused as reference


def valid_inputs():
    row = np.array([0, 2, 3], np.int32)
    source = np.array([0, 1, 0], np.int32)
    query = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], np.float32)
    key = np.array([[[1.0, 0.0]], [[2.0, 0.0]]], np.float32)
    value = np.array([[[1.0, 3.0]], [[5.0, 7.0]]], np.float32)
    return row, source, query, key, value


def test_dot_attention_weighted_value():
    output = reference(*valid_inputs())
    logits = np.array([1.0, 2.0], np.float32) / np.sqrt(np.float32(2.0))
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    expected = weights @ np.array([[1.0, 3.0], [5.0, 7.0]], np.float32)
    np.testing.assert_allclose(output[0, 0], expected, rtol=1e-6)


@pytest.mark.parametrize(
    "field", ["csr_dtype", "value_dtype", "shape", "endpoint", "source", "channels"]
)
def test_invalid_inputs(field):
    row, source, query, key, value = valid_inputs()
    if field == "csr_dtype":
        source = source.astype(np.int64)
    elif field == "value_dtype":
        query = query.astype(np.float64)
    elif field == "shape":
        key = key[:, :, :1]
    elif field == "endpoint":
        row[-1] = 2
    elif field == "source":
        source[0] = 3
    else:
        query = np.zeros((2, 1, 33), np.float32)
        key = value = query.copy()
    with pytest.raises((TypeError, ValueError)):
        reference(row, source, query, key, value)
