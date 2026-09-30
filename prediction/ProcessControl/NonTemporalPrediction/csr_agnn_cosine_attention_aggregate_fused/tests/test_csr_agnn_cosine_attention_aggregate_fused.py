# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Reference tests for CsrAgnnCosineAttentionAggregateFused."""

import numpy as np
import pytest
from reference import csr_agnn_cosine_attention_aggregate_fused as reference


def valid_inputs():
    features = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], np.float32)
    norm = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    return (
        np.array([0, 2, 3, 5], np.int32),
        np.array([0, 1, 1, 0, 2], np.int32),
        features,
        norm,
    )


def test_matches_direct_softmax():
    row, source, features, norm = valid_inputs()
    output = reference(row, source, features, norm, 1.5)
    logits = 1.5 * (norm[0] * norm[[0, 1]]).sum(-1)
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    np.testing.assert_allclose(
        output[0], (weights[:, None] * features[[0, 1]]).sum(0), rtol=1e-6
    )


def test_isolated_row_is_zero():
    features = np.ones((2, 2), np.float32)
    output = reference(
        np.array([0, 0, 1], np.int32), np.array([0], np.int32), features, features, 1.0
    )
    np.testing.assert_array_equal(output[0], 0.0)


@pytest.mark.parametrize(
    "field", ["row_dtype", "feature_dtype", "shape", "endpoint", "source", "beta"]
)
def test_invalid_inputs(field):
    row, source, features, norm = valid_inputs()
    beta = 1.0
    if field == "row_dtype":
        row = row.astype(np.int64)
    elif field == "feature_dtype":
        features = features.astype(np.float64)
    elif field == "shape":
        norm = norm[:, :1]
    elif field == "endpoint":
        row[-1] -= 1
    elif field == "source":
        source[0] = 9
    else:
        beta = np.inf
    with pytest.raises((TypeError, ValueError)):
        reference(row, source, features, norm, beta)
