# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Reference tests for CsrLightgcnK2WeightedSumFused."""

import numpy as np
import pytest

from reference import csr_lightgcn_k2_weighted_sum_fused as reference


def valid_inputs():
    return (
        np.array([0, 1, 3], np.int32),
        np.array([1, 0, 1], np.int32),
        np.array([0.5, 0.25, 0.75], np.float32),
        np.array([[1.0, 2.0], [3.0, 4.0]], np.float32),
    )


def test_two_hop_weighted_sum():
    row, source, norm, features = valid_inputs()
    output = reference(row, source, norm, features, 0.2, 0.3, 0.5)
    layer1 = np.array([[1.5, 2.0], [2.5, 3.5]], np.float32)
    layer2 = np.array([[1.25, 1.75], [2.25, 3.125]], np.float32)
    np.testing.assert_allclose(
        output, 0.2 * features + 0.3 * layer1 + 0.5 * layer2, rtol=1e-6
    )


@pytest.mark.parametrize(
    "field", ["csr_dtype", "value_dtype", "shape", "endpoint", "source", "alpha"]
)
def test_invalid_inputs(field):
    row, source, norm, features = valid_inputs()
    alpha = [1 / 3] * 3
    if field == "csr_dtype":
        source = source.astype(np.int64)
    elif field == "value_dtype":
        norm = norm.astype(np.float64)
    elif field == "shape":
        norm = norm[:-1]
    elif field == "endpoint":
        row[-1] = 2
    elif field == "source":
        source[0] = 3
    else:
        alpha[1] = np.nan
    with pytest.raises((TypeError, ValueError)):
        reference(row, source, norm, features, *alpha)
