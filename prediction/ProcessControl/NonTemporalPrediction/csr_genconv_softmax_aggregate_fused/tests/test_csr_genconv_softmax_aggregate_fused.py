# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np
import pytest
from reference import csr_genconv_softmax_aggregate


def test_matches_manual_featurewise_softmax() -> None:
    row_ptr = np.array([0, 2, 3], dtype=np.int32)
    source = np.array([0, 1, 1], dtype=np.int32)
    features = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    actual = csr_genconv_softmax_aggregate(row_ptr, source, features, 1.0)
    messages = np.maximum(features, 0.0) + np.float32(1.0e-7)
    weights = np.exp(messages - messages.max(axis=0, keepdims=True))
    expected_first = (weights / weights.sum(axis=0, keepdims=True) * messages).sum(
        axis=0
    )
    np.testing.assert_allclose(actual[0], expected_first, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual[1], messages[1], rtol=1e-6, atol=1e-6)


def test_large_logits_are_stable() -> None:
    row_ptr = np.array([0, 3, 3, 3], dtype=np.int32)
    source = np.array([0, 1, 2], dtype=np.int32)
    features = np.array([[1000.0], [1001.0], [1002.0]], dtype=np.float32)
    actual = csr_genconv_softmax_aggregate(row_ptr, source, features, 3.0)
    assert np.isfinite(actual).all()
    assert 1001.0 < actual[0, 0] <= 1002.0 + 1e-4


def test_empty_segment_returns_zero() -> None:
    row_ptr = np.array([0, 0, 1], dtype=np.int32)
    source = np.array([1], dtype=np.int32)
    features = np.array([[1.0], [2.0]], dtype=np.float32)
    actual = csr_genconv_softmax_aggregate(row_ptr, source, features, 1.0)
    assert actual[0, 0] == 0.0
    np.testing.assert_allclose(actual[1], np.array([2.0], dtype=np.float32))


def test_negative_temperature_is_supported() -> None:
    row_ptr = np.array([0, 2, 2], dtype=np.int32)
    source = np.array([0, 1], dtype=np.int32)
    features = np.array([[1.0], [4.0]], dtype=np.float32)
    positive = csr_genconv_softmax_aggregate(row_ptr, source, features, 2.0)
    negative = csr_genconv_softmax_aggregate(row_ptr, source, features, -2.0)
    assert positive[0, 0] > negative[0, 0]


@pytest.mark.parametrize(
    "row_ptr,source",
    [
        (np.array([0, 2, 1], dtype=np.int32), np.array([0], dtype=np.int32)),
        (np.array([0, 1, 1], dtype=np.int32), np.array([2], dtype=np.int32)),
    ],
)
def test_invalid_csr_is_rejected(row_ptr: np.ndarray, source: np.ndarray) -> None:
    features = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        csr_genconv_softmax_aggregate(row_ptr, source, features, 1.0)


def test_non_finite_temperature_is_rejected() -> None:
    with pytest.raises(ValueError):
        csr_genconv_softmax_aggregate(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones((1, 1), dtype=np.float32),
            float("inf"),
        )
