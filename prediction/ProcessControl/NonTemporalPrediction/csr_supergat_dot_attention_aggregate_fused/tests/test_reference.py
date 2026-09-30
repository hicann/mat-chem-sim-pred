# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import numpy as np
from reference import is_supported, reference


def inputs():
    return (
        np.array([0, 2, 3], np.int32),
        np.array([0, 1, 1], np.int32),
        np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 10,
    )


def test_matches_manual_segment_attention():
    row_ptr, source, projected = inputs()
    actual = reference(row_ptr, source, projected)
    score = (projected[0] * projected[source[:2]]).sum(-1) / np.sqrt(3.0)
    score = np.where(score >= 0, score, score * 0.2)
    weight = np.exp(score - score.max(0, keepdims=True))
    weight /= weight.sum(0, keepdims=True)
    expected = (weight[..., None] * projected[source[:2]]).sum(0)
    np.testing.assert_allclose(actual[0], expected, atol=1e-6, rtol=1e-6)


def test_dispatch_guard_rejects_training_and_non_contiguous():
    assert not is_supported(*inputs(), requires_grad=True)
    values = list(inputs())
    values[2] = np.zeros((2, 2, 6), np.float32)[..., ::2]
    assert not is_supported(*values)


def test_dispatch_guard_rejects_oversized_row():
    assert not is_supported(
        np.array([0, 513], np.int32),
        np.zeros(513, np.int32),
        np.zeros((1, 1, 1), np.float32),
    )
