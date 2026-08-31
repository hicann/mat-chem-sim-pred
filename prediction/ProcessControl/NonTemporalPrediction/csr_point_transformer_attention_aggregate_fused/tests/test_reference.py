# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from typing import NamedTuple

import numpy as np
from reference import is_supported, reference


class Inputs(NamedTuple):
    row_ptr: np.ndarray
    source: np.ndarray
    alpha_source: np.ndarray
    alpha_target: np.ndarray
    value: np.ndarray
    delta: np.ndarray


def inputs():
    row_ptr = np.array([0, 2, 3], np.int32)
    source = np.array([0, 1, 1], np.int32)
    alpha_source = np.arange(6, dtype=np.float32).reshape(2, 3) / 7
    alpha_target = alpha_source + 0.3
    value = alpha_source - 0.2
    delta = np.arange(9, dtype=np.float32).reshape(3, 3) / 11
    return Inputs(row_ptr, source, alpha_source, alpha_target, value, delta)


def test_channelwise_softmax_and_value_delta_message():
    actual = reference(*inputs())
    assert actual.shape == (2, 3)
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual[1], inputs()[4][1] + inputs()[5][2])


def test_dispatch_guard_rejects_shape_dtype_and_training():
    assert not is_supported(*inputs(), requires_grad=True)
    values = list(inputs())
    values[-1] = values[-1].astype(np.float16)
    assert not is_supported(*values)
    values = list(inputs())
    values[-1] = values[-1][:-1]
    assert not is_supported(*values)
