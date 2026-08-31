# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from typing import NamedTuple

import numpy as np
from reference import is_supported, reference


class HypergraphInputs(NamedTuple):
    edge_row_ptr: np.ndarray
    node_index: np.ndarray
    edge_scale: np.ndarray
    node_row_ptr: np.ndarray
    edge_index: np.ndarray
    incidence_position: np.ndarray
    node_scale: np.ndarray
    features: np.ndarray
    attention_logits: np.ndarray


def inputs():
    return HypergraphInputs(
        np.array([0, 2, 4], np.int32),
        np.array([0, 1, 1, 2], np.int32),
        np.array([0.5, 0.5], np.float32),
        np.array([0, 1, 3, 4], np.int32),
        np.array([0, 0, 1, 1], np.int32),
        np.array([0, 1, 2, 3], np.int32),
        np.array([1.0, 0.5, 1.0], np.float32),
        np.arange(12, dtype=np.float32).reshape(3, 2, 2) / 10,
        np.array([[0.1, 0.2], [0.4, -0.2], [0.3, 0.1], [-0.1, 0.5]], np.float32),
    )


def test_attention_reused_across_both_propagation_stages():
    actual = reference(*inputs())
    assert actual.shape == (3, 2, 2)
    assert np.all(np.isfinite(actual))
    assert not np.allclose(actual[0], actual[2])


def test_dispatch_guard_rejects_mapping_and_training():
    assert not is_supported(*inputs(), requires_grad=True)
    values = list(inputs())
    values[5] = np.array([0, 1, 2, 4], np.int32)
    assert not is_supported(*values)
