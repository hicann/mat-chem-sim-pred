# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import numpy as np
import pytest
from reference import is_supported, reference


def test_reference_matches_edgepool_order():
    order = np.asarray([2, 0, 1], dtype=np.int32)
    edge_index = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    score = np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
    cluster, selected, cluster_score, counts = reference(order, edge_index, score, 4)
    assert cluster.tolist() == [1, 1, 0, 0]
    assert selected[:2].tolist() == [2, 0]
    assert cluster_score[:2].tolist() == pytest.approx([0.4, 0.2])
    assert counts.tolist() == [2, 2]


def test_reference_handles_self_loop_and_singleton():
    order = np.asarray([0, 1], dtype=np.int32)
    edge_index = np.asarray([[0, 1], [0, 2]], dtype=np.int32)
    score = np.asarray([0.7, 0.2], dtype=np.float32)
    cluster, selected, cluster_score, counts = reference(order, edge_index, score, 3)
    assert cluster.tolist() == [0, 1, 1]
    assert selected[:2].tolist() == [0, 1]
    assert cluster_score[:2].tolist() == pytest.approx([0.7, 0.2])
    assert counts.tolist() == [2, 2]


def test_reference_keeps_unmatched_singleton():
    cluster, selected, cluster_score, counts = reference(
        np.asarray([0], dtype=np.int32),
        np.asarray([[0], [0]], dtype=np.int32),
        np.asarray([0.7], dtype=np.float32),
        2,
    )
    assert cluster.tolist() == [0, 1]
    assert selected[:1].tolist() == [0]
    assert cluster_score.tolist() == pytest.approx([0.7, 1.0])
    assert counts.tolist() == [1, 2]


def test_rejects_bad_permutation_and_indices():
    edge_index = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    score = np.asarray([0.1, 0.2], dtype=np.float32)
    assert not is_supported(np.asarray([0, 0], dtype=np.int32), edge_index, score, 2)
    assert not is_supported(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([[0, 2], [1, 0]], dtype=np.int32),
        score,
        2,
    )
