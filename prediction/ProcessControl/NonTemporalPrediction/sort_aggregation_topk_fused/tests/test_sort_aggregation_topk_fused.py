# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import importlib.util
from pathlib import Path

import numpy as np


def reference():
    spec = importlib.util.spec_from_file_location(
        "sort_reference", Path(__file__).parents[1] / "reference" / "reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sort_and_zero_pad():
    ptr = np.array([0, 3, 5], np.int32)
    features = np.array([[1, 0.2], [2, 0.8], [3, -0.1], [4, 0.4], [5, 0.3]], np.float32)
    expected = np.array([[2, 0.8, 1, 0.2, 3, -0.1], [4, 0.4, 5, 0.3, 0, 0]], np.float32)
    np.testing.assert_allclose(
        reference().sort_aggregation_topk(ptr, features, 3), expected
    )


def test_random_segments_match_stable_sort():
    generator = np.random.default_rng(20260801)
    ptr = np.array([0, 4, 11, 13], np.int32)
    features = generator.normal(size=(13, 16)).astype(np.float32)
    output = reference().sort_aggregation_topk(ptr, features, 5)
    expected = np.zeros((3, 5, 16), dtype=np.float32)
    for graph in range(3):
        begin, end = int(ptr[graph]), int(ptr[graph + 1])
        values = features[begin:end]
        order = sorted(range(end - begin), key=lambda node: -float(values[node, -1]))
        selected = values[order[:5]]
        expected[graph, : selected.shape[0]] = selected
    np.testing.assert_array_equal(output, expected.reshape(3, 80))


def test_tie_order_is_stable():
    ptr = np.array([0, 4], np.int32)
    features = np.array([[1, 0.5], [2, 0.5], [3, 0.7], [4, 0.5]], np.float32)
    expected = np.array([[3, 0.7, 1, 0.5, 2, 0.5]], np.float32)
    np.testing.assert_array_equal(
        reference().sort_aggregation_topk(ptr, features, 3), expected
    )


def test_dispatch_contract():
    dispatch = reference().use_custom_path
    assert dispatch(3371, 188, 96, 30, np.float32)
    assert not dispatch(3371, 3, 96, 30, np.float32)
    assert not dispatch(3371, 188, 1025, 30, np.float32)
    assert not dispatch(3371, 188, 96, 129, np.float32)
    assert not dispatch(3371, 188, 96, 30, np.float16)
    assert not dispatch(3371, 188, 96, 30, np.float32, False)


def test_host_api_contract():
    source = (
        Path(__file__).parents[1] / "op_host" / "sort_aggregation_topk_fused_host.h"
    ).read_text()
    assert "aclnnSortAggregationTopkFused" in source
    assert "top_k" in source
