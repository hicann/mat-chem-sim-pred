# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import importlib.util
from pathlib import Path

import numpy as np


def reference():
    spec = importlib.util.spec_from_file_location(
        "attention_reference", Path(__file__).parents[1] / "reference" / "reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_graph_attention_and_singleton():
    graph_ptr = np.array([0, 2, 3], np.int32)
    features = np.array([[1, 2], [3, 4], [5, 6]], np.float32)
    gates = np.array([0, np.log(2.0), -3], np.float32)
    expected = np.array([[7 / 3, 10 / 3], [5, 6]], np.float32)
    np.testing.assert_allclose(
        reference().attentional_aggregation(graph_ptr, features, gates), expected
    )


def test_random_segments_match_independent_stable_softmax():
    generator = np.random.default_rng(20260801)
    graph_ptr = np.array([0, 4, 11, 13], np.int32)
    features = generator.normal(size=(13, 16)).astype(np.float32)
    gates = generator.normal(size=13).astype(np.float32)
    expected = np.empty((3, 16), np.float32)
    for graph in range(3):
        begin, end = graph_ptr[graph], graph_ptr[graph + 1]
        maximum = max(float(gates[node]) for node in range(begin, end))
        denominator = sum(
            np.exp(float(gates[node]) - maximum) for node in range(begin, end)
        )
        for channel in range(16):
            expected[graph, channel] = sum(
                np.exp(float(gates[node]) - maximum)
                / denominator
                * features[node, channel]
                for node in range(begin, end)
            )
    np.testing.assert_allclose(
        reference().attentional_aggregation(graph_ptr, features, gates),
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_equal_gates_reduce_to_mean():
    graph_ptr = np.array([0, 3], np.int32)
    features = np.array([[1, 2], [4, 8], [-2, 5]], np.float32)
    gates = np.zeros(3, np.float32)
    np.testing.assert_allclose(
        reference().attentional_aggregation(graph_ptr, features, gates),
        features.mean(axis=0, keepdims=True),
    )


def test_dispatch_contract():
    dispatch = reference().use_custom_path
    assert dispatch(96, 4, 32, np.float32)
    assert not dispatch(96, 3, 32, np.float32)
    assert not dispatch(96, 4, 1025, np.float32)
    assert not dispatch(96, 4, 32, np.float16)
    assert not dispatch(96, 4, 32, np.float32, False)


def test_host_api_contains_attention_contract():
    source = (
        Path(__file__).parents[1] / "op_host" / "attentional_aggregation_fused_host.h"
    ).read_text()
    assert "aclnnAttentionalAggregationFused" in source
    assert "gates" in source
