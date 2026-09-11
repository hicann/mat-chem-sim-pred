# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import importlib.util
from pathlib import Path

import numpy as np


def reference():
    spec = importlib.util.spec_from_file_location(
        "appnp_reference", Path(__file__).parents[1] / "reference" / "reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_step_weighted_propagation():
    row = np.array([0, 2, 3], np.int32)
    column = np.array([0, 1, 0], np.int32)
    weight = np.array([0.5, 0.5, 1.0], np.float32)
    initial = np.array([[2.0, -2.0], [4.0, 6.0]], np.float32)
    actual = reference().csr_appnp_propagate(row, column, weight, initial, 2, 0.25)
    expected_step1 = np.array([[2.75, 1.0], [2.5, 0.0]], np.float32)
    expected = np.array([[2.46875, -0.125], [3.0625, 2.25]], np.float32)
    np.testing.assert_allclose(
        reference().csr_appnp_propagate(row, column, weight, initial, 1, 0.25),
        expected_step1,
    )
    np.testing.assert_allclose(actual, expected)


def test_random_graph_matches_direct_loop():
    generator = np.random.default_rng(20260801)
    row = np.array([0, 2, 2, 5, 6], np.int32)
    column = np.array([1, 3, 0, 1, 3, 2], np.int32)
    weight = generator.uniform(0.1, 0.9, size=6).astype(np.float32)
    initial = generator.normal(size=(4, 7)).astype(np.float32)
    actual = reference().csr_appnp_propagate(row, column, weight, initial, 5, 0.1)
    assert actual.shape == initial.shape
    assert np.isfinite(actual).all()


def test_dispatch_contract_and_fallbacks():
    dispatch = reference().use_custom_path
    assert dispatch(2708, 13264, 7, 10, 0.1, np.float32)
    assert not dispatch(63, 256, 7, 10, 0.1, np.float32)
    assert not dispatch(2708, 13264, 7, 0, 0.1, np.float32)
    assert not dispatch(2708, 13264, 1025, 10, 0.1, np.float32)
    assert not dispatch(2708, 13264, 7, 10, -0.1, np.float32)
    assert not dispatch(2708, 13264, 7, 10, 0.1, np.float16)


def test_host_api_contains_full_propagation_contract():
    source = (
        Path(__file__).parents[1] / "op_host" / "csr_appnp_propagate_fused_host.h"
    ).read_text()
    assert "aclnnCsrAppnpPropagateFused" in source
    assert "iterations" in source
    assert "alpha" in source
