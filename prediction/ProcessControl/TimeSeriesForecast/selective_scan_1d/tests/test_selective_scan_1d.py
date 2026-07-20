#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


THIS_DIR = Path(__file__).resolve().parent
FORECAST_DIR = THIS_DIR.parents[1]

_REFERENCE_SPEC = importlib.util.spec_from_file_location(
    "selective_scan_1d_reference",
    FORECAST_DIR / "common" / "selective_scan_1d_reference.py",
)
if _REFERENCE_SPEC is None or _REFERENCE_SPEC.loader is None:
    raise ImportError("failed to load selective_scan_1d_reference.py")
_REFERENCE_MODULE = importlib.util.module_from_spec(_REFERENCE_SPEC)
sys.modules["selective_scan_1d_reference"] = _REFERENCE_MODULE
_REFERENCE_SPEC.loader.exec_module(_REFERENCE_MODULE)
make_selective_scan_1d_case = _REFERENCE_MODULE.make_selective_scan_1d_case
selective_scan_1d = _REFERENCE_MODULE.selective_scan_1d


def scalar_reference(
    u: np.ndarray,
    delta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    batch, length, dim = u.shape
    state = a.shape[1]
    state_value = np.zeros((batch, dim, state), dtype=np.float32)
    out = np.zeros((batch, length, dim), dtype=np.float32)
    for bi in range(batch):
        for di in range(dim):
            for ti in range(length):
                for ni in range(state):
                    decay = np.exp(delta[bi, ti, di] * a[di, ni]).astype(np.float32)
                    update = delta[bi, ti, di] * b[bi, ti, ni] * u[bi, ti, di]
                    state_value[bi, di, ni] = decay * state_value[bi, di, ni] + update
                acc = np.float32(0.0)
                for ni in range(state):
                    acc = np.float32(acc + state_value[bi, di, ni] * c[bi, ti, ni])
                out[bi, ti, di] = acc + u[bi, ti, di] * d[di]
    return out


def test_deterministic_smoke_case() -> None:
    batch, length, dim, state = 1, 2, 1, 32
    u = np.array([[[2.0], [3.0]]], dtype=np.float32)
    delta = np.ones((batch, length, dim), dtype=np.float32)
    a = np.zeros((dim, state), dtype=np.float32)
    b = np.ones((batch, length, state), dtype=np.float32)
    c = np.ones((batch, length, state), dtype=np.float32)
    d = np.zeros((dim,), dtype=np.float32)

    expected = np.array([[[64.0], [160.0]]], dtype=np.float32)
    actual = selective_scan_1d(u, delta, a, b, c, d)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-5)


def test_random_case_matches_scalar_reference() -> None:
    case = make_selective_scan_1d_case(batch=2, length=7, dim=3, state=5, seed=42)

    actual = selective_scan_1d(case.u, case.delta, case.a, case.b, case.c, case.d)
    expected = scalar_reference(case.u, case.delta, case.a, case.b, case.c, case.d)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-5, atol=2.0e-5)


def test_skip_path_when_state_terms_are_zero() -> None:
    case = make_selective_scan_1d_case(batch=1, length=5, dim=4, state=3, seed=7)
    b = np.zeros_like(case.b)
    c = np.zeros_like(case.c)

    actual = selective_scan_1d(case.u, case.delta, case.a, b, c, case.d)
    expected = case.u * case.d[None, None, :]

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6)


def test_zero_input_has_zero_output_without_skip_bias() -> None:
    batch, length, dim, state = 2, 4, 3, 5
    u = np.zeros((batch, length, dim), dtype=np.float32)
    delta = np.full((batch, length, dim), 0.03, dtype=np.float32)
    a = -np.ones((dim, state), dtype=np.float32)
    b = np.ones((batch, length, state), dtype=np.float32)
    c = np.ones((batch, length, state), dtype=np.float32)
    d = np.arange(dim, dtype=np.float32)

    actual = selective_scan_1d(u, delta, a, b, c, d)

    np.testing.assert_array_equal(actual, np.zeros_like(actual))


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("u", np.zeros((2, 3), dtype=np.float32), "u must have shape"),
        ("delta", np.zeros((2, 6, 3), dtype=np.float32), "delta must have shape"),
        ("a", np.zeros((4, 5), dtype=np.float32), "a must have shape"),
        ("b", np.zeros((2, 7, 6), dtype=np.float32), "b must have shape"),
        ("c", np.zeros((2, 7, 6), dtype=np.float32), "c must have shape"),
        ("d", np.zeros((4,), dtype=np.float32), "d must have shape"),
    ],
)
def test_shape_validation(field: str, bad_value: np.ndarray, message: str) -> None:
    case = make_selective_scan_1d_case(batch=2, length=7, dim=3, state=5, seed=9)
    kwargs = {
        "u": case.u,
        "delta": case.delta,
        "a": case.a,
        "b": case.b,
        "c": case.c,
        "d": case.d,
    }
    kwargs[field] = bad_value

    with pytest.raises(ValueError, match=message):
        selective_scan_1d(**kwargs)
