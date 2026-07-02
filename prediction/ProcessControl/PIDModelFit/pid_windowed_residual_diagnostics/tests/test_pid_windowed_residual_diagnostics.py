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

import sys
from pathlib import Path

import numpy as np
import pytest


COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))

from pid_residual_diagnostics_reference import METRIC_NAMES
from pid_windowed_residual_diagnostics_reference import (
    make_windowed_residual_diagnostics_case,
    windowed_residual_diagnostics,
    windowed_residual_diagnostics_cpu_loop,
)


def test_windowed_residual_diagnostics_matches_cpu_loop():
    case = make_windowed_residual_diagnostics_case(batch=4, n=384)
    vectorized = windowed_residual_diagnostics(case.actual, case.predicted, window_size=96, stride=48, max_lag=8)
    loop = windowed_residual_diagnostics_cpu_loop(case.actual, case.predicted, window_size=96, stride=48, max_lag=8)

    np.testing.assert_allclose(vectorized.metrics, loop.metrics, rtol=3.0e-5, atol=3.0e-4)
    np.testing.assert_allclose(vectorized.autocorr, loop.autocorr, rtol=3.0e-5, atol=3.0e-4)


def test_windowed_residual_diagnostics_shapes_and_sanity():
    case = make_windowed_residual_diagnostics_case(batch=8, n=512)
    result = windowed_residual_diagnostics(case.actual, case.predicted, window_size=128, stride=64, max_lag=12)
    idx = {name: i for i, name in enumerate(METRIC_NAMES)}

    assert result.metrics.shape == (8, 7, len(METRIC_NAMES))
    assert result.autocorr.shape == (8, 7, 12)
    assert np.isfinite(result.metrics).all()
    assert np.isfinite(result.autocorr).all()
    assert np.all(result.metrics[:, :, idx["rmse"]] >= 0.0)
    assert np.all(result.metrics[:, :, idx["durbin_watson"]] >= 0.0)
    assert np.all(np.abs(result.autocorr) <= 1.0 + 1.0e-5)


def test_windowed_residual_diagnostics_known_window_count():
    actual = np.arange(16, dtype=np.float32)[None, :]
    predicted = actual.copy()
    result = windowed_residual_diagnostics(actual, predicted, window_size=4, stride=3, max_lag=2)

    assert result.metrics.shape == (1, 5, len(METRIC_NAMES))
    assert result.autocorr.shape == (1, 5, 2)
    np.testing.assert_allclose(result.metrics[:, :, 2], 0.0, atol=1.0e-6)
    np.testing.assert_allclose(result.metrics[:, :, 3], 0.0, atol=1.0e-6)


@pytest.mark.parametrize(
    ("window_size", "stride", "max_lag"),
    [(1, 1, 1), (16, 0, 1), (16, 1, 0), (1024, 1, 1)],
)
def test_windowed_residual_diagnostics_rejects_invalid_inputs(window_size, stride, max_lag):
    case = make_windowed_residual_diagnostics_case(batch=2, n=64)
    with pytest.raises(ValueError):
        windowed_residual_diagnostics(
            case.actual,
            case.predicted,
            window_size=window_size,
            stride=stride,
            max_lag=max_lag,
        )
