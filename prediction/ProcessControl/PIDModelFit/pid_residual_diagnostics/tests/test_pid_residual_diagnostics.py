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


COMMON_DIR = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON_DIR))

from pid_residual_diagnostics_reference import (
    METRIC_NAMES,
    make_residual_diagnostics_case,
    residual_diagnostics,
    residual_diagnostics_cpu_loop,
)


def test_residual_diagnostics_matches_cpu_loop():
    case = make_residual_diagnostics_case(batch=8, n=256)
    vectorized = residual_diagnostics(case.actual, case.predicted, max_lag=12)
    loop = residual_diagnostics_cpu_loop(case.actual, case.predicted, max_lag=12)

    np.testing.assert_allclose(vectorized.metrics, loop.metrics, rtol=2.0e-5, atol=2.0e-4)
    np.testing.assert_allclose(vectorized.autocorr, loop.autocorr, rtol=2.0e-5, atol=2.0e-4)


def test_residual_diagnostics_known_case():
    actual = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    predicted = np.array([[1.0, 1.0, 3.0, 5.0]], dtype=np.float32)
    result = residual_diagnostics(actual, predicted, max_lag=2)
    idx = {name: i for i, name in enumerate(METRIC_NAMES)}

    assert result.metrics.shape == (1, len(METRIC_NAMES))
    assert result.autocorr.shape == (1, 2)
    assert np.isclose(result.metrics[0, idx["mean_residual"]], 0.0, atol=1.0e-6)
    assert np.isclose(result.metrics[0, idx["mae"]], 0.5, atol=1.0e-6)
    assert np.isclose(result.metrics[0, idx["rmse"]], np.sqrt(0.5), atol=1.0e-6)
    assert np.isclose(result.metrics[0, idx["max_abs_residual"]], 1.0, atol=1.0e-6)


def test_residual_diagnostics_metric_sanity():
    case = make_residual_diagnostics_case(batch=16, n=512)
    result = residual_diagnostics(case.actual, case.predicted, max_lag=16)
    idx = {name: i for i, name in enumerate(METRIC_NAMES)}

    assert result.metrics.shape == (16, len(METRIC_NAMES))
    assert result.autocorr.shape == (16, 16)
    assert np.isfinite(result.metrics).all()
    assert np.isfinite(result.autocorr).all()
    assert np.all(result.metrics[:, idx["rmse"]] >= 0.0)
    assert np.all(result.metrics[:, idx["durbin_watson"]] >= 0.0)
    assert np.all(np.abs(result.autocorr) <= 1.0 + 1.0e-5)
