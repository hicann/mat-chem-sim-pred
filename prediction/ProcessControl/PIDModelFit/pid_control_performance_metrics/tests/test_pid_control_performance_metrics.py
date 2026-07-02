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

from pid_control_performance_reference import (
    METRIC_NAMES,
    control_performance_metrics,
    control_performance_metrics_cpu_loop,
    make_control_performance_case,
)


def test_control_performance_matches_cpu_loop():
    case = make_control_performance_case(batch=12, n=256)
    vectorized = control_performance_metrics(case.pv, case.sp, case.lsl, case.usl, case.mv_variance)
    loop = control_performance_metrics_cpu_loop(case.pv, case.sp, case.lsl, case.usl, case.mv_variance)
    np.testing.assert_allclose(vectorized, loop, rtol=2e-5, atol=2e-4)


def test_control_performance_known_case():
    pv = np.array([[50.0, 51.0, 52.0, 50.0]], dtype=np.float32)
    sp = np.full_like(pv, 50.0)
    lsl = np.array([47.0], dtype=np.float32)
    usl = np.array([53.0], dtype=np.float32)
    mv_variance = np.array([0.25], dtype=np.float32)
    metrics = control_performance_metrics(pv, sp, lsl, usl, mv_variance, sample_interval=1.0, settle_band=0.1)
    idx = {name: i for i, name in enumerate(METRIC_NAMES)}
    assert abs(metrics[0, idx["mean_pv"]] - 50.75) < 1e-5
    assert abs(metrics[0, idx["iae"]] - 3.0) < 1e-5
    assert abs(metrics[0, idx["ise"]] - 5.0) < 1e-5
    assert abs(metrics[0, idx["itae"]] - 5.0) < 1e-5
    assert abs(metrics[0, idx["max_abs_error"]] - 2.0) < 1e-5
    assert abs(metrics[0, idx["settling_time"]] - 3.0) < 1e-5
    assert abs(metrics[0, idx["final_abs_error"]] - 0.0) < 1e-5


def test_control_performance_metric_sanity():
    case = make_control_performance_case(batch=32, n=512)
    metrics = control_performance_metrics(case.pv, case.sp, case.lsl, case.usl, case.mv_variance)
    idx = {name: i for i, name in enumerate(METRIC_NAMES)}
    assert metrics.shape == (32, len(METRIC_NAMES))
    assert np.isfinite(metrics).all()
    assert np.all(metrics[:, idx["cpk"]] <= metrics[:, idx["cp"]] + 1e-6)
    assert np.all(metrics[:, idx["ppk"]] <= metrics[:, idx["pp"]] + 1e-6)
    assert np.all((metrics[:, idx["harris_index"]] >= 0.0) & (metrics[:, idx["harris_index"]] <= 1.0))
    assert np.all(metrics[:, idx["out_of_spec_ratio"]] >= 0.0)
    assert np.all(metrics[:, idx["settling_time"]] >= 0.0)
