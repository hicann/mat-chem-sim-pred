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

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from pid_process_capability_reference import (
    METRIC_NAMES,
    capability_metrics,
    capability_metrics_cpu_loop,
    make_capability_case,
)


def test_process_capability_matches_cpu_loop():
    case = make_capability_case(batch=16, n=512)
    vectorized = capability_metrics(case.values, case.lsl, case.usl)
    loop = capability_metrics_cpu_loop(case.values, case.lsl, case.usl)

    assert vectorized.shape == (16, len(METRIC_NAMES))
    assert np.allclose(vectorized, loop, atol=2.0e-4, rtol=2.0e-5)


def test_process_capability_known_small_case():
    values = np.array([[49.0, 50.0, 51.0, 54.0], [48.0, 49.0, 50.0, 51.0]], dtype=np.float32)
    lsl = np.array([47.0, 47.0], dtype=np.float32)
    usl = np.array([53.0, 53.0], dtype=np.float32)
    metrics = capability_metrics(values, lsl, usl)

    assert np.isclose(metrics[0, METRIC_NAMES.index("mean")], 51.0, atol=1.0e-6)
    assert np.isclose(metrics[0, METRIC_NAMES.index("out_of_spec_count")], 1.0, atol=1.0e-6)
    assert np.isclose(metrics[0, METRIC_NAMES.index("out_of_spec_ratio")], 0.25, atol=1.0e-6)
    assert np.isclose(metrics[1, METRIC_NAMES.index("min_value")], 48.0, atol=1.0e-6)
    assert np.isclose(metrics[1, METRIC_NAMES.index("max_value")], 51.0, atol=1.0e-6)


def test_process_capability_metric_sanity():
    case = make_capability_case(batch=8, n=1024)
    metrics = capability_metrics(case.values, case.lsl, case.usl)
    cp = metrics[:, METRIC_NAMES.index("cp")]
    cpk = metrics[:, METRIC_NAMES.index("cpk")]
    pp = metrics[:, METRIC_NAMES.index("pp")]
    ppk = metrics[:, METRIC_NAMES.index("ppk")]

    assert np.all(np.isfinite(metrics))
    assert np.all(cp > 0)
    assert np.all(pp > 0)
    assert np.all(cpk <= cp + 1.0e-6)
    assert np.all(ppk <= pp + 1.0e-6)
