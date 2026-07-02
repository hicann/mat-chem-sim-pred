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

from pid_step_response_features_reference import (
    FEATURE_NAMES,
    make_step_response_features_case,
    step_response_features,
    step_response_features_cpu_loop,
)


def test_step_response_features_matches_cpu_loop():
    case = make_step_response_features_case(batch=4, candidates=5, n=128)
    vectorized = step_response_features(case.pv_candidates, case.sp, sample_interval=0.1)
    loop = step_response_features_cpu_loop(case.pv_candidates, case.sp, sample_interval=0.1)
    np.testing.assert_allclose(vectorized, loop, rtol=2.0e-5, atol=2.0e-4)


def test_step_response_features_known_case():
    pv = np.array([[[0.0, 2.0, 5.0, 9.0, 11.0, 10.0]]], dtype=np.float32)
    sp = np.array([[0.0, 10.0, 10.0, 10.0, 10.0, 10.0]], dtype=np.float32)
    features = step_response_features(pv, sp, sample_interval=1.0, settle_band_ratio=0.05)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    assert features.shape == (1, 1, len(FEATURE_NAMES))
    assert np.isclose(features[0, 0, idx["initial_value"]], 0.0)
    assert np.isclose(features[0, 0, idx["final_value"]], 10.0)
    assert np.isclose(features[0, 0, idx["final_abs_error"]], 0.0)
    assert np.isclose(features[0, 0, idx["peak_value"]], 11.0)
    assert np.isclose(features[0, 0, idx["overshoot_ratio"]], 0.1)
    assert np.isclose(features[0, 0, idx["rise_time"]], 2.0)
    assert np.isclose(features[0, 0, idx["peak_time"]], 4.0)
    assert np.isclose(features[0, 0, idx["settling_time"]], 5.0)


def test_step_response_features_sanity():
    case = make_step_response_features_case(batch=8, candidates=7, n=256)
    features = step_response_features(case.pv_candidates, case.sp)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    assert features.shape == (8, 7, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert np.all(features[:, :, idx["final_abs_error"]] >= 0.0)
    assert np.all(features[:, :, idx["overshoot_ratio"]] >= 0.0)
    assert np.all(features[:, :, idx["iae"]] >= 0.0)
    assert np.all(features[:, :, idx["ise"]] >= 0.0)


def test_step_response_features_rejects_invalid_inputs():
    case = make_step_response_features_case(batch=2, candidates=3, n=64)
    with pytest.raises(ValueError):
        step_response_features(case.pv_candidates[0], case.sp)
    with pytest.raises(ValueError):
        step_response_features(case.pv_candidates, case.sp[:, :-1])
