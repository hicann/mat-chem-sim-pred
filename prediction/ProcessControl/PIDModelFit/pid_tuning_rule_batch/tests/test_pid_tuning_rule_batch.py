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

from pid_tuning_rule_batch_reference import (
    DIAGNOSTIC_NAMES,
    PARAM_NAMES,
    RULE_NAMES,
    make_tuning_rule_batch_case,
    tuning_rule_batch,
    tuning_rule_batch_cpu_loop,
)


def test_tuning_rule_batch_matches_loop_reference():
    case = make_tuning_rule_batch_case(batch=64)
    vectorized = tuning_rule_batch(case.process_gain, case.time_constant, case.dead_time, case.lambda_value)
    loop = tuning_rule_batch_cpu_loop(case.process_gain, case.time_constant, case.dead_time, case.lambda_value)

    np.testing.assert_allclose(vectorized.pid_params, loop.pid_params, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(vectorized.diagnostics, loop.diagnostics, rtol=1.0e-6, atol=1.0e-6)


def test_tuning_rule_batch_known_case():
    k = np.array([2.0], dtype=np.float32)
    tau = np.array([10.0], dtype=np.float32)
    theta = np.array([2.0], dtype=np.float32)
    lam = np.array([4.0], dtype=np.float32)
    result = tuning_rule_batch(k, tau, theta, lam)
    rule = {name: i for i, name in enumerate(RULE_NAMES)}
    param = {name: i for i, name in enumerate(PARAM_NAMES)}

    assert np.isclose(result.pid_params[0, rule["ziegler_nichols"], param["kp"]], 3.0, atol=1.0e-6)
    assert np.isclose(result.pid_params[0, rule["ziegler_nichols"], param["ki"]], 0.75, atol=1.0e-6)
    assert np.isclose(result.pid_params[0, rule["ziegler_nichols"], param["kd"]], 3.0, atol=1.0e-6)
    assert result.diagnostics[0, rule["imc"], DIAGNOSTIC_NAMES.index("valid")] == 1.0


def test_tuning_rule_batch_invalid_inputs_are_filtered():
    case = make_tuning_rule_batch_case(batch=8)
    result = tuning_rule_batch(case.process_gain, case.time_constant, case.dead_time, case.lambda_value)
    valid_idx = DIAGNOSTIC_NAMES.index("valid")

    assert np.all(result.diagnostics[:4, :, valid_idx] == 0.0)
    assert np.all(result.pid_params[:4] == 0.0)
    assert np.all(result.diagnostics[4:, :, valid_idx] == 1.0)


def test_tuning_rule_batch_shapes_are_finite():
    case = make_tuning_rule_batch_case(batch=32)
    result = tuning_rule_batch(case.process_gain, case.time_constant, case.dead_time, case.lambda_value)

    assert result.pid_params.shape == (32, len(RULE_NAMES), len(PARAM_NAMES))
    assert result.diagnostics.shape == (32, len(RULE_NAMES), len(DIAGNOSTIC_NAMES))
    assert np.isfinite(result.pid_params).all()
    assert np.isfinite(result.diagnostics).all()
