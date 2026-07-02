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

from pid_fopdt_batch_rollout_reference import (
    RESULT_NAMES,
    fopdt_batch_rollout_score,
    make_fopdt_batch_rollout_case,
)


def test_fopdt_batch_rollout_shape_and_sanity():
    case = make_fopdt_batch_rollout_case(batch=4, candidates=8, sim_steps=64)
    results, best_idx = fopdt_batch_rollout_score(case)
    idx = {name: i for i, name in enumerate(RESULT_NAMES)}
    assert results.shape == (4, len(RESULT_NAMES))
    assert best_idx.shape == (4,)
    assert np.isfinite(results).all()
    assert np.all(best_idx >= 0)
    assert np.all(best_idx < 8)
    assert np.all(results[:, idx["best_score"]] >= 0.0)
    assert np.all(results[:, idx["best_iae"]] >= 0.0)
    assert np.all(results[:, idx["best_ise"]] >= 0.0)
    assert np.all(results[:, idx["best_overshoot"]] >= 0.0)


def test_fopdt_batch_rollout_prefers_some_control_over_zero_gain():
    case = make_fopdt_batch_rollout_case(batch=1, candidates=3, sim_steps=96)
    case.kp[:] = np.array([0.0, 0.3, 0.8], dtype=np.float32)
    case.ki[:] = np.array([0.0, 0.03, 0.08], dtype=np.float32)
    case.kd[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    results, best_idx = fopdt_batch_rollout_score(case)
    assert int(best_idx[0]) in (1, 2)
    assert results[0, 0] > 0.0
