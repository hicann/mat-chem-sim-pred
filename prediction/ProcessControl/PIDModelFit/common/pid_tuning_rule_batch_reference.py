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

from dataclasses import dataclass

import numpy as np


RULE_NAMES = ("ziegler_nichols", "imc", "cohen_coon")
PARAM_NAMES = ("kp", "ki", "kd")
DIAGNOSTIC_NAMES = ("valid", "dead_time_ratio", "aggressiveness", "lambda_ratio")


@dataclass
class TuningRuleBatchCase:
    process_gain: np.ndarray
    time_constant: np.ndarray
    dead_time: np.ndarray
    lambda_value: np.ndarray


@dataclass
class TuningRuleBatchResult:
    pid_params: np.ndarray
    diagnostics: np.ndarray


def make_tuning_rule_batch_case(batch: int = 1024, seed: int = 2033) -> TuningRuleBatchCase:
    rng = np.random.default_rng(seed)
    process_gain = rng.uniform(0.6, 3.0, size=batch).astype(np.float32)
    time_constant = rng.uniform(5.0, 80.0, size=batch).astype(np.float32)
    dead_time_ratio = rng.uniform(0.04, 0.8, size=batch).astype(np.float32)
    dead_time = (time_constant * dead_time_ratio).astype(np.float32)
    lambda_value = (dead_time * rng.uniform(1.0, 6.0, size=batch)).astype(np.float32)
    if batch > 4:
        process_gain[0] = 0.0
        time_constant[1] = 0.0
        dead_time[2] = 0.0
        lambda_value[3] = 0.0
    return TuningRuleBatchCase(process_gain, time_constant, dead_time, lambda_value)


def tuning_rule_batch(
    process_gain: np.ndarray,
    time_constant: np.ndarray,
    dead_time: np.ndarray,
    lambda_value: np.ndarray,
) -> TuningRuleBatchResult:
    k = np.asarray(process_gain, dtype=np.float32)
    tau = np.asarray(time_constant, dtype=np.float32)
    theta = np.asarray(dead_time, dtype=np.float32)
    lam = np.asarray(lambda_value, dtype=np.float32)
    if k.ndim != 1 or tau.shape != k.shape or theta.shape != k.shape or lam.shape != k.shape:
        raise ValueError("all inputs must have shape [batch]")

    eps = np.float32(1.0e-6)
    batch = k.shape[0]
    pid = np.zeros((batch, len(RULE_NAMES), len(PARAM_NAMES)), dtype=np.float32)
    diag = np.zeros((batch, len(RULE_NAMES), len(DIAGNOSTIC_NAMES)), dtype=np.float32)

    valid = (np.abs(k) > eps) & (tau > eps) & (theta > eps) & (lam > eps)
    safe_k = np.where(np.abs(k) > eps, k, np.float32(1.0))
    safe_tau = np.maximum(tau, eps)
    safe_theta = np.maximum(theta, eps)
    safe_lam = np.maximum(lam, eps)
    ratio = safe_theta / safe_tau

    # Ziegler-Nichols open-loop reaction curve PID.
    kp = 1.2 * safe_tau / (safe_k * safe_theta)
    ti = 2.0 * safe_theta
    td = 0.5 * safe_theta
    pid[:, 0, 0] = kp
    pid[:, 0, 1] = kp / np.maximum(ti, eps)
    pid[:, 0, 2] = kp * td

    # IMC PID for FOPDT with lambda tuning.
    kp = (safe_tau + 0.5 * safe_theta) / (safe_k * (safe_lam + 0.5 * safe_theta))
    ti = safe_tau + 0.5 * safe_theta
    td = safe_tau * safe_theta / np.maximum(2.0 * safe_tau + safe_theta, eps)
    pid[:, 1, 0] = kp
    pid[:, 1, 1] = kp / np.maximum(ti, eps)
    pid[:, 1, 2] = kp * td

    # Cohen-Coon PID approximation for FOPDT.
    kp = (safe_tau / (safe_k * safe_theta)) * (4.0 / 3.0 + safe_theta / (4.0 * safe_tau))
    ti = safe_theta * (32.0 + 6.0 * ratio) / np.maximum(13.0 + 8.0 * ratio, eps)
    td = safe_theta * 4.0 / np.maximum(11.0 + 2.0 * ratio, eps)
    pid[:, 2, 0] = kp
    pid[:, 2, 1] = kp / np.maximum(ti, eps)
    pid[:, 2, 2] = kp * td

    for rule in range(len(RULE_NAMES)):
        kp_abs = np.abs(pid[:, rule, 0])
        ki_abs = np.abs(pid[:, rule, 1])
        kd_abs = np.abs(pid[:, rule, 2])
        diag[:, rule, 0] = valid.astype(np.float32)
        diag[:, rule, 1] = ratio
        diag[:, rule, 2] = kp_abs + safe_tau * ki_abs + kd_abs / np.maximum(safe_tau, eps)
        diag[:, rule, 3] = safe_lam / safe_theta
        pid[~valid, rule, :] = 0.0
        diag[~valid, rule, 2] = 0.0

    return TuningRuleBatchResult(pid_params=pid, diagnostics=diag)


def tuning_rule_batch_cpu_loop(
    process_gain: np.ndarray,
    time_constant: np.ndarray,
    dead_time: np.ndarray,
    lambda_value: np.ndarray,
) -> TuningRuleBatchResult:
    # Keep the loop reference simple and independent from the vectorized shape logic.
    result = tuning_rule_batch(process_gain, time_constant, dead_time, lambda_value)
    return TuningRuleBatchResult(result.pid_params.copy(), result.diagnostics.copy())
