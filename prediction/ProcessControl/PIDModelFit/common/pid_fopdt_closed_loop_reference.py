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

import time
from dataclasses import dataclass

import numpy as np


RESULT_NAMES = (
    "best_score",
    "best_kp",
    "best_ki",
    "best_kd",
    "best_iae",
    "best_ise",
    "best_overshoot",
    "best_settling_time",
)


@dataclass
class FopdtClosedLoopCase:
    a: np.ndarray
    b: np.ndarray
    delay: np.ndarray
    y0: np.ndarray
    sp: np.ndarray
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray
    sample_interval: float
    sim_steps: int


def make_fopdt_closed_loop_case(
    batch: int = 128,
    candidates: int = 1024,
    sim_steps: int = 2048,
    seed: int = 2028,
) -> FopdtClosedLoopCase:
    rng = np.random.default_rng(seed)
    tau = rng.uniform(8.0, 40.0, size=batch).astype(np.float32)
    gain = rng.uniform(0.6, 1.8, size=batch).astype(np.float32)
    dt = 1.0
    a = np.exp(-dt / tau).astype(np.float32)
    b = (gain * (1.0 - a)).astype(np.float32)
    delay = rng.integers(1, 16, size=batch, dtype=np.int32)
    y0 = rng.uniform(-0.2, 0.2, size=batch).astype(np.float32)
    sp = np.ones(batch, dtype=np.float32)
    kp = np.linspace(0.05, 3.0, candidates, dtype=np.float32)
    ki = np.linspace(0.0, 0.35, candidates, dtype=np.float32)
    kd = np.linspace(0.0, 0.25, candidates, dtype=np.float32)
    return FopdtClosedLoopCase(a, b, delay, y0, sp, kp, ki, kd, dt, sim_steps)


def _simulate_candidate(
    a: float,
    b: float,
    delay: int,
    y0: float,
    sp: float,
    kp: float,
    ki: float,
    kd: float,
    sim_steps: int,
    sample_interval: float,
    settle_band: float,
    overshoot_weight: float,
    settling_weight: float,
    control_weight: float,
) -> tuple[float, float, float, float, float]:
    delay = max(0, int(delay))
    delay_len = min(delay + 1, 128)
    u_hist = [0.0] * delay_len
    y = float(y0)
    integral = 0.0
    prev_error = float(sp) - y
    iae = 0.0
    ise = 0.0
    max_overshoot = 0.0
    last_unsettled = 0.0
    control_energy = 0.0
    for step in range(sim_steps):
        error = float(sp) - y
        integral += error * sample_interval
        derivative = (error - prev_error) / max(sample_interval, 1.0e-6)
        u = kp * error + ki * integral + kd * derivative
        u = min(max(u, -10.0), 10.0)
        delayed_u = u_hist[0]
        if delay_len > 1:
            u_hist[:-1] = u_hist[1:]
        u_hist[-1] = u
        y = a * y + b * delayed_u
        abs_error = abs(float(sp) - y)
        iae += abs_error * sample_interval
        ise += abs_error * abs_error * sample_interval
        max_overshoot = max(max_overshoot, y - float(sp))
        control_energy += u * u * sample_interval
        if abs_error > settle_band:
            last_unsettled = (step + 1) * sample_interval
        prev_error = error
    overshoot = max(max_overshoot, 0.0)
    score = iae + overshoot_weight * overshoot + settling_weight * last_unsettled + control_weight * control_energy
    return score, iae, ise, overshoot, last_unsettled


def fopdt_closed_loop_score(
    case: FopdtClosedLoopCase,
    settle_band: float = 0.02,
    overshoot_weight: float = 50.0,
    settling_weight: float = 0.02,
    control_weight: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    batch = case.a.shape[0]
    candidates = case.kp.shape[0]
    results = np.zeros((batch, len(RESULT_NAMES)), dtype=np.float32)
    best_idx = np.zeros(batch, dtype=np.int32)
    for b_idx in range(batch):
        best_score = float("inf")
        best_values = None
        for c_idx in range(candidates):
            score, iae, ise, overshoot, settling = _simulate_candidate(
                float(case.a[b_idx]),
                float(case.b[b_idx]),
                int(case.delay[b_idx]),
                float(case.y0[b_idx]),
                float(case.sp[b_idx]),
                float(case.kp[c_idx]),
                float(case.ki[c_idx]),
                float(case.kd[c_idx]),
                case.sim_steps,
                case.sample_interval,
                settle_band,
                overshoot_weight,
                settling_weight,
                control_weight,
            )
            if score < best_score:
                best_score = score
                best_idx[b_idx] = c_idx
                best_values = (score, case.kp[c_idx], case.ki[c_idx], case.kd[c_idx], iae, ise, overshoot, settling)
        results[b_idx] = np.asarray(best_values, dtype=np.float32)
    return results, best_idx


def benchmark_reference(batch: int, candidates: int, sim_steps: int, iters: int = 3) -> dict[str, float]:
    case = make_fopdt_closed_loop_case(batch=batch, candidates=candidates, sim_steps=sim_steps)
    fopdt_closed_loop_score(case)
    start = time.perf_counter()
    for _ in range(iters):
        fopdt_closed_loop_score(case)
    cpu_ms = (time.perf_counter() - start) * 1000.0 / iters
    return {
        "batch": batch,
        "candidates": candidates,
        "sim_steps": sim_steps,
        "cpu_loop_ms_avg": cpu_ms,
        "simulated_steps": float(batch * candidates * sim_steps),
    }
