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


FEATURE_NAMES = (
    "initial_value",
    "final_value",
    "final_abs_error",
    "peak_value",
    "trough_value",
    "overshoot_ratio",
    "undershoot_ratio",
    "rise_time",
    "peak_time",
    "settling_time",
    "iae",
    "ise",
)


@dataclass
class StepResponseFeaturesCase:
    pv_candidates: np.ndarray
    sp: np.ndarray


def make_step_response_features_case(
    batch: int = 64,
    candidates: int = 32,
    n: int = 1024,
    seed: int = 2047,
) -> StepResponseFeaturesCase:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, None, :]
    loop = np.arange(batch, dtype=np.float32)
    cand = np.arange(candidates, dtype=np.float32)[None, :, None]
    sp_initial = 45.0 + 0.05 * (loop % 7.0)
    sp_final = 50.0 + 0.1 * (loop % 5.0)
    sp_line = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    sp = sp_initial[:, None] + (sp_final - sp_initial)[:, None] * (sp_line >= 0.05)

    tau = 0.08 + 0.22 * (cand + 1.0) / max(1, candidates)
    damping = 0.04 + 0.18 * ((cand % 9.0) / 8.0)
    response = 1.0 - np.exp(-np.maximum(t - 0.05, 0.0) / tau)
    ring = damping * np.exp(-3.0 * np.maximum(t - 0.05, 0.0)) * np.sin(2.0 * np.pi * (3.0 + cand * 0.03) * t)
    noise = rng.normal(loc=0.0, scale=0.002, size=(batch, candidates, n)).astype(np.float32)
    target_delta = (sp_final - sp_initial)[:, None, None]
    pv = sp_initial[:, None, None] + target_delta * (response + ring) + noise
    return StepResponseFeaturesCase(pv_candidates=pv.astype(np.float32), sp=sp.astype(np.float32))


def _first_index(mask: np.ndarray, fallback: int) -> np.ndarray:
    found = mask.any(axis=2)
    index = np.argmax(mask, axis=2).astype(np.float32)
    return np.where(found, index, np.float32(fallback))


def step_response_features(
    pv_candidates: np.ndarray,
    sp: np.ndarray,
    sample_interval: float = 1.0,
    settle_band_ratio: float = 0.02,
) -> np.ndarray:
    pv_candidates = np.asarray(pv_candidates, dtype=np.float32)
    sp = np.asarray(sp, dtype=np.float32)
    if pv_candidates.ndim != 3:
        raise ValueError("pv_candidates must have shape [batch, candidates, n]")
    if sp.shape != (pv_candidates.shape[0], pv_candidates.shape[2]):
        raise ValueError("sp must have shape [batch, n]")

    batch, candidates, n = pv_candidates.shape
    eps = np.float32(1.0e-6)
    time_axis = (np.arange(n, dtype=np.float32) * np.float32(sample_interval))[None, None, :]
    target = sp[:, -1][:, None]
    initial = pv_candidates[:, :, 0]
    final = pv_candidates[:, :, -1]
    delta = target - initial
    abs_delta = np.maximum(np.abs(delta), eps)
    direction = np.where(delta >= 0.0, 1.0, -1.0).astype(np.float32)
    normalized = direction[:, :, None] * (pv_candidates - initial[:, :, None]) / abs_delta[:, :, None]

    peak = pv_candidates.max(axis=2)
    trough = pv_candidates.min(axis=2)
    peak_index = np.argmax(pv_candidates, axis=2).astype(np.float32)
    overshoot = np.maximum(direction * (peak - target), 0.0) / abs_delta
    undershoot = np.maximum(direction * (target - trough), 0.0) / abs_delta
    rise10 = _first_index(normalized >= 0.1, n - 1)
    rise90 = _first_index(normalized >= 0.9, n - 1)
    rise_time = np.maximum(rise90 - rise10, 0.0) * sample_interval
    abs_error = np.abs(sp[:, None, :] - pv_candidates)
    band = np.maximum(abs_delta * settle_band_ratio, np.float32(1.0e-4))
    unsettled = abs_error > band[:, :, None]
    any_unsettled = unsettled.any(axis=2)
    last_unsettled = np.where(any_unsettled, n - 1 - np.argmax(unsettled[:, :, ::-1], axis=2), -1)
    settling_time = np.where(any_unsettled, (last_unsettled + 1).astype(np.float32) * sample_interval, 0.0)

    features = np.zeros((batch, candidates, len(FEATURE_NAMES)), dtype=np.float32)
    features[:, :, 0] = initial
    features[:, :, 1] = final
    features[:, :, 2] = np.abs(target - final)
    features[:, :, 3] = peak
    features[:, :, 4] = trough
    features[:, :, 5] = overshoot
    features[:, :, 6] = undershoot
    features[:, :, 7] = rise_time
    features[:, :, 8] = peak_index * sample_interval
    features[:, :, 9] = settling_time
    features[:, :, 10] = abs_error.sum(axis=2) * sample_interval
    features[:, :, 11] = (abs_error * abs_error).sum(axis=2) * sample_interval
    return features


def step_response_features_cpu_loop(
    pv_candidates: np.ndarray,
    sp: np.ndarray,
    sample_interval: float = 1.0,
    settle_band_ratio: float = 0.02,
) -> np.ndarray:
    pv_candidates = np.asarray(pv_candidates, dtype=np.float32)
    sp = np.asarray(sp, dtype=np.float32)
    batch, candidates, n = pv_candidates.shape
    features = np.zeros((batch, candidates, len(FEATURE_NAMES)), dtype=np.float32)
    eps = 1.0e-6
    for b in range(batch):
        target = float(sp[b, -1])
        for c in range(candidates):
            series = pv_candidates[b, c]
            initial = float(series[0])
            final = float(series[-1])
            delta = target - initial
            abs_delta = max(abs(delta), eps)
            direction = 1.0 if delta >= 0.0 else -1.0
            peak_idx = int(np.argmax(series))
            peak = float(np.max(series))
            trough = float(np.min(series))
            rise10 = n - 1
            rise90 = n - 1
            for i in range(n):
                normalized = direction * (float(series[i]) - initial) / abs_delta
                if rise10 == n - 1 and normalized >= 0.1:
                    rise10 = i
                if rise90 == n - 1 and normalized >= 0.9:
                    rise90 = i
            band = max(abs_delta * settle_band_ratio, 1.0e-4)
            last_unsettled = -1
            iae = 0.0
            ise = 0.0
            for i in range(n):
                error = abs(float(sp[b, i]) - float(series[i]))
                iae += error * sample_interval
                ise += error * error * sample_interval
                if error > band:
                    last_unsettled = i
            features[b, c] = np.array(
                [
                    initial,
                    final,
                    abs(target - final),
                    peak,
                    trough,
                    max(direction * (peak - target), 0.0) / abs_delta,
                    max(direction * (target - trough), 0.0) / abs_delta,
                    max(rise90 - rise10, 0) * sample_interval,
                    peak_idx * sample_interval,
                    (last_unsettled + 1) * sample_interval if last_unsettled >= 0 else 0.0,
                    iae,
                    ise,
                ],
                dtype=np.float32,
            )
    return features


def benchmark_reference(batch: int, candidates: int, n: int, iters: int = 10) -> dict[str, float]:
    case = make_step_response_features_case(batch=batch, candidates=candidates, n=n)
    step_response_features(case.pv_candidates, case.sp)
    start = time.perf_counter()
    for _ in range(iters):
        step_response_features(case.pv_candidates, case.sp)
    numpy_ms = (time.perf_counter() - start) * 1000.0 / iters
    start = time.perf_counter()
    step_response_features_cpu_loop(case.pv_candidates, case.sp)
    loop_ms = (time.perf_counter() - start) * 1000.0
    return {
        "batch": float(batch),
        "candidates": float(candidates),
        "n": float(n),
        "numpy_ms": numpy_ms,
        "loop_ms": loop_ms,
        "speedup": loop_ms / max(numpy_ms, 1.0e-9),
    }
