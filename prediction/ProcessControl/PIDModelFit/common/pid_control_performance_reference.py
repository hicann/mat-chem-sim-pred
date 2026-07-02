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


METRIC_NAMES = (
    "mean_pv",
    "std_pv_sample",
    "std_pv_population",
    "cp",
    "cpk",
    "pp",
    "ppk",
    "harris_index",
    "iae",
    "ise",
    "itae",
    "mae",
    "rmse",
    "max_abs_error",
    "out_of_spec_ratio",
    "out_of_spec_count",
    "overshoot_ratio",
    "undershoot_ratio",
    "settling_time",
    "final_abs_error",
)


@dataclass
class ControlPerformanceCase:
    pv: np.ndarray
    sp: np.ndarray
    lsl: np.ndarray
    usl: np.ndarray
    mv_variance: np.ndarray


def make_control_performance_case(batch: int = 128, n: int = 4096, seed: int = 2027) -> ControlPerformanceCase:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    loop = np.arange(batch, dtype=np.float32)[:, None]
    sp = 50.0 + 0.35 * np.sin(2.0 * np.pi * (t * (1.0 + (loop % 5.0) * 0.05)))
    drift = -0.25 + 0.5 * loop / max(1, batch - 1)
    oscillation = 0.6 * np.sin(2.0 * np.pi * (t * (2.0 + (loop % 7.0) * 0.03) + loop * 0.013))
    noise = rng.normal(loc=0.0, scale=0.08, size=(batch, n)).astype(np.float32)
    pv = sp + drift + oscillation + noise
    if batch > 0 and n >= 16:
        pv[0, 0] = 53.8
        pv[0, 1] = 46.2
        pv[-1, -1] = 54.0
    lsl = np.full(batch, 47.0, dtype=np.float32)
    usl = np.full(batch, 53.0, dtype=np.float32)
    mv_variance = np.full(batch, 0.08, dtype=np.float32)
    return ControlPerformanceCase(pv.astype(np.float32), sp.astype(np.float32), lsl, usl, mv_variance)


def control_performance_metrics(
    pv: np.ndarray,
    sp: np.ndarray,
    lsl: np.ndarray,
    usl: np.ndarray,
    mv_variance: np.ndarray,
    sample_interval: float = 1.0,
    settle_band: float = 0.1,
) -> np.ndarray:
    pv = np.asarray(pv, dtype=np.float32)
    sp = np.asarray(sp, dtype=np.float32)
    lsl = np.asarray(lsl, dtype=np.float32)
    usl = np.asarray(usl, dtype=np.float32)
    mv_variance = np.asarray(mv_variance, dtype=np.float32)
    if pv.ndim != 2 or sp.shape != pv.shape:
        raise ValueError("pv and sp must have shape [batch, n]")
    if lsl.shape != (pv.shape[0],) or usl.shape != (pv.shape[0],) or mv_variance.shape != (pv.shape[0],):
        raise ValueError("lsl/usl/mv_variance must have shape [batch]")

    batch, n = pv.shape
    metrics = np.zeros((batch, len(METRIC_NAMES)), dtype=np.float32)
    if n <= 0:
        return metrics

    eps = np.float32(1.0e-6)
    mean = pv.mean(axis=1)
    centered = pv - mean[:, None]
    m2 = np.sum(centered * centered, axis=1)
    var_population = np.maximum(m2 / float(n), 0.0)
    var_sample = m2 / float(max(1, n - 1))
    std_population = np.sqrt(var_population)
    std_sample = np.sqrt(np.maximum(var_sample, 0.0))
    denom_sample = 3.0 * np.maximum(std_sample, eps)
    denom_population = 3.0 * np.maximum(std_population, eps)
    spec_width = np.maximum(usl - lsl, eps)
    cpu = (usl - mean) / denom_sample
    cpl = (mean - lsl) / denom_sample
    ppu = (usl - mean) / denom_population
    ppl = (mean - lsl) / denom_population

    error = sp - pv
    abs_error = np.abs(error)
    err_sq = error * error
    time_axis = (np.arange(n, dtype=np.float32) * np.float32(sample_interval))[None, :]
    out = (pv < lsl[:, None]) | (pv > usl[:, None])
    out_count = out.sum(axis=1).astype(np.float32)
    positive_deviation = np.maximum(pv - sp, 0.0)
    negative_deviation = np.maximum(sp - pv, 0.0)
    unsettled = abs_error > np.float32(settle_band)
    any_unsettled = unsettled.any(axis=1)
    last_unsettled = np.where(any_unsettled, n - 1 - np.argmax(unsettled[:, ::-1], axis=1), -1)
    settling_time = np.where(any_unsettled, (last_unsettled + 1).astype(np.float32) * sample_interval, 0.0)
    harris = np.clip(mv_variance / np.maximum(var_population, eps), 0.0, 1.0)

    metrics[:, 0] = mean
    metrics[:, 1] = std_sample
    metrics[:, 2] = std_population
    metrics[:, 3] = spec_width / (2.0 * denom_sample)
    metrics[:, 4] = np.minimum(cpu, cpl)
    metrics[:, 5] = spec_width / (2.0 * denom_population)
    metrics[:, 6] = np.minimum(ppu, ppl)
    metrics[:, 7] = harris
    metrics[:, 8] = abs_error.sum(axis=1) * sample_interval
    metrics[:, 9] = err_sq.sum(axis=1) * sample_interval
    metrics[:, 10] = (time_axis * abs_error).sum(axis=1) * sample_interval
    metrics[:, 11] = abs_error.mean(axis=1)
    metrics[:, 12] = np.sqrt(err_sq.mean(axis=1))
    metrics[:, 13] = abs_error.max(axis=1)
    metrics[:, 14] = out_count / float(n)
    metrics[:, 15] = out_count
    metrics[:, 16] = positive_deviation.max(axis=1) / spec_width
    metrics[:, 17] = negative_deviation.max(axis=1) / spec_width
    metrics[:, 18] = settling_time
    metrics[:, 19] = abs_error[:, -1]
    return metrics


def control_performance_metrics_cpu_loop(
    pv: np.ndarray,
    sp: np.ndarray,
    lsl: np.ndarray,
    usl: np.ndarray,
    mv_variance: np.ndarray,
    sample_interval: float = 1.0,
    settle_band: float = 0.1,
) -> np.ndarray:
    pv = np.asarray(pv, dtype=np.float32)
    sp = np.asarray(sp, dtype=np.float32)
    batch, n = pv.shape
    metrics = np.zeros((batch, len(METRIC_NAMES)), dtype=np.float32)
    eps = 1.0e-6
    for b in range(batch):
        mean = 0.0
        m2 = 0.0
        count = 0.0
        out_count = 0.0
        iae = 0.0
        ise = 0.0
        itae = 0.0
        max_abs_error = 0.0
        max_positive = 0.0
        max_negative = 0.0
        last_unsettled_time = 0.0
        for i in range(n):
            value = float(pv[b, i])
            target = float(sp[b, i])
            count += 1.0
            delta = value - mean
            mean += delta / count
            m2 += delta * (value - mean)
            error = target - value
            abs_error = abs(error)
            iae += abs_error * sample_interval
            ise += error * error * sample_interval
            itae += (i * sample_interval) * abs_error * sample_interval
            max_abs_error = max(max_abs_error, abs_error)
            max_positive = max(max_positive, value - target)
            max_negative = max(max_negative, target - value)
            if abs_error > settle_band:
                last_unsettled_time = (i + 1) * sample_interval
            if value < float(lsl[b]) or value > float(usl[b]):
                out_count += 1.0
        var_population = max(0.0, m2 / max(count, 1.0))
        var_sample = m2 / max(count - 1.0, 1.0)
        std_population = float(np.sqrt(var_population))
        std_sample = float(np.sqrt(max(var_sample, 0.0)))
        denom_sample = 3.0 * max(std_sample, eps)
        denom_population = 3.0 * max(std_population, eps)
        width = max(float(usl[b]) - float(lsl[b]), eps)
        cpu = (float(usl[b]) - mean) / denom_sample
        cpl = (mean - float(lsl[b])) / denom_sample
        ppu = (float(usl[b]) - mean) / denom_population
        ppl = (mean - float(lsl[b])) / denom_population
        harris = min(max(float(mv_variance[b]) / max(var_population, eps), 0.0), 1.0)
        metrics[b] = np.array(
            [
                mean,
                std_sample,
                std_population,
                width / (2.0 * denom_sample),
                min(cpu, cpl),
                width / (2.0 * denom_population),
                min(ppu, ppl),
                harris,
                iae,
                ise,
                itae,
                iae / max(count, 1.0),
                float(np.sqrt(ise / max(count * sample_interval, eps))),
                max_abs_error,
                out_count / max(count, 1.0),
                out_count,
                max(max_positive, 0.0) / width,
                max(max_negative, 0.0) / width,
                last_unsettled_time,
                abs(float(sp[b, n - 1]) - float(pv[b, n - 1])),
            ],
            dtype=np.float32,
        )
    return metrics


def benchmark_reference(batch: int, n: int, iters: int = 20) -> dict[str, float]:
    case = make_control_performance_case(batch=batch, n=n)
    control_performance_metrics(case.pv, case.sp, case.lsl, case.usl, case.mv_variance)
    start = time.perf_counter()
    for _ in range(iters):
        control_performance_metrics(case.pv, case.sp, case.lsl, case.usl, case.mv_variance)
    numpy_ms = (time.perf_counter() - start) * 1000.0 / iters

    loop_iters = max(1, min(3, iters))
    start = time.perf_counter()
    for _ in range(loop_iters):
        control_performance_metrics_cpu_loop(case.pv, case.sp, case.lsl, case.usl, case.mv_variance)
    cpu_loop_ms = (time.perf_counter() - start) * 1000.0 / loop_iters
    return {
        "batch": batch,
        "n": n,
        "numpy_vectorized_ms_avg": numpy_ms,
        "cpu_loop_ms_avg": cpu_loop_ms,
        "speedup_numpy_vs_cpu_loop": cpu_loop_ms / numpy_ms,
    }
