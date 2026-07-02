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
    "mean",
    "std_sample",
    "std_population",
    "cp",
    "cpu",
    "cpl",
    "cpk",
    "pp",
    "ppk",
    "out_of_spec_ratio",
    "out_of_spec_count",
    "min_value",
    "max_value",
)


@dataclass
class CapabilityCase:
    values: np.ndarray
    lsl: np.ndarray
    usl: np.ndarray


def make_capability_case(batch: int = 128, n: int = 4096, seed: int = 2026) -> CapabilityCase:
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=0.0, scale=1.0, size=(batch, n)).astype(np.float32)
    drift = np.linspace(-0.4, 0.4, batch, dtype=np.float32)[:, None]
    scale = (0.8 + 0.004 * (np.arange(batch, dtype=np.float32) % 37.0))[:, None]
    values = 50.0 + drift + base * scale
    if batch > 0 and n >= 16:
        values[0, 0] = 53.5
        values[0, 1] = 46.5
        values[-1, -1] = 54.0
    lsl = np.full(batch, 47.0, dtype=np.float32)
    usl = np.full(batch, 53.0, dtype=np.float32)
    return CapabilityCase(values.astype(np.float32), lsl, usl)


def capability_metrics(values: np.ndarray, lsl: np.ndarray, usl: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lsl = np.asarray(lsl, dtype=np.float32)
    usl = np.asarray(usl, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, n]")
    if lsl.shape != (values.shape[0],) or usl.shape != (values.shape[0],):
        raise ValueError("lsl/usl must have shape [batch]")

    batch, n = values.shape
    metrics = np.zeros((batch, len(METRIC_NAMES)), dtype=np.float32)
    if n <= 0:
        return metrics

    mean = values.mean(axis=1)
    centered = values - mean[:, None]
    var_population = np.mean(centered * centered, axis=1)
    var_sample = np.sum(centered * centered, axis=1) / max(1, n - 1)
    std_population = np.sqrt(np.maximum(var_population, 0.0))
    std_sample = np.sqrt(np.maximum(var_sample, 0.0))
    eps = np.float32(1.0e-6)

    cpu = (usl - mean) / (3.0 * np.maximum(std_sample, eps))
    cpl = (mean - lsl) / (3.0 * np.maximum(std_sample, eps))
    cp = (usl - lsl) / (6.0 * np.maximum(std_sample, eps))
    cpk = np.minimum(cpu, cpl)
    ppu = (usl - mean) / (3.0 * np.maximum(std_population, eps))
    ppl = (mean - lsl) / (3.0 * np.maximum(std_population, eps))
    pp = (usl - lsl) / (6.0 * np.maximum(std_population, eps))
    ppk = np.minimum(ppu, ppl)
    out = (values < lsl[:, None]) | (values > usl[:, None])
    out_count = out.sum(axis=1).astype(np.float32)
    out_ratio = out_count / float(n)

    metrics[:, 0] = mean
    metrics[:, 1] = std_sample
    metrics[:, 2] = std_population
    metrics[:, 3] = cp
    metrics[:, 4] = cpu
    metrics[:, 5] = cpl
    metrics[:, 6] = cpk
    metrics[:, 7] = pp
    metrics[:, 8] = ppk
    metrics[:, 9] = out_ratio
    metrics[:, 10] = out_count
    metrics[:, 11] = values.min(axis=1)
    metrics[:, 12] = values.max(axis=1)
    return metrics


def capability_metrics_cpu_loop(values: np.ndarray, lsl: np.ndarray, usl: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    batch, n = values.shape
    metrics = np.zeros((batch, len(METRIC_NAMES)), dtype=np.float32)
    eps = 1.0e-6
    for b in range(batch):
        mean = 0.0
        m2 = 0.0
        min_value = float(values[b, 0])
        max_value = float(values[b, 0])
        out_count = 0
        for i in range(n):
            value = float(values[b, i])
            count = float(i + 1)
            delta = value - mean
            mean += delta / count
            delta2 = value - mean
            m2 += delta * delta2
            min_value = min(min_value, value)
            max_value = max(max_value, value)
            if value < float(lsl[b]) or value > float(usl[b]):
                out_count += 1
        var_population = max(0.0, m2 / float(n))
        var_sample = m2 / float(max(1, n - 1))
        std_population = float(np.sqrt(var_population))
        std_sample = float(np.sqrt(var_sample))
        denom_sample = 3.0 * max(std_sample, eps)
        denom_population = 3.0 * max(std_population, eps)
        cpu = (float(usl[b]) - mean) / denom_sample
        cpl = (mean - float(lsl[b])) / denom_sample
        ppu = (float(usl[b]) - mean) / denom_population
        ppl = (mean - float(lsl[b])) / denom_population
        metrics[b] = np.array(
            [
                mean,
                std_sample,
                std_population,
                (float(usl[b]) - float(lsl[b])) / (2.0 * denom_sample),
                cpu,
                cpl,
                min(cpu, cpl),
                (float(usl[b]) - float(lsl[b])) / (2.0 * denom_population),
                min(ppu, ppl),
                out_count / float(n),
                float(out_count),
                min_value,
                max_value,
            ],
            dtype=np.float32,
        )
    return metrics


def benchmark_reference(batch: int, n: int, iters: int = 20) -> dict[str, float]:
    case = make_capability_case(batch=batch, n=n)
    capability_metrics(case.values, case.lsl, case.usl)
    start = time.perf_counter()
    for _ in range(iters):
        capability_metrics(case.values, case.lsl, case.usl)
    numpy_ms = (time.perf_counter() - start) * 1000.0 / iters

    start = time.perf_counter()
    for _ in range(max(1, min(3, iters))):
        capability_metrics_cpu_loop(case.values, case.lsl, case.usl)
    cpu_loop_ms = (time.perf_counter() - start) * 1000.0 / max(1, min(3, iters))
    return {
        "batch": batch,
        "n": n,
        "numpy_vectorized_ms_avg": numpy_ms,
        "cpu_loop_ms_avg": cpu_loop_ms,
        "speedup_numpy_vs_cpu_loop": cpu_loop_ms / numpy_ms,
    }
