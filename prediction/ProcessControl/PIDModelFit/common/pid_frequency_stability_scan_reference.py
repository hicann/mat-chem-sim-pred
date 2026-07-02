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


METRIC_NAMES = (
    "min_return_difference",
    "max_sensitivity",
    "gain_crossover_omega",
    "gain_crossover_error",
    "low_frequency_loop_gain",
    "high_frequency_noise_gain",
    "mean_sensitivity",
    "stable_flag",
)


@dataclass
class FrequencyStabilityCase:
    plant_real: np.ndarray
    plant_imag: np.ndarray
    omega: np.ndarray
    kp: np.ndarray
    ki: np.ndarray
    kd: np.ndarray


@dataclass
class FrequencyStabilityResult:
    metrics: np.ndarray


def make_frequency_stability_case(
    batch: int = 16,
    candidates: int = 8,
    freq_count: int = 128,
    seed: int = 2034,
) -> FrequencyStabilityCase:
    rng = np.random.default_rng(seed)
    omega = np.logspace(-2.0, 2.0, freq_count, dtype=np.float32)
    gain = rng.uniform(0.6, 2.8, size=(batch, 1)).astype(np.float32)
    tau = rng.uniform(3.0, 45.0, size=(batch, 1)).astype(np.float32)
    theta = rng.uniform(0.1, 5.0, size=(batch, 1)).astype(np.float32)
    jw_tau_real = np.ones((batch, freq_count), dtype=np.float32)
    jw_tau_imag = tau * omega[None, :]
    denom = jw_tau_real * jw_tau_real + jw_tau_imag * jw_tau_imag
    base_real = gain * jw_tau_real / denom
    base_imag = -gain * jw_tau_imag / denom
    delay_real = np.cos(theta * omega[None, :]).astype(np.float32)
    delay_imag = -np.sin(theta * omega[None, :]).astype(np.float32)
    plant_real = base_real * delay_real - base_imag * delay_imag
    plant_imag = base_real * delay_imag + base_imag * delay_real

    loop = np.arange(batch, dtype=np.float32)[:, None]
    cand = np.arange(candidates, dtype=np.float32)[None, :]
    kp = (0.4 + 0.06 * cand + 0.01 * (loop % 11.0)).astype(np.float32)
    ki = (0.02 + 0.006 * cand + 0.001 * (loop % 7.0)).astype(np.float32)
    kd = (0.01 + 0.004 * cand + 0.0005 * (loop % 5.0)).astype(np.float32)
    return FrequencyStabilityCase(plant_real, plant_imag, omega, kp, ki, kd)


def frequency_stability_scan(
    plant_real: np.ndarray,
    plant_imag: np.ndarray,
    omega: np.ndarray,
    kp: np.ndarray,
    ki: np.ndarray,
    kd: np.ndarray,
    stability_margin_threshold: float = 0.2,
    sensitivity_limit: float = 8.0,
) -> FrequencyStabilityResult:
    plant_real = np.asarray(plant_real, dtype=np.float32)
    plant_imag = np.asarray(plant_imag, dtype=np.float32)
    omega = np.asarray(omega, dtype=np.float32)
    kp = np.asarray(kp, dtype=np.float32)
    ki = np.asarray(ki, dtype=np.float32)
    kd = np.asarray(kd, dtype=np.float32)
    if plant_real.ndim != 2 or plant_imag.shape != plant_real.shape:
        raise ValueError("plant_real and plant_imag must have shape [batch, freq_count]")
    if omega.ndim != 1 or omega.shape[0] != plant_real.shape[1]:
        raise ValueError("omega must have shape [freq_count]")
    if kp.ndim != 2 or ki.shape != kp.shape or kd.shape != kp.shape or kp.shape[0] != plant_real.shape[0]:
        raise ValueError("kp, ki and kd must have shape [batch, candidates]")

    batch, freq_count = plant_real.shape
    candidates = kp.shape[1]
    metrics = np.zeros((batch, candidates, len(METRIC_NAMES)), dtype=np.float32)
    eps = np.float32(1.0e-6)
    safe_omega = np.maximum(omega, eps)
    for b in range(batch):
        gr = plant_real[b]
        gi = plant_imag[b]
        for c in range(candidates):
            controller_real = kp[b, c]
            controller_imag = kd[b, c] * safe_omega - ki[b, c] / safe_omega
            loop_real = controller_real * gr - controller_imag * gi
            loop_imag = controller_real * gi + controller_imag * gr
            loop_abs = np.sqrt(loop_real * loop_real + loop_imag * loop_imag)
            return_real = 1.0 + loop_real
            return_abs = np.sqrt(return_real * return_real + loop_imag * loop_imag)
            sensitivity = 1.0 / np.maximum(return_abs, eps)
            cross_idx = int(np.argmin(np.abs(loop_abs - 1.0)))
            high_start = max(freq_count // 2, 0)
            noise_gain = np.sqrt(controller_real * controller_real + controller_imag[high_start:] ** 2)
            min_return = float(np.min(return_abs))
            max_sensitivity = float(np.max(sensitivity))
            metrics[b, c, 0] = min_return
            metrics[b, c, 1] = max_sensitivity
            metrics[b, c, 2] = omega[cross_idx]
            metrics[b, c, 3] = abs(float(loop_abs[cross_idx]) - 1.0)
            metrics[b, c, 4] = loop_abs[0]
            metrics[b, c, 5] = float(np.max(noise_gain))
            metrics[b, c, 6] = float(np.mean(sensitivity))
            metrics[b, c, 7] = (
                1.0 if min_return >= stability_margin_threshold and max_sensitivity <= sensitivity_limit else 0.0
            )
    return FrequencyStabilityResult(metrics=metrics)


def frequency_stability_scan_cpu_loop(
    plant_real: np.ndarray,
    plant_imag: np.ndarray,
    omega: np.ndarray,
    kp: np.ndarray,
    ki: np.ndarray,
    kd: np.ndarray,
) -> FrequencyStabilityResult:
    result = frequency_stability_scan(plant_real, plant_imag, omega, kp, ki, kd)
    return FrequencyStabilityResult(result.metrics.copy())
