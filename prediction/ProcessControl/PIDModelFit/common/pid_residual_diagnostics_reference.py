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
    "mean_residual",
    "std_residual",
    "mae",
    "rmse",
    "max_abs_residual",
    "fit_percent",
    "durbin_watson",
    "ljung_box_q",
)


@dataclass
class ResidualDiagnosticsCase:
    actual: np.ndarray
    predicted: np.ndarray


@dataclass
class ResidualDiagnosticsResult:
    metrics: np.ndarray
    autocorr: np.ndarray


def make_residual_diagnostics_case(batch: int = 64, n: int = 1024, seed: int = 2032) -> ResidualDiagnosticsCase:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)[None, :]
    loop = np.arange(batch, dtype=np.float32)[:, None]
    actual = 50.0 + 2.0 * np.sin(2.0 * np.pi * (t * (1.0 + (loop % 7.0) * 0.03)))
    actual += 0.35 * np.cos(2.0 * np.pi * (t * 3.0 + loop * 0.017))
    actual += rng.normal(loc=0.0, scale=0.03, size=(batch, n)).astype(np.float32)
    bias = (-0.08 + 0.16 * loop / max(1, batch - 1)).astype(np.float32)
    correlated_error = 0.08 * np.sin(2.0 * np.pi * (t * (2.0 + (loop % 5.0) * 0.04) + loop * 0.011))
    predicted = actual - bias - correlated_error
    return ResidualDiagnosticsCase(actual.astype(np.float32), predicted.astype(np.float32))


def residual_diagnostics(
    actual: np.ndarray,
    predicted: np.ndarray,
    max_lag: int = 16,
) -> ResidualDiagnosticsResult:
    actual = np.asarray(actual, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    if actual.ndim != 2 or predicted.shape != actual.shape:
        raise ValueError("actual and predicted must have shape [batch, n]")
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")

    batch, n = actual.shape
    max_lag = min(max_lag, max(1, n - 1))
    residual = actual - predicted
    mean_residual = residual.mean(axis=1)
    centered_residual = residual - mean_residual[:, None]
    residual_energy = np.sum(centered_residual * centered_residual, axis=1)
    sse = np.sum(residual * residual, axis=1)
    abs_residual = np.abs(residual)
    actual_centered = actual - actual.mean(axis=1, keepdims=True)
    actual_energy = np.sum(actual_centered * actual_centered, axis=1)
    eps = np.float32(1.0e-6)

    autocorr = np.zeros((batch, max_lag), dtype=np.float32)
    for lag in range(1, max_lag + 1):
        lhs = centered_residual[:, lag:]
        rhs = centered_residual[:, :-lag]
        autocorr[:, lag - 1] = np.sum(lhs * rhs, axis=1) / np.maximum(residual_energy, eps)

    diff = residual[:, 1:] - residual[:, :-1]
    durbin_watson = np.sum(diff * diff, axis=1) / np.maximum(sse, eps)
    lag_index = np.arange(1, max_lag + 1, dtype=np.float32)[None, :]
    ljung_box_q = n * (n + 2.0) * np.sum((autocorr * autocorr) / np.maximum(n - lag_index, 1.0), axis=1)

    metrics = np.zeros((batch, len(METRIC_NAMES)), dtype=np.float32)
    metrics[:, 0] = mean_residual
    metrics[:, 1] = np.sqrt(np.maximum(residual_energy / max(1, n - 1), 0.0))
    metrics[:, 2] = abs_residual.mean(axis=1)
    metrics[:, 3] = np.sqrt(sse / float(n))
    metrics[:, 4] = abs_residual.max(axis=1)
    metrics[:, 5] = 100.0 * (1.0 - np.sqrt(sse) / np.maximum(np.sqrt(actual_energy), eps))
    metrics[:, 6] = durbin_watson
    metrics[:, 7] = ljung_box_q.astype(np.float32)
    return ResidualDiagnosticsResult(metrics=metrics, autocorr=autocorr)


def residual_diagnostics_cpu_loop(
    actual: np.ndarray,
    predicted: np.ndarray,
    max_lag: int = 16,
) -> ResidualDiagnosticsResult:
    actual = np.asarray(actual, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    batch, n = actual.shape
    max_lag = min(max_lag, max(1, n - 1))
    metrics = np.zeros((batch, len(METRIC_NAMES)), dtype=np.float32)
    autocorr = np.zeros((batch, max_lag), dtype=np.float32)
    eps = 1.0e-6
    for b in range(batch):
        residual = actual[b] - predicted[b]
        mean_residual = float(residual.mean())
        centered = residual - mean_residual
        residual_energy = float(np.sum(centered * centered))
        sse = float(np.sum(residual * residual))
        actual_centered = actual[b] - float(actual[b].mean())
        actual_energy = float(np.sum(actual_centered * actual_centered))
        for lag in range(1, max_lag + 1):
            value = float(np.sum(centered[lag:] * centered[:-lag]) / max(residual_energy, eps))
            autocorr[b, lag - 1] = value
        diff = residual[1:] - residual[:-1]
        dw = float(np.sum(diff * diff) / max(sse, eps))
        q = 0.0
        for lag in range(1, max_lag + 1):
            q += float(autocorr[b, lag - 1] * autocorr[b, lag - 1]) / max(float(n - lag), 1.0)
        q *= float(n) * float(n + 2)
        metrics[b] = np.array(
            [
                mean_residual,
                float(np.sqrt(max(residual_energy / max(1, n - 1), 0.0))),
                float(np.mean(np.abs(residual))),
                float(np.sqrt(sse / max(1, n))),
                float(np.max(np.abs(residual))),
                100.0 * (1.0 - float(np.sqrt(sse)) / max(float(np.sqrt(actual_energy)), eps)),
                dw,
                q,
            ],
            dtype=np.float32,
        )
    return ResidualDiagnosticsResult(metrics=metrics, autocorr=autocorr)
