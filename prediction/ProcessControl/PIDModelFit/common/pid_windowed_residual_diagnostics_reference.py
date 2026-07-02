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
from numpy.lib.stride_tricks import sliding_window_view

from pid_residual_diagnostics_reference import (
    METRIC_NAMES,
    ResidualDiagnosticsResult,
    make_residual_diagnostics_case,
    residual_diagnostics,
    residual_diagnostics_cpu_loop,
)


@dataclass
class WindowedResidualDiagnosticsCase:
    actual: np.ndarray
    predicted: np.ndarray


@dataclass
class WindowedResidualDiagnosticsResult:
    metrics: np.ndarray
    autocorr: np.ndarray


def make_windowed_residual_diagnostics_case(
    batch: int = 64,
    n: int = 2048,
    seed: int = 2041,
) -> WindowedResidualDiagnosticsCase:
    case = make_residual_diagnostics_case(batch=batch, n=n, seed=seed)
    return WindowedResidualDiagnosticsCase(actual=case.actual, predicted=case.predicted)


def _validate_inputs(
    actual: np.ndarray,
    predicted: np.ndarray,
    window_size: int,
    stride: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(actual, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    if actual.ndim != 2 or predicted.shape != actual.shape:
        raise ValueError("actual and predicted must have shape [batch, n]")
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    if window_size > actual.shape[1]:
        raise ValueError("window_size must not exceed n")
    return actual, predicted


def windowed_residual_diagnostics(
    actual: np.ndarray,
    predicted: np.ndarray,
    window_size: int = 256,
    stride: int = 128,
    max_lag: int = 16,
) -> WindowedResidualDiagnosticsResult:
    actual, predicted = _validate_inputs(actual, predicted, window_size, stride, max_lag)
    actual_windows = sliding_window_view(actual, window_shape=window_size, axis=1)[:, ::stride, :]
    predicted_windows = sliding_window_view(predicted, window_shape=window_size, axis=1)[:, ::stride, :]
    batch, num_windows, _ = actual_windows.shape

    flat_actual = actual_windows.reshape(batch * num_windows, window_size)
    flat_predicted = predicted_windows.reshape(batch * num_windows, window_size)
    flat_result = residual_diagnostics(flat_actual, flat_predicted, max_lag=max_lag)
    return WindowedResidualDiagnosticsResult(
        metrics=flat_result.metrics.reshape(batch, num_windows, len(METRIC_NAMES)),
        autocorr=flat_result.autocorr.reshape(batch, num_windows, flat_result.autocorr.shape[1]),
    )


def windowed_residual_diagnostics_cpu_loop(
    actual: np.ndarray,
    predicted: np.ndarray,
    window_size: int = 256,
    stride: int = 128,
    max_lag: int = 16,
) -> WindowedResidualDiagnosticsResult:
    actual, predicted = _validate_inputs(actual, predicted, window_size, stride, max_lag)
    batch, n = actual.shape
    num_windows = 1 + (n - window_size) // stride
    max_lag = min(max_lag, max(1, window_size - 1))
    metrics = np.zeros((batch, num_windows, len(METRIC_NAMES)), dtype=np.float32)
    autocorr = np.zeros((batch, num_windows, max_lag), dtype=np.float32)

    for window in range(num_windows):
        start = window * stride
        end = start + window_size
        result: ResidualDiagnosticsResult = residual_diagnostics_cpu_loop(
            actual[:, start:end],
            predicted[:, start:end],
            max_lag=max_lag,
        )
        metrics[:, window, :] = result.metrics
        autocorr[:, window, :] = result.autocorr

    return WindowedResidualDiagnosticsResult(metrics=metrics, autocorr=autocorr)
