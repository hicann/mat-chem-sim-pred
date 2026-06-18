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


@dataclass
class BasisGemmCase:
    family: str
    y_centered: np.ndarray
    basis_t: np.ndarray
    basis_norm: np.ndarray
    y_energy: np.ndarray
    truth_index: int


def delay_steps(delay: float, dt: float) -> int:
    if dt <= 0.0 or delay <= 0.0:
        return 0
    return max(0, int(round(delay / dt)))


def response_value(
    family: str,
    index: int,
    delay: int,
    amplitude: float,
    alpha1: float,
    alpha2: float,
    inv_n: float,
    state: list[float],
) -> float:
    du = amplitude if index >= delay else 0.0
    if family == "IPDT":
        state[0] += du * inv_n
        return state[0]
    if family == "SOPDT":
        state[0] = alpha1 * state[0] + (1.0 - alpha1) * du
        state[1] = alpha2 * state[1] + (1.0 - alpha2) * state[0]
        return state[1]
    state[0] = alpha1 * state[0] + (1.0 - alpha1) * du
    return state[0]


def make_basis_gemm_case(family: str, batch: int = 8, n: int = 256, candidates: int = 65) -> BasisGemmCase:
    family = family.upper()
    if family not in {"FOPDT", "IPDT", "SOPDT"}:
        raise ValueError("family must be FOPDT, IPDT, or SOPDT")

    y_centered = np.zeros((batch, n), dtype=np.float32)
    basis_t = np.zeros((n, candidates), dtype=np.float32)
    basis_norm = np.zeros(candidates, dtype=np.float32)
    y_energy = np.zeros(batch, dtype=np.float32)

    dt = 1.0
    truth_k = 1.2
    truth_t1 = 18.0
    truth_t2 = 7.0
    truth_l = 4.0
    truth_index = candidates // 2
    truth_delay = delay_steps(truth_l, dt)
    truth_alpha1 = float(np.exp(-dt / truth_t1))
    truth_alpha2 = float(np.exp(-dt / truth_t2))
    inv_n = 1.0 / max(1, n)

    for b in range(batch):
        state = [0.0, 0.0]
        amplitude = 1.0 + 0.03 * float(b % 11)
        for i in range(n):
            value = response_value(
                family, i, truth_delay, amplitude, truth_alpha1, truth_alpha2, inv_n, state
            )
            y_centered[b, i] = truth_k * value
            y_energy[b] += y_centered[b, i] * y_centered[b, i]

    for c in range(candidates):
        t1 = truth_t1
        t2 = truth_t2
        delay = truth_l
        if family == "IPDT":
            delay = max(0.0, truth_l + float(c % 17) - 8.0)
        elif family == "SOPDT":
            a = float(c % 13) - 6.0
            b = float((c // 13) % 11) - 5.0
            d = float((c // (13 * 11)) % 9) - 4.0
            t1 = max(1.0, truth_t1 * (1.0 + 0.04 * a))
            t2 = max(1.0, truth_t2 * (1.0 + 0.06 * b))
            delay = max(0.0, truth_l + d)
        else:
            a = float(c % 17) - 8.0
            d = float((c // 17) % 13)
            t1 = max(1.0, truth_t1 * (1.0 + 0.03 * a))
            delay = max(0.0, truth_l + d - 6.0)

        if c == truth_index:
            t1 = truth_t1
            t2 = truth_t2
            delay = truth_l
        elif abs(t1 - truth_t1) < 1.0e-6 and abs(t2 - truth_t2) < 1.0e-6 and abs(delay - truth_l) < 1.0e-6:
            delay += 1.0

        state = [0.0, 0.0]
        alpha1 = float(np.exp(-dt / t1))
        alpha2 = float(np.exp(-dt / t2))
        candidate_delay = delay_steps(delay, dt)
        for i in range(n):
            value = response_value(family, i, candidate_delay, 1.0, alpha1, alpha2, inv_n, state)
            basis_t[i, c] = value
            basis_norm[c] += value * value

    return BasisGemmCase(family, y_centered, basis_t, basis_norm, y_energy, truth_index)


def reduce_best(dot: np.ndarray, basis_norm: np.ndarray, y_energy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_norm = np.maximum(basis_norm[None, :], 1.0e-6)
    gain = dot / safe_norm
    sse = y_energy[:, None] - dot * dot / safe_norm
    sse = np.where((sse < 0.0) & (sse > -1.0e-3), 0.0, sse)
    sse = np.where(np.isfinite(sse), sse, np.finfo(np.float32).max)
    best_idx = np.argmin(sse, axis=1).astype(np.int32)
    row = np.arange(dot.shape[0])
    return sse[row, best_idx].astype(np.float32), gain[row, best_idx].astype(np.float32), best_idx


def fit_reference(case: BasisGemmCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dot = case.y_centered @ case.basis_t
    best_sse, best_k, best_idx = reduce_best(dot, case.basis_norm, case.y_energy)
    return dot.astype(np.float32), best_sse, best_k, best_idx


def benchmark_reference(family: str, batch: int, n: int, candidates: int, iters: int = 20) -> dict[str, float]:
    case = make_basis_gemm_case(family, batch=batch, n=n, candidates=candidates)
    fit_reference(case)
    start = time.perf_counter()
    for _ in range(iters):
        fit_reference(case)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / iters
    _, best_sse, _, best_idx = fit_reference(case)
    _, _, best_k, _ = fit_reference(case)
    expected_k = np.array([1.2 * (1.0 + 0.03 * float(i % 11)) for i in range(batch)], dtype=np.float32)
    return {
        "family": family.upper(),
        "batch": batch,
        "n": n,
        "candidates": candidates,
        "numpy_ms_avg": elapsed_ms,
        "max_abs_best_sse": float(np.max(np.abs(best_sse))),
        "max_gain_abs_err": float(np.max(np.abs(best_k - expected_k))),
        "best_idx_valid_rate": float(np.mean((best_idx >= 0) & (best_idx < candidates))),
    }
