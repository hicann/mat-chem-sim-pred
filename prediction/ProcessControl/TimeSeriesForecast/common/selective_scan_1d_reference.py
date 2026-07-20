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


@dataclass(frozen=True)
class SelectiveScan1DCase:
    u: np.ndarray
    delta: np.ndarray
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray


def make_selective_scan_1d_case(
    batch: int = 2,
    length: int = 16,
    dim: int = 4,
    state: int = 8,
    seed: int = 2026,
) -> SelectiveScan1DCase:
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 0.25, size=(batch, length, dim)).astype(np.float32)
    delta = rng.uniform(0.01, 0.08, size=(batch, length, dim)).astype(np.float32)
    a = -rng.uniform(0.01, 0.20, size=(dim, state)).astype(np.float32)
    b = rng.normal(0.0, 0.20, size=(batch, length, state)).astype(np.float32)
    c = rng.normal(0.0, 0.20, size=(batch, length, state)).astype(np.float32)
    d = rng.normal(0.0, 0.05, size=(dim,)).astype(np.float32)
    return SelectiveScan1DCase(u=u, delta=delta, a=a, b=b, c=c, d=d)


def selective_scan_1d(
    u: np.ndarray,
    delta: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    u = np.asarray(u, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    c = np.asarray(c, dtype=np.float32)
    d = np.asarray(d, dtype=np.float32)
    if u.ndim != 3:
        raise ValueError("u must have shape [batch, length, dim]")
    batch, length, dim = u.shape
    if delta.shape != (batch, length, dim):
        raise ValueError("delta must have shape [batch, length, dim]")
    if a.ndim != 2 or a.shape[0] != dim:
        raise ValueError("a must have shape [dim, state]")
    state = a.shape[1]
    if b.shape != (batch, length, state):
        raise ValueError("b must have shape [batch, length, state]")
    if c.shape != (batch, length, state):
        raise ValueError("c must have shape [batch, length, state]")
    if d.shape != (dim,):
        raise ValueError("d must have shape [dim]")

    state_value = np.zeros((batch, dim, state), dtype=np.float32)
    out = np.zeros((batch, length, dim), dtype=np.float32)
    for t in range(length):
        decay = np.exp(delta[:, t, :, None] * a[None, :, :]).astype(np.float32)
        update = delta[:, t, :, None] * b[:, t, None, :] * u[:, t, :, None]
        state_value = decay * state_value + update
        out[:, t, :] = np.sum(state_value * c[:, t, None, :], axis=2) + u[:, t, :] * d[None, :]
    return out.astype(np.float32)
