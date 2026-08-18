# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Dependency-free CPU reference for AutoformerInferenceAggregateFused."""

import math


def _first(sequence, name):
    if not sequence:
        raise ValueError(f"{name} must be a non-empty rank-4 tensor")
    return sequence[0]


def _tensor4(value, name):
    first_batch = _first(value, name)
    first_head = _first(first_batch, name)
    first_channel = _first(first_head, name)
    _first(first_channel, name)
    shape = (len(value), len(first_batch), len(first_head), len(first_channel))
    for batch in value:
        if len(batch) != shape[1]:
            raise ValueError(f"{name} must be rectangular")
        for head in batch:
            if len(head) != shape[2] or any(len(row) != shape[3] for row in head):
                raise ValueError(f"{name} must be rectangular")
    return shape


def reference(values, correlation, top_k):
    batch, heads, channels, length = _tensor4(values, "values")
    if _tensor4(correlation, "correlation") != (batch, heads, channels, length):
        raise ValueError("shape mismatch")
    if not 1 <= top_k <= min(16, length):
        raise ValueError("invalid top_k")
    delay, weights = [], []
    scale = float(heads * channels)
    for b in range(batch):
        mean = [
            sum(correlation[b][h][c][t] for h in range(heads) for c in range(channels))
            / scale
            for t in range(length)
        ]
        selected = sorted(range(length), key=lambda t: (-mean[t], t))[:top_k]
        maximum = mean[selected[0]]
        exponent = [math.exp(mean[t] - maximum) for t in selected]
        total = sum(exponent)
        delay.append(selected)
        weights.append([value / total for value in exponent])
    return _lag_reference(values, delay, weights)


def _lag_reference(values, delay, weights):
    batch, _, _, _ = _tensor4(values, "values")
    result = []
    for batch_index in range(batch):
        result.append(
            _lag_batch(values[batch_index], delay[batch_index], weights[batch_index])
        )
    return result


def _lag_batch(batch_values, delay, weights):
    return [
        [_lag_row(channel, delay, weights) for channel in head]
        for head in batch_values
    ]


def _lag_row(row, delay, weights):
    length = len(row)
    return [
        sum(
            weights[index] * row[(position + delay[index]) % length]
            for index in range(len(delay))
        )
        for position in range(length)
    ]
