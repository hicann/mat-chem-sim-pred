# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NumPy reference for CsrGenConvSoftmaxAggregateFused."""

from __future__ import annotations

import numpy as np


def csr_genconv_softmax_aggregate(
    row_ptr: np.ndarray,
    source_index: np.ndarray,
    features: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if row_ptr.dtype != np.int32 or source_index.dtype != np.int32:
        raise TypeError("row_ptr and source_index must be int32")
    if features.dtype != np.float32 or features.ndim != 2:
        raise TypeError("features must be a two-dimensional float32 array")
    if row_ptr.ndim != 1 or source_index.ndim != 1:
        raise ValueError("CSR inputs must be one-dimensional")
    nodes, _ = features.shape
    edges = source_index.size
    if row_ptr.size != nodes + 1 or row_ptr[0] != 0 or row_ptr[-1] != edges:
        raise ValueError("row_ptr must span exactly all source indices")
    if np.any(row_ptr[1:] < row_ptr[:-1]):
        raise ValueError("row_ptr must be non-decreasing")
    if np.any(source_index < 0) or np.any(source_index >= nodes):
        raise ValueError("source_index contains an out-of-range node")
    if not np.isfinite(temperature):
        raise ValueError("temperature must be finite")

    output = np.zeros_like(features)
    for target in range(nodes):
        begin, end = int(row_ptr[target]), int(row_ptr[target + 1])
        if begin == end:
            continue
        message = np.maximum(features[source_index[begin:end]], 0.0) + np.float32(
            1.0e-7
        )
        logits = message * np.float32(temperature)
        weights = np.exp(logits - logits.max(axis=0, keepdims=True))
        weights /= weights.sum(axis=0, keepdims=True)
        output[target] = (weights * message).sum(axis=0)
    return output
