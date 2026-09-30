# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NumPy reference for AGNN cosine attention aggregation."""

from __future__ import annotations

import numpy as np


def _validate_inputs(row_ptr, source_index, features, normalized, beta):
    if row_ptr.dtype != np.int32 or source_index.dtype != np.int32:
        raise TypeError("CSR tensors must be int32")
    if features.dtype != np.float32 or normalized.dtype != np.float32:
        raise TypeError("feature tensors must be float32")
    if features.ndim != 2 or features.shape != normalized.shape:
        raise ValueError("features and normalized must have equal [N,C] shapes")
    nodes, channels = features.shape
    if not 0 < nodes <= np.iinfo(np.int32).max or not 0 < channels <= 512:
        raise ValueError("unsupported node or channel count")
    if row_ptr.shape != (nodes + 1,) or source_index.ndim != 1:
        raise ValueError("invalid CSR shapes")
    if len(source_index) == 0 or row_ptr[0] != 0 or row_ptr[-1] != len(source_index):
        raise ValueError("invalid CSR endpoints")
    if np.any(row_ptr[1:] < row_ptr[:-1]):
        raise ValueError("row_ptr must be monotonic")
    if np.any(source_index < 0) or np.any(source_index >= nodes):
        raise ValueError("source index out of range")
    if np.max(np.diff(row_ptr)) > 2048 or not np.isfinite(beta):
        raise ValueError("unsupported segment or beta")


def csr_agnn_cosine_attention_aggregate_fused(
    row_ptr, source_index, features, normalized, beta
):
    row_ptr = np.asarray(row_ptr)
    source_index = np.asarray(source_index)
    features = np.asarray(features)
    normalized = np.asarray(normalized)
    _validate_inputs(row_ptr, source_index, features, normalized, beta)
    nodes = features.shape[0]
    output = np.zeros_like(features)
    for target in range(nodes):
        begin, end = int(row_ptr[target]), int(row_ptr[target + 1])
        if begin == end:
            continue
        sources = source_index[begin:end]
        logits = float(beta) * (normalized[target] * normalized[sources]).sum(-1)
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        output[target] = (weights[:, None] * features[sources]).sum(0)
    return output
