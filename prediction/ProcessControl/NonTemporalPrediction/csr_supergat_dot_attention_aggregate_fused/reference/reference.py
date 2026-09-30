# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NumPy reference for SuperGAT SD attention and aggregation."""

from __future__ import annotations

__all__ = ["csr_supergat_dot_attention_aggregate_fused", "is_supported", "reference"]

import numpy as np


def csr_supergat_dot_attention_aggregate_fused(
    row_ptr, source_index, projected, negative_slope=0.2
):
    row_ptr = np.asarray(row_ptr)
    source_index = np.asarray(source_index)
    projected = np.asarray(projected)
    if row_ptr.dtype != np.int32 or source_index.dtype != np.int32:
        raise TypeError("CSR tensors must be int32")
    if projected.dtype != np.float32 or projected.ndim != 3:
        raise TypeError("projected must be FP32 [N,H,C]")
    nodes, heads, channels = projected.shape
    if not 0 < heads <= 8 or not 0 < channels <= 64:
        raise ValueError("unsupported head/channel shape")
    if row_ptr.shape != (nodes + 1,) or source_index.ndim != 1:
        raise ValueError("invalid CSR shape")
    if len(source_index) == 0 or row_ptr[0] != 0 or row_ptr[-1] != len(source_index):
        raise ValueError("invalid CSR endpoints")
    if np.any(row_ptr[1:] < row_ptr[:-1]) or np.max(np.diff(row_ptr)) > 512:
        raise ValueError("invalid or oversized row")
    if np.any(source_index < 0) or np.any(source_index >= nodes):
        raise ValueError("source index out of range")
    if not np.isfinite(negative_slope) or not 0.0 <= negative_slope <= 1.0:
        raise ValueError("invalid negative_slope")
    output = np.zeros_like(projected)
    scale = np.sqrt(np.float32(channels))
    for target in range(nodes):
        begin, end = int(row_ptr[target]), int(row_ptr[target + 1])
        if begin == end:
            continue
        source = source_index[begin:end]
        score = (projected[target] * projected[source]).sum(-1) / scale
        score = np.where(score >= 0.0, score, score * negative_slope)
        weight = np.exp(score - score.max(0, keepdims=True))
        weight /= weight.sum(0, keepdims=True)
        output[target] = (weight[..., None] * projected[source]).sum(0)
    return output


def is_supported(*args, requires_grad=False, **kwargs):
    if requires_grad or any(not np.asarray(value).flags.c_contiguous for value in args):
        return False
    try:
        csr_supergat_dot_attention_aggregate_fused(*args, **kwargs)
    except (TypeError, ValueError, IndexError):
        return False
    return True


reference = csr_supergat_dot_attention_aggregate_fused
