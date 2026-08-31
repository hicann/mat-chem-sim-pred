# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""NumPy reference for the default PyG PointTransformerConv message stage."""

from __future__ import annotations

import numpy as np


def csr_point_transformer_attention_aggregate_fused(*inputs):
    row_ptr, source_index, alpha_source, alpha_target, value, delta = inputs
    row_ptr, source_index = np.asarray(row_ptr), np.asarray(source_index)
    tensors = [np.asarray(x) for x in (alpha_source, alpha_target, value, delta)]
    alpha_source, alpha_target, value, delta = tensors
    if row_ptr.dtype != np.int32 or source_index.dtype != np.int32:
        raise TypeError("CSR tensors must be int32")
    if any(tensor.dtype != np.float32 for tensor in tensors):
        raise TypeError("feature tensors must be float32")
    if alpha_source.ndim != 2 or alpha_source.shape != alpha_target.shape:
        raise ValueError("alpha tensors must have equal [N,C] shapes")
    if value.shape != alpha_source.shape:
        raise ValueError("value must match alpha shape")
    nodes, channels = value.shape
    if not 0 < channels <= 128 or delta.shape != (len(source_index), channels):
        raise ValueError("invalid delta/channel shape")
    if row_ptr.shape != (nodes + 1,) or len(source_index) == 0:
        raise ValueError("invalid CSR shape")
    if row_ptr[0] != 0 or row_ptr[-1] != len(source_index):
        raise ValueError("invalid CSR endpoints")
    if np.any(row_ptr[1:] < row_ptr[:-1]) or np.max(np.diff(row_ptr)) > 512:
        raise ValueError("invalid or oversized row")
    if np.any(source_index < 0) or np.any(source_index >= nodes):
        raise ValueError("source index out of range")
    output = np.zeros_like(value)
    for target in range(nodes):
        begin, end = int(row_ptr[target]), int(row_ptr[target + 1])
        if begin == end:
            continue
        source = source_index[begin:end]
        score = alpha_target[target] - alpha_source[source] + delta[begin:end]
        weight = np.exp(score - score.max(0, keepdims=True))
        weight /= weight.sum(0, keepdims=True)
        output[target] = (weight * (value[source] + delta[begin:end])).sum(0)
    return output


def is_supported(*args, requires_grad=False, **kwargs):
    if requires_grad or any(not np.asarray(value).flags.c_contiguous for value in args):
        return False
    try:
        csr_point_transformer_attention_aggregate_fused(*args, **kwargs)
    except (TypeError, ValueError, IndexError):
        return False
    return True


reference = csr_point_transformer_attention_aggregate_fused
