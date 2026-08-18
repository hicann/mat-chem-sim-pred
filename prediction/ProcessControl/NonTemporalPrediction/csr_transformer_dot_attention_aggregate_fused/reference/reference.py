# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""NumPy reference for TransformerConv dot-attention aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransformerAttentionInputs:
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray


def _validate_attention(query, key, value):
    query, key, value = np.asarray(query), np.asarray(key), np.asarray(value)
    for tensor in (query, key, value):
        if tensor.dtype != np.float32:
            raise TypeError("attention tensors must be float32")
        if not np.isfinite(tensor).all():
            raise ValueError("attention tensors must be finite")
    if query.ndim != 3 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("query/key/value must have equal [N,H,C] shapes")
    _, heads, channels = query.shape
    if not 0 < heads <= 8 or not 0 < channels <= 32:
        raise ValueError("unsupported head/channel shapes")
    return TransformerAttentionInputs(query, key, value)


def _validate_csr(row_ptr, source_index, nodes: int):
    row_ptr, source_index = np.asarray(row_ptr), np.asarray(source_index)
    if row_ptr.dtype != np.int32 or source_index.dtype != np.int32:
        raise TypeError("CSR tensors must be int32")
    if row_ptr.shape != (nodes + 1,) or source_index.ndim != 1:
        raise ValueError("invalid CSR shapes")
    valid_endpoints = source_index.size > 0 and row_ptr[0] == 0
    valid_endpoints = valid_endpoints and row_ptr[-1] == source_index.size
    if not valid_endpoints:
        raise ValueError("invalid CSR endpoints")
    if np.any(row_ptr[1:] < row_ptr[:-1]) or np.max(np.diff(row_ptr)) > 256:
        raise ValueError("invalid or oversized segment")
    if np.any(source_index < 0) or np.any(source_index >= nodes):
        raise ValueError("source index out of range")
    return row_ptr, source_index


def _aggregate_target(output, target, row_ptr, source_index, inputs):
    begin, end = int(row_ptr[target]), int(row_ptr[target + 1])
    if begin == end:
        return
    sources = source_index[begin:end]
    scale = np.sqrt(np.float32(inputs.query.shape[-1]))
    logits = (inputs.query[target] * inputs.key[sources]).sum(-1) / scale
    weights = np.exp(logits - logits.max(0, keepdims=True))
    weights /= weights.sum(0, keepdims=True)
    output[target] = (weights[..., None] * inputs.value[sources]).sum(0)


def csr_transformer_dot_attention_aggregate_fused(
    row_ptr, source_index, query, key, value
):
    inputs = _validate_attention(query, key, value)
    row_ptr, source_index = _validate_csr(row_ptr, source_index, inputs.query.shape[0])
    output = np.zeros_like(inputs.value)
    for target in range(inputs.query.shape[0]):
        _aggregate_target(output, target, row_ptr, source_index, inputs)
    return output
