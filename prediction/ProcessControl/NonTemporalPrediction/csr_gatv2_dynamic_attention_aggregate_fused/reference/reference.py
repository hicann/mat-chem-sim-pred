# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""NumPy reference for GATv2 dynamic attention aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gatv2AttentionInputs:
    source_projected: np.ndarray
    target_projected: np.ndarray
    attention: np.ndarray
    negative_slope: float = 0.2


def _normalize_inputs(inputs, legacy) -> Gatv2AttentionInputs:
    if isinstance(inputs, Gatv2AttentionInputs):
        if legacy:
            raise TypeError("legacy arguments cannot follow Gatv2AttentionInputs")
        return inputs
    if len(legacy) != 3:
        raise TypeError(
            "expected source projection, target projection, attention, slope"
        )
    return Gatv2AttentionInputs(inputs, legacy[0], legacy[1], legacy[2])


def _validate_projections(inputs: Gatv2AttentionInputs):
    source = np.asarray(inputs.source_projected)
    target = np.asarray(inputs.target_projected)
    attention = np.asarray(inputs.attention)
    for tensor in (source, target, attention):
        if tensor.dtype != np.float32:
            raise TypeError("attention tensors must be float32")
        if not np.isfinite(tensor).all():
            raise ValueError("attention tensors must be finite")
    if source.ndim != 3 or source.shape != target.shape:
        raise ValueError("projections must have equal [N,H,C] shapes")
    nodes, heads, channels = source.shape
    if not 0 < heads <= 8 or not 0 < channels <= 32:
        raise ValueError("unsupported head/channel shapes")
    if attention.shape != (heads, channels):
        raise ValueError("attention must have shape [H,C]")
    slope = inputs.negative_slope
    if not np.isfinite(slope) or not 0 <= slope <= 1:
        raise ValueError("invalid negative_slope")
    return source, target, attention


def _validate_csr(row_ptr, source_index, nodes: int):
    row_ptr, source_index = np.asarray(row_ptr), np.asarray(source_index)
    if row_ptr.dtype != np.int32 or source_index.dtype != np.int32:
        raise TypeError("CSR tensors must be int32")
    if row_ptr.shape != (nodes + 1,) or source_index.ndim != 1:
        raise ValueError("invalid CSR shapes")
    if source_index.size == 0 or row_ptr[0] != 0 or row_ptr[-1] != source_index.size:
        raise ValueError("invalid CSR endpoints")
    if np.any(row_ptr[1:] < row_ptr[:-1]) or np.max(np.diff(row_ptr)) > 256:
        raise ValueError("invalid or oversized segment")
    if np.any(source_index < 0) or np.any(source_index >= nodes):
        raise ValueError("source index out of range")
    return row_ptr, source_index


def _aggregate_row(output, target, row_ptr, source_index, inputs) -> None:
    begin, end = int(row_ptr[target]), int(row_ptr[target + 1])
    if begin == end:
        return
    sources = source_index[begin:end]
    dynamic = inputs.source_projected[sources] + inputs.target_projected[target]
    dynamic = np.where(dynamic >= 0, dynamic, dynamic * inputs.negative_slope)
    logits = (dynamic * inputs.attention).sum(-1)
    weights = np.exp(logits - logits.max(0, keepdims=True))
    weights /= weights.sum(0, keepdims=True)
    output[target] = (weights[..., None] * inputs.source_projected[sources]).sum(0)


def csr_gatv2_dynamic_attention_aggregate_fused(row_ptr, source_index, inputs, *legacy):
    """Run the GATv2 reference while accepting the original positional form."""
    values = _normalize_inputs(inputs, legacy)
    source, target, attention = _validate_projections(values)
    values = Gatv2AttentionInputs(source, target, attention, values.negative_slope)
    row_ptr, source_index = _validate_csr(row_ptr, source_index, source.shape[0])
    output = np.zeros_like(source)
    for target_index in range(source.shape[0]):
        _aggregate_row(output, target_index, row_ptr, source_index, values)
    return output
