# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""NumPy reference for CsrGatAttentionAggregateFused."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GatAttentionInputs:
    """Dense tensors and activation parameter used by GAT attention."""

    projected: np.ndarray
    attention_source: np.ndarray
    attention_target: np.ndarray
    negative_slope: float = 0.2


def _coerce_inputs(inputs, legacy) -> GatAttentionInputs:
    if isinstance(inputs, GatAttentionInputs):
        if legacy:
            raise TypeError("legacy arguments cannot follow GatAttentionInputs")
        return inputs
    if len(legacy) != 3:
        raise TypeError("expected projected, source attention, target attention, slope")
    return GatAttentionInputs(inputs, legacy[0], legacy[1], legacy[2])


def _validate_dense(inputs: GatAttentionInputs):
    arrays = (
        ("projected", inputs.projected),
        ("attention_source", inputs.attention_source),
        ("attention_target", inputs.attention_target),
    )
    for name, value in arrays:
        if value.dtype != np.float32:
            raise TypeError(f"{name} must be float32")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if inputs.projected.ndim != 3:
        raise ValueError("projected must have shape [N,H,C]")
    nodes, heads, channels = inputs.projected.shape
    if min(nodes, heads, channels) < 1:
        raise ValueError("N, H, and C must be positive")
    expected = (heads, channels)
    if inputs.attention_source.shape != expected:
        raise ValueError("attention_source must have shape [H,C]")
    if inputs.attention_target.shape != expected:
        raise ValueError("attention_target must have shape [H,C]")
    if not np.isfinite(inputs.negative_slope):
        raise ValueError("negative_slope must be finite")
    if not 0.0 <= inputs.negative_slope <= 1.0:
        raise ValueError("negative_slope must be in [0,1]")
    return nodes


def _validate_csr(row_ptr, source_index, nodes: int) -> None:
    if row_ptr.dtype != np.int32 or source_index.dtype != np.int32:
        raise TypeError("CSR indices must be int32")
    if row_ptr.ndim != 1 or source_index.ndim != 1:
        raise ValueError("CSR tensors must be rank one")
    edges = source_index.size
    if row_ptr.shape != (nodes + 1,) or row_ptr[0] != 0:
        raise ValueError("row_ptr does not start at zero or match N")
    if row_ptr[-1] != edges or np.any(row_ptr[1:] < row_ptr[:-1]):
        raise ValueError("row_ptr does not span a monotonic edge array")
    if np.any(source_index < 0) or np.any(source_index >= nodes):
        raise ValueError("source index is out of range")


def _aggregate_target(output, target, row_ptr, source_index, inputs) -> None:
    begin, end = int(row_ptr[target]), int(row_ptr[target + 1])
    sources = source_index[begin:end]
    if sources.size == 0:
        return
    projected = inputs.projected
    source_score = (projected[sources] * inputs.attention_source[None]).sum(-1)
    target_score = (projected[target] * inputs.attention_target).sum(-1)
    logits = source_score + target_score[None]
    logits = np.where(logits >= 0.0, logits, logits * inputs.negative_slope)
    weights = np.exp(logits - logits.max(axis=0, keepdims=True))
    weights /= weights.sum(axis=0, keepdims=True)
    output[target] = (weights[..., None] * projected[sources]).sum(axis=0)


def csr_gat_attention_aggregate(row_ptr, source_index, inputs, *legacy) -> np.ndarray:
    """Aggregate GAT messages; legacy six-positional-argument calls remain valid."""
    values = _coerce_inputs(inputs, legacy)
    nodes = _validate_dense(values)
    _validate_csr(row_ptr, source_index, nodes)
    output = np.zeros_like(values.projected)
    for target in range(nodes):
        _aggregate_target(output, target, row_ptr, source_index, values)
    return output
