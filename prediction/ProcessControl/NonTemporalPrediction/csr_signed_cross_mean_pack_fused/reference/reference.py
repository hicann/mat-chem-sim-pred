# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""NumPy reference for the SignedConv cross-sign mean pack."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignedInputs:
    negative_row_ptr: np.ndarray
    negative_source_index: np.ndarray
    features: np.ndarray
    positive_inverse_degree: np.ndarray
    negative_inverse_degree: np.ndarray


def _coerce_inputs(value, legacy):
    if isinstance(value, SignedInputs):
        if legacy:
            raise TypeError("SignedInputs cannot be combined with legacy values")
        return value
    if len(legacy) != 4:
        raise TypeError("expected negative CSR, features, and inverse degrees")
    return SignedInputs(value, legacy[0], legacy[1], legacy[2], legacy[3])


def _validate_csr(row_ptr, source, nodes, name):
    if row_ptr.dtype != np.int32 or source.dtype != np.int32:
        raise TypeError(f"{name} CSR tensors must use int32")
    if row_ptr.ndim != 1 or source.ndim != 1 or row_ptr.size != nodes + 1:
        raise ValueError(f"invalid {name} CSR shape")
    if row_ptr[0] != 0 or row_ptr[-1] != source.size:
        raise ValueError(f"invalid {name} CSR endpoints")
    if np.any(row_ptr[1:] < row_ptr[:-1]):
        raise ValueError(f"{name} row_ptr must be monotonic")
    if np.any(source < 0) or np.any(source >= nodes):
        raise ValueError(f"{name} source index is out of range")


def _validate_values(values, nodes):
    for inverse in (values.positive_inverse_degree, values.negative_inverse_degree):
        if inverse.dtype != np.float32:
            raise TypeError("inverse degree must use float32")
        if inverse.shape != (nodes,) or not np.isfinite(inverse).all():
            raise ValueError("invalid inverse degree")


def csr_signed_cross_mean_pack_fused(
    positive_row_ptr, positive_source_index, value, *legacy
):
    """Return [pos<-pos, pos<-neg, neg<-pos, neg<-neg] mean features."""
    values = _coerce_inputs(value, legacy)
    if values.features.dtype != np.float32 or values.features.ndim != 2:
        raise TypeError("features must be a rank-2 float32 array")
    nodes, width = values.features.shape
    if width == 0 or width % 2 or width > 128:
        raise ValueError("features must contain two equal channel halves")
    _validate_values(values, nodes)
    _validate_csr(positive_row_ptr, positive_source_index, nodes, "positive")
    _validate_csr(
        values.negative_row_ptr, values.negative_source_index, nodes, "negative"
    )
    channels = width // 2
    if channels % 8:
        raise ValueError("channels must be a multiple of 8")
    output = np.zeros((nodes, 4 * channels), dtype=np.float32)
    positive_first = slice(0, channels)
    positive_second = slice(2 * channels, 3 * channels)
    negative_first = slice(channels, 2 * channels)
    negative_second = slice(3 * channels, None)
    for target in range(nodes):
        begin, end = positive_row_ptr[slice(target, target + 2)]
        if end > begin:
            selected = values.features[positive_source_index[begin:end]]
            output[target, positive_first] = selected[:, positive_first].sum(0)
            output[target, positive_second] = selected[:, slice(channels, None)].sum(0)
        begin, end = values.negative_row_ptr[slice(target, target + 2)]
        if end > begin:
            selected = values.features[values.negative_source_index[begin:end]]
            output[target, negative_first] = selected[:, slice(channels, None)].sum(0)
            output[target, negative_second] = selected[:, positive_first].sum(0)
        output[target, positive_first] *= values.positive_inverse_degree[target]
        output[target, positive_second] *= values.positive_inverse_degree[target]
        output[target, negative_first] *= values.negative_inverse_degree[target]
        output[target, negative_second] *= values.negative_inverse_degree[target]
    return output
