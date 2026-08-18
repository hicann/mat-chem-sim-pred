# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""NumPy reference for two-hop LightGCN propagation and weighted sum."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LightGcnInputs:
    norm: np.ndarray
    features: np.ndarray
    weights: tuple[float, float, float]


def _coerce_inputs(value, legacy):
    if isinstance(value, LightGcnInputs):
        if legacy:
            raise TypeError("LightGcnInputs cannot be combined with legacy values")
        return value
    if len(legacy) != 4:
        raise TypeError("expected norm, features, alpha0, alpha1, alpha2")
    return LightGcnInputs(value, legacy[0], tuple(legacy[1:]))


def _validate_graph(row_ptr, source_index, values):
    nodes = values.features.shape[0]
    if (
        row_ptr.shape != (nodes + 1,)
        or source_index.ndim != 1
        or values.norm.shape != source_index.shape
    ):
        raise ValueError("invalid CSR shapes")
    if len(source_index) == 0 or row_ptr[0] != 0 or row_ptr[-1] != len(source_index):
        raise ValueError("invalid CSR endpoints")
    if (
        np.any(row_ptr[1:] < row_ptr[:-1])
        or np.any(source_index < 0)
        or np.any(source_index >= nodes)
    ):
        raise ValueError("invalid CSR content")


def csr_lightgcn_k2_weighted_sum_fused(row_ptr, source_index, value, *legacy):
    values = _coerce_inputs(value, legacy)
    row_ptr, source_index = np.asarray(row_ptr), np.asarray(source_index)
    norm, features = np.asarray(values.norm), np.asarray(values.features)
    values = LightGcnInputs(norm, features, values.weights)
    if row_ptr.dtype != np.int32 or source_index.dtype != np.int32:
        raise TypeError("CSR tensors must be int32")
    if norm.dtype != np.float32 or features.dtype != np.float32:
        raise TypeError("value tensors must be float32")
    if features.ndim != 2 or not 0 < features.shape[1] <= 512:
        raise ValueError("features must be supported [N,C]")
    _validate_graph(row_ptr, source_index, values)
    if not np.all(np.isfinite(values.weights)):
        raise ValueError("weights must be finite")

    def propagate(current):
        output = np.zeros_like(current)
        for target in range(features.shape[0]):
            begin, end = int(row_ptr[target]), int(row_ptr[target + 1])
            output[target] = (
                norm[begin:end, None] * current[source_index[begin:end]]
            ).sum(0)
        return output

    layer1 = propagate(features)
    layer2 = propagate(layer1)
    alpha0, alpha1, alpha2 = values.weights
    return alpha0 * features + alpha1 * layer1 + alpha2 * layer2
