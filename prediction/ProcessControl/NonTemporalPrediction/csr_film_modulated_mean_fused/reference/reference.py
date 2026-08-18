# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""NumPy reference and dispatch eligibility for FiLM modulated mean."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FilmInputs:
    projected: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    apply_relu: bool = True


def _coerce_inputs(value, legacy, options):
    unknown = set(options) - {"apply_relu", "requires_grad"}
    if unknown:
        raise TypeError(f"unexpected keyword arguments: {sorted(unknown)}")
    if isinstance(value, FilmInputs):
        if legacy or "apply_relu" in options:
            raise TypeError("FilmInputs cannot be combined with legacy values")
        return value
    if len(legacy) not in (2, 3):
        raise TypeError("expected projected, beta, gamma, and optional apply_relu")
    apply_relu = legacy[2] if len(legacy) == 3 else options.get("apply_relu", True)
    return FilmInputs(value, legacy[0], legacy[1], apply_relu)


def _valid_arrays(row_ptr, source_index, values):
    arrays = (row_ptr, source_index, values.projected, values.beta, values.gamma)
    if not all(isinstance(value, np.ndarray) for value in arrays):
        return False
    if not all(value.flags.c_contiguous for value in arrays):
        return False
    if row_ptr.dtype != np.int32 or source_index.dtype != np.int32:
        return False
    return all(
        value.dtype == np.float32
        for value in (values.projected, values.beta, values.gamma)
    )


def _valid_shapes(row_ptr, source_index, values):
    if row_ptr.ndim != 1 or source_index.ndim != 1:
        return False
    if any(value.ndim != 2 for value in (values.projected, values.beta, values.gamma)):
        return False
    nodes, channels = values.projected.shape
    edges = source_index.size
    return (
        nodes > 0
        and edges > 0
        and 0 < channels <= 512
        and values.beta.shape == values.projected.shape
        and values.gamma.shape == values.projected.shape
        and row_ptr.shape == (nodes + 1,)
    )


def _valid_graph(row_ptr, source_index, nodes):
    if row_ptr[0] != 0 or row_ptr[-1] != source_index.size:
        return False
    degrees = np.diff(row_ptr)
    if np.any(degrees < 0) or np.max(degrees) > 2048:
        return False
    return not np.any(source_index < 0) and not np.any(source_index >= nodes)


def is_supported(row_ptr, source_index, value, *legacy, **options):
    values = _coerce_inputs(value, legacy, options)
    if options.get("requires_grad", False) or not _valid_arrays(
        row_ptr, source_index, values
    ):
        return False
    if not _valid_shapes(row_ptr, source_index, values):
        return False
    if not _valid_graph(row_ptr, source_index, values.projected.shape[0]):
        return False
    if not isinstance(values.apply_relu, (bool, np.bool_)):
        return False
    return all(
        np.all(np.isfinite(item))
        for item in (values.projected, values.beta, values.gamma)
    )


def csr_film_modulated_mean_fused(row_ptr, source_index, value, *legacy, **options):
    values = _coerce_inputs(value, legacy, options)
    if not is_supported(row_ptr, source_index, values):
        raise ValueError("unsupported or malformed FiLM modulated-mean inputs")
    output = np.zeros_like(values.projected)
    for target in range(values.projected.shape[0]):
        begin = int(row_ptr[target])
        end = int(row_ptr[target + 1])
        if begin == end:
            continue
        messages = (
            values.gamma[target] * values.projected[source_index[begin:end]]
            + values.beta[target]
        )
        if values.apply_relu:
            messages = np.maximum(messages, 0.0)
        output[target] = messages.mean(axis=0)
    return output
