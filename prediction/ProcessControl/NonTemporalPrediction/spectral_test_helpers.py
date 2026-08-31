# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Shared invalid-input checks for spectral CSR reference implementations."""

from __future__ import annotations

import numpy as np
import pytest


def invalid_csr_cases() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (
            np.array([0, 2], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones(1, np.float32),
        ),
        (
            np.array([0, 1], dtype=np.int32),
            np.array([1], dtype=np.int32),
            np.ones(1, np.float32),
        ),
        (
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.ones(2, np.float32),
        ),
    ]


def check_basis_invalid(operator) -> None:
    for row_ptr, source, norm in invalid_csr_cases():
        with pytest.raises(ValueError):
            operator(row_ptr, source, norm, np.ones((1, 2), np.float32))


def check_basis_nonfinite(operator) -> None:
    with pytest.raises(ValueError):
        operator(
            np.array([0, 1], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([np.inf], dtype=np.float32),
            np.ones((1, 1), np.float32),
        )


def check_residual_invalid(operator) -> None:
    for row_ptr, source, weight in invalid_csr_cases():
        with pytest.raises(ValueError):
            operator(
                row_ptr,
                source,
                weight,
                np.ones((1, 2), np.float32),
                np.ones((1, 2), np.float32),
                0.1,
            )
