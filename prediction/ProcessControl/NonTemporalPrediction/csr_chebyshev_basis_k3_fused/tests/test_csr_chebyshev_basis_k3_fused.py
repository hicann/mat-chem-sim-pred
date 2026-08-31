# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import sys
from pathlib import Path

import numpy as np
from reference import csr_chebyshev_basis_k3

sys.path.insert(0, str(Path(__file__).parents[2]))
from spectral_test_helpers import (
    check_basis_invalid,
    check_basis_nonfinite,
)


def test_two_node_recurrence() -> None:
    row_ptr = np.array([0, 1, 2], dtype=np.int32)
    source = np.array([1, 0], dtype=np.int32)
    norm = np.ones(2, dtype=np.float32)
    features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    actual = csr_chebyshev_basis_k3(row_ptr, source, norm, features)
    np.testing.assert_allclose(actual[0], features)
    np.testing.assert_allclose(actual[1], features[::-1])
    np.testing.assert_allclose(actual[2], features)


def test_weighted_rows_and_empty_row() -> None:
    row_ptr = np.array([0, 2, 2, 3], dtype=np.int32)
    source = np.array([0, 1, 2], dtype=np.int32)
    norm = np.array([0.25, 0.75, -1.0], dtype=np.float32)
    features = np.arange(6, dtype=np.float32).reshape(3, 2)
    actual = csr_chebyshev_basis_k3(row_ptr, source, norm, features)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[1, 1], 0.0)


def test_negative_norm_is_supported() -> None:
    result = csr_chebyshev_basis_k3(
        np.array([0, 1], dtype=np.int32),
        np.array([0], dtype=np.int32),
        np.array([-0.5], dtype=np.float32),
        np.array([[2.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(result[:, 0, 0], [2.0, -1.0, -1.0])


def test_invalid_csr_is_rejected() -> None:
    check_basis_invalid(csr_chebyshev_basis_k3)


def test_non_finite_norm_is_rejected() -> None:
    check_basis_nonfinite(csr_chebyshev_basis_k3)
