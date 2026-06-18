#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / "common"))
from pid_basis_gemm_reference import fit_reference, make_basis_gemm_case, reduce_best


def test_sopdt_basis_gemm_best_candidate():
    case = make_basis_gemm_case("SOPDT", batch=8, n=256, candidates=65)
    dot, best_sse, best_k, best_idx = fit_reference(case)
    reduced_sse, reduced_k, reduced_idx = reduce_best(dot, case.basis_norm, case.y_energy)

    assert dot.shape == (8, 65)
    assert best_sse.shape == (8,)
    assert np.all(best_idx == case.truth_index)
    assert np.allclose(best_sse, reduced_sse, atol=1.0e-6)
    assert np.allclose(best_k, reduced_k, atol=1.0e-6)
    assert np.array_equal(best_idx, reduced_idx)


def test_sopdt_gain_is_close_to_truth():
    case = make_basis_gemm_case("SOPDT", batch=4, n=512, candidates=97)
    _, best_sse, best_k, best_idx = fit_reference(case)

    assert np.max(np.abs(best_sse)) < 1.0e-2
    assert np.all(best_idx == case.truth_index)
    assert np.allclose(best_k, np.array([1.2, 1.236, 1.272, 1.308], dtype=np.float32), atol=2.0e-3)
