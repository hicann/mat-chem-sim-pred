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

from __future__ import annotations

from pid_fopdt_closed_loop_reference import (
    RESULT_NAMES,
    FopdtClosedLoopCase,
    benchmark_reference,
    fopdt_closed_loop_score as fopdt_batch_rollout_score,
    make_fopdt_closed_loop_case as make_fopdt_batch_rollout_case,
)

__all__ = [
    "RESULT_NAMES",
    "FopdtClosedLoopCase",
    "benchmark_reference",
    "fopdt_batch_rollout_score",
    "make_fopdt_batch_rollout_case",
]
