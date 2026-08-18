# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Dependency-free CPU reference for ReformerLshQkvGather."""


def _matrix(value, name):
    if not isinstance(value, (list, tuple)) or not value or not value[0]:
        raise ValueError(f"{name} must be a non-empty matrix")
    width = len(value[0])
    if any(len(row) != width for row in value):
        raise ValueError(f"{name} must be rectangular")
    return len(value), width


def _tensor3(value, name):
    rows, length = _matrix(value, name)
    if not isinstance(value[0][0], (list, tuple)) or not value[0][0]:
        raise ValueError(f"{name} must be rank 3")
    width = len(value[0][0])
    if any(len(item) != width for row in value for item in row):
        raise ValueError(f"{name} must be rectangular")
    return rows, length, width


def _indices(indices, rows, source_length):
    _matrix(indices, "indices")
    if len(indices) != rows or any(
        i < 0 or i >= source_length for row in indices for i in row
    ):
        raise ValueError("indices outside source")


def reference(query_key, value, indices):
    rows, source_length, width = _tensor3(query_key, "query_key")
    if _tensor3(value, "value") != (rows, source_length, width):
        raise ValueError("source shape mismatch")
    _indices(indices, rows, source_length)
    return (
        [[query_key[r][i][:] for i in indices[r]] for r in range(rows)],
        [[value[r][i][:] for i in indices[r]] for r in range(rows)],
    )
