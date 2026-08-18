# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Guarded framework adapter; unsupported cases call the exact fallback."""

_UINT32_MAX = (1 << 32) - 1
_MAX_WIDTH = 16384


def _resident_contiguous(*tensors):
    return all(
        getattr(tensor, "device", None) is not None
        and tensor.device.type == "npu"
        and tensor.is_contiguous()
        for tensor in tensors
    )


def supports_custom(query_key, value, indices):
    try:
        return (
            _resident_contiguous(query_key, value, indices)
            and tuple(query_key.shape) == tuple(value.shape)
            and len(query_key.shape) == 3
            and len(indices.shape) == 2
            and str(query_key.dtype) == "torch.float32"
            and str(value.dtype) == "torch.float32"
            and str(indices.dtype) == "torch.int64"
            and all(0 < dim <= _UINT32_MAX for dim in query_key.shape)
            and all(0 < dim <= _UINT32_MAX for dim in indices.shape)
            and query_key.shape[0] == indices.shape[0]
            and query_key.shape[2] <= _MAX_WIDTH
            and query_key.shape[2] % 8 == 0
        )
    except (AttributeError, IndexError, TypeError):
        return False


def dispatch(query_key, value, indices, *, custom_call, fallback):
    """Use custom_call only for the audited contract; fallback owns all other shapes."""
    args = (
        query_key,
        value,
        indices,
    )
    if supports_custom(query_key, value, indices) and custom_call is not None:
        return custom_call(*args)
    return fallback(*args)
