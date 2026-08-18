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
_MAX_SEQUENCE_LENGTH = 4096


def _resident_contiguous(*tensors):
    return all(
        getattr(tensor, "device", None) is not None
        and tensor.device.type == "npu"
        and tensor.is_contiguous()
        for tensor in tensors
    )


def supports_custom(values, correlation, top_k):
    try:
        return (
            _resident_contiguous(values, correlation)
            and len(values.shape) == 4
            and tuple(values.shape) == tuple(correlation.shape)
            and str(values.dtype) == "torch.float32"
            and str(correlation.dtype) == "torch.float32"
            and all(0 < dim <= _UINT32_MAX for dim in values.shape)
            and values.shape[3] <= _MAX_SEQUENCE_LENGTH
            and values.shape[3] % 8 == 0
            and 0 < top_k <= min(16, values.shape[3])
            and not (values.shape[0] == 4 and values.shape[3] == 336)
        )
    except (AttributeError, IndexError, TypeError):
        return False


def dispatch(values, correlation, top_k, *, custom_call, fallback):
    """Use custom_call only for the audited contract; fallback owns all other shapes."""
    args = (
        values,
        correlation,
        top_k,
    )
    if supports_custom(values, correlation, top_k) and custom_call is not None:
        return custom_call(*args)
    return fallback(*args)
