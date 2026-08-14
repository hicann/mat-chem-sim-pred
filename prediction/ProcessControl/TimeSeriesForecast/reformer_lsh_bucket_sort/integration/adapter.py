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
_INT32_MAX = (1 << 31) - 1


def _resident_contiguous(*tensors):
    return all(
        getattr(tensor, "device", None) is not None
        and tensor.device.type == "npu"
        and tensor.is_contiguous()
        for tensor in tensors
    )


def supports_custom(keys, sequence_length, total_buckets):
    try:
        return (
            _resident_contiguous(keys)
            and len(keys.shape) == 2
            and str(keys.dtype) == "torch.int64"
            and 0 < keys.shape[0] <= _UINT32_MAX
            and 0 < keys.shape[1] <= _INT32_MAX
            and 0 < sequence_length <= _UINT32_MAX
            and 0 < total_buckets <= 4096
        )
    except (AttributeError, IndexError, TypeError):
        return False


def dispatch(keys, sequence_length, total_buckets, *, custom_call, fallback):
    """Use custom_call only for the audited contract; fallback owns all other shapes."""
    args = (
        keys,
        sequence_length,
        total_buckets,
    )
    if (
        supports_custom(keys, sequence_length, total_buckets)
        and custom_call is not None
    ):
        return custom_call(*args)
    return fallback(*args)
