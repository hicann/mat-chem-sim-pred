# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Shape and dtype guard for the PPF geometry fast path."""


def supports(position, normal, source_index, target_index):
    tensors = (position, normal, source_index, target_index)
    return (
        all(
            t.device.type == "npu" and t.is_contiguous() and t.storage_offset() == 0
            for t in tensors
        )
        and position.dtype == "torch.float32"
        and normal.dtype == "torch.float32"
        and source_index.dtype == "torch.int32"
        and target_index.dtype == "torch.int32"
        and position.ndim == 2
        and position.shape[1] == 3
        and normal.shape == position.shape
        and source_index.ndim == 1
        and target_index.shape == source_index.shape
        and source_index.numel() > 0
    )


def dispatch(custom_call, fallback, *tensors):
    if supports(*tensors):
        return custom_call()
    return fallback()
