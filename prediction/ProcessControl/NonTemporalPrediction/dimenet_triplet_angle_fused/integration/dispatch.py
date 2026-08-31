# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Shape and dtype guard for the DimeNet angle fast path."""


def supports(position, idx_i, idx_j, idx_k):
    tensors = (position, idx_i, idx_j, idx_k)
    return (
        all(
            t.device.type == "npu" and t.is_contiguous() and t.storage_offset() == 0
            for t in tensors
        )
        and position.dtype == "torch.float32"
        and all(t.dtype == "torch.int32" for t in tensors[1:])
        and position.ndim == 2
        and position.shape[1] == 3
        and idx_i.ndim == 1
        and idx_j.shape == idx_i.shape
        and idx_k.shape == idx_i.shape
        and idx_i.numel() > 0
    )


def dispatch(custom_call, fallback, *inputs):
    if supports(*inputs):
        return custom_call()
    return fallback()
