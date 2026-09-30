# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Guarded integration policy; the concrete ACLNN binding is injected."""


def supports(
    row_ptr,
    source_index,
    projected,
    max_segment_size,
    **options,
):
    csr_validated = options.get("csr_validated", False)
    tensors = (row_ptr, source_index, projected)
    return (
        csr_validated
        and all(
            tensor.device.type == "npu" and tensor.is_contiguous() for tensor in tensors
        )
        and str(row_ptr.dtype) == "torch.int32"
        and str(source_index.dtype) == "torch.int32"
        and str(projected.dtype) in {"torch.float32", "torch.float16", "torch.bfloat16"}
        and projected.ndim == 3
        and row_ptr.numel() == projected.shape[0] + 1
        and source_index.ndim == 1
        and 0 < projected.shape[1] <= 8
        and 0 < projected.shape[2] <= 64
        and 0 < max_segment_size <= 512
    )


def dispatch(
    custom_call,
    fallback,
    *args,
    max_segment_size,
    training=False,
    csr_validated=False,
):
    if supports(
        *args,
        max_segment_size=max_segment_size,
        training=training,
        csr_validated=csr_validated,
    ):
        return custom_call()
    return fallback()
