# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Guarded dispatch policy for fixed-capacity DimeNet triplet outputs."""


def supports(row_ptr, source_index, capacity, max_degree, csr_validated=False):
    return (
        csr_validated
        and row_ptr.device.type == "npu"
        and source_index.device.type == "npu"
        and row_ptr.is_contiguous()
        and source_index.is_contiguous()
        and str(row_ptr.dtype) == "torch.int32"
        and str(source_index.dtype) == "torch.int32"
        and row_ptr.ndim == 1
        and source_index.ndim == 1
        and row_ptr.numel() >= 2
        and source_index.numel() > 0
        and max_degree > 0
        and capacity >= source_index.numel() * max_degree
        and capacity <= 2_147_483_647
    )


def dispatch(custom_call, fallback, *inputs, csr_validated=False):
    row_ptr, source_index, capacity, max_degree = inputs
    if supports(
        row_ptr,
        source_index,
        capacity,
        max_degree,
        csr_validated=csr_validated,
    ):
        return custom_call()
    return fallback()
