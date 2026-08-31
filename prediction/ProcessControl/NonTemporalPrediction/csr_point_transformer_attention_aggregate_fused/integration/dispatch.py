# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Guarded integration policy for the default PointTransformer message."""


def supports(
    *tensors,
    max_segment_size,
    training=False,
    has_attention_mlp=False,
    csr_validated=False,
):
    del training
    if len(tensors) != 6:
        return False
    row_ptr, source_index, alpha_source, alpha_target, value, delta = tensors
    feature_dtype = str(alpha_source.dtype)
    nodes = value.shape[0] if value.ndim == 2 else -1
    channels = value.shape[1] if value.ndim == 2 else -1
    return (
        csr_validated
        and not has_attention_mlp
        and all(
            tensor.device.type == "npu" and tensor.is_contiguous() for tensor in tensors
        )
        and str(row_ptr.dtype) == str(source_index.dtype) == "torch.int32"
        and feature_dtype in {"torch.float32", "torch.float16", "torch.bfloat16"}
        and all(str(tensor.dtype) == feature_dtype for tensor in tensors[2:])
        and alpha_source.shape == alpha_target.shape == value.shape
        and row_ptr.numel() == nodes + 1
        and source_index.ndim == 1
        and delta.shape == (source_index.numel(), channels)
        and 0 < channels <= 128
        and 0 < max_segment_size <= 512
    )


def dispatch(custom_call, fallback, *args, **policy):
    return custom_call() if supports(*args, **policy) else fallback()
