# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""PyTorch custom-op binding for the direct ACLNN PointTransformer kernel."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

_RUNTIME = None
_DTYPE_IDS = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


def _find_host_library(build, filename):
    for candidate in (build / filename, build / "lib" / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find {filename} under {build}")


class _Runtime:
    def __init__(self, build_dir):
        build = Path(build_dir)
        if (build / "build").is_dir():
            build = build / "build"
        toolkit = Path(
            os.environ.get(
                "ASCEND_CANN_PACKAGE_PATH",
                os.environ.get(
                    "ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest"
                ),
            )
        )
        ascendcl = toolkit / "lib64" / "libascendcl.so"
        if ascendcl.exists():
            ctypes.CDLL(str(ascendcl), mode=ctypes.RTLD_GLOBAL)
        for directory in (build / "lib", build):
            for library in directory.glob("lib*_kernel*.so"):
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        host = _find_host_library(
            build, "libcsr_point_transformer_attention_aggregate_fused_host.so"
        )
        library = ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
        self.query = (
            library.aclnnCsrPointTransformerAttentionAggregateFusedGetWorkspaceSize
        )
        self.query.argtypes = [ctypes.c_int64] * 4
        self.query.restype = ctypes.c_uint64
        self.operation = library.aclnnCsrPointTransformerAttentionAggregateFused
        self.operation.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int64] * 5
            + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        self.operation.restype = ctypes.c_int32


def configure(build_dir):
    global _RUNTIME
    _RUNTIME = _Runtime(build_dir)


def _check(tensors):
    if _RUNTIME is None:
        raise RuntimeError("configure(build_dir) must be called before the custom op")
    row_ptr, source_index, *features = tensors
    if row_ptr.dtype != torch.int32 or source_index.dtype != torch.int32:
        raise TypeError("CSR indices must be int32")
    if features[0].dtype not in _DTYPE_IDS or any(
        value.dtype != features[0].dtype for value in features
    ):
        raise TypeError("feature tensors must share a supported floating dtype")
    if any(
        value.device.type != "npu" or not value.is_contiguous() for value in tensors
    ):
        raise ValueError("all tensors must be contiguous NPU tensors")


def _padded_csr(row_ptr, source_index, max_segment_size):
    offsets = torch.arange(max_segment_size, device=source_index.device)
    counts = row_ptr[1:].long() - row_ptr[:-1].long()
    positions = row_ptr[:-1].long().unsqueeze(1) + offsets
    valid = offsets.unsqueeze(0) < counts.unsqueeze(1)
    safe_positions = positions.clamp(max=source_index.numel() - 1)
    return source_index[safe_positions].long(), safe_positions, valid


def _gather_rows(values, index):
    return values.index_select(0, index.reshape(-1)).reshape(
        *index.shape, *values.shape[1:]
    )


def _reference(*inputs):
    (
        row_ptr,
        source_index,
        alpha_source,
        alpha_target,
        value,
        delta,
        max_segment_size,
    ) = inputs
    source, positions, valid = _padded_csr(row_ptr, source_index, max_segment_size)
    delta = _gather_rows(delta, positions)
    alpha = torch.softmax(
        (
            alpha_target.unsqueeze(1) - _gather_rows(alpha_source, source) + delta
        ).masked_fill(~valid.unsqueeze(-1), torch.finfo(value.dtype).min),
        dim=1,
    )
    alpha = torch.where(valid.unsqueeze(-1), alpha, torch.zeros_like(alpha))
    return (alpha * (_gather_rows(value, source) + delta)).sum(1)


@torch.library.custom_op(
    "cann_prediction::csr_point_transformer_attention_aggregate_fused",
    mutates_args=(),
    device_types="npu",
    schema=(
        "(Tensor row_ptr, Tensor source_index, Tensor alpha_source, "
        "Tensor alpha_target, Tensor value, Tensor delta, "
        "int max_segment_size) -> Tensor"
    ),
)
def csr_point_transformer_attention_aggregate_fused(*inputs) -> torch.Tensor:
    (
        row_ptr,
        source_index,
        alpha_source,
        alpha_target,
        value,
        delta,
        max_segment_size,
    ) = inputs
    tensors = (row_ptr, source_index, alpha_source, alpha_target, value, delta)
    _check(tensors)
    nodes, channels = value.shape
    query = (nodes, source_index.numel(), channels, max_segment_size)
    workspace_size = int(_RUNTIME.query(*query))
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=value.device)
    output = torch.empty_like(value)
    stream = torch.npu.current_stream()
    result = _RUNTIME.operation(
        *(item.data_ptr() for item in tensors),
        output.data_ptr(),
        *query,
        _DTYPE_IDS[value.dtype],
        workspace.data_ptr(),
        workspace_size,
        stream.npu_stream,
    )
    if result != 0:
        raise RuntimeError(f"ACLNN PointTransformer operation returned {result}")
    for tensor in (*tensors, workspace, output):
        tensor.record_stream(stream)
    return output


@csr_point_transformer_attention_aggregate_fused.register_fake
def _fake(*inputs):
    (
        row_ptr,
        source_index,
        alpha_source,
        alpha_target,
        value,
        delta,
        max_segment_size,
    ) = inputs
    del row_ptr, source_index, alpha_source, alpha_target, delta, max_segment_size
    return torch.empty_like(value)


def _setup_context(ctx, inputs, output):
    del output
    (
        row_ptr,
        source_index,
        alpha_source,
        alpha_target,
        value,
        delta,
        max_segment_size,
    ) = inputs
    ctx.save_for_backward(
        row_ptr, source_index, alpha_source, alpha_target, value, delta
    )
    ctx.max_segment_size = max_segment_size


def _backward(ctx, grad_output):
    row_ptr, source_index, alpha_source, alpha_target, value, delta = ctx.saved_tensors
    with torch.enable_grad():
        tensors = tuple(
            tensor.detach().requires_grad_(True)
            for tensor in (alpha_source, alpha_target, value, delta)
        )
        reference = _reference(row_ptr, source_index, *tensors, ctx.max_segment_size)
        grads = torch.autograd.grad(reference, tensors, grad_output)
    return None, None, *grads, None


torch.library.register_autograd(
    "cann_prediction::csr_point_transformer_attention_aggregate_fused",
    _backward,
    setup_context=_setup_context,
)
