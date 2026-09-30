# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""PyTorch custom-op binding for the direct ACLNN SuperGAT kernel."""

from __future__ import annotations

__all__ = ["configure", "csr_supergat_dot_attention_aggregate_fused"]

import os
import ctypes
from pathlib import Path

import torch

_RUNTIME = None
_DTYPE_IDS = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}


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
        host = _find_host_library(build)
        library = ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
        self.query = library.aclnnCsrSuperGatDotAttentionAggregateFusedGetWorkspaceSize
        self.query.argtypes = [ctypes.c_int64] * 5
        self.query.restype = ctypes.c_uint64
        self.operation = library.aclnnCsrSuperGatDotAttentionAggregateFused
        self.operation.argtypes = (
            [ctypes.c_void_p] * 4
            + [ctypes.c_int64] * 6
            + [ctypes.c_float, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        self.operation.restype = ctypes.c_int32


def configure(build_dir):
    global _RUNTIME
    _RUNTIME = _Runtime(build_dir)


def _find_host_library(build):
    candidates = (
        build / "libcsr_supergat_dot_attention_aggregate_fused_host.so",
        build / "lib" / "libcsr_supergat_dot_attention_aggregate_fused_host.so",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("SuperGAT host library was not found")


def _check(row_ptr, source_index, projected):
    if _RUNTIME is None:
        raise RuntimeError("configure(build_dir) must be called before the custom op")
    if projected.dtype not in _DTYPE_IDS:
        raise TypeError(f"unsupported dtype: {projected.dtype}")
    if row_ptr.dtype != torch.int32 or source_index.dtype != torch.int32:
        raise TypeError("CSR indices must be int32")
    tensors = (row_ptr, source_index, projected)
    if any(
        tensor.device.type != "npu" or not tensor.is_contiguous() for tensor in tensors
    ):
        raise ValueError("all tensors must be contiguous NPU tensors")


def _padded_csr(row_ptr, source_index, max_segment_size):
    offsets = torch.arange(max_segment_size, device=source_index.device)
    counts = row_ptr[1:].long() - row_ptr[:-1].long()
    positions = row_ptr[:-1].long().unsqueeze(1) + offsets
    valid = offsets.unsqueeze(0) < counts.unsqueeze(1)
    safe_positions = positions.clamp(max=source_index.numel() - 1)
    return source_index[safe_positions].long(), valid


def _gather_rows(values, index):
    return values.index_select(0, index.reshape(-1)).reshape(
        *index.shape, *values.shape[1:]
    )


def _reference(row_ptr, source_index, projected, max_segment_size, negative_slope):
    source, valid = _padded_csr(row_ptr, source_index, max_segment_size)
    _, _, channels = projected.shape
    messages = _gather_rows(projected, source)
    score = (projected.unsqueeze(1) * messages).sum(-1) / channels**0.5
    score = torch.nn.functional.leaky_relu(score, negative_slope)
    alpha = torch.softmax(
        score.masked_fill(~valid.unsqueeze(-1), torch.finfo(score.dtype).min), dim=1
    )
    alpha = torch.where(valid.unsqueeze(-1), alpha, torch.zeros_like(alpha))
    return (messages * alpha.unsqueeze(-1)).sum(1)


@torch.library.custom_op(
    "cann_prediction::csr_supergat_dot_attention_aggregate_fused",
    mutates_args=(),
    device_types="npu",
)
def csr_supergat_dot_attention_aggregate_fused(
    row_ptr: torch.Tensor,
    source_index: torch.Tensor,
    projected: torch.Tensor,
    max_segment_size: int,
    negative_slope: float = 0.2,
) -> torch.Tensor:
    tensors = (row_ptr, source_index, projected)
    _check(*tensors)
    nodes, heads, channels = projected.shape
    edges = source_index.numel()
    query = (nodes, edges, heads, channels, max_segment_size)
    workspace_size = int(_RUNTIME.query(*query))
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=projected.device)
    output = torch.empty_like(projected)
    stream = torch.npu.current_stream()
    result = _RUNTIME.operation(
        row_ptr.data_ptr(),
        source_index.data_ptr(),
        projected.data_ptr(),
        output.data_ptr(),
        *query,
        _DTYPE_IDS[projected.dtype],
        ctypes.c_float(negative_slope),
        workspace.data_ptr(),
        workspace_size,
        stream.npu_stream,
    )
    if result != 0:
        raise RuntimeError(f"ACLNN SuperGAT operation returned {result}")
    for tensor in (*tensors, workspace, output):
        tensor.record_stream(stream)
    return output


@csr_supergat_dot_attention_aggregate_fused.register_fake
def _fake(row_ptr, source_index, projected, max_segment_size, negative_slope=0.2):
    del row_ptr, source_index, max_segment_size, negative_slope
    return torch.empty_like(projected)


def _setup_context(ctx, inputs, output):
    del output
    row_ptr, source_index, projected, max_segment_size, negative_slope = inputs
    ctx.save_for_backward(row_ptr, source_index, projected)
    ctx.max_segment_size = max_segment_size
    ctx.negative_slope = negative_slope


def _backward(ctx, grad_output):
    row_ptr, source_index, projected = ctx.saved_tensors
    with torch.enable_grad():
        differentiable = projected.detach().requires_grad_(True)
        reference = _reference(
            row_ptr,
            source_index,
            differentiable,
            ctx.max_segment_size,
            ctx.negative_slope,
        )
        (grad_projected,) = torch.autograd.grad(reference, differentiable, grad_output)
    return None, None, grad_projected, None, None


torch.library.register_autograd(
    "cann_prediction::csr_supergat_dot_attention_aggregate_fused",
    _backward,
    setup_context=_setup_context,
)
