# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""PyTorch custom-op binding for direct ACLNN DimeNet triplet enumeration."""

from __future__ import annotations

import ctypes
import importlib
import sys
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).parents[2] / "dimenet_triplet_angle_fused" / "integration")
)
load_host_library = importlib.import_module("geometry_binding_common").load_host_library

_RUNTIME = None


class _Runtime:
    def __init__(self, build_dir):
        library = load_host_library(
            build_dir,
            "libdimenet_triplet_enumerate_fused_host.so",
            "lib*_kernel*.so",
        )
        self.query = library.aclnnDimeNetTripletEnumerateFusedGetWorkspaceSize
        self.query.argtypes = [ctypes.c_int64] * 3
        self.query.restype = ctypes.c_uint64
        self.operation = library.aclnnDimeNetTripletEnumerateFused
        self.operation.argtypes = (
            [ctypes.c_void_p] * 8
            + [ctypes.c_int64] * 3
            + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        self.operation.restype = ctypes.c_int32


def configure(build_dir):
    global _RUNTIME
    _RUNTIME = _Runtime(build_dir)


def _check(row_ptr, source_index, capacity):
    if _RUNTIME is None:
        raise RuntimeError("configure(build_dir) must be called before the custom op")
    if row_ptr.dtype != torch.int32 or source_index.dtype != torch.int32:
        raise TypeError("CSR tensors must be int32")
    if any(
        tensor.device.type != "npu" or not tensor.is_contiguous()
        for tensor in (row_ptr, source_index)
    ):
        raise ValueError("CSR tensors must be contiguous NPU tensors")
    if row_ptr.ndim != 1 or source_index.ndim != 1 or row_ptr.numel() < 2:
        raise ValueError("invalid CSR ranks or node count")
    if source_index.numel() <= 0 or capacity <= 0:
        raise ValueError("edges and capacity must be positive")


@torch.library.custom_op(
    "cann_prediction::dimenet_triplet_enumerate_fused",
    mutates_args=(),
    device_types="npu",
)
def dimenet_triplet_enumerate_fused(
    row_ptr: torch.Tensor,
    source_index: torch.Tensor,
    capacity: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    tensors = (row_ptr, source_index)
    _check(*tensors, capacity)
    nodes = row_ptr.numel() - 1
    edges = source_index.numel()
    workspace_size = int(_RUNTIME.query(nodes, edges, capacity))
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=row_ptr.device)
    outputs = tuple(
        torch.empty(capacity, dtype=torch.int32, device=row_ptr.device)
        for _ in range(5)
    )
    counts = torch.empty(2, dtype=torch.int32, device=row_ptr.device)
    stream = torch.npu.current_stream()
    result = _RUNTIME.operation(
        row_ptr.data_ptr(),
        source_index.data_ptr(),
        *(output.data_ptr() for output in outputs),
        counts.data_ptr(),
        nodes,
        edges,
        capacity,
        workspace.data_ptr(),
        workspace_size,
        stream.npu_stream,
    )
    if result != 0:
        raise RuntimeError(f"ACLNN DimeNet triplet operation returned {result}")
    for tensor in (*tensors, *outputs, counts, workspace):
        tensor.record_stream(stream)
    return (*outputs, counts)


@dimenet_triplet_enumerate_fused.register_fake
def _fake(row_ptr, source_index, capacity):
    del source_index
    outputs = tuple(
        torch.empty(capacity, dtype=torch.int32, device=row_ptr.device)
        for _ in range(5)
    )
    counts = torch.empty(2, dtype=torch.int32, device=row_ptr.device)
    return (*outputs, counts)
