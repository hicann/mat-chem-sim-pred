# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Direct PyTorch/ACLNN binding for DimeNet triplet angles."""

from __future__ import annotations

import ctypes

import torch
from geometry_binding_common import load_host_library

_RUNTIME = None


class _Runtime:
    def __init__(self, build_dir):
        library = load_host_library(
            build_dir,
            "libdimenet_triplet_angle_fused_host.so",
            "lib*_kernel*.so",
        )
        self.query = library.aclnnDimeNetTripletAngleFusedGetWorkspaceSize
        self.query.argtypes = [ctypes.c_int64, ctypes.c_int64]
        self.query.restype = ctypes.c_uint64
        self.operation = library.aclnnDimeNetTripletAngleFused
        self.operation.argtypes = (
            [ctypes.c_void_p] * 5
            + [ctypes.c_int64] * 2
            + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        self.operation.restype = ctypes.c_int32


def configure(build_dir):
    global _RUNTIME
    _RUNTIME = _Runtime(build_dir)


def _valid_inputs(tensors):
    position, idx_i, idx_j, idx_k = tensors
    checks = [
        all(t.device.type == "npu" for t in tensors),
        position.dtype == torch.float32,
        all(t.dtype == torch.int32 for t in tensors[1:]),
        position.ndim == 2,
        position.shape[1] == 3,
        idx_i.ndim == 1,
        idx_j.shape == idx_i.shape,
        idx_k.shape == idx_i.shape,
        all(t.is_contiguous() and t.storage_offset() == 0 for t in tensors),
        idx_i.numel() > 0,
    ]
    return all(checks)


@torch.library.custom_op(
    "cann_prediction::dimenet_triplet_angle_fused",
    mutates_args=(),
    device_types="npu",
    schema="(Tensor position, Tensor idx_i, Tensor idx_j, Tensor idx_k) -> Tensor",
)
def dimenet_triplet_angle_fused(*inputs) -> torch.Tensor:
    position, idx_i, idx_j, idx_k = inputs
    if _RUNTIME is None:
        raise RuntimeError("configure(build_dir) must be called first")
    tensors = (position, idx_i, idx_j, idx_k)
    if not _valid_inputs(tensors):
        raise ValueError("unsupported DimeNet angle contract")
    output = torch.empty(idx_i.numel(), dtype=torch.float32, device=position.device)
    workspace_size = int(_RUNTIME.query(position.shape[0], idx_i.numel()))
    stream = torch.npu.current_stream()
    result = _RUNTIME.operation(
        *(tensor.data_ptr() for tensor in tensors),
        output.data_ptr(),
        position.shape[0],
        idx_i.numel(),
        None,
        workspace_size,
        stream.npu_stream,
    )
    if result != 0:
        raise RuntimeError(f"ACLNN DimeNet angle returned {result}")
    for tensor in (*tensors, output):
        tensor.record_stream(stream)
    return output


@dimenet_triplet_angle_fused.register_fake
def _fake(*inputs):
    position, idx_i, idx_j, idx_k = inputs
    del idx_j, idx_k
    return torch.empty(idx_i.shape, dtype=torch.float32, device=position.device)
