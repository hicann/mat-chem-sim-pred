# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Direct PyTorch/ACLNN binding for fused PPF geometry generation."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

_RUNTIME = None


def _find_host_library(build):
    filename = "libppf_point_pair_features_fused_host.so"
    for candidate in (build / filename, build / "lib" / filename):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"cannot find {filename} under {build}")


class _Runtime:
    def __init__(self, build_dir):
        build = Path(build_dir)
        toolkit = Path(
            os.environ.get(
                "ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest"
            )
        )
        ctypes.CDLL(str(toolkit / "lib64" / "libascendcl.so"), mode=ctypes.RTLD_GLOBAL)
        for directory in (build / "lib", build):
            for library in directory.glob("lib*_kernel*.so"):
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        host = _find_host_library(build)
        library = ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
        self.query = library.aclnnPpfPointPairFeaturesFusedGetWorkspaceSize
        self.query.argtypes = [ctypes.c_int64, ctypes.c_int64]
        self.query.restype = ctypes.c_uint64
        self.operation = library.aclnnPpfPointPairFeaturesFused
        self.operation.argtypes = (
            [ctypes.c_void_p] * 5
            + [ctypes.c_int64] * 2
            + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        self.operation.restype = ctypes.c_int32


def configure(build_dir):
    global _RUNTIME
    _RUNTIME = _Runtime(build_dir)


def _validate_tensors(tensors):
    position, normal, source_index, target_index = tensors
    if any(tensor.device.type != "npu" for tensor in tensors):
        raise ValueError("all inputs must be NPU tensors")
    feature_dtypes_match = all(
        tensor.dtype == torch.float32 for tensor in (position, normal)
    )
    index_dtypes_match = all(
        tensor.dtype == torch.int32 for tensor in (source_index, target_index)
    )
    if not feature_dtypes_match or not index_dtypes_match:
        raise ValueError("expected FP32 features and int32 indices")
    shapes_match = (
        position.ndim == 2
        and position.shape[1] == 3
        and normal.shape == position.shape
        and source_index.ndim == 1
        and target_index.shape == source_index.shape
        and source_index.numel() > 0
    )
    if not shapes_match:
        raise ValueError("expected feature shape [N,3] and index shape [E]")
    contiguous = all(tensor.is_contiguous() for tensor in tensors)
    zero_offset = all(tensor.storage_offset() == 0 for tensor in tensors)
    if not contiguous or not zero_offset:
        raise ValueError("all inputs must be contiguous with zero storage offset")


@torch.library.custom_op(
    "cann_prediction::ppf_point_pair_features_fused",
    mutates_args=(),
    device_types="npu",
)
def ppf_point_pair_features_fused(
    position: torch.Tensor,
    normal: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
) -> torch.Tensor:
    if _RUNTIME is None:
        raise RuntimeError("configure(build_dir) must be called first")
    tensors = (position, normal, source_index, target_index)
    _validate_tensors(tensors)
    nodes = position.shape[0]
    edges = source_index.numel()
    output = torch.empty((edges, 4), dtype=torch.float32, device=position.device)
    workspace_size = int(_RUNTIME.query(nodes, edges))
    stream = torch.npu.current_stream()
    result = _RUNTIME.operation(
        position.data_ptr(),
        normal.data_ptr(),
        source_index.data_ptr(),
        target_index.data_ptr(),
        output.data_ptr(),
        nodes,
        edges,
        None,
        workspace_size,
        stream.npu_stream,
    )
    if result != 0:
        raise RuntimeError(f"ACLNN PPF operation returned {result}")
    for tensor in (*tensors, output):
        tensor.record_stream(stream)
    return output


@ppf_point_pair_features_fused.register_fake
def _fake(position, normal, source_index, target_index):
    del normal, target_index
    return torch.empty(
        (source_index.shape[0], 4), dtype=torch.float32, device=position.device
    )
