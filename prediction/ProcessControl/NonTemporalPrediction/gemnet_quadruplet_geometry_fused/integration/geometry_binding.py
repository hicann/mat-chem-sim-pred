# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Direct PyTorch/ACLNN binding for official GemNet geometry caches."""

from __future__ import annotations

import ctypes
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch

_COMMON_PATH = (
    Path(__file__).parents[2]
    / "dimenet_triplet_angle_fused"
    / "integration"
    / "geometry_binding_common.py"
)
_COMMON_SPEC = spec_from_file_location("molecular_geometry_common", _COMMON_PATH)
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise ImportError(f"cannot load common geometry helper from {_COMMON_PATH}")
_COMMON_MODULE = module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON_MODULE)
load_host_library = _COMMON_MODULE.load_host_library

_RUNTIME = None


class _Runtime:
    def __init__(self, build_dir):
        library = load_host_library(
            build_dir,
            "libgemnet_quadruplet_geometry_fused_host.so",
            "lib*geometry*kernel*.so",
        )
        self.query = library.aclnnGemNetQuadrupletGeometryFusedGetWorkspaceSize
        self.query.argtypes = [ctypes.c_int64] * 6
        self.query.restype = ctypes.c_uint64
        self.operation = library.aclnnGemNetQuadrupletGeometryFused
        self.operation.argtypes = (
            [ctypes.c_void_p] * 14
            + [ctypes.c_int64] * 6
            + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        self.operation.restype = ctypes.c_int32


def configure(build_dir):
    global _RUNTIME
    _RUNTIME = _Runtime(build_dir)


def _valid_inputs(tensors):
    (
        position,
        source_index,
        target_index,
        interaction_source,
        interaction_target,
        reduce_ca,
        expand_db,
        reduce_intermediate_ca,
        expand_intermediate_db,
        reduce_intermediate_ab,
        expand_intermediate_ab,
    ) = tensors
    checks = [
        all(t.device.type == "npu" for t in tensors),
        position.dtype == torch.float32,
        all(t.dtype == torch.int32 for t in tensors[1:]),
        position.ndim == 2,
        position.shape[1] == 3,
        source_index.ndim == 1,
        target_index.shape == source_index.shape,
        interaction_source.ndim == 1,
        interaction_target.shape == interaction_source.shape,
        reduce_ca.ndim == 1,
        expand_db.shape == reduce_ca.shape,
        reduce_intermediate_ca.ndim == 1,
        reduce_intermediate_ab.shape == reduce_intermediate_ca.shape,
        expand_intermediate_db.ndim == 1,
        expand_intermediate_ab.shape == expand_intermediate_db.shape,
        all(t.is_contiguous() and t.storage_offset() == 0 for t in tensors),
        reduce_ca.numel() > 0,
        reduce_intermediate_ca.numel() > 0,
        expand_intermediate_db.numel() > 0,
    ]
    return all(checks)


def _allocate_outputs(
    position, reduce_ca, reduce_intermediate_ca, expand_intermediate_db
):
    return (
        torch.empty(
            reduce_intermediate_ca.shape, dtype=torch.float32, device=position.device
        ),
        torch.empty(
            expand_intermediate_db.shape, dtype=torch.float32, device=position.device
        ),
        torch.empty(reduce_ca.shape, dtype=torch.float32, device=position.device),
    )


def _execute(tensors, outputs, sizes):
    workspace_size = int(_RUNTIME.query(*sizes))
    stream = torch.npu.current_stream()
    result = _RUNTIME.operation(
        *(tensor.data_ptr() for tensor in tensors),
        *(tensor.data_ptr() for tensor in outputs),
        *sizes,
        None,
        workspace_size,
        stream.npu_stream,
    )
    if result != 0:
        raise RuntimeError(f"ACLNN GemNet geometry returned {result}")
    for tensor in (*tensors, *outputs):
        tensor.record_stream(stream)
    return outputs


@torch.library.custom_op(
    "cann_prediction::gemnet_quadruplet_geometry_fused",
    mutates_args=(),
    device_types="npu",
    schema=(
        "(Tensor position, Tensor source_index, Tensor target_index, "
        "Tensor interaction_source, Tensor interaction_target, Tensor reduce_ca, "
        "Tensor expand_db, Tensor reduce_intermediate_ca, Tensor expand_intermediate_db, "
        "Tensor reduce_intermediate_ab, Tensor expand_intermediate_ab) -> (Tensor, Tensor, Tensor)"
    ),
)
def gemnet_quadruplet_geometry_fused(*inputs):
    if _RUNTIME is None:
        raise RuntimeError("configure(build_dir) must be called first")
    if not _valid_inputs(inputs):
        raise ValueError("unsupported GemNet geometry contract")
    position, reduce_ca = inputs[0], inputs[5]
    reduce_intermediate_ca, expand_intermediate_db = inputs[7], inputs[8]
    outputs = _allocate_outputs(
        position, reduce_ca, reduce_intermediate_ca, expand_intermediate_db
    )
    sizes = (
        position.shape[0],
        inputs[1].numel(),
        inputs[3].numel(),
        reduce_ca.numel(),
        reduce_intermediate_ca.numel(),
        expand_intermediate_db.numel(),
    )
    return _execute(inputs, outputs, sizes)


@gemnet_quadruplet_geometry_fused.register_fake
def _fake(*inputs):
    return _allocate_outputs(inputs[0], inputs[5], inputs[7], inputs[8])
