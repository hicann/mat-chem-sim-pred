# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""PyTorch custom-op binding for the direct ACLNN Hypergraph attention kernel."""

from __future__ import annotations

__all__ = ["configure", "csr_hypergraph_attention_two_stage_propagate_fused"]

import ctypes
import os
from pathlib import Path

import torch

_RUNTIME = None
_DTYPE_IDS = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_TENSOR_COUNT = 9


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
        host = build / "libcsr_hypergraph_attention_two_stage_propagate_fused_host.so"
        if not host.exists():
            host = (
                build
                / "lib"
                / "libcsr_hypergraph_attention_two_stage_propagate_fused_host.so"
            )
        if not host.exists():
            raise FileNotFoundError("Hypergraph attention host library not found")
        library = ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
        self.query = (
            library.aclnnCsrHypergraphAttentionTwoStagePropagateFusedGetWorkspaceSize
        )
        self.query.argtypes = [ctypes.c_int64] * 7
        self.query.restype = ctypes.c_uint64
        self.operation = library.aclnnCsrHypergraphAttentionTwoStagePropagateFused
        self.operation.argtypes = (
            [ctypes.c_void_p] * 10
            + [ctypes.c_int64] * 8
            + [ctypes.c_float, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        self.operation.restype = ctypes.c_int32


def configure(build_dir):
    global _RUNTIME
    _RUNTIME = _Runtime(build_dir)


def _check(tensors):
    if _RUNTIME is None:
        raise RuntimeError("configure(build_dir) must be called before the custom op")
    integer_positions = (0, 1, 3, 4, 5)
    floating_positions = (2, 6, 7, 8)
    if any(tensors[index].dtype != torch.int32 for index in integer_positions):
        raise TypeError("incidence and CSR indices must be int32")
    dtype = tensors[2].dtype
    if dtype not in _DTYPE_IDS or any(
        tensors[index].dtype != dtype for index in floating_positions
    ):
        raise TypeError("floating tensors must share a supported dtype")
    if any(
        value.device.type != "npu" or not value.is_contiguous() for value in tensors
    ):
        raise ValueError("all tensors must be contiguous NPU tensors")


def _padded_csr(row_ptr, index, max_segment_size):
    offsets = torch.arange(max_segment_size, device=index.device)
    counts = row_ptr[1:].long() - row_ptr[:-1].long()
    positions = row_ptr[:-1].long().unsqueeze(1) + offsets
    valid = offsets.unsqueeze(0) < counts.unsqueeze(1)
    safe_positions = positions.clamp(max=index.numel() - 1)
    return index[safe_positions].long(), safe_positions, valid


def _gather_rows(values, index):
    return values.index_select(0, index.reshape(-1)).reshape(
        *index.shape, *values.shape[1:]
    )


def _parameters(inputs):
    values = inputs[_TENSOR_COUNT:]
    return values[0], values[1], values[2] if len(values) > 2 else 0.2


def _reference(*inputs):
    (
        edge_row_ptr,
        node_index,
        edge_scale,
        node_row_ptr,
        edge_index,
        incidence_position,
        node_scale,
        features,
        attention_logits,
        max_edge_size,
        max_node_degree,
        negative_slope,
    ) = inputs
    node, edge_positions, edge_valid = _padded_csr(
        edge_row_ptr, node_index, max_edge_size
    )
    scores = torch.nn.functional.leaky_relu(
        _gather_rows(attention_logits, edge_positions), negative_slope
    )
    alpha = torch.softmax(
        scores.masked_fill(~edge_valid.unsqueeze(-1), torch.finfo(scores.dtype).min),
        dim=1,
    )
    alpha = torch.where(edge_valid.unsqueeze(-1), alpha, torch.zeros_like(alpha))
    edge_features = (
        _gather_rows(features, node)
        * alpha.unsqueeze(-1)
        * edge_scale.view(-1, 1, 1, 1)
    ).sum(1)

    hyperedge, node_positions, node_valid = _padded_csr(
        node_row_ptr, edge_index, max_node_degree
    )
    incidence = incidence_position[node_positions].long()
    alpha_by_incidence = alpha[edge_valid]
    node_alpha = _gather_rows(alpha_by_incidence, incidence)
    node_alpha = torch.where(
        node_valid.unsqueeze(-1), node_alpha, torch.zeros_like(node_alpha)
    )
    return (_gather_rows(edge_features, hyperedge) * node_alpha.unsqueeze(-1)).sum(
        1
    ) * node_scale.view(-1, 1, 1)


@torch.library.custom_op(
    "cann_prediction::csr_hypergraph_attention_two_stage_propagate_fused",
    mutates_args=(),
    device_types="npu",
    schema=(
        "(Tensor edge_row_ptr, Tensor node_index, Tensor edge_scale, "
        "Tensor node_row_ptr, Tensor edge_index, Tensor incidence_position, "
        "Tensor node_scale, Tensor features, Tensor attention_logits, "
        "int max_edge_size, int max_node_degree, float negative_slope=0.2) -> Tensor"
    ),
)
def csr_hypergraph_attention_two_stage_propagate_fused(*inputs) -> torch.Tensor:
    tensors = inputs[:_TENSOR_COUNT]
    max_edge_size, max_node_degree, negative_slope = _parameters(inputs)
    edge_row_ptr, node_index, features = tensors[0], tensors[1], tensors[7]
    _check(tensors)
    nodes, heads, channels = features.shape
    query = (
        nodes,
        edge_row_ptr.numel() - 1,
        node_index.numel(),
        heads,
        channels,
        max_edge_size,
        max_node_degree,
    )
    workspace_size = int(_RUNTIME.query(*query))
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=features.device)
    output = torch.empty_like(features)
    stream = torch.npu.current_stream()
    result = _RUNTIME.operation(
        *(item.data_ptr() for item in tensors),
        output.data_ptr(),
        *query,
        _DTYPE_IDS[features.dtype],
        ctypes.c_float(negative_slope),
        workspace.data_ptr(),
        workspace_size,
        stream.npu_stream,
    )
    if result != 0:
        raise RuntimeError(f"ACLNN Hypergraph operation returned {result}")
    for tensor in (*tensors, workspace, output):
        tensor.record_stream(stream)
    return output


@csr_hypergraph_attention_two_stage_propagate_fused.register_fake
def _fake(*inputs):
    features = inputs[7]
    return torch.empty_like(features)


def _setup_context(ctx, inputs, output):
    del output
    ctx.save_for_backward(*inputs[:_TENSOR_COUNT])
    ctx.max_edge_size, ctx.max_node_degree, ctx.negative_slope = _parameters(inputs)


def _backward(ctx, grad_output):
    saved = ctx.saved_tensors
    with torch.enable_grad():
        tensors = tuple(
            saved[index].detach().requires_grad_(True) for index in (2, 6, 7, 8)
        )
        reference_inputs = (
            *saved[:2],
            tensors[0],
            *saved[3:6],
            *tensors[1:],
            ctx.max_edge_size,
            ctx.max_node_degree,
            ctx.negative_slope,
        )
        reference = _reference(*reference_inputs)
        grads = torch.autograd.grad(reference, tensors, grad_output)
    result = [None] * 12
    for index, value in zip((2, 6, 7, 8), grads):
        result[index] = value
    return tuple(result)


torch.library.register_autograd(
    "cann_prediction::csr_hypergraph_attention_two_stage_propagate_fused",
    _backward,
    setup_context=_setup_context,
)
