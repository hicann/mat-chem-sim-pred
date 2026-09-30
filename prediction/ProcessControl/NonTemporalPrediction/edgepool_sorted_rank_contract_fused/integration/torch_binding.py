# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""ctypes binding for the sorted-rank EdgePool contraction ACLNN op."""

from __future__ import annotations

__all__ = [
    "configure",
    "edgepool_sorted_rank_contract_fused",
    "edgepool_sorted_rank_contract_fused_static",
]

import ctypes
import os
from pathlib import Path

import torch

_RUNTIME = None


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
            for library in directory.glob("lib*kernel*.so"):
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        name = "libedgepool_sorted_rank_contract_fused_host.so"
        host = build / name
        if not host.exists():
            host = build / "lib" / name
        if not host.exists():
            raise FileNotFoundError(f"EdgePool host library not found in {build}")
        library = ctypes.CDLL(str(host), mode=ctypes.RTLD_GLOBAL)
        self.query = library.aclnnEdgepoolSortedRankContractFusedGetWorkspaceSize
        self.query.argtypes = [ctypes.c_int64] * 2
        self.query.restype = ctypes.c_uint64
        self.operation = library.aclnnEdgepoolSortedRankContractFused
        self.operation.argtypes = (
            [ctypes.c_void_p] * 7
            + [ctypes.c_int64] * 2
            + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        self.operation.restype = ctypes.c_int32


def _validate_inputs(order, edge_index, score, nodes):
    if order.dtype != torch.int32 or edge_index.dtype != torch.int32:
        raise TypeError("order and edge_index must be int32")
    if score.dtype != torch.float32:
        raise TypeError("score must be float32")
    tensors = (order, edge_index, score)
    if any(
        tensor.device.type != "npu" or not tensor.is_contiguous() for tensor in tensors
    ):
        raise ValueError("all tensors must be contiguous NPU tensors")
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2,E]")
    edges = edge_index.size(1)
    if order.numel() != edges:
        raise ValueError("order and edge_index disagree on edge count")
    if score.numel() != edges:
        raise ValueError("score and edge_index disagree on edge count")
    if nodes <= 0 or nodes > 2**31 - 1:
        raise ValueError("invalid node count")
    return edges


def configure(build_dir):
    global _RUNTIME
    _RUNTIME = _Runtime(build_dir)


def edgepool_sorted_rank_contract_fused_static(order, edge_index, score, nodes):
    """Return fixed-capacity outputs and an NPU-resident count tensor."""

    if _RUNTIME is None:
        raise RuntimeError("configure(build_dir) must be called first")
    tensors = order, edge_index, score
    edges = _validate_inputs(order, edge_index, score, nodes)
    workspace_size = int(_RUNTIME.query(nodes, edges))
    if workspace_size == 2**64 - 1:
        raise ValueError("unsupported EdgePool contraction shape")
    workspace = torch.empty(
        max(workspace_size, 1), dtype=torch.uint8, device=score.device
    )
    cluster = torch.empty(nodes, dtype=torch.int32, device=score.device)
    selected_edge = torch.empty(nodes, dtype=torch.int32, device=score.device)
    cluster_score = torch.empty(nodes, dtype=torch.float32, device=score.device)
    counts = torch.empty(2, dtype=torch.int32, device=score.device)
    stream = torch.npu.current_stream()
    result = _RUNTIME.operation(
        order.data_ptr(),
        edge_index.data_ptr(),
        score.data_ptr(),
        cluster.data_ptr(),
        selected_edge.data_ptr(),
        cluster_score.data_ptr(),
        counts.data_ptr(),
        nodes,
        edges,
        workspace.data_ptr(),
        max(workspace_size, 1),
        stream.npu_stream,
    )
    if result != 0:
        raise RuntimeError(f"EdgePool contraction returned {result}")
    for tensor in (*tensors, workspace, cluster, selected_edge, cluster_score, counts):
        tensor.record_stream(stream)
    return cluster, selected_edge, cluster_score, counts


def edgepool_sorted_rank_contract_fused(order, edge_index, score, nodes):
    cluster, selected_edge, cluster_score, counts = (
        edgepool_sorted_rank_contract_fused_static(order, edge_index, score, nodes)
    )
    counts_host = counts.cpu()
    return (
        cluster,
        selected_edge[: int(counts_host[0].item())],
        cluster_score[: int(counts_host[1].item())],
        int(counts_host[0].item()),
        int(counts_host[1].item()),
    )
