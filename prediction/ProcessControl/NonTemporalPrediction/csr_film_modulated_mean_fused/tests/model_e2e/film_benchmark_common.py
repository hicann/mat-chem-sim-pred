#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Shared real-PPI FiLM benchmark utilities and resident-NPU baseline."""

from __future__ import annotations

import ctypes
import importlib
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch_npu

sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("graph_message_model_e2e_common")
load_host_library, log_json, timed = (
    _common.load_host_library,
    _common.log_json,
    _common.timed,
)


@dataclass
class Layout:
    row_ptr: torch.Tensor
    source: torch.Tensor
    target: torch.Tensor
    inverse_degree: torch.Tensor
    max_degree: int

    def to(self, device):
        return Layout(
            self.row_ptr.to(device),
            self.source.to(device),
            self.target.to(device),
            self.inverse_degree.to(device),
            self.max_degree,
        )


def build_layout(edge_index, nodes):
    order = torch.argsort(edge_index[1], stable=True)
    source = edge_index[0, order]
    target = edge_index[1, order]
    degree = torch.bincount(target, minlength=nodes)
    row_ptr = torch.cat((torch.zeros(1, dtype=torch.int64), degree.cumsum(0)))
    inverse = degree.clamp_min(1).to(torch.float32).reciprocal()
    inverse[degree == 0] = 0
    return Layout(
        row_ptr.to(torch.int32),
        source.to(torch.int32),
        target.to(torch.int64),
        inverse,
        int(degree.max()),
    )


def native_mean(projected, beta, gamma, layout, apply_relu):
    message = gamma[layout.target] * projected[layout.source.to(torch.int64)]
    message = message + beta[layout.target]
    if apply_relu:
        message = torch.relu(message)
    output = projected.new_zeros(projected.shape)
    output.index_add_(0, layout.target, message)
    return output * layout.inverse_degree[:, None]


class CustomOperator:
    def __init__(self, build: Path, device):
        self.library = load_host_library(build, "csr_film_modulated_mean_fused")
        self.device = device
        workspace = self.library.aclnnCsrFilmModulatedMeanFusedGetWorkspaceSize
        workspace.argtypes = [ctypes.c_int64] * 4
        workspace.restype = ctypes.c_uint64
        operation = self.library.aclnnCsrFilmModulatedMeanFused
        operation.argtypes = (
            [ctypes.c_void_p] * 6
            + [ctypes.c_int64] * 5
            + [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        )
        operation.restype = ctypes.c_int32
        self.cache = {}

    def __call__(self, projected, beta, gamma, layout, apply_relu=True):
        nodes, channels = projected.shape
        edges = layout.source.numel()
        values = (nodes, edges, channels, layout.max_degree)
        if values not in self.cache:
            size = self.library.aclnnCsrFilmModulatedMeanFusedGetWorkspaceSize(*values)
            self.cache[values] = (
                torch.empty((size,), dtype=torch.uint8, device=self.device),
                torch.empty_like(projected),
            )
        workspace, output = self.cache[values]
        result = self.library.aclnnCsrFilmModulatedMeanFused(
            layout.row_ptr.data_ptr(),
            layout.source.data_ptr(),
            projected.data_ptr(),
            beta.data_ptr(),
            gamma.data_ptr(),
            output.data_ptr(),
            *values,
            int(apply_relu),
            workspace.data_ptr(),
            workspace.numel(),
            torch_npu.npu.current_stream().npu_stream,
        )
        if result != 0:
            raise RuntimeError(f"custom operator returned {result}")
        return output


def main():
    from torch_geometric.datasets import PPI
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import FiLMConv

    dataset = PPI("/home/huawei/hot_model_15_20260801/data/PPI", split="train")
    data = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    layout = build_layout(data.edge_index, data.num_nodes)
    device = torch.device("npu:0")
    torch_npu.npu.set_device(device)
    data = data.to(device)
    layout = layout.to(device)
    layer = FiLMConv(dataset.num_features, 320).eval().to(device)
    with torch.no_grad():
        projected = layer.lins[0](data.x).contiguous()
        beta, gamma = layer.films[0](data.x).split(320, dim=-1)
        beta, gamma = beta.contiguous(), gamma.contiguous()
    custom = CustomOperator(Path(__file__).resolve().parent / "build", device)
    baseline = timed(partial(native_mean, projected, beta, gamma, layout, True), 5, 20)
    custom_result = timed(partial(custom, projected, beta, gamma, layout, True), 5, 20)
    log_json(
        {
            "nodes": data.num_nodes,
            "edges": data.num_edges,
            "channels": 320,
            "max_degree": layout.max_degree,
            "baseline_ms": baseline.median_ms,
            "custom_ms": custom_result.median_ms,
            "speedup": baseline.median_ms / custom_result.median_ms,
            "max_abs_error": float(
                (baseline.output - custom_result.output).abs().max().cpu()
            ),
        },
        indent=2,
    )


if __name__ == "__main__":
    main()
