# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared model, graph preparation, native binding and timing for GENConv."""

from __future__ import annotations

import ctypes
import hashlib
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv


class DeeperGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.node_encoder = Linear(in_channels, hidden_channels)
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GENConv(
                hidden_channels,
                hidden_channels,
                aggr="softmax",
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm="layer",
            )
            self.layers.append(
                DeepGCNLayer(
                    conv,
                    LayerNorm(hidden_channels, elementwise_affine=True),
                    ReLU(inplace=False),
                    block="res+",
                    dropout=0.0,
                    ckpt_grad=False,
                )
            )
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.node_encoder(x)
        x = self.layers[0].conv(x, edge_index)
        for layer in self.layers[1:]:
            x = layer(x, edge_index)
        x = self.layers[0].act(self.layers[0].norm(x))
        return self.lin(x)


class CustomGenConvAggregate:
    def __init__(self, build: Path, device: torch.device) -> None:
        for library in sorted((build / "lib").glob("lib*_kernel_lib.so")):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
        self.library = ctypes.CDLL(
            str(build / "libcsr_genconv_softmax_aggregate_fused_host.so"),
            mode=ctypes.RTLD_GLOBAL,
        )
        self.workspace_size = (
            self.library.aclnnCsrGenConvSoftmaxAggregateFusedGetWorkspaceSize
        )
        self.workspace_size.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_float]
        self.workspace_size.restype = ctypes.c_uint64
        self.operator = self.library.aclnnCsrGenConvSoftmaxAggregateFused
        self.operator.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 3
        self.operator.argtypes += [
            ctypes.c_float,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self.operator.restype = ctypes.c_int32
        self.device = device
        self.cache: dict[
            tuple[int, int, int, float], tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def __call__(
        self,
        row_ptr: torch.Tensor,
        source_index: torch.Tensor,
        features: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if row_ptr.dtype != torch.int32 or source_index.dtype != torch.int32:
            raise TypeError("custom CSR indices must be int32")
        if features.dtype != torch.float32 or not features.is_contiguous():
            raise TypeError("custom features must be contiguous float32")
        dimensions = (features.size(0), source_index.numel(), features.size(1))
        key = (*dimensions, temperature)
        self.prepare(*dimensions, temperature)
        buffers = self.cache.get(key)
        if buffers is None:
            raise RuntimeError("custom workspace was not prepared")
        workspace, output = buffers
        result = self.operator(
            ctypes.c_void_p(row_ptr.data_ptr()),
            ctypes.c_void_p(source_index.data_ptr()),
            ctypes.c_void_p(features.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            *dimensions,
            ctypes.c_float(temperature),
            ctypes.c_void_p(workspace.data_ptr()),
            workspace.numel(),
            ctypes.c_void_p(torch.npu.current_stream().npu_stream),
        )
        if result:
            raise RuntimeError(f"custom operator returned {result}")
        return output

    def prepare(
        self, nodes: int, edges: int, channels: int, temperature: float
    ) -> None:
        key = (nodes, edges, channels, temperature)
        if key not in self.cache:
            size = int(self.workspace_size(*key))
            self.cache[key] = (
                torch.empty(size, dtype=torch.uint8, device=self.device),
                torch.empty((nodes, channels), dtype=torch.float32, device=self.device),
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def disjoint_copies(
    x: torch.Tensor, edge_index: torch.Tensor, copies: int
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes = x.size(0)
    return x.repeat(copies, 1), torch.cat(
        [edge_index + copy * nodes for copy in range(copies)], dim=1
    )


def make_csr(
    edge_index: torch.Tensor, nodes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source, target = edge_index
    order = target.argsort(stable=True)
    source = source[order]
    target = target[order]
    counts = torch.bincount(target, minlength=nodes)
    row_ptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), counts.to(torch.int32).cumsum(0)]
    ).to(torch.int32)
    return row_ptr, source.to(torch.int32), counts


def staged_forward(
    model: DeeperGCN,
    x: torch.Tensor,
    aggregate: Callable[[torch.Tensor, float], torch.Tensor],
    temperatures: list[float],
) -> torch.Tensor:
    x = model.node_encoder(x)
    first = model.layers[0].conv
    out = aggregate(x, temperatures[0]) + x
    x = first.mlp(out)
    for layer, temperature in zip(model.layers[1:], temperatures[1:]):
        normalized = layer.act(layer.norm(x))
        out = aggregate(normalized, temperature) + normalized
        x = x + layer.conv.mlp(out)
    x = model.layers[0].act(model.layers[0].norm(x))
    return model.lin(x)


def median_cpu_ms(function, warmup: int, iterations: int) -> tuple[float, torch.Tensor]:
    return measure_median(function, warmup, iterations)


def median_npu_ms(function, warmup: int, iterations: int) -> tuple[float, torch.Tensor]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    return measure_median(function, warmup, iterations, torch.npu.synchronize)


def measure_median(function, warmup, iterations, synchronize=None):
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations positive")
    output = None
    for _ in range(warmup):
        output = function()
    if synchronize is not None:
        synchronize()
    samples = []
    for _ in range(iterations):
        if synchronize is not None:
            synchronize()
        start = time.perf_counter()
        output = function()
        if synchronize is not None:
            synchronize()
        samples.append((time.perf_counter() - start) * 1_000.0)
    if output is None:
        raise RuntimeError("benchmark produced no output")
    return statistics.median(samples), output


def accuracy(output: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return float((output.argmax(dim=-1)[mask] == labels[mask]).float().mean())


def make_padded_csr(edge_index: torch.Tensor, nodes: int):
    row_ptr, source, counts = make_csr(edge_index, nodes)
    padded = torch.zeros((nodes, int(counts.max())), dtype=torch.long)
    mask = torch.zeros_like(padded, dtype=torch.bool)
    for node in range(nodes):
        begin, end = int(row_ptr[node]), int(row_ptr[node + 1])
        degree = end - begin
        padded[node, :degree] = source[begin:end]
        mask[node, :degree] = True
    return row_ptr, source, padded, mask


def native_dense(features, padded_source, mask, temperature):
    message = features[padded_source].relu() + 1.0e-7
    logits = (message * temperature).masked_fill(~mask.unsqueeze(-1), float("-inf"))
    return (torch.softmax(logits, dim=1) * message).sum(dim=1)


class BoundCustomAggregate:
    """Bind one graph without mutable defaults or loop-dependent closures."""

    def __init__(self, custom, row_ptr, source):
        self.custom = custom
        self.row_ptr = row_ptr
        self.source = source

    def __call__(self, features, temperature):
        return self.custom(
            self.row_ptr, self.source, features.contiguous(), temperature
        )
