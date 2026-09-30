#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run the same-host CPU/NPU value gate for the GENConv fused aggregate."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
from pathlib import Path

import torch
import torch_geometric
from genconv_common import (
    BoundCustomAggregate,
    CustomGenConvAggregate,
    DeeperGCN,
    accuracy,
    disjoint_copies,
    make_csr,
    median_cpu_ms,
    median_npu_ms,
    native_dense,
    sha256,
    staged_forward,
)
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

LOGGER = logging.getLogger(__name__)


class GlobalPaddedAggregate:
    def __init__(
        self,
        row_ptr: torch.Tensor,
        source: torch.Tensor,
        counts: torch.Tensor,
        device: torch.device,
    ) -> None:
        nodes = counts.numel()
        max_degree = int(counts.max())
        padded = torch.zeros((nodes, max_degree), dtype=torch.long)
        mask = torch.zeros((nodes, max_degree), dtype=torch.bool)
        for node in range(nodes):
            degree = int(counts[node])
            if degree:
                begin = int(row_ptr[node])
                end = begin + degree
                padded[node, :degree] = source[begin:end]
                mask[node, :degree] = True
        self.padded = padded.to(device)
        self.mask = mask.to(device)
        self.padding_ratio = float(padded.numel() / source.numel())

    def __call__(self, features: torch.Tensor, temperature: float) -> torch.Tensor:
        return native_dense(features, self.padded, self.mask, temperature)


class DegreeBucketAggregate:
    def __init__(
        self,
        row_ptr: torch.Tensor,
        source: torch.Tensor,
        counts: torch.Tensor,
        device: torch.device,
    ) -> None:
        buckets: list[tuple[torch.Tensor, torch.Tensor]] = []
        target_order: list[torch.Tensor] = []
        padded_elements = 0
        for exponent in range(math.ceil(math.log2(int(counts.max()))) + 1):
            padded_degree = 1 << exponent
            lower = 0 if exponent == 0 else (1 << (exponent - 1)) + 1
            targets = torch.nonzero(
                (counts >= lower) & (counts <= padded_degree), as_tuple=False
            ).flatten()
            if targets.numel() == 0:
                continue
            padded = torch.zeros((targets.numel(), padded_degree), dtype=torch.long)
            mask = torch.zeros_like(padded, dtype=torch.bool)
            for row, target in enumerate(targets.tolist()):
                degree = int(counts[target])
                begin = int(row_ptr[target])
                end = begin + degree
                padded[row, :degree] = source[begin:end]
                mask[row, :degree] = True
            buckets.append((padded.to(device), mask.to(device)))
            target_order.append(targets)
            padded_elements += padded.numel()
        ordered_targets = torch.cat(target_order)
        if ordered_targets.numel() != counts.numel():
            raise RuntimeError("degree buckets do not cover every node")
        self.buckets = buckets
        self.inverse_order = ordered_targets.argsort().to(device)
        self.padding_ratio = float(padded_elements / source.numel())

    def __call__(self, features: torch.Tensor, temperature: float) -> torch.Tensor:
        parts = []
        for padded, mask in self.buckets:
            parts.append(native_dense(features, padded, mask, temperature))
        return torch.cat(parts, dim=0)[self.inverse_order]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--cpu-threads", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    return parser.parse_args()


class ValueGate:
    def __init__(self, args):
        self.args = args
        dataset = Planetoid(str(args.data_root), name="Cora")
        self.data = dataset[0]
        self.edges, _ = add_self_loops(
            self.data.edge_index, num_nodes=self.data.num_nodes
        )
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        self.cpu_model = DeeperGCN(
            dataset.num_features, 64, dataset.num_classes, args.layers
        )
        self.cpu_model.load_state_dict(checkpoint["state_dict"])
        self.cpu_model.eval()
        self.device = torch.device("npu:0")
        self.npu_model = DeeperGCN(
            dataset.num_features, 64, dataset.num_classes, args.layers
        )
        self.npu_model.load_state_dict(checkpoint["state_dict"])
        self.npu_model = self.npu_model.to(self.device).eval()
        self.temperatures = [
            float(layer.conv.aggr_module.t.detach().cpu())
            for layer in self.npu_model.layers
        ]
        self.custom = CustomGenConvAggregate(args.build.resolve(), self.device)


class ValueGateCase:
    def __init__(self, gate, copies):
        self.gate = gate
        self.args = gate.args
        self.copies = copies
        self.x_cpu, self.edges_cpu = disjoint_copies(gate.data.x, gate.edges, copies)
        self.labels = gate.data.y.repeat(copies)
        self.test_mask = gate.data.test_mask.repeat(copies)
        self.cpu_timings = {}
        self.cpu_output = None
        self.resident_timings = {}
        self.resident_outputs = {}
        self.host_timings = {}
        self.host_outputs = {}
        self.edge_count = 0
        self.x_npu = None
        self.edges_npu = None
        self.custom_aggregate = None
        self.global_padded = None
        self.bucketed = None

    def measure_cpu(self):
        for thread_count in self.args.cpu_threads:
            torch.set_num_threads(thread_count)
            elapsed, output = median_cpu_ms(
                lambda: self.gate.cpu_model(self.x_cpu, self.edges_cpu),
                self.args.warmup,
                self.args.iterations,
            )
            self.cpu_timings[str(thread_count)] = elapsed
            if self.cpu_output is None:
                self.cpu_output = output.detach().clone()
        if self.cpu_output is None:
            raise RuntimeError("CPU thread sweep produced no output")

    def prepare_npu(self):
        row_ptr, source, counts = make_csr(self.edges_cpu, self.x_cpu.size(0))
        self.edge_count = source.numel()
        device = self.gate.device
        self.x_npu = self.x_cpu.to(device)
        self.edges_npu = self.edges_cpu.to(device)
        self.custom_aggregate = BoundCustomAggregate(
            self.gate.custom, row_ptr.to(device), source.to(device)
        )
        for temperature in self.gate.temperatures:
            self.gate.custom.prepare(
                self.x_cpu.size(0), self.edge_count, 64, temperature
            )
        self.global_padded = GlobalPaddedAggregate(row_ptr, source, counts, device)
        self.bucketed = DegreeBucketAggregate(row_ptr, source, counts, device)

    def resident_forward(self, aggregate):
        return staged_forward(
            self.gate.npu_model, self.x_npu, aggregate, self.gate.temperatures
        )

    def host_to_host(self, aggregate):
        output = staged_forward(
            self.gate.npu_model,
            self.x_cpu.to(self.gate.device),
            aggregate,
            self.gate.temperatures,
        )
        return output.cpu()

    def measure_npu(self):
        resident_functions = {
            "custom": lambda: self.resident_forward(self.custom_aggregate),
            "power2_degree_bucketed": lambda: self.resident_forward(self.bucketed),
            "global_max_degree_padded": lambda: self.resident_forward(
                self.global_padded
            ),
            # Keep CPU fallback last to preserve resident candidate allocator state.
            "official_pyg_eager": lambda: self.gate.npu_model(
                self.x_npu, self.edges_npu
            ),
        }
        for name, function in resident_functions.items():
            elapsed, output = median_npu_ms(
                function, self.args.warmup, self.args.iterations
            )
            self.resident_timings[name] = elapsed
            self.resident_outputs[name] = output.detach().cpu()
        host_functions = {
            "power2_degree_bucketed": lambda: self.host_to_host(self.bucketed),
            "custom": lambda: self.host_to_host(self.custom_aggregate),
        }
        for name, function in host_functions.items():
            elapsed, output = median_npu_ms(
                function, self.args.warmup, self.args.iterations
            )
            self.host_timings[name] = elapsed
            self.host_outputs[name] = output

    def timing_results(self):
        native_candidates = {
            name: elapsed
            for name, elapsed in self.resident_timings.items()
            if name != "custom"
        }
        strongest_name, strongest_ms = min(
            native_candidates.items(), key=lambda item: item[1]
        )
        fastest_threads, fastest_ms = min(
            self.cpu_timings.items(), key=lambda item: item[1]
        )
        custom_resident_ms = required_result(self.resident_timings, "custom")
        custom_host_ms = required_result(self.host_timings, "custom")
        return {
            "cpu_official_pyg_e2e_ms_by_threads": self.cpu_timings,
            "fastest_cpu_threads": int(fastest_threads),
            "fastest_cpu_e2e_ms": fastest_ms,
            "resident_npu_e2e_ms": self.resident_timings,
            "strongest_native_npu": strongest_name,
            "strongest_native_npu_e2e_ms": strongest_ms,
            "custom_vs_strongest_npu_speedup": strongest_ms / custom_resident_ms,
            "custom_vs_strongest_npu_reduction_pct": 100.0
            * (strongest_ms - custom_resident_ms)
            / strongest_ms,
            "host_to_host_e2e_ms": self.host_timings,
            "custom_host_to_host_vs_cpu_ratio": custom_host_ms / fastest_ms,
            "custom_host_to_host_vs_cpu_reduction_pct": 100.0
            * (fastest_ms - custom_host_ms)
            / fastest_ms,
        }

    def accuracy_results(self):
        custom_output = required_result(self.resident_outputs, "custom")
        bucketed_output = required_result(
            self.resident_outputs, "power2_degree_bucketed"
        )
        padded_output = required_result(
            self.resident_outputs, "global_max_degree_padded"
        )
        host_output = required_result(self.host_outputs, "custom")
        return {
            "max_abs_error_vs_cpu_official": float(
                (custom_output - self.cpu_output).abs().max()
            ),
            "max_abs_error_bucketed_vs_cpu_official": float(
                (bucketed_output - self.cpu_output).abs().max()
            ),
            "max_abs_error_global_padded_vs_cpu_official": float(
                (padded_output - self.cpu_output).abs().max()
            ),
            "prediction_agreement_vs_cpu_official": float(
                (custom_output.argmax(dim=-1) == self.cpu_output.argmax(dim=-1))
                .float()
                .mean()
            ),
            "cpu_test_accuracy": accuracy(self.cpu_output, self.labels, self.test_mask),
            "custom_test_accuracy": accuracy(
                custom_output, self.labels, self.test_mask
            ),
            "host_output_max_abs_error_vs_cpu_official": float(
                (host_output - self.cpu_output).abs().max()
            ),
        }

    def run(self):
        self.measure_cpu()
        self.prepare_npu()
        self.measure_npu()
        row = {
            "copies": self.copies,
            "nodes": int(self.x_cpu.size(0)),
            "edges": int(self.edge_count),
            "global_padding_ratio": self.global_padded.padding_ratio,
            "power2_bucket_padding_ratio": self.bucketed.padding_ratio,
        }
        row.update(self.timing_results())
        row.update(self.accuracy_results())
        return row


def required_result(results, name):
    value = results.get(name)
    if value is None:
        raise RuntimeError(f"missing benchmark result: {name}")
    return value


def make_payload(args, results):
    return {
        "schema_version": 1,
        "scope": "same-host GENConv complete-model value gate",
        "model": "eight-layer PyG DeeperGCN with exact GENConv weights",
        "dataset": "Cora",
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_origin": "fixed-seed locally trained; not an official checkpoint",
        "environment": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
            "torch_geometric_version": torch_geometric.__version__,
            "npu": torch.npu.get_device_name(0),
        },
        "timing": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "statistic": "median wall time",
            "synchronization": "before and after every measured NPU forward",
            "cpu_boundary": "official PyG model forward, CPU input to CPU logits",
            "resident_npu_boundary": "resident model, graph, input and output",
            "host_to_host_boundary": (
                "CPU feature input through H2D, full model, and D2H logits; "
                "static graph and model weights remain resident"
            ),
        },
        "baseline_policy": (
            "minimum correct resident NPU latency among official PyG eager, "
            "global maximum-degree padding, and power-of-two degree buckets"
        ),
        "torchair": {
            "status": "not_measured",
            "reason": "official PyG scatter_reduce/index_copy path was not compiled in this run",
        },
        "results": results,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260802)
    torch.set_num_interop_threads(1)
    torch.npu.set_device("npu:0")
    gate = ValueGate(args)
    results = []
    with torch.no_grad():
        for copies in args.copies:
            row = ValueGateCase(gate, copies).run()
            results.append(row)
            LOGGER.info("%s", json.dumps(row, sort_keys=True))
    payload = make_payload(args, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
