#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Train and benchmark an exact PyG DeeperGCN around the fused aggregate."""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from genconv_common import (
    BoundCustomAggregate,
    CustomGenConvAggregate,
    DeeperGCN,
    accuracy,
    disjoint_copies,
    median_npu_ms,
    native_dense,
    sha256,
    staged_forward,
)
from genconv_common import make_padded_csr as make_csr
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

LOGGER = logging.getLogger(__name__)


def train_checkpoint(
    model: DeeperGCN,
    data,
    edge_index: torch.Tensor,
    checkpoint: Path,
    epochs: int,
) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data.x, edge_index)
        loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            LOGGER.info("train epoch=%d loss=%.6f", epoch + 1, float(loss))
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "epochs": epochs,
            "seed": 20260802,
        },
        checkpoint,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


class CheckedAggregate:
    def __init__(self, aggregate, copies):
        self.aggregate = aggregate
        self.copies = copies
        self.calls = 0

    def __call__(self, features, temperature):
        output = self.aggregate(features, temperature)
        torch.npu.synchronize()
        LOGGER.info(
            "custom validation copies=%d layer=%d temperature=%.9f",
            self.copies,
            self.calls,
            temperature,
        )
        self.calls += 1
        return output


class ModelBenchmark:
    def __init__(self, args):
        self.args = args
        dataset = Planetoid(str(args.data_root), name="Cora")
        self.data = dataset[0]
        self.edges, _ = add_self_loops(
            self.data.edge_index, num_nodes=self.data.num_nodes
        )
        cpu_model = DeeperGCN(
            dataset.num_features, 64, dataset.num_classes, args.layers
        )
        if not args.checkpoint.exists():
            train_checkpoint(
                cpu_model, self.data, self.edges, args.checkpoint, args.epochs
            )
        self.checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
        cpu_model.load_state_dict(self.checkpoint["state_dict"])
        self.device = torch.device("npu:0")
        self.model = cpu_model.to(self.device).eval()
        self.temperatures = [
            float(layer.conv.aggr_module.t.detach().cpu())
            for layer in self.model.layers
        ]
        self.custom = CustomGenConvAggregate(args.build.resolve(), self.device)

    def run_case(self, copies):
        x, edges = disjoint_copies(self.data.x, self.edges, copies)
        x = x.to(self.device)
        row_ptr, source, padded, mask = (
            tensor.to(self.device) for tensor in make_csr(edges, x.size(0))
        )

        def baseline_aggregate(features, temperature):
            return native_dense(features, padded, mask, temperature)

        custom_aggregate = BoundCustomAggregate(self.custom, row_ptr, source)
        baseline = partial(
            staged_forward, self.model, x, baseline_aggregate, self.temperatures
        )
        fused = partial(
            staged_forward, self.model, x, custom_aggregate, self.temperatures
        )
        staged_forward(
            self.model, x, CheckedAggregate(custom_aggregate, copies), self.temperatures
        )
        baseline_output, custom_output = baseline(), fused()
        torch.npu.synchronize()
        exact_reference_error = None
        if copies == 1:
            exact_output = self.model(x, edges.to(self.device))
            exact_reference_error = float(
                (exact_output - baseline_output).abs().max().cpu()
            )
        baseline_ms, _ = median_npu_ms(baseline, self.args.warmup, self.args.iterations)
        custom_ms, _ = median_npu_ms(fused, self.args.warmup, self.args.iterations)
        row = {
            "copies": copies,
            "nodes": int(x.size(0)),
            "edges": int(source.numel()),
            "baseline_e2e_ms": baseline_ms,
            "custom_e2e_ms": custom_ms,
            "e2e_speedup": baseline_ms / custom_ms,
            "e2e_reduction_pct": 100.0 * (baseline_ms - custom_ms) / baseline_ms,
            "exact_pyg_reference_max_error": exact_reference_error,
        }
        row.update(self.compare_outputs(copies, baseline_output, custom_output))
        return row

    def compare_outputs(self, copies, baseline_output, custom_output):
        labels = self.data.y.repeat(copies).to(self.device)
        test_mask = self.data.test_mask.repeat(copies).to(self.device)
        return {
            "max_model_abs_error": float(
                (baseline_output - custom_output).abs().max().cpu()
            ),
            "prediction_agreement": float(
                (baseline_output.argmax(dim=-1) == custom_output.argmax(dim=-1))
                .float()
                .mean()
                .cpu()
            ),
            "baseline_test_accuracy": accuracy(baseline_output, labels, test_mask),
            "custom_test_accuracy": accuracy(custom_output, labels, test_mask),
        }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260802)
    torch.npu.set_device("npu:0")
    benchmark = ModelBenchmark(args)
    results = []
    with torch.no_grad():
        for copies in args.copies:
            row = benchmark.run_case(copies)
            results.append(row)
            LOGGER.info("%s", json.dumps(row, sort_keys=True))
    payload = {
        "model": "8-layer PyG DeeperGCN with exact GENConv weights",
        "dataset": "Cora",
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_epochs": int(benchmark.checkpoint["epochs"]),
        "baseline": "all-NPU padded GENConv message/softmax aggregate",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
