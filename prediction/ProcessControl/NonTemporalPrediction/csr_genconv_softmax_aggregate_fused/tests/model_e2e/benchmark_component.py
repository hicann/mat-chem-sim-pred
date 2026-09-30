#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Benchmark fused GENConv softmax aggregation against an all-NPU baseline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from genconv_common import CustomGenConvAggregate, disjoint_copies, native_dense
from genconv_common import make_padded_csr as make_csr
from genconv_common import median_npu_ms as timed
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


class ComponentBenchmark:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("npu:0")
        self.data = Planetoid(str(args.data_root), name="Cora")[0]
        self.edges, _ = add_self_loops(
            self.data.edge_index, num_nodes=self.data.num_nodes
        )
        self.projector = (
            torch.nn.Linear(self.data.num_features, args.channels)
            .to(self.device)
            .eval()
        )
        self.custom = CustomGenConvAggregate(args.build.resolve(), self.device)

    def run_case(self, copies):
        features, edges = disjoint_copies(self.data.x, self.edges, copies)
        features = self.projector(features.to(self.device)).contiguous()
        row_ptr, source, padded, mask = (
            tensor.to(self.device) for tensor in make_csr(edges, features.size(0))
        )
        baseline_ms, expected = timed(
            lambda: native_dense(features, padded, mask, self.args.temperature),
            self.args.warmup,
            self.args.iterations,
        )
        custom_ms, actual = timed(
            lambda: self.custom(row_ptr, source, features, self.args.temperature),
            self.args.warmup,
            self.args.iterations,
        )
        return {
            "copies": copies,
            "nodes": int(features.size(0)),
            "edges": int(source.numel()),
            "channels": self.args.channels,
            "max_degree": int(padded.size(1)),
            "native_all_npu_ms": baseline_ms,
            "custom_ms": custom_ms,
            "speedup": baseline_ms / custom_ms,
            "max_abs_error": float((expected - actual).abs().max().cpu()),
        }


def main() -> None:
    args = parse_args()
    torch.manual_seed(20260802)
    torch.npu.set_device("npu:0")
    benchmark = ComponentBenchmark(args)
    results = []
    with torch.no_grad():
        for copies in args.copies:
            row = benchmark.run_case(copies)
            results.append(row)
            LOGGER.info("%s", json.dumps(row, sort_keys=True))
    payload = {
        "model_stage": "PyG GENConv message + feature-wise softmax aggregate",
        "dataset": "Cora",
        "baseline": "all-NPU padded gather, stable softmax, weighted reduction",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
