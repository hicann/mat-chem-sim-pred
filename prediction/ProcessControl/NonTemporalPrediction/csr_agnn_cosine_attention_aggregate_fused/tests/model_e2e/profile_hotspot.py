#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Profile the complete resident-NPU model for this attention package."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path

import torch
from profiler_common import parse_profile, profile_once, warmup
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures


def detect_package(root: Path):
    candidates = {
        "agnn": (
            "benchmark_agnn_cora_e2e",
            [1, 4, 16],
            {"Index", "Mul", "ReduceSum", "SoftmaxV2", "MaskedFill"},
        ),
        "gatv2": (
            "benchmark_gatv2_cora_e2e",
            [1, 4, 8],
            {
                "Index",
                "LeakyRelu",
                "Mul",
                "ReduceSum",
                "SoftmaxV2",
                "MaskedFill",
            },
        ),
        "transformer": (
            "benchmark_transformer_cora_e2e",
            [1, 4, 8],
            {"Index", "Mul", "ReduceSum", "SoftmaxV2", "MaskedFill"},
        ),
    }
    for kind, (module_name, copies, stage_types) in candidates.items():
        if (root / f"{module_name}.py").exists():
            return kind, importlib.import_module(module_name), copies, stage_types
    raise RuntimeError("no supported benchmark module found")


def load_model(kind, module, dataset, checkpoint):
    model = module.Model(dataset.num_features, dataset.num_classes)
    state = checkpoint["state_dict"]
    if kind == "agnn":
        model.lin1.load_state_dict(
            {"weight": state["lin1.weight"], "bias": state["lin1.bias"]}
        )
        model.lin2.load_state_dict(
            {"weight": state["lin2.weight"], "bias": state["lin2.bias"]}
        )
        model.beta.data.copy_(state["prop2.beta"])
    else:
        module.load_weights(model, state)
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", nargs="+", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(__file__).resolve().parent
    kind, module, default_copies, stage_types = detect_package(root)
    dataset = Planetoid(str(args.dataset_root), "Cora", transform=NormalizeFeatures())
    data = dataset[0]
    layout = module.build_layout(data.edge_index, data.num_nodes)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    device = torch.device("npu:0")
    model = load_model(kind, module, dataset, checkpoint).eval().to(device)
    results = []
    for copies in args.copies or default_copies:
        features, repeated = module.repeat_layout(data.x, layout, copies)
        features, repeated = features.to(device), repeated.to(device)

        def complete_model(features=features, repeated=repeated):
            aggregate = (
                module.resident_stage if kind == "agnn" else module.resident_aggregate
            )
            return model(features, repeated, aggregate)

        warmup(complete_model)
        trace_dir = args.trace_root / f"{kind}_copies_{copies}"
        profile_once(complete_model, trace_dir)
        result = {
            "copies": copies,
            "nodes": features.size(0),
            "edges": repeated.source.numel(),
            "replaceable_calls": 2,
            **parse_profile(trace_dir, stage_types),
        }
        results.append(result)
        logging.getLogger(__name__).info("%s", json.dumps(result))
    args.output.write_text(
        json.dumps(
            {
                "candidate": kind,
                "model": "complete maintained PyG two-layer model",
                "profiled_baseline": "exact resident padded NPU implementation",
                "dataset": "Cora",
                "results": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
