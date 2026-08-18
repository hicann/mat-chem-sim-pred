#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Profile the exact full-graph resident NPU GAT baseline."""

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from benchmark_gat_cora_e2e import (
    GATClassifier,
    build_layout,
    padded_node_aggregate,
    repeat_graph,
)


COMMON_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(COMMON_ROOT))
COMMON = importlib.import_module("attention_model_e2e_common")
LOGGER = logging.getLogger("profile_gat_hotspot")
STAGE_KERNEL_TYPES = {
    "Index",
    "LeakyRelu",
    "MaskedFill",
    "Mul",
    "ReduceSum",
    "Softmax",
    "SoftmaxV2",
}


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--max-degree", type=int, default=169)
    return parser.parse_args()


def _profile_case(args, model, data, layout, copies: int):
    features, _, repeated = repeat_graph(data.x, data.y, layout, copies)
    features = features.to(torch.device("npu:0"))
    repeated = repeated.to(torch.device("npu:0"))
    call = COMMON.ResidentModelCall(model, features, repeated, padded_node_aggregate)
    COMMON.warmup(call)
    trace_dir = args.trace_root / f"cora_gat_copies_{copies}"
    COMMON.profile_once(call, trace_dir)
    return {
        "copies": copies,
        "nodes": features.size(0),
        "edges": repeated.source.numel(),
        "replaceable_calls": 2,
        **COMMON.parse_profile(trace_dir, STAGE_KERNEL_TYPES),
    }


def main() -> None:
    COMMON.configure_logging()
    args = _arguments()
    dataset = Planetoid(str(args.dataset_root), "Cora", transform=NormalizeFeatures())
    data = dataset[0]
    layout = build_layout(data.edge_index, data.num_nodes, args.max_degree)
    if layout.dropped_edges:
        raise RuntimeError(f"truncated {layout.dropped_edges} graph edges")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = GATClassifier(data.num_node_features, dataset.num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("npu:0")
    model = model.eval().to(device)
    results = []
    for copies in args.copies:
        result = _profile_case(args, model, data, layout, copies)
        results.append(result)
        LOGGER.info("%s", json.dumps(result))
    payload = {
        "model": "maintained PyG examples/gat.py topology",
        "profiled_baseline": "exact full-graph resident padded NPU model",
        "dataset": "Cora",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
