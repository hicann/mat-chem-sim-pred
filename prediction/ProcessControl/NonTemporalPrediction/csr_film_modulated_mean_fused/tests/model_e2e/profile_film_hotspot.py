#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Profile the resident PPI FiLM model and exact replaceable aggregation."""

import argparse
import importlib
import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

from benchmark_film_ppi_e2e import Net, layer_inputs, manual_model, repeat_graph
from film_benchmark_common import build_layout, native_mean

sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("graph_message_model_e2e_common")


def _first_stage(prepared, layout):
    return native_mean(*prepared, layout, True)


def _profile_summary(trace_dir):
    raw = _common.parse_profile(trace_dir)
    by_type = []
    for row in raw["by_type"]:
        by_type.append(
            {
                "type": row["type"],
                "count": row["count"],
                "duration_us": row["duration_us"],
            }
        )
    return {
        "kernel_count": raw["kernel_count"],
        "kernel_us": raw["full_model_kernel_us"],
        "by_type": by_type,
    }


def profile_case(model, data, copies, trace_root):
    features, _, edge_index = repeat_graph(data, copies)
    layout = build_layout(edge_index, features.size(0)).to(torch.device("npu:0"))
    features = features.to("npu:0")
    with torch.inference_mode():
        prepared = layer_inputs(features, model.convs[0])
    first_stage = partial(_first_stage, prepared, layout)
    complete_model = partial(manual_model, model, features, layout, native_mean)
    _common.warmup(complete_model)
    _common.warmup(first_stage)
    full_trace = trace_root / f"copies_{copies}_full"
    stage_trace = trace_root / f"copies_{copies}_stage"
    _common.profile_once(complete_model, full_trace)
    _common.profile_once(first_stage, stage_trace)
    full = _profile_summary(full_trace)
    stage = _profile_summary(stage_trace)
    return {
        "copies": copies,
        "graphs": 2 * copies,
        "nodes": data.num_nodes * copies,
        "edges": data.num_edges * copies,
        "full_model": full,
        "replaceable_stage": stage,
        "stage_hotspot_pct": 100.0 * stage["kernel_us"] / full["kernel_us"],
        "attribution": (
            "isolated exact first-layer native FiLM modulation and mean "
            "divided by exact resident full-model NPU kernel time"
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", type=int, nargs="+", default=[1, 2, 4])
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = PPI(args.dataset_root, split="train")
    data = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Net(dataset.num_features, dataset.num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.eval().to(torch.device("npu:0"))
    results = []
    for copies in args.copies:
        result = profile_case(model, data, copies, args.trace_root)
        results.append(result)
        _common.log_json(result)
    payload = {
        "model": "maintained PyG examples/film.py PPI topology",
        "dataset": "PPI deterministic two-graph training batch",
        "profiled_baseline": "strong exact resident NPU baseline",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
