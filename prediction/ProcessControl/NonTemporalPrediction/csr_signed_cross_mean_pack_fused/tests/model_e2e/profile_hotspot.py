#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Profile the second-layer signed cross-mean pack in complete SignedGCN."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_signedgcn_bitcoin_e2e import (
    load_signed_setup,
    native_pack,
    prepare_case,
    second_layer,
)

sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("graph_message_model_e2e_common")
log_json, parse_operator_details, parse_profile, profile_once, warmup = (
    _common.log_json,
    _common.parse_operator_details,
    _common.parse_profile,
    _common.profile_once,
    _common.warmup,
)


@dataclass
class ProfileCase:
    copies: int
    features: torch.Tensor
    positive: object
    negative: object
    positive_edge: torch.Tensor
    negative_edge: torch.Tensor
    test_edges: torch.Tensor
    hidden: torch.Tensor


def clean_shapes(row):
    return row["Input Shapes"].replace("\r", "").replace("\n", "")


def _complete_model(model, case):
    hidden = F.relu(model.conv1(case.features, case.positive_edge, case.negative_edge))
    embedding = second_layer(model, hidden, case.positive, case.negative, native_pack)
    return model.discriminate(embedding, case.test_edges)


def _component(case):
    return native_pack(case.hidden, case.positive, case.negative)


def _select_details(rows, nodes, channels, positive_edges, negative_edges):
    node_shape = f"{nodes},{channels};"
    edge_shapes = (f"{positive_edges},{channels};", f"{negative_edges},{channels};")
    selected_by_name = {}
    for name in ("aten::index", "aten::index_add_", "aten::mul"):
        matches = []
        for row in rows:
            shapes = clean_shapes(row)
            if row["Name"] != name or not shapes.startswith(node_shape):
                continue
            matches_index_add = name == "aten::index_add_" and any(
                shape in shapes for shape in edge_shapes
            )
            matches_multiply = name == "aten::mul" and f"{nodes},1" in shapes
            if name == "aten::index" or matches_index_add or matches_multiply:
                matches.append(row)
        selected_by_name[name] = matches
    if any(len(value) != 4 for value in selected_by_name.values()):
        counts = {name: len(value) for name, value in selected_by_name.items()}
        raise RuntimeError(f"expected four calls per stage op, got {counts}")
    selected = []
    for values in selected_by_name.values():
        selected.extend(values)
    return selected


def _detail_payload(selected):
    details = []
    for row in selected:
        details.append(
            {
                "name": row["Name"],
                "input_shapes": row["Input Shapes"],
                "device_total_us": float(row["Device Total Duration(us)"]),
            }
        )
    return details


def profile_case(payload, base_values, copies, trace_root, reuse_traces):
    base_case = prepare_case(payload, base_values, copies)
    hidden = F.relu(
        payload["model"].conv1(
            base_case.features, base_case.positive_edge, base_case.negative_edge
        )
    )
    case = ProfileCase(**vars(base_case), hidden=hidden)
    complete_model = partial(_complete_model, payload["model"], case)
    component = partial(_component, case)
    warmup(complete_model)
    full_trace = trace_root / f"copies_{copies}_full"
    component_trace = trace_root / f"copies_{copies}_component"
    if not reuse_traces:
        profile_once(complete_model, full_trace)
        profile_once(component, component_trace)
    full = parse_profile(full_trace)
    component_profile = parse_profile(component_trace)
    channels = case.hidden.size(1) // 2
    selected = _select_details(
        parse_operator_details(full_trace),
        case.features.size(0),
        channels,
        case.positive.source.numel(),
        case.negative.source.numel(),
    )
    stage_us = sum(float(row["Device Total Duration(us)"]) for row in selected)
    return {
        "copies": copies,
        "nodes": case.features.size(0),
        "positive_edges": case.positive.source.numel(),
        "negative_edges": case.negative.source.numel(),
        "full_model_kernel_us": full["full_model_kernel_us"],
        "conservative_stage_kernel_us": stage_us,
        "conservative_stage_hotspot_pct": 100.0
        * stage_us
        / full["full_model_kernel_us"],
        "exact_stage_framework_ops": _detail_payload(selected),
        "full_by_type": full["by_type"],
        "one_stage_component_kernel_us": component_profile["full_model_kernel_us"],
        "component_by_type": component_profile["by_type"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--reuse-traces", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    setup = load_signed_setup(args.checkpoint)
    setup.payload["model"] = setup.model
    results = []
    for copies in args.copies:
        result = profile_case(
            setup.payload,
            setup.base_values,
            copies,
            args.trace_root,
            args.reuse_traces,
        )
        results.append(result)
        log_json(result)
    output = {
        "operator": "CsrSignedCrossMeanPackFused",
        "model": "maintained PyG SignedGCN on Bitcoin-OTC",
        "profiled_baseline": "complete native replacement-equivalent model",
        "results": results,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
