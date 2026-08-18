#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Profile the complete resident-NPU LightGCN recommendation model."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

from benchmark_lightgcn_movielens_e2e import (
    build_graph,
    load_ratings,
    repeat_graph,
    resident_embedding,
)

sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("graph_message_model_e2e_common")
log_json, parse_profile, profile_once, warmup = (
    _common.log_json,
    _common.parse_profile,
    _common.profile_once,
    _common.warmup,
)


STAGE_TYPES = {"InplaceIndexAdd", "ZerosLike"}


def _complete_model(embedding, graph, pairs):
    values = resident_embedding(embedding, graph)
    return (values[pairs[0]] * values[pairs[1]]).sum(-1)


def profile_case(base_graph, base_embedding, base_pairs, copies, trace_root):
    graph = repeat_graph(base_graph, copies).to("npu:0")
    embedding = base_embedding.repeat(copies, 1).to("npu:0")
    pair_values = [base_pairs + index * base_graph.nodes for index in range(copies)]
    pairs = torch.cat(pair_values, dim=1).to("npu:0")
    complete_model = partial(_complete_model, embedding, graph, pairs)
    warmup(complete_model)
    trace_dir = trace_root / f"lightgcn_copies_{copies}"
    profile_once(complete_model, trace_dir)
    return {
        "copies": copies,
        "nodes": graph.nodes,
        "edges": graph.source.numel(),
        "pairs": pairs.size(1),
        "replaceable_calls": 1,
        "attribution_note": (
            "Only propagation-exclusive index-add/zero kernels are counted; "
            "shared Index/Mul types used by pair scoring are excluded."
        ),
        **parse_profile(trace_dir, STAGE_TYPES),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copies", nargs="+", type=int, default=[1, 2, 4])
    return parser.parse_args()


def _load_inputs(args):
    train_users, train_items, _ = load_ratings(args.dataset_dir / "u1.base")
    test_users, test_items, _ = load_ratings(args.dataset_dir / "u1.test")
    user_count = int(max(train_users.max(), test_users.max()) + 1)
    item_count = int(max(train_items.max(), test_items.max()) + 1)
    graph = build_graph(train_users, train_items, user_count, item_count)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    negative_items = (test_items * 131 + test_users * 17 + 29) % item_count
    pairs = torch.stack(
        (
            torch.from_numpy(np.concatenate((test_users, test_users))),
            torch.from_numpy(np.concatenate((test_items, negative_items)) + user_count),
        )
    )
    return graph, checkpoint["embedding"], pairs


def main():
    args = parse_args()
    graph, embedding, pairs = _load_inputs(args)
    results = []
    for copies in args.copies:
        result = profile_case(graph, embedding, pairs, copies, args.trace_root)
        results.append(result)
        log_json(result)
    payload = {
        "model": "maintained PyG LightGCN K=2 recommendation inference",
        "profiled_baseline": "exact resident NPU model with pair scoring",
        "dataset": "MovieLens 100K official u1 split",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
