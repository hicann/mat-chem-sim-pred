#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Compare official GemNet geometry calculation with the fused cache kernel."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import torch
from benchmark_gemnet_qm9_e2e import (
    _graph_inputs,
    build_gemnet_model,
    timed,
)
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9

LOGGER = logging.getLogger(__name__)


class GeometryGraph:
    def __init__(self, source, int_source, topology, common):
        self.source = source
        self.int_source = int_source
        self.topology = topology
        self.common = common


def _custom_calculate(*inputs, common, geometry_binding):
    (
        position,
        id_c,
        id_a,
        id4_int_b,
        id4_int_a,
        id4_expand_abd,
        id4_reduce_cab,
        id4_expand_intm_db,
        id4_reduce_intm_ca,
        id4_expand_intm_ab,
        id4_reduce_intm_ab,
    ) = inputs
    del id4_expand_abd, id4_expand_intm_db, id4_reduce_intm_ca
    angle_cab, angle_abd, theta = geometry_binding.gemnet_quadruplet_geometry_fused(
        position,
        id_c,
        id_a,
        id4_int_b,
        id4_int_a,
        common["id4_reduce_ca"],
        common["id4_expand_db"],
        common["id4_reduce_intm_ca"],
        common["id4_expand_intm_db"],
        id4_reduce_intm_ab,
        id4_expand_intm_ab,
    )
    return angle_cab[id4_reduce_cab], angle_abd, theta


def _run_model(model, inputs, calculate_angles):
    model.calculate_angles = calculate_angles
    return model(inputs)


def _load_components(args):
    torch.npu.set_device(0)
    sys.path.insert(0, str(args.official_source))
    import types

    from torch_geometric.utils import scatter

    torch_scatter = types.ModuleType("torch_scatter")
    torch_scatter.scatter = scatter
    sys.modules["torch_scatter"] = torch_scatter
    from gemnet.model.gemnet import GemNet

    sys.path.insert(0, str(Path(__file__).parents[2] / "integration"))
    import geometry_binding

    geometry_binding.configure(args.build_dir)
    scale_file = args.output.with_suffix(".scales.json")
    scale_file.write_text("{}\n", encoding="utf-8")
    model = build_gemnet_model(GemNet, scale_file)
    return model, model.calculate_angles, geometry_binding, QM9(str(args.dataset_root))


def _graph_common(batch):
    graph = _graph_inputs(batch)
    topology = graph.topology
    common = dict(graph.common)
    common.update(
        {
            "id4_reduce_ca": topology[0],
            "id4_expand_db": topology[1],
            "id4_reduce_cab": topology[2],
            "id4_expand_abd": topology[3],
            "Kidx4": topology[4],
            "id4_reduce_intm_ca": topology[5],
            "id4_expand_intm_db": topology[6],
            "id4_reduce_intm_ab": topology[7],
            "id4_expand_intm_ab": topology[8],
        }
    )
    return GeometryGraph(graph.source, graph.int_source, topology, common)


def _angle_values(common, angle_function):
    return angle_function(
        common["R"],
        common["id_c"],
        common["id_a"],
        common["id4_int_b"],
        common["id4_int_a"],
        common["id4_expand_abd"],
        common["id4_reduce_cab"],
        common["id4_expand_intm_db"],
        common["id4_reduce_intm_ca"],
        common["id4_expand_intm_ab"],
        common["id4_reduce_intm_ab"],
    )


def _run_graph(graph_count, dataset, model, original_angles, geometry_binding):
    batch = Batch.from_data_list([dataset[2048 + i] for i in range(graph_count)])
    graph = _graph_common(batch)
    source, int_source = graph.source, graph.int_source
    topology, common = graph.topology, graph.common
    custom_angles = partial(
        _custom_calculate, common=common, geometry_binding=geometry_binding
    )
    native = partial(
        _run_model, model=model, inputs=common, calculate_angles=original_angles
    )
    custom = partial(
        _run_model, model=model, inputs=common, calculate_angles=custom_angles
    )
    native_angles = _angle_values(common, original_angles)
    custom_values = _angle_values(common, custom_angles)
    native_ms, native_output = timed(native)
    custom_ms, custom_output = timed(custom)
    angle_diffs = [
        float((a - b).abs().max().cpu()) for a, b in zip(native_angles, custom_values)
    ]
    model_diff = max(
        float((a - b).abs().max().cpu()) for a, b in zip(native_output, custom_output)
    )
    return {
        "graphs": graph_count,
        "nodes": int(batch.num_nodes),
        "embedding_edges": int(source.numel()),
        "interaction_edges": int(int_source.numel()),
        "quadruplets": int(topology[0].numel()),
        "intermediate_ca": int(topology[5].numel()),
        "intermediate_db": int(topology[6].numel()),
        "native_e2e_ms": native_ms,
        "custom_e2e_ms": custom_ms,
        "speedup": native_ms / custom_ms,
        "angle_max_abs_diff": max(angle_diffs),
        "angle_component_max_abs_diff": angle_diffs,
        "model_max_abs_diff": model_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--graphs", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, original_angles, geometry_binding, dataset = _load_components(args)
    results = [
        _run_graph(count, dataset, model, original_angles, geometry_binding)
        for count in args.graphs
    ]
    payload = {
        "candidate": "GemNet official cached quadruplet geometry",
        "random_weights": True,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("%s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
