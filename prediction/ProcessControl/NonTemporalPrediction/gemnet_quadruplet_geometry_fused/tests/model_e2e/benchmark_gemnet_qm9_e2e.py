#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Run a small official GemNet forward with native and custom topology paths."""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from functools import partial
from pathlib import Path
from typing import NamedTuple

import torch
from torch_geometric.data import Batch
from torch_geometric.datasets import QM9

LOGGER = logging.getLogger(__name__)


class QuadrupletTopology(NamedTuple):
    edge_ca: torch.Tensor
    edge_db: torch.Tensor
    ca_map: torch.Tensor
    db_map: torch.Tensor
    kidx: torch.Tensor
    intermediate_ca: torch.Tensor
    intermediate_db: torch.Tensor
    intermediate_ca_ab: torch.Tensor
    intermediate_db_ab: torch.Tensor


class GraphInputs:
    def __init__(self, values):
        (
            self.source,
            self.int_source,
            self.row_npu,
            self.source_npu,
            self.int_source_npu,
            self.int_target_npu,
            self.common,
            self.topology,
            self.capacities,
            self.lengths,
        ) = values


def _make_graph_inputs(values):
    return GraphInputs(values)


class TopologyContext:
    def __init__(
        self, row_ptr, source, int_source, int_target, interaction, degree_a, degree_b
    ):
        self.row_ptr, self.source = row_ptr, source
        self.int_source, self.int_target = int_source, int_target
        self.interaction, self.degree_a, self.degree_b = interaction, degree_a, degree_b


GEOMETRY_KEYS = (
    "id4_reduce_ca",
    "id4_expand_db",
    "id4_reduce_cab",
    "id4_expand_abd",
    "Kidx4",
    "id4_reduce_intm_ca",
    "id4_expand_intm_db",
    "id4_reduce_intm_ab",
    "id4_expand_intm_ab",
)


def _populate_inputs(common, values):
    inputs = dict(common)
    for key, value in zip(GEOMETRY_KEYS, values):
        inputs[key] = value
    return inputs


def _populate_custom_inputs(common, values, capacities, intermediate_lengths):
    inputs = dict(common)
    for key, value in zip(GEOMETRY_KEYS, values[:-1]):
        if key in (
            "id4_reduce_ca",
            "id4_expand_db",
            "id4_reduce_cab",
            "id4_expand_abd",
            "Kidx4",
        ):
            length = capacities[0]
        elif key in ("id4_reduce_intm_ca", "id4_reduce_intm_ab"):
            length = intermediate_lengths[0]
        else:
            length = intermediate_lengths[1]
        inputs[key] = value.narrow(0, 0, length)
    return inputs


def _run_model(model, inputs):
    return model(inputs)


def radius_edges(position, batch, cutoff):
    edges = []
    for graph in range(int(batch.max()) + 1):
        nodes = (batch == graph).nonzero(as_tuple=False).view(-1)
        distance = torch.cdist(position[nodes], position[nodes])
        target, source = ((distance < cutoff) & (distance > 0)).nonzero(as_tuple=True)
        edges.append(torch.stack([nodes[source], nodes[target]]))
    source, target = torch.cat(edges, dim=1)
    order = torch.argsort(target * position.size(0) + source, stable=True)
    return source[order].to(torch.int32), target[order].to(torch.int32)


def csr(target, nodes):
    degree = torch.bincount(target.long(), minlength=nodes)
    row_ptr = torch.cat(
        [torch.zeros(1, dtype=torch.int32), degree.cumsum(0).to(torch.int32)]
    )
    return row_ptr


def reverse_edges(source, target):
    mapping = {
        (int(s), int(t)): index
        for index, (s, t) in enumerate(zip(source.tolist(), target.tolist()))
    }
    return torch.tensor(
        [mapping[(int(t), int(s))] for s, t in zip(source.tolist(), target.tolist())],
        dtype=torch.int32,
    )


def triplets(row_ptr, source, target):
    reduce_ca = []
    expand_ba = []
    kidx = []
    for edge_ca, atom_a in enumerate(target.tolist()):
        local = 0
        atom_c = int(source[edge_ca])
        for edge_ba in range(int(row_ptr[atom_a]), int(row_ptr[atom_a + 1])):
            if int(source[edge_ba]) == atom_c:
                continue
            reduce_ca.append(edge_ca)
            expand_ba.append(edge_ba)
            kidx.append(local)
            local += 1
    return tuple(
        torch.tensor(value, dtype=torch.int32) for value in (reduce_ca, expand_ba, kidx)
    )


def native_quadruplets(row_ptr, source, int_source, int_target):
    degree = row_ptr[1:].long() - row_ptr[:-1].long()
    interaction = torch.arange(int_source.numel(), device=row_ptr.device)
    degree_a = degree.index_select(0, int_target.long())
    degree_b = degree.index_select(0, int_source.long())
    context = TopologyContext(
        row_ptr, source, int_source, int_target, interaction, degree_a, degree_b
    )
    intermediate = _intermediate_indices(context)
    quadruplets = _quadruplet_indices(context)
    return QuadrupletTopology(*quadruplets, *intermediate)


def _intermediate_indices(context):
    row_ptr, int_source, int_target = (
        context.row_ptr,
        context.int_source,
        context.int_target,
    )
    interaction, degree_a, degree_b = (
        context.interaction,
        context.degree_a,
        context.degree_b,
    )
    ca_interaction = torch.repeat_interleave(interaction, degree_a)
    ca_group = torch.cumsum(degree_a, dim=0) - degree_a
    ca_rel = torch.arange(ca_interaction.numel(), device=row_ptr.device)
    ca_rel -= torch.repeat_interleave(ca_group, degree_a)
    intm_ca = (
        row_ptr.long().index_select(
            0, int_target.long().index_select(0, ca_interaction)
        )
        + ca_rel
    )
    intm_ca_ab = torch.repeat_interleave(interaction, degree_a)
    db_interaction = torch.repeat_interleave(interaction, degree_b)
    db_group = torch.cumsum(degree_b, dim=0) - degree_b
    db_rel = torch.arange(db_interaction.numel(), device=row_ptr.device)
    db_rel -= torch.repeat_interleave(db_group, degree_b)
    intm_db = (
        row_ptr.long().index_select(
            0, int_source.long().index_select(0, db_interaction)
        )
        + db_rel
    )
    intm_db_ab = torch.repeat_interleave(interaction, degree_b)
    return intm_ca, intm_db, intm_ca_ab, intm_db_ab


def _quadruplet_indices(context):
    row_ptr, source, int_source, int_target = (
        context.row_ptr,
        context.source,
        context.int_source,
        context.int_target,
    )
    interaction, degree_a, degree_b = (
        context.interaction,
        context.degree_a,
        context.degree_b,
    )
    pair_count = degree_a * degree_b
    quad_interaction = torch.repeat_interleave(interaction, pair_count)
    pair_group = torch.cumsum(pair_count, dim=0) - pair_count
    pair_rel = torch.arange(quad_interaction.numel(), device=row_ptr.device)
    pair_rel -= torch.repeat_interleave(pair_group, pair_count)
    ca_rel = pair_rel // degree_b.index_select(0, quad_interaction)
    db_rel = pair_rel - ca_rel * degree_b.index_select(0, quad_interaction)
    edge_ca = (
        row_ptr.long().index_select(
            0, int_target.long().index_select(0, quad_interaction)
        )
        + ca_rel
    )
    edge_db = (
        row_ptr.long().index_select(
            0, int_source.long().index_select(0, quad_interaction)
        )
        + db_rel
    )
    valid = _quadruplet_mask(context, edge_ca, edge_db, quad_interaction)
    edge_ca, edge_db = edge_ca[valid], edge_db[valid]
    ca_map = (torch.cumsum(degree_a, 0) - degree_a).index_select(
        0, quad_interaction[valid]
    ) + ca_rel[valid]
    db_map = (torch.cumsum(degree_b, 0) - degree_b).index_select(
        0, quad_interaction[valid]
    ) + db_rel[valid]
    order = torch.argsort(edge_ca, stable=True)
    edge_ca, edge_db = edge_ca[order], edge_db[order]
    ca_map, db_map = ca_map[order], db_map[order]
    counts = torch.bincount(edge_ca, minlength=source.numel())
    starts = torch.cumsum(counts, 0) - counts
    kidx = torch.arange(
        edge_ca.numel(), device=edge_ca.device
    ) - torch.repeat_interleave(starts, counts)
    return edge_ca, edge_db, ca_map, db_map, kidx


def _quadruplet_mask(context, edge_ca, edge_db, interactions):
    source, int_source, int_target = (
        context.source,
        context.int_source,
        context.int_target,
    )
    atom_c = source.long().index_select(0, edge_ca)
    atom_d = source.long().index_select(0, edge_db)
    atom_a = int_target.long().index_select(0, interactions)
    atom_b = int_source.long().index_select(0, interactions)
    return (atom_c != atom_b) & (atom_a != atom_d) & (atom_c != atom_d)


def timed(fn, warmup=1, repeat=3):
    for _ in range(warmup):
        result = fn()
    torch.npu.synchronize()
    values = []
    for _ in range(repeat):
        torch.npu.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.npu.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(values), result


def _graph_inputs(batch):
    pos = batch.pos.float()
    source, target = radius_edges(pos, batch.batch, 5.0)
    int_source, int_target = radius_edges(pos, batch.batch, 10.0)
    row_ptr = csr(target, batch.num_nodes)
    id3_reduce, id3_expand, kidx3 = triplets(row_ptr, source, target)
    row_npu, source_npu, target_npu, int_source_npu, int_target_npu = (
        value.npu() for value in (row_ptr, source, target, int_source, int_target)
    )
    topo = native_quadruplets(row_npu, source_npu, int_source_npu, int_target_npu)
    common = {
        "Z": batch.z.to(torch.int64).npu(),
        "R": pos.npu(),
        "id_a": target_npu,
        "id_c": source_npu,
        "id_undir": torch.arange(source.numel(), dtype=torch.int32).npu(),
        "id_swap": reverse_edges(source, target).npu(),
        "id3_expand_ba": id3_expand.npu(),
        "id3_reduce_ca": id3_reduce.npu(),
        "Kidx3": kidx3.npu(),
        "batch_seg": batch.batch.to(torch.int32).npu(),
        "id4_int_b": int_source_npu,
        "id4_int_a": int_target_npu,
    }
    capacities = (int(topo[0].numel()), max(int(topo[5].numel()), int(topo[6].numel())))
    lengths = (int(topo[5].numel()), int(topo[6].numel()))
    return _make_graph_inputs(
        (
            source,
            int_source,
            row_npu,
            source_npu,
            int_source_npu,
            int_target_npu,
            common,
            topo,
            capacities,
            lengths,
        )
    )


def _run_graph(model, torch_binding, dataset, graph_count):
    batch = Batch.from_data_list([dataset[2048 + i] for i in range(graph_count)])
    graph = _graph_inputs(batch)
    source, int_source = graph.source, graph.int_source
    row_npu, source_npu = graph.row_npu, graph.source_npu
    int_source_npu, int_target_npu = graph.int_source_npu, graph.int_target_npu
    common, topo = graph.common, graph.topology
    capacities, lengths = graph.capacities, graph.lengths
    native_inputs = _populate_inputs(common, topo)
    custom_values = torch_binding.gemnet_quadruplet_enumerate_fused(
        row_npu,
        source_npu,
        int_source_npu,
        int_target_npu,
        capacities[0],
        capacities[1],
    )
    custom_inputs = _populate_custom_inputs(common, custom_values, capacities, lengths)
    native_ms, native_out = timed(
        partial(_run_model, model=model, inputs=native_inputs)
    )
    custom_ms, custom_out = timed(
        partial(_run_model, model=model, inputs=custom_inputs)
    )
    max_diff = max(
        float((a - b).abs().max().cpu()) for a, b in zip(native_out, custom_out)
    )
    return {
        "graphs": graph_count,
        "nodes": int(batch.num_nodes),
        "embedding_edges": int(source.numel()),
        "interaction_edges": int(int_source.numel()),
        "quadruplets": capacities[0],
        "native_e2e_ms": native_ms,
        "custom_e2e_ms": custom_ms,
        "speedup": native_ms / custom_ms,
        "max_abs_diff": max_diff,
    }


def build_gemnet_model(model_type, scale_file):
    return (
        model_type(
            num_spherical=3,
            num_radial=3,
            num_blocks=1,
            emb_size_atom=16,
            emb_size_edge=16,
            emb_size_trip=8,
            emb_size_quad=8,
            emb_size_rbf=8,
            emb_size_cbf=8,
            emb_size_sbf=8,
            emb_size_bil_quad=8,
            emb_size_bil_trip=8,
            num_before_skip=1,
            num_after_skip=1,
            num_concat=1,
            num_atom=2,
            triplets_only=False,
            num_targets=1,
            direct_forces=True,
            cutoff=5.0,
            int_cutoff=10.0,
            extensive=True,
            scale_file=str(scale_file),
        )
        .eval()
        .npu()
    )


def _load_runtime(args):
    scale_file = args.output.with_suffix(".scales.json")
    scale_file.write_text("{}\n", encoding="utf-8")
    torch.npu.set_device(0)
    sys.path.insert(0, str(args.official_source))
    import types

    from torch_geometric.utils import scatter

    torch_scatter = types.ModuleType("torch_scatter")
    torch_scatter.scatter = scatter
    sys.modules["torch_scatter"] = torch_scatter
    from gemnet.model.gemnet import GemNet

    sys.path.insert(0, str(args.build_dir.parent.parent / "unused"))
    sys.path.insert(
        0, str(args.build_dir.parent / "sorted_kidx_source" / "integration")
    )
    import torch_binding

    torch_binding.configure(args.build_dir)
    torch.manual_seed(20260817)
    model = build_gemnet_model(GemNet, scale_file)
    return model, torch_binding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--graphs", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model, torch_binding = _load_runtime(args)
    dataset = QM9(str(args.dataset_root))
    results = [
        _run_graph(model, torch_binding, dataset, count) for count in args.graphs
    ]
    payload = {
        "candidate": "GemNet official model quadruplet topology",
        "random_weights": True,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("%s", json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
