#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Profile residual propagation kernels in the complete 64-layer GCNII."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch_geometric.transforms as T
from benchmark_gcn2_cora_e2e import (
    Gcn2Net,
    native_residual,
    normalized_layout,
    staged_forward,
)
from torch_geometric.datasets import Planetoid

sys.path.insert(0, str(Path(__file__).parents[3]))
from spectral_benchmark_common import disjoint_copies
from spectral_profile_common import (
    HotspotSpec,
    ProfileCase,
    ProfileSpec,
    load_model,
    profile_parser,
    run_profiles,
)

RESIDUAL_KERNEL_TYPES = {"Add", "Index", "InplaceIndexAdd", "Mul", "Muls", "ZerosLike"}


@dataclass
class GcnProfileContext:
    data: object
    model: Gcn2Net
    device: torch.device
    layers: int


def build_case(context: GcnProfileContext, copies: int) -> ProfileCase:
    features, edge_index = disjoint_copies(
        context.data.x, context.data.edge_index, copies
    )
    features = features.to(context.device)
    graph = normalized_layout(edge_index, features.size(0)).to(context.device)
    residual = partial(native_residual, graph=graph, alpha=0.1)
    complete_model = partial(staged_forward, context.model, features, residual)
    return ProfileCase(
        complete_model,
        int(features.size(0)),
        int(graph.source.numel()),
        {"layers": context.layers},
    )


def main() -> None:
    parser = profile_parser()
    parser.add_argument("--layers", type=int, default=64)
    args = parser.parse_args()
    dataset = Planetoid(
        str(args.data_root), name="Cora", transform=T.NormalizeFeatures()
    )
    device = torch.device("npu:0")
    model = load_model(
        Gcn2Net(dataset.num_features, 64, dataset.num_classes, layers=args.layers),
        args.checkpoint,
        device,
    )
    context = GcnProfileContext(dataset[0], model, device, args.layers)
    run_profiles(
        args,
        ProfileSpec(
            partial(build_case, context),
            "maintained PyG examples/gcn2_cora.py, 64-layer GCNII",
            "cora_gcn2_copies",
            HotspotSpec("residual", RESIDUAL_KERNEL_TYPES),
            warmup=3,
        ),
    )


if __name__ == "__main__":
    main()
