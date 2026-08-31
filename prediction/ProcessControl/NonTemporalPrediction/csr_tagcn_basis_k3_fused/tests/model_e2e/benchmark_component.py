#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Compare K=3 TAGCN basis generation on real Cora graphs."""

from __future__ import annotations

__all__ = [
    "CustomTagcnBasis",
    "disjoint_copies",
    "native_basis",
    "normalized_csr",
    "timed",
]

import sys
from pathlib import Path

import torch
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm

sys.path.insert(0, str(Path(__file__).parents[3]))
import spectral_benchmark_common as benchmark_common

disjoint_copies = benchmark_common.disjoint_copies
timed = benchmark_common.timed


class CustomTagcnBasis(benchmark_common.CustomBasisOperator):
    def __init__(self, build: Path, device: torch.device) -> None:
        super().__init__(
            build,
            device,
            benchmark_common.BasisOperatorSpec(
                "libcsr_tagcn_basis_k3_fused_host.so",
                "aclnnCsrTagcnBasisK3FusedGetWorkspaceSize",
                "aclnnCsrTagcnBasisK3Fused",
                4,
            ),
        )


def normalized_csr(
    edge_index: torch.Tensor, nodes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    normalized_edges, norm = gcn_norm(
        edge_index,
        num_nodes=nodes,
        improved=False,
        add_self_loops=False,
        flow="source_to_target",
        dtype=torch.float32,
    )
    return benchmark_common.sorted_csr(normalized_edges, norm, nodes)


def native_basis(
    features: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    norm: torch.Tensor,
) -> torch.Tensor:
    bases = [features]
    for _ in range(3):
        propagated = torch.zeros_like(features)
        propagated.index_add_(0, target, bases[-1][source] * norm.view(-1, 1))
        bases.append(propagated)
    return torch.stack(bases, dim=0)


def main() -> None:
    args = benchmark_common.component_parser().parse_args()
    benchmark_common.run_component_cli(
        args,
        benchmark_common.ComponentCliSpec(
            CustomTagcnBasis,
            normalized_csr,
            native_basis,
            T.NormalizeFeatures(),
            "PyG TAGConv K=3 three-hop basis generation",
        ),
    )


if __name__ == "__main__":
    main()
