#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Compare K=3 Chebyshev basis generation on real Cora graphs."""

from __future__ import annotations

__all__ = [
    "CustomChebBasis",
    "disjoint_copies",
    "native_basis",
    "normalized_csr",
    "timed",
]

import sys
from pathlib import Path

import torch
from torch_geometric.nn import ChebConv

sys.path.insert(0, str(Path(__file__).parents[3]))
import spectral_benchmark_common as benchmark_common

disjoint_copies = benchmark_common.disjoint_copies
timed = benchmark_common.timed


class CustomChebBasis(benchmark_common.CustomBasisOperator):
    def __init__(self, build: Path, device: torch.device) -> None:
        super().__init__(
            build,
            device,
            benchmark_common.BasisOperatorSpec(
                "libcsr_chebyshev_basis_k3_fused_host.so",
                "aclnnCsrChebyshevBasisK3FusedGetWorkspaceSize",
                "aclnnCsrChebyshevBasisK3Fused",
                3,
            ),
        )


def normalized_csr(
    edge_index: torch.Tensor, nodes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    helper = ChebConv(1, 1, K=3)
    normalized_edges, norm = helper.__norm__(
        edge_index, nodes, None, "sym", None, dtype=torch.float32, batch=None
    )
    return benchmark_common.sorted_csr(normalized_edges, norm, nodes)


def native_basis(
    features: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
    norm: torch.Tensor,
) -> torch.Tensor:
    first = torch.zeros_like(features)
    first.index_add_(0, target, features[source] * norm.view(-1, 1))
    second = torch.zeros_like(features)
    second.index_add_(0, target, first[source] * norm.view(-1, 1))
    return torch.stack([features, first, 2.0 * second - features], dim=0)


def main() -> None:
    args = benchmark_common.component_parser().parse_args()
    benchmark_common.run_component_cli(
        args,
        benchmark_common.ComponentCliSpec(
            CustomChebBasis,
            normalized_csr,
            native_basis,
            None,
            "PyG ChebConv K=3 Chebyshev basis generation",
        ),
    )


if __name__ == "__main__":
    main()
