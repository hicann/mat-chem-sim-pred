#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Train and benchmark the maintained two-layer K=3 TAGCN on Cora."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from benchmark_component import CustomTagcnBasis, native_basis, normalized_csr
from torch_geometric.nn import TAGConv

sys.path.insert(0, str(Path(__file__).parents[3]))
from spectral_benchmark_common import E2ESpec, e2e_parser, run_e2e


class TagNet(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = TAGConv(in_channels, hidden, K=3)
        self.conv2 = TAGConv(hidden, out_channels, K=3)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv1(features, edge_index))
        return F.log_softmax(self.conv2(hidden, edge_index), dim=1)


def apply_basis(conv: TAGConv, basis: torch.Tensor) -> torch.Tensor:
    output = conv.lins[0](basis[0])
    for index, linear in enumerate(conv.lins[1:], start=1):
        output += linear(basis[index])
    return output + conv.bias if conv.bias is not None else output


def staged_forward(
    model: TagNet, features: torch.Tensor, basis_function
) -> torch.Tensor:
    hidden = F.relu(apply_basis(model.conv1, basis_function(features)))
    output = apply_basis(model.conv2, basis_function(hidden))
    return F.log_softmax(output, dim=1)


def main() -> None:
    args = e2e_parser().parse_args()
    spec = E2ESpec(
        TagNet,
        CustomTagcnBasis,
        normalized_csr,
        native_basis,
        staged_forward,
        T.NormalizeFeatures(),
        "maintained PyG examples/tagcn.py two-layer TAGCN, K=3",
    )
    run_e2e(args, spec)


if __name__ == "__main__":
    main()
