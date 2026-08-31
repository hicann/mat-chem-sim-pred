#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Train and benchmark a two-layer K=3 ChebNet on Cora."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from benchmark_component import CustomChebBasis, native_basis, normalized_csr
from torch_geometric.nn import ChebConv

sys.path.insert(0, str(Path(__file__).parents[3]))
from spectral_benchmark_common import E2ESpec, e2e_parser, run_e2e


class ChebNet(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ChebConv(in_channels, hidden, K=3)
        self.conv2 = ChebConv(hidden, out_channels, K=3)

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv1(features, edge_index))
        return F.log_softmax(self.conv2(hidden, edge_index), dim=1)


def apply_basis(conv: ChebConv, basis: torch.Tensor) -> torch.Tensor:
    output = conv.lins[0](basis[0])
    output += conv.lins[1](basis[1])
    output += conv.lins[2](basis[2])
    return output + conv.bias if conv.bias is not None else output


def staged_forward(
    model: ChebNet, features: torch.Tensor, basis_function
) -> torch.Tensor:
    hidden = F.relu(apply_basis(model.conv1, basis_function(features)))
    output = apply_basis(model.conv2, basis_function(hidden))
    return F.log_softmax(output, dim=1)


def main() -> None:
    args = e2e_parser().parse_args()
    spec = E2ESpec(
        ChebNet,
        CustomChebBasis,
        normalized_csr,
        native_basis,
        staged_forward,
        None,
        "two-layer PyG ChebNet, K=3",
    )
    run_e2e(args, spec)


if __name__ == "__main__":
    main()
