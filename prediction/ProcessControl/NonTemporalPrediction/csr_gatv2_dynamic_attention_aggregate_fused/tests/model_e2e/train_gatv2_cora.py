#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Train a two-layer maintained PyG GATv2Conv model on Cora."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


SHARED_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SHARED_ROOT))
COMMON = importlib.import_module("attention_model_e2e_common")


class Net(torch.nn.Module):
    def __init__(self, input_channels, output_channels, gatv2_conv):
        super().__init__()
        self.conv1 = gatv2_conv(input_channels, 8, heads=8, dropout=0.0)
        self.conv2 = gatv2_conv(64, output_channels, heads=1, concat=False)

    def forward(self, features, edge_index):
        features = F.elu(self.conv1(features, edge_index))
        return self.conv2(features, edge_index)


def main():
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.transforms import NormalizeFeatures

    COMMON.configure_logging()
    args = COMMON.training_arguments("Train GATv2Conv on Cora")
    torch.manual_seed(20260802)
    dataset = Planetoid(str(args.dataset_root), "Cora", transform=NormalizeFeatures())
    data = dataset[0]
    model = Net(data.num_features, dataset.num_classes, GATv2Conv)
    result = COMMON.train_cora_model(model, data, args.epochs)
    COMMON.save_training_result(result, args.output, args.epochs)


if __name__ == "__main__":
    main()
