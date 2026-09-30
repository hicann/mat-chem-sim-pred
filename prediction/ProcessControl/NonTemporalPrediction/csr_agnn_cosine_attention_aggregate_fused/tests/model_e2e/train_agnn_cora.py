#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Train the maintained two-layer PyG AGNN example on Cora."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, input_channels: int, output_channels: int, agnn_conv):
        super().__init__()
        self.lin1 = torch.nn.Linear(input_channels, 16)
        self.prop1 = agnn_conv(requires_grad=False)
        self.prop2 = agnn_conv(requires_grad=True)
        self.lin2 = torch.nn.Linear(16, output_channels)

    def forward(self, features, edge_index):
        features = F.relu(self.lin1(features))
        features = self.prop1(features, edge_index)
        features = self.prop2(features, edge_index)
        return self.lin2(features)


def train(model, optimizer, data, epochs):
    best_validation = -1.0
    best_test = 0.0
    best_state = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            predictions = logits.argmax(-1)
            validation = float(
                (predictions[data.val_mask] == data.y[data.val_mask]).float().mean()
            )
            test = float(
                (predictions[data.test_mask] == data.y[data.test_mask]).float().mean()
            )
        if validation > best_validation:
            best_validation = validation
            best_test = test
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    return best_validation, best_test, best_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()

    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import AGNNConv
    from torch_geometric.transforms import NormalizeFeatures

    torch.manual_seed(20260802)
    dataset = Planetoid(str(args.dataset_root), "Cora", transform=NormalizeFeatures())
    data = dataset[0]
    model = Net(data.num_features, dataset.num_classes, AGNNConv)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_validation, best_test, best_state = train(model, optimizer, data, args.epochs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "epochs": args.epochs,
            "seed": 20260802,
            "validation_accuracy": best_validation,
            "test_accuracy": best_test,
        },
        args.output,
    )
    logging.getLogger(__name__).info(
        "saved=%s validation=%.4f test=%.4f", args.output, best_validation, best_test
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
