#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Train the EdgePooling classifier used by the value-gate benchmark."""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

NON_TEMPORAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NON_TEMPORAL_ROOT))
EdgePoolNet = importlib.import_module(
    "benchmark_edgepool_sorted_rank_contract_e2e"
).EdgePoolNet


def accuracy(model, loader):
    model.eval()
    correct = 0
    graphs = 0
    with torch.no_grad():
        for batch in loader:
            prediction = model(batch).argmax(-1)
            correct += int((prediction == batch.y).sum())
            graphs += batch.num_graphs
    return correct / graphs


def train(model, optimizer, train_loader, test_loader, epochs):
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0 or epoch == 1:
            current_accuracy = accuracy(model, test_loader)
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_epoch = epoch
                best_state = {
                    name: value.detach().clone()
                    for name, value in model.state_dict().items()
                }

    return best_accuracy, best_epoch, best_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def main():
    args = parse_args()

    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    dataset = TUDataset(str(args.dataset_root), "PROTEINS")
    permutation = torch.randperm(
        len(dataset), generator=torch.Generator().manual_seed(args.seed)
    )
    split = int(len(dataset) * 0.8)
    train_indices = permutation[:split].tolist()
    test_indices = permutation[split:].tolist()
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = EdgePoolNet(dataset.num_features, dataset.num_classes, path="native")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_accuracy, best_epoch, best_state = train(
        model, optimizer, train_loader, test_loader, args.epochs
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "seed": args.seed,
            "train_indices": train_indices,
            "test_indices": test_indices,
            "test_accuracy": best_accuracy,
            "epoch": best_epoch,
            "model": ("GraphConv(F,32)-EdgePooling-GraphConv(32,32)-mean-linear"),
        },
        args.output,
    )
    logging.getLogger(__name__).info(
        f"best epoch={best_epoch} test_accuracy={best_accuracy:.6f}"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
