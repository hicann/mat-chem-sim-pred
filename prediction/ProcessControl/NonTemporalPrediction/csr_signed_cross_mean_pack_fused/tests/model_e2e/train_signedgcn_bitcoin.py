#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Train maintained PyG SignedGCN on the real Bitcoin-OTC graph."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[3]))
_common = importlib.import_module("graph_message_model_e2e_common")
LOGGER, configure_logging = _common.LOGGER, _common.configure_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    return parser.parse_args()


def _load_edges(dataset):
    positive, negative = [], []
    for data in dataset:
        positive.append(data.edge_index[:, data.edge_attr > 0])
        negative.append(data.edge_index[:, data.edge_attr < 0])
    return torch.cat(positive, dim=1), torch.cat(negative, dim=1)


def _train(model, features, edges, epochs):
    train_positive, train_negative, test_positive, test_negative = edges
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best = {"auc": -1.0, "f1": 0.0, "state": None}
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(features, train_positive, train_negative)
        model.loss(embeddings, train_positive, train_negative).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            embeddings = model(features, train_positive, train_negative)
        auc, f1 = model.test(embeddings, test_positive, test_negative)
        if auc > best["auc"]:
            best = {
                "auc": auc,
                "f1": f1,
                "state": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
    return best


def _checkpoint(model, positive, negative, epochs):
    train_positive, test_positive = model.split_edges(positive)
    train_negative, test_negative = model.split_edges(negative)
    features = model.create_spectral_features(train_positive, train_negative)
    edges = (train_positive, train_negative, test_positive, test_negative)
    best = _train(model, features, edges, epochs)
    return {
        "state_dict": best["state"],
        "features": features,
        "train_positive": train_positive,
        "train_negative": train_negative,
        "test_positive": test_positive,
        "test_negative": test_negative,
        "epochs": epochs,
        "auc": float(best["auc"]),
        "f1": float(best["f1"]),
        "source_commit": "pytorch_geometric@003c3cd8a10520567ceaeda619f0315e30ec2f66",
    }


def main():
    args = parse_args()
    from torch_geometric.datasets import BitcoinOTC
    from torch_geometric.nn import SignedGCN

    torch.manual_seed(20260802)
    positive, negative = _load_edges(
        BitcoinOTC(str(args.dataset_root), edge_window_size=1)
    )
    payload = _checkpoint(
        SignedGCN(64, 64, num_layers=2, lamb=5), positive, negative, args.epochs
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    configure_logging()
    LOGGER.info(
        "auc=%.4f f1=%.4f nodes=%d positive=%d negative=%d",
        payload["auc"],
        payload["f1"],
        payload["features"].size(0),
        payload["train_positive"].size(1),
        payload["train_negative"].size(1),
    )


if __name__ == "__main__":
    main()
