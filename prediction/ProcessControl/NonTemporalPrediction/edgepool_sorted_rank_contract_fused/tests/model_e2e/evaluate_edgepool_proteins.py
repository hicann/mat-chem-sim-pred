#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Evaluate CPU, resident NPU, and custom NPU on the full PROTEINS test split."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

import torch
import torch_npu
from torch.utils.data import Subset

NON_TEMPORAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NON_TEMPORAL_ROOT))
benchmark_module = importlib.import_module(
    "benchmark_edgepool_sorted_rank_contract_e2e"
)
EdgePoolNet = benchmark_module.EdgePoolNet
configure = benchmark_module.configure


def evaluate(loader, cpu_model, resident_model, custom_model, device):
    correct_cpu = 0
    correct_resident = 0
    correct_custom = 0
    cpu_custom_agreement = 0
    resident_custom_agreement = 0
    graphs = 0
    cpu_custom_max_error = 0.0
    resident_custom_max_error = 0.0
    with torch.no_grad():
        for batch_cpu in loader:
            cpu_output = cpu_model(batch_cpu)
            batch_npu = batch_cpu.clone().to(device)
            resident_output = resident_model(batch_npu)
            custom_output = custom_model(batch_npu)
            torch_npu.npu.synchronize()
            resident_cpu = resident_output.cpu()
            custom_cpu = custom_output.cpu()
            cpu_prediction = cpu_output.argmax(-1)
            resident_prediction = resident_cpu.argmax(-1)
            custom_prediction = custom_cpu.argmax(-1)
            correct_cpu += int((cpu_prediction == batch_cpu.y).sum())
            correct_resident += int((resident_prediction == batch_cpu.y).sum())
            correct_custom += int((custom_prediction == batch_cpu.y).sum())
            cpu_custom_agreement += int((cpu_prediction == custom_prediction).sum())
            resident_custom_agreement += int(
                (resident_prediction == custom_prediction).sum()
            )
            graphs += batch_cpu.num_graphs
            cpu_custom_max_error = max(
                cpu_custom_max_error,
                float((cpu_output - custom_cpu).abs().max()),
            )
            resident_custom_max_error = max(
                resident_custom_max_error,
                float((resident_cpu - custom_cpu).abs().max()),
            )

    return {
        "graphs": graphs,
        "cpu_accuracy": correct_cpu / graphs,
        "resident_npu_accuracy": correct_resident / graphs,
        "custom_npu_accuracy": correct_custom / graphs,
        "cpu_custom_prediction_agreement": cpu_custom_agreement / graphs,
        "resident_custom_prediction_agreement": (resident_custom_agreement / graphs),
        "cpu_custom_logits_max_abs_diff": cpu_custom_max_error,
        "resident_custom_logits_max_abs_diff": resident_custom_max_error,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--build-name", default="build_clean")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    return parser.parse_args()


def main():
    args = parse_args()

    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = TUDataset(str(args.dataset_root), "PROTEINS")
    test_dataset = Subset(dataset, checkpoint["test_indices"])
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    cpu_model = EdgePoolNet(
        dataset.num_features, dataset.num_classes, path="native"
    ).eval()
    cpu_model.load_state_dict(checkpoint["state_dict"])

    device = torch.device(args.device)
    torch_npu.npu.set_device(device)
    configure(args.operator_root / args.build_name)
    resident_model = (
        EdgePoolNet(dataset.num_features, dataset.num_classes, path="resident")
        .to(device)
        .eval()
    )
    custom_model = (
        EdgePoolNet(dataset.num_features, dataset.num_classes, path="custom")
        .to(device)
        .eval()
    )
    resident_model.load_state_dict(checkpoint["state_dict"])
    custom_model.load_state_dict(checkpoint["state_dict"])

    result = evaluate(loader, cpu_model, resident_model, custom_model, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logging.getLogger(__name__).info(json.dumps(result, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
