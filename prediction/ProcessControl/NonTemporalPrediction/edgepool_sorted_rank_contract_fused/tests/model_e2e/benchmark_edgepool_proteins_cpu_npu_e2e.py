#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Run the same-host complete-model EdgePooling CPU/NPU value gate."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch_npu
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

NON_TEMPORAL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(NON_TEMPORAL_ROOT))
benchmark_module = importlib.import_module(
    "benchmark_edgepool_sorted_rank_contract_e2e"
)
EdgePoolNet = benchmark_module.EdgePoolNet
configure = benchmark_module.configure


def timed_cpu(call, warmup, repeat):
    result = None
    with torch.no_grad():
        for _ in range(warmup):
            result = call()
        samples = []
        for _ in range(repeat):
            begin = time.perf_counter()
            result = call()
            samples.append((time.perf_counter() - begin) * 1000.0)
    return float(statistics.median(samples)), result, samples


def timed_npu(call, warmup, repeat):
    result = None
    with torch.no_grad():
        for _ in range(warmup):
            result = call()
            torch_npu.npu.synchronize()
        samples = []
        for _ in range(repeat):
            torch_npu.npu.synchronize()
            begin = time.perf_counter()
            result = call()
            torch_npu.npu.synchronize()
            samples.append((time.perf_counter() - begin) * 1000.0)
    return float(statistics.median(samples)), result, samples


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_cpu(args, dataset, checkpoint, batch_cpu):
    cpu_model = EdgePoolNet(
        dataset.num_features, dataset.num_classes, path="native"
    ).eval()
    cpu_model.load_state_dict(checkpoint["state_dict"])

    cpu_sweep = []
    cpu_outputs = {}
    for thread_count in args.cpu_threads:
        torch.set_num_threads(thread_count)
        median_ms, output, samples = timed_cpu(
            lambda model=cpu_model, batch=batch_cpu: model(batch),
            args.warmup,
            args.repeat,
        )
        cpu_sweep.append(
            {
                "threads": thread_count,
                "median_ms": median_ms,
                "samples_ms": samples,
            }
        )
        cpu_outputs[thread_count] = output
    best_cpu = min(cpu_sweep, key=lambda item: item["median_ms"])
    cpu_output = cpu_outputs.get(best_cpu["threads"])

    if cpu_output is None:
        raise RuntimeError("CPU sweep produced no selected output")
    return cpu_sweep, best_cpu, cpu_output


def load_npu_models(dataset, checkpoint, device):
    models = {}
    for path in ("native", "resident", "custom"):
        model = (
            EdgePoolNet(dataset.num_features, dataset.num_classes, path=path)
            .to(device)
            .eval()
        )
        model.load_state_dict(checkpoint["state_dict"])
        models[path] = model

    return models


def run_npu(args, dataset, checkpoint, batch_cpu, device):
    batch_npu = batch_cpu.clone().to(device)
    models = load_npu_models(dataset, checkpoint, device)

    with torch.no_grad():
        encoded = F.relu(models["native"].encoder(batch_npu.x, batch_npu.edge_index))
        native_pool = models["native"].pool(
            encoded, batch_npu.edge_index, batch_npu.batch
        )
        custom_pool = models["custom"].custom_pool(
            encoded, batch_npu.edge_index, batch_npu.batch
        )

    native_ms, native_output, native_samples = timed_npu(
        lambda model=models["native"], batch=batch_npu: model(batch),
        args.warmup,
        args.repeat,
    )
    resident_ms, resident_output, resident_samples = timed_npu(
        lambda model=models["resident"], batch=batch_npu: model(batch),
        args.warmup,
        args.repeat,
    )
    custom_ms, custom_output, custom_samples = timed_npu(
        lambda model=models["custom"], batch=batch_npu: model(batch),
        args.warmup,
        args.repeat,
    )

    def custom_host_to_host(model=models["custom"], host_batch=batch_cpu, npu=device):
        inputs = SimpleNamespace(
            x=host_batch.x.to(npu),
            edge_index=host_batch.edge_index.to(npu),
            batch=host_batch.batch.to(npu),
        )
        return model(inputs).cpu()

    host_to_host_ms, host_to_host_output, host_to_host_samples = timed_npu(
        custom_host_to_host, args.warmup, args.repeat
    )
    result_tuple = (
        (native_ms, native_output, native_samples),
        (resident_ms, resident_output, resident_samples),
        (custom_ms, custom_output, custom_samples),
        (host_to_host_ms, host_to_host_output, host_to_host_samples),
        native_pool,
        custom_pool,
    )
    return result_tuple


def accuracy_metrics(cpu_output, npu_runs, batch_cpu):
    native_run, resident_run, custom_run, host_run, native_pool, custom_pool = npu_runs
    native_output, resident_output, custom_output = (
        native_run[1],
        resident_run[1],
        custom_run[1],
    )
    host_to_host_output = host_run[1]
    custom_cpu = custom_output.cpu()
    native_cluster = native_pool[3].cluster
    return {
        "cluster_exact": bool(torch.equal(native_cluster, custom_pool[3].long())),
        "features_max_abs_diff": float(
            (native_pool[0] - custom_pool[0]).abs().max().cpu()
        ),
        "edge_index_exact": bool(torch.equal(native_pool[1], custom_pool[1].long())),
        "batch_exact": bool(torch.equal(native_pool[2], custom_pool[2])),
        "cpu_custom_logits_max_abs_diff": float((cpu_output - custom_cpu).abs().max()),
        "native_custom_logits_max_abs_diff": float(
            (native_output - custom_output).abs().max().cpu()
        ),
        "resident_custom_logits_max_abs_diff": float(
            (resident_output - custom_output).abs().max().cpu()
        ),
        "host_to_host_custom_max_abs_diff": float(
            (host_to_host_output - custom_cpu).abs().max()
        ),
        "cpu_custom_prediction_agreement": float(
            (cpu_output.argmax(-1) == custom_cpu.argmax(-1)).float().mean()
        ),
        "cpu_accuracy": float((cpu_output.argmax(-1) == batch_cpu.y).float().mean()),
        "custom_accuracy": float((custom_cpu.argmax(-1) == batch_cpu.y).float().mean()),
    }


def summarize_batch(cpu_run, npu_runs, batch_cpu, batch_size):
    cpu_sweep, best_cpu, cpu_output = cpu_run
    native_run, resident_run, custom_run, host_run, native_pool, custom_pool = npu_runs
    native_ms, native_output, native_samples = native_run
    resident_ms, resident_output, resident_samples = resident_run
    custom_ms, custom_output, custom_samples = custom_run
    host_to_host_ms, host_to_host_output, host_to_host_samples = host_run
    strongest_non_custom = min(best_cpu["median_ms"], native_ms, resident_ms)
    result = {
        "batch_size": batch_size,
        "nodes": int(batch_cpu.num_nodes),
        "edges": int(batch_cpu.edge_index.size(1)),
        "graphs": int(batch_cpu.num_graphs),
        "cpu_thread_sweep": cpu_sweep,
        "best_cpu_threads": best_cpu["threads"],
        "best_cpu_model_ms": best_cpu["median_ms"],
        "native_npu_model_ms": native_ms,
        "resident_npu_model_ms": resident_ms,
        "custom_npu_resident_model_ms": custom_ms,
        "custom_npu_host_to_host_ms": host_to_host_ms,
        "strongest_non_custom_model_ms": strongest_non_custom,
        "resident_reduction_vs_strongest_percent": (
            (strongest_non_custom - custom_ms) / strongest_non_custom * 100.0
        ),
        "host_to_host_speedup_vs_cpu": (best_cpu["median_ms"] / host_to_host_ms),
        **accuracy_metrics(cpu_output, npu_runs, batch_cpu),
        "native_npu_samples_ms": native_samples,
        "resident_npu_samples_ms": resident_samples,
        "custom_npu_samples_ms": custom_samples,
        "custom_host_to_host_samples_ms": host_to_host_samples,
    }

    return result


def run_batch(args, context, batch_size):
    checkpoint, dataset, test_dataset, device = context
    batch_cpu = next(
        iter(DataLoader(test_dataset, batch_size=batch_size, shuffle=False))
    )
    cpu_run = run_cpu(args, dataset, checkpoint, batch_cpu)
    npu_runs = run_npu(args, dataset, checkpoint, batch_cpu, device)
    return summarize_batch(cpu_run, npu_runs, batch_cpu, batch_size)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--build-name", default="build_clean")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 128])
    parser.add_argument("--cpu-threads", nargs="+", type=int, default=[1, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    from torch_geometric.datasets import TUDataset

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    dataset = TUDataset(str(args.dataset_root), "PROTEINS")
    test_dataset = Subset(dataset, checkpoint["test_indices"])
    device = torch.device(args.device)
    torch_npu.npu.set_device(device)
    configure(args.operator_root / args.build_name)

    results = []
    for batch_size in args.batch_sizes:
        results.append(
            run_batch(args, (checkpoint, dataset, test_dataset, device), batch_size)
        )

    payload = {
        "operator": "EdgepoolSortedRankContractFused",
        "scope": "complete trained PyG EdgePooling PROTEINS classifier",
        "dataset": "TUDataset PROTEINS deterministic random 80/20 split",
        "checkpoint_sha256": digest(args.checkpoint),
        "checkpoint_test_accuracy": checkpoint["test_accuracy"],
        "checkpoint_epoch": checkpoint["epoch"],
        "host": "node202",
        "device": torch_npu.npu.get_device_name(device),
        "timing": "FP32 no-grad host wall clock median with synchronization",
        "host_to_host_boundary": (
            "host x/edge_index/batch -> H2D -> resident model -> D2H logits"
        ),
        "excluded_from_host_to_host": [
            "dataset loading",
            "checkpoint loading",
            "one-time model weight transfer",
        ],
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logging.getLogger(__name__).info(json.dumps(payload, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
