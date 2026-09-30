# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from integration.torch_binding import (
    configure,
    edgepool_sorted_rank_contract_fused,
)
from reference import reference


def _require_npu():
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("EdgePool binding requires an Ascend torch_npu runtime")
    importlib.import_module("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("EdgePool binding requires an available Ascend NPU")


def test_contract_matches_python_reference():
    _require_npu()
    configure(Path(__file__).parents[1] / "build_clean")
    order = torch.tensor([2, 0, 1], dtype=torch.int32, device="npu")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int32, device="npu")
    score = torch.tensor([0.2, 0.3, 0.4], device="npu")
    cluster, selected, cluster_score, selected_count, cluster_count = (
        edgepool_sorted_rank_contract_fused(order, edge_index, score, 4)
    )
    torch.npu.synchronize()
    assert cluster.cpu().tolist() == [1, 1, 0, 0]
    assert selected.cpu().tolist()[:selected_count] == [2, 0]
    assert torch.allclose(cluster_score.cpu()[:cluster_count], torch.tensor([0.4, 0.2]))
    assert (selected_count, cluster_count) == (2, 2)


def test_random_graph_matches_numpy_oracle():
    _require_npu()
    configure(Path(__file__).parents[1] / "build_clean")
    generator = torch.Generator().manual_seed(20260815)
    nodes, edges = 32, 96
    edge_index = torch.randint(nodes, (2, edges), generator=generator)
    score = torch.randn(edges, generator=generator).abs().to("npu")
    order = torch.argsort(score, descending=True).to(torch.int32)
    cluster, selected, cluster_score, selected_count, cluster_count = (
        edgepool_sorted_rank_contract_fused(
            order, edge_index.int().to("npu"), score, nodes
        )
    )
    torch.npu.synchronize()
    expected_cluster, expected_selected, expected_score, expected_counts = reference(
        order.cpu().numpy(),
        edge_index.numpy().astype(np.int32),
        score.cpu().numpy(),
        nodes,
    )
    assert cluster.cpu().numpy().tolist() == expected_cluster.tolist()
    assert selected.cpu().numpy()[:selected_count].tolist() == (
        expected_selected[: expected_counts[0]].tolist()
    )
    assert torch.allclose(
        cluster_score.cpu()[:cluster_count],
        torch.from_numpy(expected_score[: expected_counts[1]]),
    )
    assert (selected_count, cluster_count) == tuple(expected_counts.tolist())
