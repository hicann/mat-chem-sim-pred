# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU regression coverage for shared GENConv benchmark semantics."""

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
common = importlib.import_module("genconv_common")
gate = importlib.import_module("benchmark_genconv_value_gate")
deep = importlib.import_module("benchmark_deepergcn_cora_e2e")


@pytest.fixture
def graph():
    # Every destination has a self-loop; degrees cross power-of-two buckets.
    features = torch.tensor([[1.0, -2.0], [3.0, 4.0], [-1.0, 2.0]])
    edges = torch.tensor([[0, 1, 1, 2, 0, 1], [0, 0, 1, 2, 2, 2]])
    return features, edges


@pytest.mark.parametrize("temperature", [-2.0, 0.0, 1.5])
def test_shared_baselines_match_featurewise_reference(graph, temperature):
    features, edges = graph
    row_ptr, source, counts = common.make_csr(edges, features.size(0))
    padded_row, padded_source, padded, mask = common.make_padded_csr(
        edges, features.size(0)
    )
    torch.testing.assert_close(padded_row, row_ptr)
    torch.testing.assert_close(padded_source, source)
    expected = torch.zeros_like(features)
    for target in range(features.size(0)):
        messages = features[edges[0, edges[1] == target]].relu() + 1.0e-7
        expected[target] = (
            messages * torch.softmax(messages * temperature, dim=0)
        ).sum(dim=0)
    actual = common.native_dense(features, padded, mask, temperature)
    torch.testing.assert_close(actual, expected)
    for aggregate_type in (gate.GlobalPaddedAggregate, gate.DegreeBucketAggregate):
        aggregate = aggregate_type(row_ptr, source, counts, torch.device("cpu"))
        torch.testing.assert_close(aggregate(features, temperature), expected)


def test_staged_model_matches_official_pyg(graph):
    features, edges = graph
    torch.manual_seed(19)
    model = common.DeeperGCN(2, 8, 3, 3).eval()
    temperatures = [-1.25, 0.0, 2.0]
    for layer, temperature in zip(model.layers, temperatures):
        layer.conv.aggr_module.t.data.fill_(temperature)
    _, _, padded, mask = common.make_padded_csr(edges, features.size(0))

    def aggregate(values, temperature):
        return common.native_dense(values, padded, mask, temperature)

    with torch.no_grad():
        expected = model(features, edges)
        actual = common.staged_forward(model, features, aggregate, temperatures)
    torch.testing.assert_close(actual, expected)


def test_disjoint_graphs_preserve_output(graph):
    features, edges = graph
    repeated, copied_edges = common.disjoint_copies(features, edges, 3)
    _, _, padded, mask = common.make_padded_csr(edges, features.size(0))
    _, _, copied_padded, copied_mask = common.make_padded_csr(
        copied_edges, repeated.size(0)
    )
    expected = common.native_dense(features, padded, mask, 1.0).repeat(3, 1)
    actual = common.native_dense(repeated, copied_padded, copied_mask, 1.0)
    torch.testing.assert_close(actual, expected)


def test_validation_callbacks_keep_graph_and_counter_independent(monkeypatch):
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False
    )
    calls = []

    def custom(row_ptr, source, features, temperature):
        calls.append((row_ptr.item(), source.item(), temperature))
        return features

    callbacks = [
        deep.CheckedAggregate(
            common.BoundCustomAggregate(
                custom, torch.tensor(index), torch.tensor(index + 1)
            ),
            index,
        )
        for index in range(3)
    ]
    for callback in reversed(callbacks):
        callback(torch.ones(2, 2).t(), 1.0)
    callbacks[0](torch.ones(2, 2), 2.0)
    assert calls == [(2, 3, 1.0), (1, 2, 1.0), (0, 1, 1.0), (0, 1, 2.0)]
    assert [callback.calls for callback in callbacks] == [2, 1, 1]


@pytest.mark.parametrize("timing", [common.median_cpu_ms, common.median_npu_ms])
def test_timing_rejects_empty_measurements(timing):
    with pytest.raises(ValueError, match="iterations positive"):
        timing(lambda: torch.ones(1), 0, 0)


def test_missing_benchmark_result_is_explicit():
    with pytest.raises(RuntimeError, match="missing benchmark result: custom"):
        gate.required_result({}, "custom")
