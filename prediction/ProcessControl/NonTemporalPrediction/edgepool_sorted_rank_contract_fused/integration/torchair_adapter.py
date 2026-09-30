# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""TorchAir adapter for fixed-capacity EdgePool contraction outputs."""

from __future__ import annotations

__all__ = ["register_torchair_operator"]

from typing import Any

import torch

from .torch_binding import edgepool_sorted_rank_contract_fused_static

_LIBRARIES: list[torch.library.Library] = []
_REGISTERED = False


def register_torchair_operator() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    namespace = "cann_edgepool_ge"
    definition = torch.library.Library(namespace, "DEF")
    definition.define(
        "edgepool_sorted_rank_contract_fused(Tensor order, Tensor edge_index, "
        "Tensor score, int nodes) -> (Tensor, Tensor, Tensor, Tensor)"
    )
    npu_impl = torch.library.Library(namespace, "IMPL", "PrivateUse1")
    npu_impl.impl(
        "edgepool_sorted_rank_contract_fused",
        edgepool_sorted_rank_contract_fused_static,
    )
    meta_impl = torch.library.Library(namespace, "IMPL", "Meta")

    def meta(order, edge_index, score, nodes):
        del order, edge_index
        return (
            torch.empty(nodes, dtype=torch.int32, device="meta"),
            torch.empty(nodes, dtype=torch.int32, device="meta"),
            torch.empty(nodes, dtype=score.dtype, device="meta"),
            torch.empty(2, dtype=torch.int32, device="meta"),
        )

    meta_impl.impl("edgepool_sorted_rank_contract_fused", meta)
    _LIBRARIES.extend((definition, npu_impl, meta_impl))

    try:
        from torch_npu.dynamo.torchair import register_fx_node_ge_converter
    except ImportError:
        from torchair._ge_concrete_graph.fx2ge_converter import (
            register_fx_node_ge_converter,
        )
    from torchair import ge

    @register_fx_node_ge_converter(
        torch.ops.cann_edgepool_ge.edgepool_sorted_rank_contract_fused.default
    )
    def convert(order: Any, edge_index: Any, score: Any, nodes: Any, meta_outputs=None):
        del meta_outputs
        return ge.custom_op(
            "EdgepoolSortedRankContractFused",
            inputs={"order": order, "edge_index": edge_index, "score": score},
            outputs=["cluster", "selected_edge", "cluster_score", "counts"],
            attrs={"nodes": ge.attr.Int(int(nodes))},
        )

    _REGISTERED = True
