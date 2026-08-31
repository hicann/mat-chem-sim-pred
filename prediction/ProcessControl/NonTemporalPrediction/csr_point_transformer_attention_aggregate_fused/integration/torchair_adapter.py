# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""TorchAir/GE adapter for the fixed-shape PointTransformer fused operator."""

from __future__ import annotations

from typing import Any

import torch

from .torch_binding import csr_point_transformer_attention_aggregate_fused

_LIBRARIES: list[torch.library.Library] = []
_REGISTERED = False


def register_torchair_operator() -> None:
    """Register eager, fake, and TorchAir converter implementations once."""

    global _REGISTERED
    if _REGISTERED:
        return

    namespace = "cann_point_transformer_ge"
    definition = torch.library.Library(namespace, "DEF")
    definition.define(
        "csr_point_transformer_attention_aggregate_fused(Tensor row_ptr, "
        "Tensor source_index, Tensor alpha_source, Tensor alpha_target, "
        "Tensor value, Tensor delta, int max_segment_size) -> Tensor"
    )
    npu_impl = torch.library.Library(namespace, "IMPL", "PrivateUse1")
    npu_impl.impl(
        "csr_point_transformer_attention_aggregate_fused",
        csr_point_transformer_attention_aggregate_fused,
    )
    meta_impl = torch.library.Library(namespace, "IMPL", "Meta")
    meta_impl.impl(
        "csr_point_transformer_attention_aggregate_fused",
        lambda row_ptr, source_index, alpha_source, alpha_target, value, delta, max_segment_size: (
            torch.empty_like(value)
        ),
    )
    _LIBRARIES.extend((definition, npu_impl, meta_impl))

    try:
        from torch_npu.dynamo.torchair import register_fx_node_ge_converter
    except ImportError:
        from torchair._ge_concrete_graph.fx2ge_converter import (
            register_fx_node_ge_converter,
        )
    from torchair import ge

    attr = getattr(ge, "attr", None)

    @register_fx_node_ge_converter(
        torch.ops.cann_point_transformer_ge.csr_point_transformer_attention_aggregate_fused.default
    )
    def convert_point_transformer(
        *inputs: Any,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        (
            row_ptr,
            source_index,
            alpha_source,
            alpha_target,
            value,
            delta,
            max_segment_size,
        ) = inputs
        attrs = {"max_segment_size": int(max_segment_size)}
        if attr is not None:
            attrs = {"max_segment_size": attr.Int(int(max_segment_size))}
        return ge.custom_op(
            "CsrPointTransformerAttentionAggregateFused",
            inputs={
                "row_ptr": row_ptr,
                "source_index": source_index,
                "alpha_source": alpha_source,
                "alpha_target": alpha_target,
                "value": value,
                "delta": delta,
            },
            outputs=["output"],
            attrs=attrs,
        )

    _REGISTERED = True


def point_transformer_graph_op(*inputs: Any) -> torch.Tensor:
    """Call the graph-facing operator; eager NPU uses the direct binding."""

    (
        row_ptr,
        source_index,
        alpha_source,
        alpha_target,
        value,
        delta,
        max_segment_size,
    ) = inputs
    register_torchair_operator()
    return torch.ops.cann_point_transformer_ge.csr_point_transformer_attention_aggregate_fused(
        row_ptr,
        source_index,
        alpha_source,
        alpha_target,
        value,
        delta,
        max_segment_size,
    )
