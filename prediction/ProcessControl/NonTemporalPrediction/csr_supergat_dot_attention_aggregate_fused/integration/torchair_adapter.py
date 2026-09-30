# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""TorchAir/GE adapter for the fixed-shape SuperGAT fused operator.

The direct binding remains the eager/ACLNN implementation.  This module adds
the graph-facing ``torch.library`` schema and maps it to the packaged CANN
operator through ``ge.custom_op``.  The adapter is intentionally separate so
loading it cannot change the existing eager dispatch path.
"""

from __future__ import annotations

__all__ = ["register_torchair_operator", "supergat_graph_op"]

from typing import Any

import torch

from .torch_binding import csr_supergat_dot_attention_aggregate_fused

_LIBRARIES: list[torch.library.Library] = []
_REGISTERED = False


def _register_converter():
    try:
        from torch_npu.dynamo.torchair import register_fx_node_ge_converter
    except ImportError:
        from torchair._ge_concrete_graph.fx2ge_converter import (
            register_fx_node_ge_converter,
        )
    register_fx_node_ge_converter(
        torch.ops.cann_supergat_ge.csr_supergat_dot_attention_aggregate_fused.default
    )(_convert_supergat)


def _convert_supergat(
    row_ptr: Any,
    source_index: Any,
    projected: Any,
    max_segment_size: Any,
    negative_slope: Any = 0.2,
    **kwargs: Any,
) -> Any:
    from torchair import ge

    del kwargs
    attr = getattr(ge, "attr", None)
    attributes = {
        "max_segment_size": int(max_segment_size),
        "negative_slope": float(negative_slope),
    }
    if attr is not None:
        attributes = {
            "max_segment_size": attr.Int(int(max_segment_size)),
            "negative_slope": attr.Float(float(negative_slope)),
        }
    return ge.custom_op(
        "CsrSuperGatDotAttentionAggregateFused",
        inputs={
            "row_ptr": row_ptr,
            "source_index": source_index,
            "projected": projected,
        },
        outputs=["output"],
        attrs=attributes,
    )


def register_torchair_operator() -> None:
    """Register eager, fake, and TorchAir converter implementations once."""

    global _REGISTERED
    if _REGISTERED:
        return

    namespace = "cann_supergat_ge"
    definition = torch.library.Library(namespace, "DEF")
    definition.define(
        "csr_supergat_dot_attention_aggregate_fused(Tensor row_ptr, "
        "Tensor source_index, Tensor projected, int max_segment_size, "
        "float negative_slope=0.2) -> Tensor"
    )
    npu_impl = torch.library.Library(namespace, "IMPL", "PrivateUse1")
    npu_impl.impl(
        "csr_supergat_dot_attention_aggregate_fused",
        csr_supergat_dot_attention_aggregate_fused,
    )
    meta_impl = torch.library.Library(namespace, "IMPL", "Meta")
    meta_impl.impl(
        "csr_supergat_dot_attention_aggregate_fused",
        lambda row_ptr, source_index, projected, max_segment_size, negative_slope=0.2: (
            torch.empty_like(projected)
        ),
    )
    _LIBRARIES.extend((definition, npu_impl, meta_impl))

    _register_converter()

    _REGISTERED = True


def supergat_graph_op(
    row_ptr: torch.Tensor,
    source_index: torch.Tensor,
    projected: torch.Tensor,
    max_segment_size: int,
    negative_slope: float = 0.2,
) -> torch.Tensor:
    """Call the graph-facing operator; eager NPU uses the direct binding."""

    register_torchair_operator()
    return torch.ops.cann_supergat_ge.csr_supergat_dot_attention_aggregate_fused(
        row_ptr, source_index, projected, max_segment_size, negative_slope
    )
