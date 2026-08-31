# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Guarded node-attention HypergraphConv two-stage dispatch policy."""

__all__ = ["dispatch", "supports"]


def _basic_contract(tensors) -> bool:
    integer_tensors = tuple(tensors[index] for index in (0, 1, 3, 4, 5))
    feature_tensors = tuple(tensors[index] for index in (2, 6, 7, 8))
    feature_dtype = str(tensors[7].dtype)
    return (
        all(
            tensor.device.type == "npu" and tensor.is_contiguous() for tensor in tensors
        )
        and all(str(tensor.dtype) == "torch.int32" for tensor in integer_tensors)
        and feature_dtype in {"torch.float32", "torch.float16", "torch.bfloat16"}
        and all(str(tensor.dtype) == feature_dtype for tensor in feature_tensors)
    )


def _shape_contract(tensors, max_edge_size, max_node_degree) -> bool:
    edge_row_ptr, node_index, edge_scale = tensors[:3]
    node_row_ptr, edge_index, incidence_position = tensors[3:6]
    node_scale, features, attention_logits = tensors[6:]
    if features.ndim != 3 or attention_logits.ndim != 2:
        return False
    nodes, heads, channels = features.shape
    incidences = node_index.numel()
    hyperedges = edge_row_ptr.numel() - 1
    return (
        node_row_ptr.numel() == nodes + 1
        and edge_index.numel() == incidence_position.numel() == incidences
        and edge_scale.shape == (hyperedges,)
        and node_scale.shape == (nodes,)
        and attention_logits.shape == (incidences, heads)
        and 0 < heads <= 4
        and 0 < channels <= 32
        and 0 < max_edge_size <= 512
        and 0 < max_node_degree <= 512
    )


def supports(*args, **policy) -> bool:
    max_edge_size = policy.get("max_edge_size", 0)
    max_node_degree = policy.get("max_node_degree", 0)
    return (
        policy.get("structure_validated", False)
        and policy.get("attention_mode", "node") == "node"
        and len(args) == 9
        and _basic_contract(args)
        and _shape_contract(args, max_edge_size, max_node_degree)
    )


def dispatch(custom_call, fallback, *args, **policy):
    return custom_call() if supports(*args, **policy) else fallback()
