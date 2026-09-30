# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""Guarded integration policy for EdgePool sorted-rank contraction."""


def supports(order, edge_index, score, nodes, permutation_validated=False):
    return (
        permutation_validated
        and isinstance(nodes, int)
        and 0 < nodes <= 2**31 - 1
        and order.device.type == "npu"
        and edge_index.device.type == "npu"
        and score.device.type == "npu"
        and order.is_contiguous()
        and edge_index.is_contiguous()
        and score.is_contiguous()
        and str(order.dtype) == "torch.int32"
        and str(edge_index.dtype) == "torch.int32"
        and str(score.dtype) == "torch.float32"
        and order.ndim == 1
        and edge_index.ndim == 2
        and edge_index.shape[0] == 2
        and score.ndim == 1
        and order.numel() == edge_index.shape[1] == score.numel()
        and order.numel() > 0
    )


def dispatch(custom_call, fallback, *args, permutation_validated=False):
    if supports(*args, permutation_validated=permutation_validated):
        return custom_call()
    return fallback()
