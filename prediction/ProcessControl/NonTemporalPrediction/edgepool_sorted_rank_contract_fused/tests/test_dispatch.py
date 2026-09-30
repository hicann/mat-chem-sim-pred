# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from integration.dispatch import dispatch, supports


class Tensor:
    def __init__(self, shape, dtype, device="npu", contiguous=True):
        self.shape, self.ndim, self.dtype = shape, len(shape), dtype
        self.device = type("Device", (), {"type": device})()
        self._contiguous = contiguous

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def is_contiguous(self):
        return self._contiguous


def test_supports_only_prevalidated_npu_contract_inputs():
    order = Tensor((2,), "torch.int32")
    edge_index = Tensor((2, 2), "torch.int32")
    score = Tensor((2,), "torch.float32")
    assert supports(order, edge_index, score, 2, permutation_validated=True)
    assert not supports(order, edge_index, score, 2)
    assert not supports(
        Tensor((2,), "torch.float32"),
        edge_index,
        score,
        2,
        permutation_validated=True,
    )


def test_dispatch_selects_custom_only_when_supported():
    order = Tensor((2,), "torch.int32")
    edge_index = Tensor((2, 2), "torch.int32")
    score = Tensor((2,), "torch.float32")
    assert (
        dispatch(
            lambda: "custom",
            lambda: "fallback",
            order,
            edge_index,
            score,
            2,
            permutation_validated=True,
        )
        == "custom"
    )
    assert (
        dispatch(lambda: "custom", lambda: "fallback", order, edge_index, score, 2)
        == "fallback"
    )
