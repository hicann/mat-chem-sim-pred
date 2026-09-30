# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
from integration import dispatch, supports


class Tensor:
    def __init__(self, shape, dtype, device="npu", contiguous=True):
        self.shape, self.ndim, self.dtype = shape, len(shape), dtype
        self.device = type("Device", (), {"type": device})()
        self.requires_grad = False
        self._contiguous = contiguous

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def is_contiguous(self):
        return self._contiguous


def test_guarded_dispatch_and_fallback():
    args = (
        Tensor((3,), "torch.int32"),
        Tensor((4,), "torch.int32"),
        Tensor((2, 2, 8), "torch.float32"),
    )
    assert supports(*args, max_segment_size=2, csr_validated=True)
    assert (
        dispatch(
            lambda: "custom",
            lambda: "fallback",
            *args,
            max_segment_size=2,
            csr_validated=True,
        )
        == "custom"
    )
    assert (
        dispatch(
            lambda: "custom",
            lambda: "fallback",
            *args,
            max_segment_size=513,
            csr_validated=True,
        )
        == "fallback"
    )


def test_mixed_precision_and_training_policy():
    for dtype in ("torch.float32", "torch.float16", "torch.bfloat16"):
        args = (
            Tensor((3,), "torch.int32"),
            Tensor((4,), "torch.int32"),
            Tensor((2, 2, 8), dtype),
        )
        assert supports(*args, max_segment_size=2, csr_validated=True)
        args[-1].requires_grad = True
        assert supports(*args, max_segment_size=2, training=True, csr_validated=True)
