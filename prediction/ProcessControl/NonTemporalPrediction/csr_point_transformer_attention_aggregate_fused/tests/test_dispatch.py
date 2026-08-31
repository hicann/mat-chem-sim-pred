# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from integration import dispatch, supports


class Tensor:
    def __init__(self, shape, dtype, device="npu"):
        self.shape, self.ndim, self.dtype = shape, len(shape), dtype
        self.device = type("Device", (), {"type": device})()
        self.requires_grad = False

    @staticmethod
    def is_contiguous():
        return True

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result


def test_default_semantics_dispatch_and_attention_mlp_fallback():
    args = (Tensor((3,), "torch.int32"), Tensor((4,), "torch.int32"))
    args += tuple(
        Tensor(shape, "torch.float32") for shape in ((2, 8), (2, 8), (2, 8), (4, 8))
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
            max_segment_size=2,
            has_attention_mlp=True,
            csr_validated=True,
        )
        == "fallback"
    )
    assert (
        dispatch(
            lambda: "custom",
            lambda: "fallback",
            *args,
            max_segment_size=2,
            training=True,
            csr_validated=True,
        )
        == "custom"
    )


def test_mixed_precision_requires_a_shared_dtype():
    for dtype in ("torch.float32", "torch.float16", "torch.bfloat16"):
        args = (Tensor((3,), "torch.int32"), Tensor((4,), "torch.int32"))
        args += tuple(
            Tensor(shape, dtype) for shape in ((2, 8), (2, 8), (2, 8), (4, 8))
        )
        assert supports(*args, max_segment_size=2, csr_validated=True)
    mismatched = list(args)
    mismatched[-1] = Tensor((4, 8), "torch.float32")
    assert not supports(*mismatched, max_segment_size=2, csr_validated=True)
