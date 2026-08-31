# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from integration import dispatch, supports


class Tensor:
    def __init__(self, shape, dtype):
        self.shape, self.ndim, self.dtype = shape, len(shape), dtype
        self.device = type("Device", (), {"type": "npu"})()
        self.requires_grad = False

    @staticmethod
    def is_contiguous():
        return True

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result


def test_node_attention_dispatch_and_edge_mode_fallback():
    i32 = "torch.int32"
    f32 = "torch.float32"
    args = (
        Tensor((3,), i32),
        Tensor((4,), i32),
        Tensor((2,), f32),
        Tensor((4,), i32),
        Tensor((4,), i32),
        Tensor((4,), i32),
        Tensor((3,), f32),
        Tensor((3, 2, 8), f32),
        Tensor((4, 2), f32),
    )
    assert supports(*args, max_edge_size=2, max_node_degree=2, structure_validated=True)
    assert (
        dispatch(
            lambda: "custom",
            lambda: "fallback",
            *args,
            max_edge_size=2,
            max_node_degree=2,
            training=True,
            structure_validated=True,
        )
        == "custom"
    )
    assert (
        dispatch(
            lambda: "custom",
            lambda: "fallback",
            *args,
            max_edge_size=2,
            max_node_degree=2,
            attention_mode="edge",
            structure_validated=True,
        )
        == "fallback"
    )


def test_mixed_precision_requires_a_shared_dtype():
    i32 = "torch.int32"
    for dtype in ("torch.float32", "torch.float16", "torch.bfloat16"):
        args = (
            Tensor((3,), i32),
            Tensor((4,), i32),
            Tensor((2,), dtype),
            Tensor((4,), i32),
            Tensor((4,), i32),
            Tensor((4,), i32),
            Tensor((3,), dtype),
            Tensor((3, 2, 8), dtype),
            Tensor((4, 2), dtype),
        )
        assert supports(
            *args, max_edge_size=2, max_node_degree=2, structure_validated=True
        )
    mismatched = list(args)
    mismatched[-1] = Tensor((4, 2), "torch.float32")
    assert not supports(
        *mismatched, max_edge_size=2, max_node_degree=2, structure_validated=True
    )
