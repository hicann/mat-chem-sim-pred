# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from __future__ import annotations

import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "integration" / "dispatch.py"
SPEC = importlib.util.spec_from_file_location("ppf_dispatch", PATH)
DISPATCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DISPATCH)


class Tensor:
    def __init__(self, shape, dtype, device="npu", contiguous=True, offset=0):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = type("Device", (), {"type": device})()
        self._contiguous = contiguous
        self._offset = offset

    def is_contiguous(self):
        return self._contiguous

    def storage_offset(self):
        return self._offset

    def numel(self):
        size = 1
        for value in self.shape:
            size *= value
        return size


def test_contract_accepts_contiguous_npu_inputs():
    position = Tensor((8, 3), "torch.float32")
    normal = Tensor((8, 3), "torch.float32")
    source = Tensor((16,), "torch.int32")
    target = Tensor((16,), "torch.int32")
    assert DISPATCH.supports(position, normal, source, target)


def test_contract_falls_back_for_unsupported_layout_or_empty_edges():
    position = Tensor((8, 3), "torch.float32", contiguous=False)
    normal = Tensor((8, 3), "torch.float32")
    source = Tensor((16,), "torch.int32")
    target = Tensor((16,), "torch.int32")
    assert not DISPATCH.supports(position, normal, source, target)
    assert not DISPATCH.supports(
        position, normal, Tensor((0,), "torch.int32"), Tensor((0,), "torch.int32")
    )
