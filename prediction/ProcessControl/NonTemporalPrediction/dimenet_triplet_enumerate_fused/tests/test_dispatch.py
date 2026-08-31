# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from __future__ import annotations

import importlib.util
from pathlib import Path

PATH = Path(__file__).parents[1] / "integration" / "dispatch.py"
SPEC = importlib.util.spec_from_file_location("dimenet_triplet_dispatch", PATH)
DISPATCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(DISPATCH)


class Tensor:
    def __init__(self, size, dtype="torch.int32", device="npu", contiguous=True):
        self._size = size
        self.ndim = 1
        self.dtype = dtype
        self.device = type("Device", (), {"type": device})()
        self._contiguous = contiguous

    def numel(self):
        return self._size

    def is_contiguous(self):
        return self._contiguous


def test_capacity_guard_accepts_proven_upper_bound():
    row_ptr, source = Tensor(101), Tensor(800)
    assert DISPATCH.supports(
        row_ptr, source, capacity=12_800, max_degree=16, csr_validated=True
    )


def test_unvalidated_or_undersized_contract_falls_back():
    row_ptr, source = Tensor(101), Tensor(800)
    assert not DISPATCH.supports(row_ptr, source, 12_800, 16)
    assert not DISPATCH.supports(
        row_ptr, source, capacity=12_799, max_degree=16, csr_validated=True
    )
