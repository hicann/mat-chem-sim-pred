# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import importlib.util
import unittest
from pathlib import Path


def _load_adapter():
    path = Path(__file__).resolve().parents[1] / "integration" / "adapter.py"
    spec = importlib.util.spec_from_file_location(
        "reformer_lsh_bucket_sort_adapter", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ADAPTER = _load_adapter()
dispatch = _ADAPTER.dispatch
supports_custom = _ADAPTER.supports_custom


class Device:
    type = "npu"


class FakeTensor:
    def __init__(self, shape, dtype="torch.float32", contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.device = Device()
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous


class AdapterTest(unittest.TestCase):
    def test_supported_uses_custom(self):
        tensor = FakeTensor((2, 8), "torch.int64")
        self.assertTrue(supports_custom(tensor, 4, 3))
        selected = dispatch(
            tensor,
            4,
            3,
            custom_call=lambda *unused: "custom",
            fallback=lambda *unused: "fallback",
        )
        self.assertEqual(selected, "custom")

    def test_unsupported_uses_fallback(self):
        bad = FakeTensor((2, 8), "torch.int32")
        self.assertFalse(supports_custom(bad, 4, 3))
        selected = dispatch(
            bad,
            4,
            3,
            custom_call=lambda *unused: "custom",
            fallback=lambda *unused: "fallback",
        )
        self.assertEqual(selected, "fallback")

    def test_oversized_dimension_uses_fallback(self):
        oversized = FakeTensor(((1 << 32), 8), "torch.int64")
        self.assertFalse(supports_custom(oversized, 4, 3))
        oversized_length = FakeTensor((1, (1 << 31)), "torch.int64")
        self.assertFalse(supports_custom(oversized_length, 4, 3))


if __name__ == "__main__":
    unittest.main()
