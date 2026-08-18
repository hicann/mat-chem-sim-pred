# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from adapter_test_utils import FakeTensor, load_adapter  # noqa: E402

_ADAPTER = load_adapter(
    Path(__file__).resolve().parents[1],
    "autoformer_inference_aggregate_fused_adapter",
)
dispatch = _ADAPTER.dispatch
supports_custom = _ADAPTER.supports_custom


class AdapterTest(unittest.TestCase):
    def test_supported_uses_custom(self):
        tensors = (FakeTensor((2, 4, 8, 96)), FakeTensor((2, 4, 8, 96)))
        self.assertTrue(supports_custom(*tensors, 4))
        selected = dispatch(
            *tensors,
            4,
            custom_call=lambda *unused: "custom",
            fallback=lambda *unused: "fallback",
        )
        self.assertEqual(selected, "custom")

    def test_unsupported_uses_fallback(self):
        bad = (FakeTensor((2, 4, 8, 95)), FakeTensor((2, 4, 8, 95)))
        self.assertFalse(supports_custom(*bad, 4))
        selected = dispatch(
            *bad,
            4,
            custom_call=lambda *unused: "custom",
            fallback=lambda *unused: "fallback",
        )
        self.assertEqual(selected, "fallback")

    def test_low_value_b4_l336_uses_fallback(self):
        tensors = (FakeTensor((4, 4, 16, 336)), FakeTensor((4, 4, 16, 336)))
        self.assertFalse(supports_custom(*tensors, 4))
        selected = dispatch(
            *tensors,
            4,
            custom_call=lambda *unused: "custom",
            fallback=lambda *unused: "fallback",
        )
        self.assertEqual(selected, "fallback")

    def test_oversized_dimension_uses_fallback(self):
        tensors = (
            FakeTensor((2, 4, 8, (1 << 32))),
            FakeTensor((2, 4, 8, (1 << 32))),
        )
        self.assertFalse(supports_custom(*tensors, 4))
        over_local_memory = (
            FakeTensor((2, 4, 8, 4104)),
            FakeTensor((2, 4, 8, 4104)),
        )
        self.assertFalse(supports_custom(*over_local_memory, 4))


if __name__ == "__main__":
    unittest.main()
