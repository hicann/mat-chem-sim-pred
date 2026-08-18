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
import random
import unittest
from pathlib import Path


def _load_reference():
    path = Path(__file__).resolve().parents[1] / "reference" / "reference.py"
    spec = importlib.util.spec_from_file_location(
        "autoformer_inference_aggregate_fused_reference", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference


reference = _load_reference()


class ReferenceTest(unittest.TestCase):
    def test_reference_and_tie_order(self):
        values = [[[[0.0, 1.0, 2.0, 3.0]]]]
        correlation = [[[[5.0, 0.0, 5.0, 0.0]]]]
        actual = reference(values, correlation, 2)
        self.assertEqual(actual, [[[[1.0, 2.0, 1.0, 2.0]]]])

    def test_invalid_topk(self):
        with self.assertRaises(ValueError):
            reference([[[[1.0]]]], [[[[1.0]]]], 2)

    def test_random_batch_shared_delay(self):
        generator = random.Random(47)
        values = [
            [
                [[generator.random() for _ in range(8)] for _ in range(2)]
                for _ in range(2)
            ]
        ]
        correlation = [[[[0.0 for _ in range(8)] for _ in range(2)] for _ in range(2)]]
        for head in range(2):
            for channel in range(2):
                correlation[0][head][channel][3] = 7.0
        actual = reference(values, correlation, 1)
        for head in range(2):
            for channel in range(2):
                expected = values[0][head][channel][3:] + values[0][head][channel][:3]
                self.assertEqual(actual[0][head][channel], expected)

    def test_single_lag_minimum_aligned_length(self):
        values = [[[[float(i) for i in range(8)]]]]
        correlation = [[[[0, 0, 0, 0, 1, 0, 0, 0]]]]
        self.assertEqual(
            reference(values, correlation, 1)[0][0][0], [4, 5, 6, 7, 0, 1, 2, 3]
        )


if __name__ == "__main__":
    unittest.main()
