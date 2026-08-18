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
        "reformer_lsh_qkv_gather_reference", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference


reference = _load_reference()


class ReferenceTest(unittest.TestCase):
    def test_reference(self):
        qk = [[[0], [1], [2], [3]]]
        value = [[[10], [11], [12], [13]]]
        self.assertEqual(
            reference(qk, value, [[2, 0, 3]]), ([[[2], [0], [3]]], [[[12], [10], [13]]])
        )

    def test_invalid_index(self):
        with self.assertRaises(ValueError):
            reference([[[0]]], [[[1]]], [[1]])

    def test_random_gather(self):
        generator = random.Random(23)
        query_key = [
            [[generator.random() for _ in range(8)] for _ in range(11)]
            for _ in range(2)
        ]
        value = [
            [[generator.random() for _ in range(8)] for _ in range(11)]
            for _ in range(2)
        ]
        indices = [[10, 0, 4, 4], [3, 8, 1, 6]]
        actual_qk, actual_value = reference(query_key, value, indices)
        self.assertEqual(actual_qk[0][2], query_key[0][4])
        self.assertEqual(actual_qk[0][3], query_key[0][4])
        self.assertEqual(actual_value[1][1], value[1][8])

    def test_minimum_aligned_width(self):
        source = [[[float(i) for i in range(8)]]]
        self.assertEqual(reference(source, source, [[0]]), (source, source))


if __name__ == "__main__":
    unittest.main()
