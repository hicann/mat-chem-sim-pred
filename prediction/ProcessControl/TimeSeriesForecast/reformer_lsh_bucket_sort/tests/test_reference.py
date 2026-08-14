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
        "reformer_lsh_bucket_sort_reference", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reference from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference


reference = _load_reference()


class ReferenceTest(unittest.TestCase):
    def test_reference(self):
        keys = [[9, 1, 8, 0, 5, 4, 2, 10]]
        sorted_keys, sticker, inverse = reference(keys, 4, 3)
        self.assertEqual(sorted_keys, [[1, 0, 2, 5, 4, 9, 8, 10]])
        self.assertEqual(sticker, [[1, 3, 6, 4, 5, 0, 2, 7]])
        self.assertEqual(inverse, [[5, 0, 6, 1, 3, 4, 2, 7]])

    def test_invalid_key(self):
        with self.assertRaises(ValueError):
            reference([[12]], 4, 3)

    def test_random_stability_and_inverse(self):
        generator = random.Random(17)
        keys = [[generator.randrange(0, 32) for _ in range(37)] for _ in range(3)]
        sorted_keys, stickers, inverses = reference(keys, 8, 4)
        for row, sorted_row, sticker, inverse in zip(
            keys, sorted_keys, stickers, inverses
        ):
            self.assertEqual(sorted(sticker), list(range(len(row))))
            self.assertEqual(
                [inverse[index] for index in sticker], list(range(len(row)))
            )
            self.assertEqual(
                [value // 8 for value in sorted_row],
                sorted(value // 8 for value in row),
            )
            for bucket in range(4):
                sources = [index for index in sticker if row[index] // 8 == bucket]
                self.assertEqual(sources, sorted(sources))

    def test_minimum(self):
        self.assertEqual(reference([[0]], 1, 1), ([[0]], [[0]], [[0]]))


if __name__ == "__main__":
    unittest.main()
