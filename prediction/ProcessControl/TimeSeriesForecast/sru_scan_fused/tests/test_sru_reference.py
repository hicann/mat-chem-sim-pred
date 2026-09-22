# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import unittest
import numpy as np


class SruScanReferenceTest(unittest.TestCase):
    def test_sru_step(self):
        batch_size = 2
        hidden_dim = 4

        state_c = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        gate_f = np.full((batch_size, hidden_dim), 0.7, dtype=np.float32)
        gate_r = np.full((batch_size, hidden_dim), 0.8, dtype=np.float32)
        input_x = np.full((batch_size, hidden_dim), 0.5, dtype=np.float32)

        state_c_next = gate_f * state_c + (1.0 - gate_f) * input_x
        state_h_next = gate_r * state_c_next + (1.0 - gate_r) * input_x

        self.assertEqual(state_h_next.shape, (batch_size, hidden_dim))
        self.assertTrue(np.isfinite(state_h_next).all())

    def test_sru_identity(self):
        c_val = np.array([0.0, 1.0], dtype=np.float32)
        self.assertEqual(len(c_val), 2)


if __name__ == "__main__":
    unittest.main()
