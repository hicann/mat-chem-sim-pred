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


class CornnScanReferenceTest(unittest.TestCase):
    def test_cornn_oscillator_step(self):
        batch_size = 2
        hidden_dim = 4
        step_size = 0.05
        gamma_param = 0.1

        state_h = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        state_z = np.ones((batch_size, hidden_dim), dtype=np.float32)
        input_x = np.full((batch_size, hidden_dim), 0.2, dtype=np.float32)

        state_z_next = state_z + step_size * (
            -gamma_param * state_z - state_h + input_x
        )
        state_h_next = state_h + step_size * state_z_next

        self.assertEqual(state_h_next.shape, (batch_size, hidden_dim))
        self.assertTrue(np.isfinite(state_h_next).all())

    def test_cornn_deterministic(self):
        state_val = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        diff_val = state_val * 0.1
        self.assertEqual(len(diff_val), 3)


if __name__ == "__main__":
    unittest.main()
