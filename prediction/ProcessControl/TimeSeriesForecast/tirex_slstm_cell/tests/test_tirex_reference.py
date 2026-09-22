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


class TirexSlstmReferenceTest(unittest.TestCase):
    def test_slstm_step(self):
        batch_size = 2
        hidden_dim = 4

        state_c = np.zeros((batch_size, hidden_dim), dtype=np.float32)
        normalizer_n = np.ones((batch_size, hidden_dim), dtype=np.float32)

        gate_i = np.exp(np.array([0.1], dtype=np.float32))
        gate_f = np.exp(np.array([-0.1], dtype=np.float32))
        gate_z = np.tanh(np.array([0.5], dtype=np.float32))
        gate_o = 1.0 / (1.0 + np.exp(np.array([0.0], dtype=np.float32)))

        state_c_next = gate_f * state_c + gate_i * gate_z
        normalizer_next = gate_f * normalizer_n + gate_i
        state_h_next = gate_o * (state_c_next / (normalizer_next + 1e-6))

        self.assertEqual(state_h_next.shape, (batch_size, hidden_dim))
        self.assertTrue(np.isfinite(state_h_next).all())


if __name__ == "__main__":
    unittest.main()
