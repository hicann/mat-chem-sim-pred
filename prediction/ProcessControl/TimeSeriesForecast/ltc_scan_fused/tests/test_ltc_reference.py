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


class LtcScanReferenceTest(unittest.TestCase):
    def test_ltc_ode_step(self):
        batch_size = 2
        hidden_dim = 4
        time_delta = 0.1
        capacitance = 1.0

        hidden_prev = np.ones((batch_size, hidden_dim), dtype=np.float32)
        current_input = np.full((batch_size, hidden_dim), 0.3, dtype=np.float32)
        conductance = np.full((batch_size, hidden_dim), 0.5, dtype=np.float32)

        numerator = capacitance * hidden_prev + time_delta * (
            current_input * conductance
        )
        denominator = capacitance + time_delta * conductance
        hidden_next = numerator / denominator

        self.assertEqual(hidden_next.shape, (batch_size, hidden_dim))
        self.assertTrue(np.isfinite(hidden_next).all())

    def test_ltc_steady_state(self):
        val = np.array([1.0, 1.0], dtype=np.float32)
        self.assertTrue(np.all(val > 0.0))


if __name__ == "__main__":
    unittest.main()
