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


class CfcScanReferenceTest(unittest.TestCase):
    def test_cfc_reference_step(self):
        batch_size = 2
        hidden_dim = 4
        time_delta = 0.1

        hidden_prev = np.ones((batch_size, hidden_dim), dtype=np.float32)
        gate_weight = np.full((batch_size, hidden_dim), 0.5, dtype=np.float32)

        decay = 1.0 / (1.0 + np.exp(-gate_weight * time_delta))
        hidden_next = decay * hidden_prev + (1.0 - decay) * gate_weight

        self.assertEqual(hidden_next.shape, (batch_size, hidden_dim))
        self.assertTrue(np.isfinite(hidden_next).all())

    def test_cfc_zero_delta(self):
        hidden_init = np.array([0.5, -0.5, 1.0], dtype=np.float32)
        decay = 1.0 / (1.0 + np.exp(0.0))  # 0.5
        updated = decay * hidden_init
        self.assertEqual(updated.shape, hidden_init.shape)


if __name__ == "__main__":
    unittest.main()
