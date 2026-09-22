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


class AutoCorrReferenceTest(unittest.TestCase):
    def test_autocorr_reference_shape(self):
        batch_size = 2
        seq_len = 16
        num_heads = 4
        head_dim = 8
        top_k = 2

        value = np.arange(seq_len, dtype=np.float32).reshape(1, 1, 1, seq_len)
        value = np.broadcast_to(
            value, (batch_size, num_heads, head_dim, seq_len)
        ).copy()

        # Simple lag roll reference
        lags = [0, 1]
        weight_val = 1.0 / float(top_k)
        aggregated = np.zeros_like(value)
        for lag in lags:
            rolled = np.roll(value, shift=lag, axis=-1)
            aggregated += rolled * weight_val

        self.assertEqual(aggregated.shape, (batch_size, num_heads, head_dim, seq_len))
        self.assertTrue(np.isfinite(aggregated).all())

    def test_autocorr_identity_lag(self):
        input_signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        rolled_zero = np.roll(input_signal, shift=0)
        np.testing.assert_allclose(input_signal, rolled_zero)


if __name__ == "__main__":
    unittest.main()
