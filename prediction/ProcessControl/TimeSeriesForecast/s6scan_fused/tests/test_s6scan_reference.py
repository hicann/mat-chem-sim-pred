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


class S6ScanReferenceTest(unittest.TestCase):
    def test_s6_recurrence_step(self):
        batch_size = 2
        seq_len = 4
        in_features = 3
        hidden_dim = 4

        input_x = np.ones((batch_size, seq_len, in_features), dtype=np.float32) * 0.01
        weight_mat = np.ones((in_features * 3, hidden_dim), dtype=np.float32) * 0.01
        bias_vec = np.ones((hidden_dim * 4,), dtype=np.float32) * 0.01

        end_b = 2 * in_features
        start_x = 2 * in_features
        weight_delta = weight_mat[:in_features, :]
        weight_b = weight_mat[in_features:end_b, :]
        weight_x = weight_mat[start_x:, :]

        mid_dim = 2 * hidden_dim
        three_dim = 3 * hidden_dim
        four_dim = 4 * hidden_dim
        bias_delta = bias_vec[:hidden_dim]
        bias_b = bias_vec[hidden_dim:mid_dim]
        bias_x = bias_vec[mid_dim:three_dim]
        param_a = bias_vec[three_dim:four_dim]

        output_arr = np.zeros((batch_size, seq_len, hidden_dim), dtype=np.float32)
        for b_idx in range(batch_size):
            state_h = np.zeros((hidden_dim,), dtype=np.float32)
            for t_idx in range(seq_len):
                x_vec = input_x[b_idx, t_idx, :]
                pre_delta = x_vec @ weight_delta + bias_delta
                delta_val = np.log1p(np.exp(pre_delta))
                disc_a = np.exp(-delta_val * param_a)
                gate_b = 1.0 / (1.0 + np.exp(-(x_vec @ weight_b + bias_b)))
                proj_x = x_vec @ weight_x + bias_x
                state_h = disc_a * state_h + gate_b * proj_x
                output_arr[b_idx, t_idx, :] = state_h

        self.assertEqual(output_arr.shape, (batch_size, seq_len, hidden_dim))
        self.assertTrue(np.isfinite(output_arr).all())


if __name__ == "__main__":
    unittest.main()
