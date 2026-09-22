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


class BatchSpdInvReferenceTest(unittest.TestCase):
    def test_spd_inv_identity(self):
        batch_size = 4
        matrix_dim = 8
        identity_batch = np.zeros(
            (batch_size, matrix_dim, matrix_dim), dtype=np.float32
        )
        for idx in range(batch_size):
            identity_batch[idx] = np.eye(matrix_dim, dtype=np.float32)

        inverted = np.zeros_like(identity_batch)
        for idx in range(batch_size):
            inverted[idx] = np.linalg.inv(identity_batch[idx])

        np.testing.assert_allclose(inverted, identity_batch, atol=1e-5)

    def test_spd_inv_reconstruction(self):
        matrix_dim = 4
        random_state = np.random.RandomState(42)
        random_mat = random_state.randn(matrix_dim, matrix_dim).astype(np.float32)
        spd_mat = random_mat.T @ random_mat + np.eye(matrix_dim, dtype=np.float32) * 0.1

        inv_mat = np.linalg.inv(spd_mat)
        product = spd_mat @ inv_mat
        np.testing.assert_allclose(
            product, np.eye(matrix_dim, dtype=np.float32), atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
