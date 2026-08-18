# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "reference"))
from reference import farthest_point_sampling


def test_farthest_point_sampling_is_deterministic():
    points = np.array(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=np.float32,
    )
    actual = farthest_point_sampling(points, 3)
    np.testing.assert_array_equal(actual, np.array([[0, 2, 3]], dtype=np.int32))
