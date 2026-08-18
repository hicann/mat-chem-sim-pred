# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import importlib.util
from pathlib import Path

import numpy as np


def load_reference():
    path = Path(__file__).parents[1] / "reference" / "reference.py"
    spec = importlib.util.spec_from_file_location("point_ball_query_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_point_ball_query_reference():
    points = np.array([[[0, 0, 0], [1, 0, 0], [3, 0, 0]]], dtype=np.float32)
    queries = np.array([[[0, 0, 0], [2, 0, 0]]], dtype=np.float32)
    indices, counts = load_reference().point_ball_query(points, queries, 2, 1.1)
    np.testing.assert_array_equal(counts, [[2, 2]])
    np.testing.assert_array_equal(indices, [[[0, 1], [1, 2]]])
