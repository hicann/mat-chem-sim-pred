# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

"""NumPy reference for deterministic farthest-point sampling."""

from __future__ import annotations

import numpy as np


def farthest_point_sampling(points, sample_count):
    batch, point_count, _ = points.shape
    output = np.empty((batch, sample_count), dtype=np.int32)
    for batch_index in range(batch):
        minimum_distance = np.full((point_count,), np.inf, dtype=np.float32)
        selected = 0
        for sample in range(sample_count):
            output[batch_index, sample] = selected
            distance = ((points[batch_index] - points[batch_index, selected]) ** 2).sum(
                axis=-1
            )
            minimum_distance = np.minimum(minimum_distance, distance)
            selected = int(minimum_distance.argmax())
    return output
