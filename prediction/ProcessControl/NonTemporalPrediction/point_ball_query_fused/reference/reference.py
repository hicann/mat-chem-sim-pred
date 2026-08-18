# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import numpy as np


def point_ball_query(points, queries, sample_count, radius):
    batch, _, _ = points.shape
    query_count = queries.shape[1]
    indices = np.full((batch, query_count, sample_count), -1, dtype=np.int32)
    counts = np.zeros((batch, query_count), dtype=np.int32)
    for batch_index in range(batch):
        for query in range(query_count):
            distance = np.sum(
                (points[batch_index] - queries[batch_index, query]) ** 2, axis=-1
            )
            selected = np.flatnonzero(distance <= radius * radius)[:sample_count]
            counts[batch_index, query] = selected.size
            indices[batch_index, query, : selected.size] = selected
    return indices, counts
