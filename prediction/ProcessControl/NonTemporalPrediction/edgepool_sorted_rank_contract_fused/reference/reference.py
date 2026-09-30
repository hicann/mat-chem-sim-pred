# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""NumPy oracle for PyG EdgePooling's sorted-rank edge contraction."""

from __future__ import annotations

import numpy as np


def is_supported(order, edge_index, score, nodes):
    return (
        isinstance(nodes, int)
        and 0 < nodes <= 2**31 - 1
        and order.ndim == 1
        and edge_index.ndim == 2
        and edge_index.shape[0] == 2
        and score.ndim == 1
        and order.shape[0] == edge_index.shape[1] == score.shape[0]
        and order.size > 0
        and np.array_equal(np.sort(order), np.arange(order.size))
        and np.all((edge_index >= 0) & (edge_index < nodes))
    )


def reference(order, edge_index, score, nodes):
    """Return cluster, selected edge, cluster score, and dynamic counts."""
    if not is_supported(order, edge_index, score, nodes):
        raise ValueError("unsupported EdgePool contraction inputs")

    cluster = np.full(nodes, -1, dtype=np.int32)
    selected_edge = np.empty(nodes, dtype=np.int32)
    cluster_score = np.empty(nodes, dtype=np.float32)
    selected = 0
    for edge in order.tolist():
        source = int(edge_index[0, edge])
        target = int(edge_index[1, edge])
        if cluster[source] != -1 or cluster[target] != -1:
            continue
        cluster[source] = selected
        if source != target:
            cluster[target] = selected
        selected_edge[selected] = edge
        cluster_score[selected] = score[edge]
        selected += 1

    clusters = selected
    for node in range(nodes):
        if cluster[node] == -1:
            cluster[node] = clusters
            cluster_score[clusters] = np.float32(1.0)
            clusters += 1
    return (
        cluster,
        selected_edge,
        cluster_score,
        np.asarray([selected, clusters], dtype=np.int32),
    )
