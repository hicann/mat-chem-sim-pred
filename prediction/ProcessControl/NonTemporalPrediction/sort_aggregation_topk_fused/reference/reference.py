# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import numpy as np


def sort_aggregation_topk(graph_ptr, features, top_k):
    graphs, channels = graph_ptr.size - 1, features.shape[1]
    output = np.zeros((graphs, top_k, channels), dtype=np.float32)
    for graph in range(graphs):
        begin, end = int(graph_ptr[graph]), int(graph_ptr[graph + 1])
        values = features[begin:end]
        order = np.argsort(-values[:, -1], kind="stable")[:top_k]
        output[graph, : order.size] = values[order]
    return output.reshape(graphs, top_k * channels)


def use_custom_path(*values):
    nodes, graphs, channels, top_k, dtype = values[:5]
    scores_are_finite = values[5] if len(values) > 5 else True
    return (
        nodes > 0
        and graphs >= 4
        and 0 < channels <= 1024
        and 0 < top_k <= 128
        and np.dtype(dtype) == np.dtype(np.float32)
        and scores_are_finite
    )
