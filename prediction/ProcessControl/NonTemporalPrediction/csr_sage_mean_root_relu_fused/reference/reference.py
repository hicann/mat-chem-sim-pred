# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import numpy as np


def csr_sage_mean_root_relu(row_ptr, column_index, neighbor_features, root_features):
    output = np.empty_like(root_features, dtype=np.float32)
    for node in range(row_ptr.size - 1):
        begin, end = int(row_ptr[node]), int(row_ptr[node + 1])
        output[node] = root_features[node]
        if end > begin:
            output[node] += neighbor_features[column_index[begin:end]].mean(axis=0)
    return np.maximum(output, 0.0)


def use_custom_path(nodes, edges, channels, dtype):
    return (
        nodes >= 64
        and edges > 0
        and 0 < channels <= 1024
        and channels % 2 == 0
        and np.dtype(dtype) == np.dtype(np.float32)
    )
