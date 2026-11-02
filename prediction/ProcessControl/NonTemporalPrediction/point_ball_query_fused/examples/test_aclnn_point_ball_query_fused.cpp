/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cstdint>
#include <vector>

#include "../../pointnet2_geometry_example_common.h"
#include "point_ball_query_fused_host.h"

int main(int argc, char** argv)
{
    int device = 0;
    if (!PointNet2GeometryExample::ParseDevice(argc, argv, &device))
    {
        return 1;
    }
    PointNet2GeometryExample::Runtime runtime(device);
    PointNet2GeometryExample::Buffers buffers;
    const std::vector<float> points = {0, 0, 0, 1, 0, 0, 3, 0, 0};
    const std::vector<float> queries = {0, 0, 0, 2, 0, 0};
    std::vector<int32_t> indices(4, -1);
    std::vector<int32_t> counts(2, 0);
    void* devicePoints = buffers.Copy(points);
    void* deviceQueries = buffers.Copy(queries);
    void* deviceIndices = buffers.Allocate(indices.size() * sizeof(int32_t));
    void* deviceCounts = buffers.Allocate(counts.size() * sizeof(int32_t));
    const uint64_t workspaceSize = aclnnPointBallQueryFusedGetWorkspaceSize(1, 3, 2, 2);
    void* workspace = buffers.Allocate(workspaceSize);
    PointNet2GeometryExample::Check(
        aclnnPointBallQueryFused(devicePoints, deviceQueries, deviceIndices, deviceCounts, 1, 3, 2, 2, 1.1F, workspace,
                                 workspaceSize, runtime.Stream()),
        "launch");
    buffers.CopyOut(&indices, deviceIndices, runtime.Stream());
    buffers.CopyOut(&counts, deviceCounts, runtime.Stream());
    const bool indicesMatch = indices == std::vector<int32_t>({0, 1, 1, 2});
    return indicesMatch && counts == std::vector<int32_t>({2, 2}) ? 0 : 1;
}
