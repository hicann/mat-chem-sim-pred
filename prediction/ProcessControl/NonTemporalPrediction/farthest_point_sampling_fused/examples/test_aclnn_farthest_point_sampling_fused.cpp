/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cstdint>
#include <vector>

#include "../../pointnet2_geometry_example_common.h"
#include "farthest_point_sampling_fused_host.h"

int main(int argc, char** argv)
{
    int device = 0;
    if (!PointNet2GeometryExample::ParseDevice(argc, argv, &device))
    {
        return 1;
    }
    PointNet2GeometryExample::Runtime runtime(device);
    PointNet2GeometryExample::Buffers buffers;
    const std::vector<float> points = {0.0F, 0.0F, 0.0F, 1.0F, 0.0F, 0.0F, 4.0F, 0.0F, 0.0F, 2.0F, 0.0F, 0.0F};
    std::vector<int32_t> indices(3, -1);
    void* devicePoints = buffers.Copy(points);
    void* deviceIndices = buffers.Allocate(indices.size() * sizeof(int32_t));
    const uint64_t workspaceSize = aclnnFarthestPointSamplingFusedGetWorkspaceSize(1, 4, 3);
    void* workspace = buffers.Allocate(workspaceSize);
    PointNet2GeometryExample::Check(aclnnFarthestPointSamplingFused(devicePoints, deviceIndices, 1, 4, 3, workspace,
                                                                    workspaceSize, runtime.Stream()),
                                    "launch");
    buffers.CopyOut(&indices, deviceIndices, runtime.Stream());
    return indices == std::vector<int32_t>({0, 2, 3}) ? 0 : 1;
}
