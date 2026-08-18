/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <cstdint>
#include <vector>

#include "../../graph_message_example_common.h"
#include "csr_signed_cross_mean_pack_fused_host.h"

namespace
{
std::vector<float> BuildFeatures(int channels)
{
    std::vector<float> features(4 * channels);
    for (int channel = 0; channel < channels; ++channel)
    {
        features[channel] = 1.0F;
        features[channels + channel] = 10.0F;
        features[2 * channels + channel] = 2.0F;
        features[3 * channels + channel] = 20.0F;
    }
    return features;
}

std::vector<float> BuildExpected(int channels)
{
    std::vector<float> expected(8 * channels);
    const float values[8] = {1.5F, 20.0F, 15.0F, 2.0F, 1.0F, 15.0F, 10.0F, 1.5F};
    for (int block = 0; block < 8; ++block)
    {
        std::fill(expected.begin() + block * channels, expected.begin() + (block + 1) * channels, values[block]);
    }
    return expected;
}
}  // namespace

int main(int argc, char** argv)
{
    int device = 0;
    if (!GraphMessageExample::ParseDevice(argc, argv, &device))
    {
        return 1;
    }
    GraphMessageExample::Runtime runtime(device);
    GraphMessageExample::Buffers buffers;
    const std::vector<int32_t> positiveRow{0, 2, 3}, positiveSource{0, 1, 0};
    const std::vector<int32_t> negativeRow{0, 1, 3}, negativeSource{1, 0, 1};
    const std::vector<float> positiveInverse{0.5F, 1.0F}, negativeInverse{1.0F, 0.5F};
    constexpr int channels = 8;
    const std::vector<float> features = BuildFeatures(channels);
    const std::vector<float> expected = BuildExpected(channels);
    std::vector<float> output(expected.size());
    void* dPositiveRow = buffers.Copy(positiveRow);
    void* dPositiveSource = buffers.Copy(positiveSource);
    void* dNegativeRow = buffers.Copy(negativeRow);
    void* dNegativeSource = buffers.Copy(negativeSource);
    void* dFeatures = buffers.Copy(features);
    void* dPositiveInverse = buffers.Copy(positiveInverse);
    void* dNegativeInverse = buffers.Copy(negativeInverse);
    void* dOutput = buffers.Allocate(output.size() * sizeof(float));
    const uint64_t bytes = aclnnCsrSignedCrossMeanPackFusedGetWorkspaceSize(2, 3, 3, channels, 2);
    void* workspace = buffers.Allocate(bytes);
    bool passed = aclnnCsrSignedCrossMeanPackFused(dPositiveRow, dPositiveSource, dNegativeRow, dNegativeSource,
                                                   dFeatures, dPositiveInverse, dNegativeInverse, dOutput, 2, 3, 3, 1,
                                                   2, workspace, bytes, runtime.Stream()) == ACL_ERROR_INVALID_PARAM;
    passed &= aclnnCsrSignedCrossMeanPackFused(dPositiveRow, dPositiveSource, dNegativeRow, dNegativeSource, dFeatures,
                                               dPositiveInverse, dNegativeInverse, dOutput, 2, 3, 3, channels, 2,
                                               workspace, bytes, runtime.Stream()) == ACL_SUCCESS;
    GraphMessageExample::CopyOut(&output, dOutput, runtime.Stream());
    passed &= GraphMessageExample::Matches(output, expected);
    return passed ? 0 : 1;
}
