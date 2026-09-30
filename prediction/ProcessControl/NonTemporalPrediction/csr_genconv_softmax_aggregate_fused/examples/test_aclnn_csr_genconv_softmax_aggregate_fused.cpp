/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "csr_genconv_softmax_aggregate_fused_host.h"

namespace
{
void Check(aclError error, const char* operation)
{
    if (error != ACL_SUCCESS)
    {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(error));
    }
}

std::vector<float> Reference(const std::vector<int32_t>& rowPtr, const std::vector<int32_t>& source,
                             const std::vector<float>& features, int32_t nodes, int32_t channels, float temperature)
{
    std::vector<float> output(static_cast<size_t>(nodes) * channels, 0.0F);
    std::vector<float> maximum(channels);
    std::vector<float> denominator(channels);
    for (int32_t target = 0; target < nodes; ++target)
    {
        if (rowPtr[target] == rowPtr[target + 1])
        {
            continue;
        }
        std::fill(maximum.begin(), maximum.end(), -std::numeric_limits<float>::max());
        for (int32_t edge = rowPtr[target]; edge < rowPtr[target + 1]; ++edge)
        {
            for (int32_t channel = 0; channel < channels; ++channel)
            {
                const float message =
                    std::max(features[static_cast<size_t>(source[edge]) * channels + channel], 0.0F) + 1.0e-7F;
                maximum[channel] = std::max(maximum[channel], temperature * message);
            }
        }
        std::fill(denominator.begin(), denominator.end(), 0.0F);
        for (int32_t edge = rowPtr[target]; edge < rowPtr[target + 1]; ++edge)
        {
            for (int32_t channel = 0; channel < channels; ++channel)
            {
                const float message =
                    std::max(features[static_cast<size_t>(source[edge]) * channels + channel], 0.0F) + 1.0e-7F;
                const float weight = std::exp(temperature * message - maximum[channel]);
                denominator[channel] += weight;
                output[static_cast<size_t>(target) * channels + channel] += weight * message;
            }
        }
        for (int32_t channel = 0; channel < channels; ++channel)
        {
            output[static_cast<size_t>(target) * channels + channel] /= denominator[channel];
        }
    }
    return output;
}
void CheckCleanup(aclError error, const char* operation)
{
    if (error != ACL_SUCCESS)
    {
        std::fprintf(stderr, "%s failed during cleanup: %d\n", operation, static_cast<int>(error));
    }
}

template <typename T>
void Upload(void* destination, const std::vector<T>& values)
{
    const size_t bytes = values.size() * sizeof(T);
    Check(aclrtMemcpy(destination, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
}

class SmokeTest
{
   public:
    SmokeTest() = default;
    SmokeTest(const SmokeTest&) = delete;
    SmokeTest& operator=(const SmokeTest&) = delete;

    // CWE-772 fix: release every resource acquired before a failing ACL call.
    ~SmokeTest()
    {
        for (void* pointer : {workspace_, dOutput_, dFeatures_, dSource_, dRowPtr_})
        {
            if (pointer != nullptr)
            {
                CheckCleanup(aclrtFree(pointer), "aclrtFree");
            }
        }
        if (stream_ != nullptr)
        {
            CheckCleanup(aclrtDestroyStream(stream_), "aclrtDestroyStream");
        }
        if (deviceReady_)
        {
            CheckCleanup(aclrtResetDevice(device_), "aclrtResetDevice");
        }
        if (aclReady_)
        {
            CheckCleanup(aclFinalize(), "aclFinalize");
        }
    }

    void Initialize(int device)
    {
        device_ = device;
        Check(aclInit(nullptr), "aclInit");
        aclReady_ = true;
        Check(aclrtSetDevice(device_), "aclrtSetDevice");
        deviceReady_ = true;
        Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
        Allocate(&dRowPtr_, rowPtr_.size() * sizeof(int32_t));
        Allocate(&dSource_, source_.size() * sizeof(int32_t));
        Allocate(&dFeatures_, features_.size() * sizeof(float));
        Allocate(&dOutput_, output_.size() * sizeof(float));
        workspaceSize_ = aclnnCsrGenConvSoftmaxAggregateFusedGetWorkspaceSize(kNodes, kEdges, kChannels, kTemperature);
        Allocate(&workspace_, workspaceSize_);
    }

    void ValidateParameters()
    {
        const int64_t tooManyNodes = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
        if (aclnnCsrGenConvSoftmaxAggregateFused(dRowPtr_, dSource_, dFeatures_, dOutput_, tooManyNodes, kEdges,
                                                 kChannels, kTemperature, workspace_, workspaceSize_,
                                                 stream_) != ACL_ERROR_INVALID_PARAM)
        {
            throw std::runtime_error("unrepresentable node count was not rejected");
        }
        if (aclnnCsrGenConvSoftmaxAggregateFused(dRowPtr_, dSource_, dFeatures_, dOutput_, kNodes, kEdges, kChannels,
                                                 std::numeric_limits<float>::infinity(), workspace_, workspaceSize_,
                                                 stream_) != ACL_ERROR_INVALID_PARAM)
        {
            throw std::runtime_error("non-finite temperature was not rejected");
        }
    }

    void CheckOutput()
    {
        const std::vector<float> expected = Reference(rowPtr_, source_, features_, kNodes, kChannels, kTemperature);
        Upload(dRowPtr_, rowPtr_);
        Upload(dSource_, source_);
        Upload(dFeatures_, features_);
        LaunchAndDownload();
        for (size_t index = 0; index < output_.size(); ++index)
        {
            if (std::fabs(output_[index] - expected[index]) > 2.0e-5F)
            {
                throw std::runtime_error("output mismatch at " + std::to_string(index));
            }
        }
    }

    void CheckMalformedRows()
    {
        const std::vector<int32_t> invalidRowPtr{0, 5, 3, 4};
        std::fill(output_.begin(), output_.end(), 7.0F);
        Upload(dRowPtr_, invalidRowPtr);
        Upload(dOutput_, output_);
        LaunchAndDownload();
        for (size_t index = 0; index < 4; ++index)
        {
            if (std::fabs(output_[index]) > 1.0e-6F)
            {
                throw std::runtime_error("malformed CSR row was not guarded");
            }
        }
    }

   private:
    static void Allocate(void** pointer, size_t bytes)
    {
        Check(aclrtMalloc(pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }

    void LaunchAndDownload()
    {
        Check(aclnnCsrGenConvSoftmaxAggregateFused(dRowPtr_, dSource_, dFeatures_, dOutput_, kNodes, kEdges, kChannels,
                                                   kTemperature, workspace_, workspaceSize_, stream_),
              "launch");
        Check(aclrtSynchronizeStream(stream_), "synchronize");
        const size_t bytes = output_.size() * sizeof(float);
        Check(aclrtMemcpy(output_.data(), bytes, dOutput_, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
    }

    static constexpr int64_t kNodes = 3;
    static constexpr int64_t kEdges = 4;
    static constexpr int64_t kChannels = 2;
    static constexpr float kTemperature = 1.0F;
    const std::vector<int32_t> rowPtr_{0, 2, 3, 4};
    const std::vector<int32_t> source_{0, 1, 1, 2};
    const std::vector<float> features_{1.0F, -2.0F, 3.0F, 4.0F, -1.0F, 2.0F};
    std::vector<float> output_ = std::vector<float>(static_cast<size_t>(kNodes * kChannels), 0.0F);
    void* dRowPtr_ = nullptr;
    void* dSource_ = nullptr;
    void* dFeatures_ = nullptr;
    void* dOutput_ = nullptr;
    void* workspace_ = nullptr;
    uint64_t workspaceSize_ = 0;
    aclrtStream stream_ = nullptr;
    int device_ = 0;
    bool aclReady_ = false;
    bool deviceReady_ = false;
};
}  // namespace

int main(int argc, char** argv)
{
    try
    {
        SmokeTest test;
        test.Initialize(argc > 1 ? std::stoi(argv[1]) : 0);
        test.ValidateParameters();
        test.CheckOutput();
        test.CheckMalformedRows();
        std::puts("PASSED");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
