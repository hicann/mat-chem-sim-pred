/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TIME_SERIES_FORECAST_ACLNN_EXAMPLE_UTILS_H
#define TIME_SERIES_FORECAST_ACLNN_EXAMPLE_UTILS_H

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "acl/acl.h"

namespace timeseries_examples {

class Runtime {
   public:
    Runtime() = default;
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    ~Runtime()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (opApi_ != nullptr) {
            dlclose(opApi_);
        }
        if (allOps_ != nullptr) {
            dlclose(allOps_);
        }
        if (deviceActive_) {
            aclrtResetDevice(0);
        }
        if (aclInitialized_) {
            aclFinalize();
        }
    }

    bool Init()
    {
        if (aclInit(nullptr) != ACL_SUCCESS) {
            return false;
        }
        aclInitialized_ = true;
        if (aclrtSetDevice(0) != ACL_SUCCESS) {
            return false;
        }
        deviceActive_ = true;
        allOps_ = dlopen("libascend_all_ops.so", RTLD_NOW | RTLD_GLOBAL);
        opApi_ = dlopen("libcust_opapi.so", RTLD_NOW | RTLD_GLOBAL);
        if (allOps_ == nullptr || opApi_ == nullptr) {
            std::cerr << "custom operator libraries unavailable" << std::endl;
            return false;
        }
        return aclrtCreateStream(&stream_) == ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

   private:
    bool aclInitialized_ = false;
    bool deviceActive_ = false;
    void* allOps_ = nullptr;
    void* opApi_ = nullptr;
    aclrtStream stream_ = nullptr;
};

class DeviceBuffer {
   public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer()
    {
        if (data_ != nullptr) {
            aclrtFree(data_);
        }
    }

    bool Allocate(size_t bytes)
    {
        return bytes == 0U || aclrtMalloc(&data_, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) == ACL_SUCCESS;
    }

    template <typename T>
    bool CopyFrom(const std::vector<T>& host)
    {
        const size_t bytes = host.size() * sizeof(T);
        return Allocate(bytes) &&
               aclrtMemcpy(data_, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }

    template <typename T>
    bool CopyTo(std::vector<T>& host) const
    {
        const size_t bytes = host.size() * sizeof(T);
        return aclrtMemcpy(host.data(), bytes, data_, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    }

    void* Get() const { return data_; }

   private:
    void* data_ = nullptr;
};

inline aclTensor* CreateTensor(const std::vector<int64_t>& dimensions, aclDataType dataType, void* data)
{
    std::vector<int64_t> strides(dimensions.size());
    int64_t stride = 1;
    for (size_t offset = 0; offset < dimensions.size(); ++offset) {
        const size_t index = dimensions.size() - offset - 1U;
        strides[index] = stride;
        stride *= dimensions[index];
    }
    return aclCreateTensor(dimensions.data(), dimensions.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                           dimensions.data(), dimensions.size(), data);
}

class Tensor {
   public:
    Tensor(const std::vector<int64_t>& dimensions, aclDataType dataType, void* data)
        : tensor_(CreateTensor(dimensions, dataType, data))
    {
    }
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    ~Tensor()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
        }
    }

    aclTensor* Get() const { return tensor_; }
    bool IsValid() const { return tensor_ != nullptr; }

   private:
    aclTensor* tensor_ = nullptr;
};

template <typename Prepare, typename Launch>
bool Benchmark(aclrtStream stream, Prepare prepare, Launch launch, double& meanMilliseconds)
{
    constexpr int kWarmup = 3;
    constexpr int kRepeat = 10;
    std::vector<double> samples;
    for (int iteration = 0; iteration < kWarmup + kRepeat; ++iteration) {
        uint64_t workspaceSize = 0U;
        aclOpExecutor* executor = nullptr;
        aclnnStatus status = prepare(&workspaceSize, &executor);
        if (status != 0) {
            std::cerr << "GetWorkspaceSize failed: " << status << std::endl;
            return false;
        }
        DeviceBuffer workspace;
        if (!workspace.Allocate(workspaceSize)) {
            return false;
        }
        const auto start = std::chrono::steady_clock::now();
        status = launch(workspace.Get(), workspaceSize, executor, stream);
        if (status != 0 || aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            return false;
        }
        if (iteration >= kWarmup) {
            const auto elapsed = std::chrono::steady_clock::now() - start;
            samples.push_back(std::chrono::duration<double, std::milli>(elapsed).count());
        }
    }
    meanMilliseconds = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    return true;
}

inline float MaxAbsDiff(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
    float result = 0.0F;
    for (size_t index = 0; index < lhs.size(); ++index) {
        result = std::max(result, std::fabs(lhs[index] - rhs[index]));
    }
    return result;
}

template <typename T>
size_t MismatchCount(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    size_t count = 0U;
    for (size_t index = 0; index < lhs.size(); ++index) {
        count += lhs[index] != rhs[index] ? 1U : 0U;
    }
    return count;
}

}  // namespace timeseries_examples

#endif  // TIME_SERIES_FORECAST_ACLNN_EXAMPLE_UTILS_H
