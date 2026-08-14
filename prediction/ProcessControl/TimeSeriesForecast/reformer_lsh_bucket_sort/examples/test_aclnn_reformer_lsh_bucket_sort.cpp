/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "acl/acl.h"
#include "aclnn_reformer_lsh_bucket_sort.h"

namespace
{
constexpr size_t kElementCount = 8U;
constexpr int kWarmup = 3;
constexpr int kRepeat = 10;

aclTensor* MakeTensor(const std::vector<int64_t>& dims, aclDataType dtype, void* data)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for (int64_t i = static_cast<int64_t>(dims.size()) - 2; i >= 0; --i)
    {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * dims[static_cast<size_t>(i + 1)];
    }
    return aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, dims.data(), dims.size(),
                           data);
}

template <typename T>
bool CopyToDevice(void** device, const std::vector<T>& host)
{
    const size_t bytes = host.size() * sizeof(T);
    return aclrtMalloc(device, bytes, ACL_MEM_MALLOC_NORMAL_ONLY) == ACL_SUCCESS &&
           aclrtMemcpy(*device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
}

template <typename T>
bool AllocateOutput(void** device, size_t elements)
{
    return aclrtMalloc(device, elements * sizeof(T), ACL_MEM_MALLOC_NORMAL_ONLY) == ACL_SUCCESS;
}

template <typename T>
bool CopyToHost(std::vector<T>& host, void* device)
{
    const size_t bytes = host.size() * sizeof(T);
    return aclrtMemcpy(host.data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

template <typename T>
size_t MismatchCount(const std::vector<T>& lhs, const std::vector<T>& rhs)
{
    size_t count = 0;
    for (size_t i = 0; i < lhs.size(); ++i)
    {
        count += lhs[i] != rhs[i] ? 1U : 0U;
    }
    return count;
}

class RuntimeContext
{
   public:
    bool Initialize()
    {
        if (aclInit(nullptr) != ACL_SUCCESS)
        {
            return false;
        }
        initialized_ = true;
        if (aclrtSetDevice(0) != ACL_SUCCESS)
        {
            return false;
        }
        deviceSet_ = true;
        allOps_ = dlopen("libascend_all_ops.so", RTLD_NOW | RTLD_GLOBAL);
        opApi_ = dlopen("libcust_opapi.so", RTLD_NOW | RTLD_GLOBAL);
        if (allOps_ == nullptr || opApi_ == nullptr)
        {
            std::cerr << "custom operator libraries unavailable" << std::endl;
            return false;
        }
        return aclrtCreateStream(&stream_) == ACL_SUCCESS;
    }

    ~RuntimeContext()
    {
        if (stream_ != nullptr)
        {
            aclrtDestroyStream(stream_);
        }
        if (opApi_ != nullptr)
        {
            dlclose(opApi_);
        }
        if (allOps_ != nullptr)
        {
            dlclose(allOps_);
        }
        if (deviceSet_)
        {
            aclrtResetDevice(0);
        }
        if (initialized_)
        {
            aclFinalize();
        }
    }

    aclrtStream Stream() const { return stream_; }

   private:
    void* allOps_ = nullptr;
    void* opApi_ = nullptr;
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

class SortFixture
{
   public:
    bool Initialize()
    {
        if (!CopyToDevice(&dKeys_, keys_) || !AllocateOutput<int64_t>(&dSorted_, kElementCount) ||
            !AllocateOutput<int64_t>(&dSticker_, kElementCount) || !AllocateOutput<int64_t>(&dInverse_, kElementCount))
        {
            return false;
        }
        keysTensor_ = MakeTensor({1, 8}, ACL_INT64, dKeys_);
        sortedTensor_ = MakeTensor({1, 8}, ACL_INT64, dSorted_);
        stickerTensor_ = MakeTensor({1, 8}, ACL_INT64, dSticker_);
        inverseTensor_ = MakeTensor({1, 8}, ACL_INT64, dInverse_);
        return keysTensor_ != nullptr && sortedTensor_ != nullptr && stickerTensor_ != nullptr &&
               inverseTensor_ != nullptr;
    }

    ~SortFixture()
    {
        DestroyTensor(keysTensor_);
        DestroyTensor(sortedTensor_);
        DestroyTensor(stickerTensor_);
        DestroyTensor(inverseTensor_);
        FreeDevice(dKeys_);
        FreeDevice(dSorted_);
        FreeDevice(dSticker_);
        FreeDevice(dInverse_);
    }

    aclTensor* Keys() const { return keysTensor_; }

    aclTensor* Sorted() const { return sortedTensor_; }

    aclTensor* Sticker() const { return stickerTensor_; }

    aclTensor* Inverse() const { return inverseTensor_; }

    bool CountMismatches(size_t* mismatch)
    {
        if (!CopyToHost(sorted_, dSorted_) || !CopyToHost(sticker_, dSticker_) || !CopyToHost(inverse_, dInverse_))
        {
            return false;
        }
        *mismatch = MismatchCount(sorted_, expectedSorted_) + MismatchCount(sticker_, expectedSticker_) +
                    MismatchCount(inverse_, expectedInverse_);
        return true;
    }

   private:
    static void DestroyTensor(aclTensor* tensor)
    {
        if (tensor != nullptr)
        {
            aclDestroyTensor(tensor);
        }
    }

    static void FreeDevice(void* device)
    {
        if (device != nullptr)
        {
            aclrtFree(device);
        }
    }

    const std::vector<int64_t> keys_{9, 1, 8, 0, 5, 4, 2, 10};
    const std::vector<int64_t> expectedSorted_{1, 0, 2, 5, 4, 9, 8, 10};
    const std::vector<int64_t> expectedSticker_{1, 3, 6, 4, 5, 0, 2, 7};
    const std::vector<int64_t> expectedInverse_{5, 0, 6, 1, 3, 4, 2, 7};
    std::vector<int64_t> sorted_ = std::vector<int64_t>(kElementCount);
    std::vector<int64_t> sticker_ = std::vector<int64_t>(kElementCount);
    std::vector<int64_t> inverse_ = std::vector<int64_t>(kElementCount);
    void* dKeys_ = nullptr;
    void* dSorted_ = nullptr;
    void* dSticker_ = nullptr;
    void* dInverse_ = nullptr;
    aclTensor* keysTensor_ = nullptr;
    aclTensor* sortedTensor_ = nullptr;
    aclTensor* stickerTensor_ = nullptr;
    aclTensor* inverseTensor_ = nullptr;
};

bool RunBenchmark(const SortFixture& fixture, aclrtStream stream, double* meanMs)
{
    std::vector<double> hotMs;
    for (int iteration = 0; iteration < kWarmup + kRepeat; ++iteration)
    {
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        const aclnnStatus prepareStatus = aclnnReformerLshBucketSortGetWorkspaceSize(
            fixture.Keys(), 4, 3, fixture.Sorted(), fixture.Sticker(), fixture.Inverse(), &workspaceSize, &executor);
        if (prepareStatus != 0)
        {
            std::cerr << "GetWorkspaceSize failed: " << prepareStatus << std::endl;
            return false;
        }
        void* workspace = nullptr;
        if (workspaceSize > 0 && aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_NORMAL_ONLY) != ACL_SUCCESS)
        {
            return false;
        }
        const auto start = std::chrono::steady_clock::now();
        const aclnnStatus launchStatus = aclnnReformerLshBucketSort(workspace, workspaceSize, executor, stream);
        const aclError synchronizeStatus = aclrtSynchronizeStream(stream);
        const auto stop = std::chrono::steady_clock::now();
        if (workspace != nullptr)
        {
            aclrtFree(workspace);
        }
        if (launchStatus != 0 || synchronizeStatus != ACL_SUCCESS)
        {
            return false;
        }
        if (iteration >= kWarmup)
        {
            hotMs.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        }
    }
    *meanMs = std::accumulate(hotMs.begin(), hotMs.end(), 0.0) / hotMs.size();
    return true;
}

}  // namespace

int main()
{
    RuntimeContext runtime;
    SortFixture fixture;
    if (!runtime.Initialize() || !fixture.Initialize())
    {
        return 1;
    }
    double meanMs = 0.0;
    size_t mismatch = 0U;
    if (!RunBenchmark(fixture, runtime.Stream(), &meanMs) || !fixture.CountMismatches(&mismatch))
    {
        return 1;
    }
    std::cout << "mismatch_count=" << mismatch << std::endl;
    std::cout << "operator=ReformerLshBucketSort,npu_hot_mean_ms=" << meanMs << std::endl;
    return mismatch == 0U ? 0 : 2;
}
