/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <iostream>
#include <vector>

#include "aclnn_reformer_lsh_qkv_gather.h"
#include "../../common/aclnn_example_utils.h"

namespace
{
using timeseries_examples::Benchmark;
using timeseries_examples::DeviceBuffer;
using timeseries_examples::MaxAbsDiff;
using timeseries_examples::Runtime;
using timeseries_examples::Tensor;

struct HostData
{
    std::vector<float> queryKey = std::vector<float>(32);
    std::vector<float> value = std::vector<float>(32);
    std::vector<int64_t> indices{2, 0, 3};
    std::vector<float> expectedQueryKey = std::vector<float>(24);
    std::vector<float> expectedValue = std::vector<float>(24);
};

HostData MakeHostData()
{
    HostData data;
    for (size_t index = 0; index < data.queryKey.size(); ++index)
    {
        data.queryKey[index] = static_cast<float>(index);
        data.value[index] = 100.0F + static_cast<float>(index);
    }
    for (size_t row = 0; row < data.indices.size(); ++row)
    {
        for (size_t column = 0; column < 8U; ++column)
        {
            const size_t source = static_cast<size_t>(data.indices[row]) * 8U + column;
            data.expectedQueryKey[row * 8U + column] = data.queryKey[source];
            data.expectedValue[row * 8U + column] = data.value[source];
        }
    }
    return data;
}

int RunExample()
{
    Runtime runtime;
    if (!runtime.Init())
    {
        return 1;
    }
    const HostData host = MakeHostData();
    DeviceBuffer queryKeyMemory, valueMemory, indicesMemory, sortedQueryKeyMemory, sortedValueMemory;
    if (!queryKeyMemory.CopyFrom(host.queryKey) || !valueMemory.CopyFrom(host.value) ||
        !indicesMemory.CopyFrom(host.indices) || !sortedQueryKeyMemory.Allocate(24U * sizeof(float)) ||
        !sortedValueMemory.Allocate(24U * sizeof(float)))
    {
        return 1;
    }
    Tensor queryKey({1, 4, 8}, ACL_FLOAT, queryKeyMemory.Get());
    Tensor value({1, 4, 8}, ACL_FLOAT, valueMemory.Get());
    Tensor indices({1, 3}, ACL_INT64, indicesMemory.Get());
    Tensor sortedQueryKey({1, 3, 8}, ACL_FLOAT, sortedQueryKeyMemory.Get());
    Tensor sortedValue({1, 3, 8}, ACL_FLOAT, sortedValueMemory.Get());
    if (!queryKey.IsValid() || !value.IsValid() || !indices.IsValid() || !sortedQueryKey.IsValid() ||
        !sortedValue.IsValid())
    {
        return 1;
    }
    auto prepare = [&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnReformerLshQkvGatherGetWorkspaceSize(queryKey.Get(), value.Get(), indices.Get(),
                                                         sortedQueryKey.Get(), sortedValue.Get(), size, executor);
    };
    auto launch = [](void* workspace, uint64_t size, aclOpExecutor* executor, aclrtStream stream) {
        return aclnnReformerLshQkvGather(workspace, size, executor, stream);
    };
    double meanMilliseconds = 0.0;
    if (!Benchmark(runtime.Stream(), prepare, launch, meanMilliseconds))
    {
        return 1;
    }
    std::vector<float> sortedQueryKeyHost(24), sortedValueHost(24);
    if (!sortedQueryKeyMemory.CopyTo(sortedQueryKeyHost) || !sortedValueMemory.CopyTo(sortedValueHost))
    {
        return 1;
    }
    const float maxAbs = std::max(MaxAbsDiff(sortedQueryKeyHost, host.expectedQueryKey),
                                  MaxAbsDiff(sortedValueHost, host.expectedValue));
    std::cout << "max_abs_err=" << maxAbs << std::endl;
    std::cout << "operator=ReformerLshQkvGather,npu_hot_mean_ms=" << meanMilliseconds << std::endl;
    return maxAbs == 0.0F ? 0 : 2;
}
}  // namespace

int main()
{
    return RunExample();
}
