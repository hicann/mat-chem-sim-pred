/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>

#include "aclnn_autoformer_inference_aggregate_fused.h"
#include "../../common/aclnn_example_utils.h"

namespace
{
using timeseries_examples::Benchmark;
using timeseries_examples::DeviceBuffer;
using timeseries_examples::MaxAbsDiff;
using timeseries_examples::Runtime;
using timeseries_examples::Tensor;

int RunExample()
{
    Runtime runtime;
    if (!runtime.Init())
    {
        return 1;
    }
    const std::vector<float> hostValues{0, 1, 2, 3, 4, 5, 6, 7};
    const std::vector<float> hostCorrelation{5, 0, 5, 0, 0, 0, 0, 0};
    const std::vector<float> expected{1, 2, 3, 4, 5, 6, 3, 4};
    DeviceBuffer valuesMemory, correlationMemory, outputMemory;
    if (!valuesMemory.CopyFrom(hostValues) || !correlationMemory.CopyFrom(hostCorrelation) ||
        !outputMemory.Allocate(expected.size() * sizeof(float)))
    {
        return 1;
    }
    Tensor values({1, 1, 1, 8}, ACL_FLOAT, valuesMemory.Get());
    Tensor correlation({1, 1, 1, 8}, ACL_FLOAT, correlationMemory.Get());
    Tensor output({1, 1, 1, 8}, ACL_FLOAT, outputMemory.Get());
    if (!values.IsValid() || !correlation.IsValid() || !output.IsValid())
    {
        return 1;
    }
    auto prepare = [&](uint64_t* size, aclOpExecutor** executor) {
        return aclnnAutoformerInferenceAggregateFusedGetWorkspaceSize(values.Get(), correlation.Get(), 2, output.Get(),
                                                                       size, executor);
    };
    auto launch = [](void* workspace, uint64_t size, aclOpExecutor* executor, aclrtStream stream) {
        return aclnnAutoformerInferenceAggregateFused(workspace, size, executor, stream);
    };
    double meanMilliseconds = 0.0;
    if (!Benchmark(runtime.Stream(), prepare, launch, meanMilliseconds))
    {
        return 1;
    }
    std::vector<float> hostOutput(expected.size());
    if (!outputMemory.CopyTo(hostOutput))
    {
        return 1;
    }
    const float maxAbs = MaxAbsDiff(hostOutput, expected);
    std::cout << "max_abs_err=" << maxAbs << std::endl;
    std::cout << "operator=AutoformerInferenceAggregateFused,npu_hot_mean_ms=" << meanMilliseconds << std::endl;
    return maxAbs <= 1.0e-5F ? 0 : 2;
}
}  // namespace

int main()
{
    return RunExample();
}
