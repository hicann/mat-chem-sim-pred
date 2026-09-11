/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <acl/acl.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>

inline bool CheckAcl(aclError status, const char* stage)
{
    if (status == ACL_SUCCESS) return true;
    std::cerr << stage << " failed with ACL error " << status << std::endl;
    return false;
}

inline void* AllocateAcl(std::size_t bytes)
{
    void* pointer = nullptr;
    return CheckAcl(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc") ? pointer : nullptr;
}

inline bool CopyToDevice(void* destination, std::size_t bytes, const void* source, const char* stage)
{
    return CheckAcl(aclrtMemcpy(destination, bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE), stage);
}

inline bool AllChecks(std::initializer_list<bool> checks)
{
    bool passed = true;
    for (bool check : checks) passed &= check;
    return passed;
}

inline bool InitializeAclDevice(int device, aclrtStream& stream)
{
    return CheckAcl(aclInit(nullptr), "aclInit") && CheckAcl(aclrtSetDevice(device), "aclrtSetDevice") &&
           CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream");
}

template <std::size_t N>
inline bool CopyAndCheckOutput(std::array<float, N>& output, void* deviceOutput, aclrtStream stream,
                               const std::array<float, N>& expected)
{
    bool passed = CheckAcl(aclrtSynchronizeStream(stream), "synchronize");
    passed &= CheckAcl(aclrtMemcpy(output.data(), output.size() * sizeof(float), deviceOutput,
                                   output.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST),
                       "copy output");
    for (std::size_t index = 0; index < output.size(); ++index)
    {
        passed &= std::fabs(output[index] - expected[index]) < 1.0e-5F;
    }
    return passed;
}
