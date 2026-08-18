/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GRAPH_MESSAGE_EXAMPLE_COMMON_H
#define GRAPH_MESSAGE_EXAMPLE_COMMON_H

#include <acl/acl.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace GraphMessageExample
{
inline void Check(aclError error, const char* operation)
{
    if (error != ACL_SUCCESS)
    {
        std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(error));
        std::exit(1);
    }
}

inline bool ParseDevice(int argc, char** argv, int* device)
{
    *device = 0;
    if (argc <= 1)
    {
        return true;
    }
    char* end = nullptr;
    const long parsed = std::strtol(argv[1], &end, 10);
    if (end == argv[1] || *end != '\0' || parsed < 0 || parsed > 1024)
    {
        std::fprintf(stderr, "invalid device id: %s\n", argv[1]);
        return false;
    }
    *device = static_cast<int>(parsed);
    return true;
}

class Runtime
{
   public:
    explicit Runtime(int device) : device_(device)
    {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device_), "aclrtSetDevice");
        Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }

    ~Runtime()
    {
        aclrtDestroyStream(stream_);
        aclrtResetDevice(device_);
        aclFinalize();
    }

    aclrtStream Stream() const { return stream_; }

   private:
    int device_;
    aclrtStream stream_ = nullptr;
};

class Buffers
{
   public:
    ~Buffers()
    {
        for (void* value : values_)
        {
            aclrtFree(value);
        }
    }

    void* Allocate(size_t bytes)
    {
        void* device = nullptr;
        Check(aclrtMalloc(&device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        values_.push_back(device);
        return device;
    }

    template <typename T>
    void* Copy(const std::vector<T>& host)
    {
        const size_t bytes = host.size() * sizeof(T);
        void* device = Allocate(bytes);
        Check(aclrtMemcpy(device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy to device");
        return device;
    }

   private:
    std::vector<void*> values_;
};

template <typename T>
void CopyOut(std::vector<T>* host, void* device, aclrtStream stream)
{
    Check(aclrtSynchronizeStream(stream), "synchronize");
    const size_t bytes = host->size() * sizeof(T);
    Check(aclrtMemcpy(host->data(), bytes, device, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
}

inline bool Matches(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance = 1.0e-5F)
{
    if (actual.size() != expected.size())
    {
        return false;
    }
    for (size_t index = 0; index < actual.size(); ++index)
    {
        if (std::fabs(actual[index] - expected[index]) > tolerance)
        {
            std::fprintf(stderr, "mismatch at %zu: %.6f != %.6f\n", index, actual[index], expected[index]);
            return false;
        }
    }
    return true;
}
}  // namespace GraphMessageExample

#endif
