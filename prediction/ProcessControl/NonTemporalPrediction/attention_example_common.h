/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef ATTENTION_EXAMPLE_COMMON_H
#define ATTENTION_EXAMPLE_COMMON_H

#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace AttentionExample
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
        for (void* pointer : pointers_)
        {
            aclrtFree(pointer);
        }
    }

    void* Allocate(size_t bytes)
    {
        void* pointer = nullptr;
        Check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
        pointers_.push_back(pointer);
        return pointer;
    }

    template <typename T>
    void* Copy(const std::vector<T>& values)
    {
        const size_t bytes = values.size() * sizeof(T);
        void* pointer = Allocate(bytes);
        Check(aclrtMemcpy(pointer, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
        return pointer;
    }

   private:
    std::vector<void*> pointers_;
};

inline bool Matches(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance)
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

inline void CopyOutput(std::vector<float>* output, const void* deviceOutput)
{
    const size_t bytes = output->size() * sizeof(float);
    Check(aclrtMemcpy(output->data(), bytes, deviceOutput, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
}
}  // namespace AttentionExample

#endif  // ATTENTION_EXAMPLE_COMMON_H
