/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#pragma once

#include <acl/acl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace SparseSmoke
{
inline void Check(aclError error, const char* operation)
{
    if (error != ACL_SUCCESS)
    {
        std::fprintf(stderr, "%s failed: %d\n", operation, static_cast<int>(error));
        std::exit(1);
    }
}

class Session
{
   public:
    Session(int argc, char** argv) : device_(argc > 1 ? std::atoi(argv[1]) : 0)
    {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(device_), "set device");
        Check(aclrtCreateStream(&stream_), "create stream");
    }
    ~Session()
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

class Buffer
{
   public:
    explicit Buffer(size_t bytes) : bytes_(bytes)
    {
        Check(aclrtMalloc(&data_, bytes_, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
    }
    ~Buffer() { aclrtFree(data_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    template <typename T>
    void CopyFrom(const std::vector<T>& values)
    {
        Check(aclrtMemcpy(data_, bytes_, values.data(), values.size() * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE),
              "copy in");
    }

    template <typename T>
    void CopyTo(std::vector<T>& values) const
    {
        Check(aclrtMemcpy(values.data(), values.size() * sizeof(T), data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST),
              "copy out");
    }

    void* Data() const { return data_; }
    size_t Size() const { return bytes_; }

   private:
    void* data_ = nullptr;
    size_t bytes_;
};

inline int Validate(const std::vector<float>& actual, const std::vector<float>& expected)
{
    for (size_t index = 0; index < actual.size(); ++index)
    {
        if (std::fabs(actual[index] - expected[index]) > 1.0e-5F)
        {
            std::fprintf(stderr, "mismatch at %zu\n", index);
            return 1;
        }
    }
    std::puts("PASSED");
    return 0;
}
}  // namespace SparseSmoke
