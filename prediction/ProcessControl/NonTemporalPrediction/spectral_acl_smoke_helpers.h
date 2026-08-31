/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#pragma once

#include <acl/acl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

namespace SpectralSmoke
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
        Check(aclrtSetDevice(device_), "aclrtSetDevice");
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
              "copy to device");
    }

    template <typename T>
    void CopyTo(std::vector<T>& values) const
    {
        Check(aclrtMemcpy(values.data(), values.size() * sizeof(T), data_, bytes_, ACL_MEMCPY_DEVICE_TO_HOST),
              "copy from device");
    }

    void* Data() const { return data_; }
    size_t Size() const { return bytes_; }

   private:
    void* data_ = nullptr;
    size_t bytes_;
};

using BasisWorkspace = uint64_t (*)(int64_t, int64_t, int64_t);
using BasisLaunch = int32_t (*)(void*, void*, void*, void*, void*, int64_t, int64_t, int64_t, void*, uint64_t, void*);

struct BasisApi
{
    BasisWorkspace workspace;
    BasisLaunch launch;
    uint32_t planes;
};

inline std::vector<float> ExpectedBasis(uint32_t planes)
{
    const std::vector<float> first{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> second{3.0F, 4.0F, 1.0F, 2.0F};
    std::vector<float> expected;
    for (uint32_t plane = 0U; plane < planes; ++plane)
    {
        const std::vector<float>& values = plane % 2U == 0U ? first : second;
        expected.insert(expected.end(), values.begin(), values.end());
    }
    return expected;
}

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

inline int RunBasis(int argc, char** argv, const BasisApi& api)
{
    Session session(argc, argv);
    const std::vector<int32_t> rowPtr{0, 1, 2}, source{1, 0};
    const std::vector<float> norm{1.0F, 1.0F}, features{1.0F, 2.0F, 3.0F, 4.0F};
    const std::vector<float> expected = ExpectedBasis(api.planes);
    std::vector<float> output(expected.size(), 0.0F);
    Buffer row(rowPtr.size() * sizeof(int32_t)), sourceData(source.size() * sizeof(int32_t));
    Buffer normData(norm.size() * sizeof(float)), featureData(features.size() * sizeof(float));
    Buffer outputData(output.size() * sizeof(float)), workspace(api.workspace(2, 2, 2));
    row.CopyFrom(rowPtr);
    sourceData.CopyFrom(source);
    normData.CopyFrom(norm);
    featureData.CopyFrom(features);
    const int64_t overflow = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    if (api.launch(row.Data(), sourceData.Data(), normData.Data(), featureData.Data(), outputData.Data(), overflow, 2,
                   2, workspace.Data(), workspace.Size(), session.Stream()) != ACL_ERROR_INVALID_PARAM)
    {
        return 1;
    }
    Check(api.launch(row.Data(), sourceData.Data(), normData.Data(), featureData.Data(), outputData.Data(), 2, 2, 2,
                     workspace.Data(), workspace.Size(), session.Stream()),
          "launch");
    Check(aclrtSynchronizeStream(session.Stream()), "sync");
    outputData.CopyTo(output);
    return Validate(output, expected);
}
}  // namespace SpectralSmoke
