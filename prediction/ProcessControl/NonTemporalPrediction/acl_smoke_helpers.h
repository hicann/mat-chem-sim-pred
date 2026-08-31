/** Copyright (c) 2026 Huawei Technologies Co., Ltd. */
// Licensed under the CANN Open Software License Agreement Version 2.0.
#pragma once

#include <acl/acl.h>

#include <cstdio>
#include <cstdlib>

namespace MolecularGeometrySmoke
{
inline void Check(aclError value, const char* name)
{
    if (value != ACL_SUCCESS)
    {
        std::fprintf(stderr, "%s failed\n", name);
        std::exit(1);
    }
}

inline void CopyIn(void** device, const void* host, size_t bytes)
{
    Check(aclrtMalloc(device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
    Check(aclrtMemcpy(*device, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "copy");
}
}  // namespace MolecularGeometrySmoke
