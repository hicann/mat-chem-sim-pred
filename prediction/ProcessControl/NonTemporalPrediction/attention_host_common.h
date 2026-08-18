/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef ATTENTION_HOST_COMMON_H
#define ATTENTION_HOST_COMMON_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "acl/acl.h"
#include "attention_fused_tiling.h"

namespace AttentionHost
{
inline uint64_t WorkspaceSize()
{
    const uint64_t size = sizeof(AttentionFusedTiling);
    return (size + 31U) / 32U * 32U;
}

inline bool HasNull(std::initializer_list<const void*> pointers)
{
    for (const void* pointer : pointers)
    {
        if (pointer == nullptr)
        {
            return true;
        }
    }
    return false;
}

inline bool OutputAliases(const void* output, std::initializer_list<const void*> inputs)
{
    for (const void* input : inputs)
    {
        if (output == input)
        {
            return true;
        }
    }
    return false;
}

inline bool ValidShape(int64_t nodes, int64_t edges, int64_t heads, int64_t channels, int64_t maxSegmentSize)
{
    const int64_t int32Max = std::numeric_limits<int32_t>::max();
    const bool validGraph = nodes > 0 && nodes <= int32Max && edges > 0 && edges <= int32Max;
    const bool validFeatures = heads > 0 && heads <= 8 && channels > 0 && channels <= 32;
    const bool validSegment = maxSegmentSize > 0 && maxSegmentSize <= 256;
    return validGraph && validFeatures && validSegment;
}

inline bool ValidSlope(float slope) { return std::isfinite(slope) && slope >= 0.0F && slope <= 1.0F; }

inline bool ValidInvocation(std::initializer_list<const void*> required, const void* output,
                            std::initializer_list<const void*> inputs, int64_t nodes, int64_t edges, int64_t heads,
                            int64_t channels, int64_t maxSegmentSize, uint64_t workspaceSize)
{
    if (HasNull(required) || OutputAliases(output, inputs))
    {
        return false;
    }
    return ValidShape(nodes, edges, heads, channels, maxSegmentSize) && workspaceSize >= WorkspaceSize();
}

inline uint32_t CoreCount(int64_t nodes)
{
    return static_cast<uint32_t>(std::max<int64_t>(1, std::min<int64_t>(nodes, 40)));
}

inline AttentionFusedTiling MakeTiling(int64_t nodes, int64_t edges, int64_t heads, int64_t channels,
                                       int64_t maxSegmentSize, float parameter)
{
    return {static_cast<uint32_t>(nodes),
            static_cast<uint32_t>(edges),
            static_cast<uint32_t>(heads),
            static_cast<uint32_t>(channels),
            CoreCount(nodes),
            static_cast<uint32_t>(maxSegmentSize),
            parameter,
            0U};
}

inline int32_t CopyTiling(void* workspace, uint64_t workspaceSize, const AttentionFusedTiling& tiling)
{
    if (workspaceSize < WorkspaceSize())
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    return aclrtMemcpy(workspace, sizeof(tiling), &tiling, sizeof(tiling), ACL_MEMCPY_HOST_TO_DEVICE);
}
}  // namespace AttentionHost

#endif  // ATTENTION_HOST_COMMON_H
