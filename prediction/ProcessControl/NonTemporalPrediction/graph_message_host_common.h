/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#ifndef GRAPH_MESSAGE_HOST_COMMON_H
#define GRAPH_MESSAGE_HOST_COMMON_H

#include <cstdint>
#include <initializer_list>

namespace GraphMessageHost
{
inline uint64_t AlignUp32(uint64_t value) { return (value + 31U) / 32U * 32U; }

inline bool HasNull(std::initializer_list<const void*> values)
{
    for (const void* value : values)
    {
        if (value == nullptr)
        {
            return true;
        }
    }
    return false;
}

inline bool Aliases(const void* output, std::initializer_list<const void*> inputs)
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

inline bool PositiveInt32(int64_t value) { return value > 0 && value <= INT32_MAX; }

inline bool InRange(int64_t value, int64_t maximum) { return value > 0 && value <= maximum; }
}  // namespace GraphMessageHost

#endif
