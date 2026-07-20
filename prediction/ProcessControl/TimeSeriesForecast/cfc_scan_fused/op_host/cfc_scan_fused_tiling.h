/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CFC_SCAN_FUSED_TILING_H
#define CFC_SCAN_FUSED_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CfcScanFusedTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, length);
  TILING_DATA_FIELD_DEF(uint32_t, in_size);
  TILING_DATA_FIELD_DEF(uint32_t, hidden);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CfcScanFused, CfcScanFusedTilingData)
}  // namespace optiling

#endif  // CFC_SCAN_FUSED_TILING_H
