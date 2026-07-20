#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TirexSlstmCellTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, seq_len);
  TILING_DATA_FIELD_DEF(uint32_t, hidden_dim);
  TILING_DATA_FIELD_DEF(uint32_t, num_heads);
  TILING_DATA_FIELD_DEF(uint32_t, head_dim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TirexSlstmCell, TirexSlstmCellTilingData)
}
