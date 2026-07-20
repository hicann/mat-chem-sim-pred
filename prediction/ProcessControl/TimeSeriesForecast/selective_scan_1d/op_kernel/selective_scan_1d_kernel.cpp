/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct SelectiveScanTiling {
  uint32_t batch;
  uint32_t length;
  uint32_t dim;
  uint32_t state;
};

// Selective scan (Mamba SSM core):
//   state[n] = exp(delta*A[d,n]) * state[n] + delta*B[t,n]*u
//   y        = sum_n state[n]*C[t,n] + u*D[d]
// Vectorized over the N (state) dimension: the per-channel A-row stays resident
// in UB, B/C rows are streamed via DataCopy (coalesced) instead of 32 scalar
// GM loads/step, and exp/mul/add/reduce run on the vector unit. Hardware Exp
// also removes the polynomial-approx error of the previous scalar kernel.
class SelectiveScan1DKernel {
 public:
  __aicore__ inline SelectiveScan1DKernel() = default;

  __aicore__ inline void Init(
      GM_ADDR u, GM_ADDR delta, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d,
      GM_ADDR output, GM_ADDR tiling) {
    const __gm__ SelectiveScanTiling* tilingData =
        reinterpret_cast<const __gm__ SelectiveScanTiling*>(tiling);
    batch_ = tilingData->batch;
    length_ = tilingData->length;
    dim_ = tilingData->dim;
    state_ = tilingData->state;
    uGm_.SetGlobalBuffer((__gm__ float*)u, batch_ * length_ * dim_);
    deltaGm_.SetGlobalBuffer((__gm__ float*)delta, batch_ * length_ * dim_);
    aGm_.SetGlobalBuffer((__gm__ float*)a, dim_ * state_);
    bGm_.SetGlobalBuffer((__gm__ float*)b, batch_ * length_ * state_);
    cGm_.SetGlobalBuffer((__gm__ float*)c, batch_ * length_ * state_);
    dGm_.SetGlobalBuffer((__gm__ float*)d, dim_);
    outputGm_.SetGlobalBuffer((__gm__ float*)output, batch_ * length_ * dim_);

    pipe_.InitBuffer(aQue_, 1, state_ * sizeof(float));
    pipe_.InitBuffer(bQue_, 2, state_ * sizeof(float));
    pipe_.InitBuffer(cQue_, 2, state_ * sizeof(float));
    pipe_.InitBuffer(stateBuf_, state_ * sizeof(float));
    pipe_.InitBuffer(tmpBuf_, state_ * sizeof(float));
    pipe_.InitBuffer(workBuf_, 256U);
    pipe_.InitBuffer(redBuf_, 32U);
  }

  __aicore__ inline void Process() {
    const uint32_t groups = batch_ * dim_;
    const uint32_t blockNum = GetBlockNum();
    if (blockNum == 0U) {
      return;
    }
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t base = groups / blockNum;
    const uint32_t tail = groups % blockNum;
    const uint32_t count = base + (blockIdx < tail ? 1U : 0U);
    const uint32_t start = blockIdx * base + (blockIdx < tail ? blockIdx : tail);
    for (uint32_t i = 0; i < count; ++i) {
      ProcessGroup(start + i);
    }
  }

 private:
  __aicore__ inline void ProcessGroup(uint32_t group) {
    const uint32_t batch = group / dim_;
    const uint32_t dim = group % dim_;

    LocalTensor<float> aRow = aQue_.AllocTensor<float>();
    DataCopy(aRow, aGm_[dim * state_], state_);
    aQue_.EnQue(aRow);
    aRow = aQue_.DeQue<float>();

    const float dval = dGm_.GetValue(dim);
    LocalTensor<float> state = stateBuf_.Get<float>();
    Duplicate(state, 0.0f, state_);
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> red = redBuf_.Get<float>();
    LocalTensor<float> work = workBuf_.Get<float>();

    for (uint32_t t = 0; t < length_; ++t) {
      const uint32_t uOff = (batch * length_ + t) * dim_ + dim;
      const uint32_t sOff = (batch * length_ + t) * state_;
      const float uv = uGm_.GetValue(uOff);
      const float dv = deltaGm_.GetValue(uOff);

      LocalTensor<float> bRow = bQue_.AllocTensor<float>();
      DataCopy(bRow, bGm_[sOff], state_);
      bQue_.EnQue(bRow);
      bRow = bQue_.DeQue<float>();

      LocalTensor<float> cRow = cQue_.AllocTensor<float>();
      DataCopy(cRow, cGm_[sOff], state_);
      cQue_.EnQue(cRow);
      cRow = cQue_.DeQue<float>();

      // dA = exp(dv * A[d,:]);  state = dA*state
      Muls(tmp, aRow, dv, state_);
      Exp(tmp, tmp, state_);
      Mul(state, state, tmp, state_);
      // state += (dv*uv) * B[t,:]
      Muls(tmp, bRow, dv * uv, state_);
      Add(state, state, tmp, state_);
      // y = sum_n state*C[t,:] + uv*D[d]
      Mul(tmp, state, cRow, state_);
      ReduceSum<float>(red, tmp, work, state_);
      outputGm_.SetValue(uOff, red.GetValue(0) + uv * dval);

      bQue_.FreeTensor(bRow);
      cQue_.FreeTensor(cRow);
    }
    aQue_.FreeTensor(aRow);
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, 1> aQue_;
  TQue<TPosition::VECIN, 2> bQue_;
  TQue<TPosition::VECIN, 2> cQue_;
  TBuf<TPosition::VECCALC> stateBuf_;
  TBuf<TPosition::VECCALC> tmpBuf_;
  TBuf<TPosition::VECCALC> workBuf_;
  TBuf<TPosition::VECCALC> redBuf_;

  GlobalTensor<float> uGm_;
  GlobalTensor<float> deltaGm_;
  GlobalTensor<float> aGm_;
  GlobalTensor<float> bGm_;
  GlobalTensor<float> cGm_;
  GlobalTensor<float> dGm_;
  GlobalTensor<float> outputGm_;
  uint32_t batch_ = 0;
  uint32_t length_ = 0;
  uint32_t dim_ = 0;
  uint32_t state_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void selective_scan1_d(
    GM_ADDR u, GM_ADDR delta, GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR d,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  SelectiveScan1DKernel op;
  op.Init(u, delta, a, b, c, d, output, tiling);
  op.Process();
}
