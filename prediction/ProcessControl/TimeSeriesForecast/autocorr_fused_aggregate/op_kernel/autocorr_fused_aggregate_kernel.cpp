#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t kMaxTopK = 32U;

__aicore__ inline float FastExp(float x) {
  if (x < -20.0f) {
    return 0.0f;
  }
  constexpr float kInvLn2 = 1.4426950409f;
  constexpr float kLn2 = 0.6931471806f;
  const float scaled = x * kInvLn2;
  const int32_t exponent = scaled >= 0.0f
      ? static_cast<int32_t>(scaled + 0.5f)
      : static_cast<int32_t>(scaled - 0.5f);
  const float reduced = x - static_cast<float>(exponent) * kLn2;
  const float polynomial =
      1.0f + reduced *
      (1.0f + reduced *
      (0.5f + reduced *
      (0.1666666667f + reduced *
      (0.0416666667f + reduced *
      (0.0083333333f + reduced * 0.0013888889f)))));
  float scale = 1.0f;
  if (exponent < 0) {
    for (int32_t i = 0; i < -exponent; ++i) {
      scale *= 0.5f;
    }
  } else {
    for (int32_t i = 0; i < exponent; ++i) {
      scale *= 2.0f;
    }
  }
  return polynomial * scale;
}

struct AutoCorrFusedAggregateTiling {
  uint32_t batch;
  uint32_t heads;
  uint32_t embed;
  uint32_t length;
  uint32_t top_k;
};

class AutoCorrFusedAggregateKernel {
 public:
  __aicore__ inline AutoCorrFusedAggregateKernel() = default;

  __aicore__ inline void Init(
      GM_ADDR query,
      GM_ADDR key,
      GM_ADDR value,
      GM_ADDR output,
      const AutoCorrFusedAggregateTiling* tiling) {
    batch_ = tiling->batch;
    heads_ = tiling->heads;
    embed_ = tiling->embed;
    length_ = tiling->length;
    topK_ = tiling->top_k < kMaxTopK ? tiling->top_k : kMaxTopK;
    const uint32_t total = batch_ * heads_ * embed_ * length_;
    queryGm_.SetGlobalBuffer((__gm__ float*)query, total);
    keyGm_.SetGlobalBuffer((__gm__ float*)key, total);
    valueGm_.SetGlobalBuffer((__gm__ float*)value, total);
    outputGm_.SetGlobalBuffer((__gm__ float*)output, total);
  }

  __aicore__ inline void Process() {
    const uint32_t groups = batch_ * heads_ * embed_;
    const uint32_t blockNum = GetBlockNum();
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
  __aicore__ inline float CorrAtLag(uint32_t group, uint32_t lag) const {
    float sum = 0.0f;
    const uint32_t base = group * length_;
    const uint32_t noWrap = length_ - lag;
    for (uint32_t t = 0; t < noWrap; ++t) {
      sum += queryGm_.GetValue(base + t) * keyGm_.GetValue(base + t + lag);
    }
    for (uint32_t t = noWrap; t < length_; ++t) {
      sum += queryGm_.GetValue(base + t) * keyGm_.GetValue(base + t - noWrap);
    }
    return sum;
  }

  __aicore__ inline void InsertTopK(
      float score,
      int32_t lag,
      uint32_t count,
      float* scores,
      int32_t* lags) const {
    if (count == 0U || score <= scores[count - 1U]) {
      return;
    }
    uint32_t pos = count - 1U;
    while (pos > 0U && score > scores[pos - 1U]) {
      scores[pos] = scores[pos - 1U];
      lags[pos] = lags[pos - 1U];
      --pos;
    }
    scores[pos] = score;
    lags[pos] = lag;
  }

  __aicore__ inline void SoftmaxTopK(uint32_t count, float* scores) const {
    if (count == 0U) {
      return;
    }
    float maxValue = scores[0];
    for (uint32_t i = 1; i < count; ++i) {
      const float v = scores[i];
      if (v > maxValue) {
        maxValue = v;
      }
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
      const float e = FastExp(scores[i] - maxValue);
      scores[i] = e;
      sum += e;
    }
    const float inv = sum > 0.0f ? (1.0f / sum) : 0.0f;
    for (uint32_t i = 0; i < count; ++i) {
      scores[i] *= inv;
    }
  }

  __aicore__ inline void Aggregate(uint32_t group, uint32_t count, const float* weights, const int32_t* lags) {
    const uint32_t base = group * length_;
    for (uint32_t t = 0; t < length_; ++t) {
      float acc = 0.0f;
      for (uint32_t k = 0; k < count; ++k) {
        const int32_t lag = lags[k];
        if (lag < 0) {
          continue;
        }
        uint32_t srcT = t + static_cast<uint32_t>(lag);
        if (srcT >= length_) {
          srcT -= length_;
        }
        acc += weights[k] * valueGm_.GetValue(base + srcT);
      }
      outputGm_.SetValue(base + t, acc);
    }
  }

  __aicore__ inline void ProcessGroup(uint32_t group) {
    const uint32_t count = topK_ < length_ ? topK_ : length_;
    float scores[kMaxTopK];
    int32_t lags[kMaxTopK];
    for (uint32_t i = 0; i < kMaxTopK; ++i) {
      scores[i] = -1.0e30f;
      lags[i] = -1;
    }
    for (uint32_t lag = 0; lag < length_; ++lag) {
      InsertTopK(CorrAtLag(group, lag), static_cast<int32_t>(lag), count, scores, lags);
    }
    SoftmaxTopK(count, scores);
    Aggregate(group, count, scores, lags);
  }

  GlobalTensor<float> queryGm_;
  GlobalTensor<float> keyGm_;
  GlobalTensor<float> valueGm_;
  GlobalTensor<float> outputGm_;
  uint32_t batch_ = 0;
  uint32_t heads_ = 0;
  uint32_t embed_ = 0;
  uint32_t length_ = 0;
  uint32_t topK_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void auto_corr_fused_aggregate(
    GM_ADDR query,
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  AutoCorrFusedAggregateKernel op;
  op.Init(query, key, value, output, reinterpret_cast<const AutoCorrFusedAggregateTiling*>(&tilingData));
  op.Process();
}
