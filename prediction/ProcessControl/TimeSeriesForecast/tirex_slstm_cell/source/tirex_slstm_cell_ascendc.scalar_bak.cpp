#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr float kLn2 = 0.69314718055994531f;
constexpr float kLog2e = 1.44269504088896341f;
constexpr float kEps = 1.0e-6f;

struct TirexSlstmCellTiling {
  uint32_t batch;
  uint32_t seq_len;
  uint32_t hidden_dim;
  uint32_t num_heads;
  uint32_t head_dim;
};

__aicore__ inline float FastExp(float x) {
  if (x < -80.0f) return 0.0f;
  if (x > 80.0f) x = 80.0f;
  const float kf = x * kLog2e;
  int32_t k = static_cast<int32_t>(kf >= 0.0f ? kf + 0.5f : kf - 0.5f);
  const float r = x - static_cast<float>(k) * kLn2;
  const float er = 1.0f + r * (1.0f + r * (0.5f + r * (0.16666666667f +
      r * (0.04166666667f + r * (0.00833333333f + r * 0.00138888889f)))));
  int32_t e = k + 127;
  if (e <= 0) return 0.0f;
  if (e >= 255) e = 254;
  union { uint32_t u; float f; } cvt;
  cvt.u = static_cast<uint32_t>(e) << 23;
  return er * cvt.f;
}

__aicore__ inline float FastLog(float x) {
  if (x <= 0.0f) return -80.0f;
  union { float f; uint32_t u; } v;
  v.f = x;
  int32_t e = static_cast<int32_t>((v.u >> 23) & 0xFFU) - 127;
  v.u = (v.u & 0x007FFFFFU) | 0x3F800000U;
  const float m = v.f;
  const float y = (m - 1.0f) / (m + 1.0f);
  const float y2 = y * y;
  const float poly = y * (1.0f + y2 * (0.33333333333f + y2 * (0.2f + y2 * (0.14285714286f + y2 * 0.11111111111f))));
  return 2.0f * poly + static_cast<float>(e) * kLn2;
}

__aicore__ inline float Sigmoid(float x) {
  return 1.0f / (1.0f + FastExp(-x));
}

__aicore__ inline float FastTanh(float x) {
  if (x > 10.0f) return 1.0f;
  if (x < -10.0f) return -1.0f;
  const float e2 = FastExp(2.0f * x);
  return (e2 - 1.0f) / (e2 + 1.0f);
}

__aicore__ inline float LogSigmoid(float x) {
  if (x > 15.0f) x = 15.0f;
  return -FastLog(1.0f + FastExp(-x));
}

class TirexSlstmCellKernel {
 public:
  __aicore__ inline TirexSlstmCellKernel() = default;

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR recurrentKernel, GM_ADDR bias,
                              GM_ADDR initialState, GM_ADDR output, GM_ADDR finalState,
                              const TirexSlstmCellTiling* tiling) {
    batch_ = tiling->batch;
    seqLen_ = tiling->seq_len;
    hidden_ = tiling->hidden_dim;
    numHeads_ = tiling->num_heads;
    headDim_ = tiling->head_dim;
    inputGm_.SetGlobalBuffer((__gm__ float*)input, batch_ * seqLen_ * 4U * hidden_);
    recurrentGm_.SetGlobalBuffer((__gm__ float*)recurrentKernel, numHeads_ * headDim_ * 4U * headDim_);
    biasGm_.SetGlobalBuffer((__gm__ float*)bias, 4U * hidden_);
    initialStateGm_.SetGlobalBuffer((__gm__ float*)initialState, 4U * batch_ * hidden_);
    outputGm_.SetGlobalBuffer((__gm__ float*)output, batch_ * seqLen_ * hidden_);
    finalStateGm_.SetGlobalBuffer((__gm__ float*)finalState, 4U * batch_ * hidden_);
  }

  __aicore__ inline void Process() {
    const uint32_t blockNum = GetBlockNum() == 0U ? 1U : GetBlockNum();
    for (uint32_t b = GetBlockIdx(); b < batch_; b += blockNum) {
      InitState(b);
      for (uint32_t s = 0; s < seqLen_; ++s) {
        ProcessStep(b, s);
      }
    }
  }

 private:
  __aicore__ inline uint32_t StateIndex(uint32_t state, uint32_t b, uint32_t h) const {
    return (state * batch_ + b) * hidden_ + h;
  }

  __aicore__ inline uint32_t OutputIndex(uint32_t b, uint32_t s, uint32_t h) const {
    return (b * seqLen_ + s) * hidden_ + h;
  }

  __aicore__ inline uint32_t InputIndex(uint32_t b, uint32_t s, uint32_t gate, uint32_t h) const {
    return (b * seqLen_ + s) * (4U * hidden_) + gate * hidden_ + h;
  }

  __aicore__ inline float PrevH(uint32_t b, uint32_t s, uint32_t h) const {
    if (s == 0U) return initialStateGm_.GetValue(StateIndex(0U, b, h));
    return outputGm_.GetValue(OutputIndex(b, s - 1U, h));
  }

  __aicore__ inline void InitState(uint32_t b) {
    for (uint32_t st = 0; st < 4U; ++st) {
      for (uint32_t h = 0; h < hidden_; ++h) {
        finalStateGm_.SetValue(StateIndex(st, b, h), initialStateGm_.GetValue(StateIndex(st, b, h)));
      }
    }
  }

  __aicore__ inline float RecurrentContribution(uint32_t b, uint32_t s, uint32_t gate, uint32_t h) const {
    const uint32_t head = h / headDim_;
    const uint32_t pos = h - head * headDim_;
    float acc = 0.0f;
    const uint32_t rBase = head * headDim_ * (4U * headDim_);
    for (uint32_t j = 0; j < headDim_; ++j) {
      const float prev = PrevH(b, s, head * headDim_ + j);
      const float r = recurrentGm_.GetValue(rBase + j * (4U * headDim_) + gate * headDim_ + pos);
      acc += prev * r;
    }
    return acc;
  }

  __aicore__ inline bool IsInitialNAllZero(uint32_t b) const {
    for (uint32_t h = 0; h < hidden_; ++h) {
      const float n = initialStateGm_.GetValue(StateIndex(2U, b, h));
      if (n != 0.0f) return false;
    }
    return true;
  }

  __aicore__ inline void ProcessStep(uint32_t b, uint32_t s) {
    const bool useIrawForM = (s == 0U) && IsInitialNAllZero(b);
    for (uint32_t h = 0; h < hidden_; ++h) {
      float raw0 = inputGm_.GetValue(InputIndex(b, s, 0U, h)) + biasGm_.GetValue(0U * hidden_ + h);
      float raw1 = inputGm_.GetValue(InputIndex(b, s, 1U, h)) + biasGm_.GetValue(1U * hidden_ + h);
      float raw2 = inputGm_.GetValue(InputIndex(b, s, 2U, h)) + biasGm_.GetValue(2U * hidden_ + h);
      float raw3 = inputGm_.GetValue(InputIndex(b, s, 3U, h)) + biasGm_.GetValue(3U * hidden_ + h);
      const uint32_t head = h / headDim_;
      const uint32_t pos = h - head * headDim_;
      const uint32_t rBase = head * headDim_ * (4U * headDim_);
      for (uint32_t j = 0; j < headDim_; ++j) {
        const float prev = PrevH(b, s, head * headDim_ + j);
        const uint32_t rOff = rBase + j * (4U * headDim_) + pos;
        raw0 += prev * recurrentGm_.GetValue(rOff + 0U * headDim_);
        raw1 += prev * recurrentGm_.GetValue(rOff + 1U * headDim_);
        raw2 += prev * recurrentGm_.GetValue(rOff + 2U * headDim_);
        raw3 += prev * recurrentGm_.GetValue(rOff + 3U * headDim_);
      }
      const float iraw = raw0;
      const float fraw = raw1;
      const float zraw = raw2;
      const float oraw = raw3;

      const float prevC = finalStateGm_.GetValue(StateIndex(1U, b, h));
      const float prevN = finalStateGm_.GetValue(StateIndex(2U, b, h));
      const float prevM = finalStateGm_.GetValue(StateIndex(3U, b, h));
      const float logfplusm = prevM + LogSigmoid(fraw);
      const float mnew = useIrawForM ? iraw : (iraw > logfplusm ? iraw : logfplusm);
      float igate = FastExp(iraw - mnew);
      if (igate > 1.0f) igate = 1.0f;
      float fgate = FastExp(logfplusm - mnew);
      if (fgate > 1.0f) fgate = 1.0f;
      const float ogate = Sigmoid(oraw);
      const float zgate = FastTanh(zraw);
      const float cnew = fgate * prevC + igate * zgate;
      const float nnew = fgate * prevN + igate;
      const float denom = nnew == 0.0f ? kEps : nnew;
      const float hnew = ogate * cnew / denom;
      outputGm_.SetValue(OutputIndex(b, s, h), hnew);
      finalStateGm_.SetValue(StateIndex(0U, b, h), hnew);
      finalStateGm_.SetValue(StateIndex(1U, b, h), cnew);
      finalStateGm_.SetValue(StateIndex(2U, b, h), nnew);
      finalStateGm_.SetValue(StateIndex(3U, b, h), mnew);
    }
  }

  GlobalTensor<float> inputGm_;
  GlobalTensor<float> recurrentGm_;
  GlobalTensor<float> biasGm_;
  GlobalTensor<float> initialStateGm_;
  GlobalTensor<float> outputGm_;
  GlobalTensor<float> finalStateGm_;
  uint32_t batch_ = 0;
  uint32_t seqLen_ = 0;
  uint32_t hidden_ = 0;
  uint32_t numHeads_ = 0;
  uint32_t headDim_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void tirex_slstm_cell(
    GM_ADDR input,
    GM_ADDR recurrentKernel,
    GM_ADDR bias,
    GM_ADDR initialState,
    GM_ADDR output,
    GM_ADDR finalState,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  TirexSlstmCellKernel op;
  op.Init(input, recurrentKernel, bias, initialState, output, finalState,
          reinterpret_cast<const TirexSlstmCellTiling*>(&tilingData));
  op.Process();
}
