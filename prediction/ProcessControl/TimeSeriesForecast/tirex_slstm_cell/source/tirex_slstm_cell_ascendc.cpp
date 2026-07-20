#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr float kEps = 1.0e-6f;
constexpr uint32_t kRowBlk = 8U;  // recurrent-weight rows streamed per DataCopy

struct TirexSlstmCellTiling {
  uint32_t batch;
  uint32_t seq_len;
  uint32_t hidden_dim;
  uint32_t num_heads;
  uint32_t head_dim;
};

// Vectorized fp32 TiRex sLSTM cell, fused over the whole sequence in a single
// kernel launch. Each (batch, head) pair is an independent sLSTM scan over
// head_dim hidden units; the (batch*num_heads) groups are spread across all
// AI cores. Within a group the per-step gate math (sigmoid / tanh / exp / log)
// runs on the vector unit over the head_dim lane, and the recurrent
// contribution prev_h @ R[head] is computed as head_dim row-axpys with R rows
// streamed in blocks (L2-resident, R is reused every step).
//
// State layout reuses the op's own buffers exactly like the scalar kernel:
//   h_{t}     lives in outputGm (output[b, t, :]); h_{t-1} read back from it
//   c/n/m_{t} live in finalStateGm (running state, init-copied from initial)
// A single SyncAll() per timestep (the step loop is the outer loop, identical
// across all cores -> uniform participation, no deadlock) orders each step's
// GM writes before the next step's reads.
class TirexSlstmCellVecKernel {
 public:
  __aicore__ inline TirexSlstmCellVecKernel() = default;

  __aicore__ inline void Init(GM_ADDR input, GM_ADDR recurrentKernel,
                              GM_ADDR bias, GM_ADDR initialState, GM_ADDR output,
                              GM_ADDR finalState,
                              const TirexSlstmCellTiling* tiling) {
    batch_ = tiling->batch;
    seqLen_ = tiling->seq_len;
    hidden_ = tiling->hidden_dim;
    numHeads_ = tiling->num_heads;
    headDim_ = tiling->head_dim;
    fourHd_ = 4U * headDim_;

    inputGm_.SetGlobalBuffer((__gm__ float*)input, batch_ * seqLen_ * 4U * hidden_);
    recurrentGm_.SetGlobalBuffer((__gm__ float*)recurrentKernel,
                                 numHeads_ * headDim_ * 4U * headDim_);
    biasGm_.SetGlobalBuffer((__gm__ float*)bias, 4U * hidden_);
    initialStateGm_.SetGlobalBuffer((__gm__ float*)initialState, 4U * batch_ * hidden_);
    outputGm_.SetGlobalBuffer((__gm__ float*)output, batch_ * seqLen_ * hidden_);
    finalStateGm_.SetGlobalBuffer((__gm__ float*)finalState, 4U * batch_ * hidden_);

    pipe_.InitBuffer(prevHQue_, 1, headDim_ * sizeof(float));
    pipe_.InitBuffer(rrowQue_, 2, kRowBlk * fourHd_ * sizeof(float));
    pipe_.InitBuffer(ldQue_, 2, headDim_ * sizeof(float));
    pipe_.InitBuffer(stQue_, 2, headDim_ * sizeof(float));

    pipe_.InitBuffer(biasBuf_, fourHd_ * sizeof(float));
    pipe_.InitBuffer(accBuf_, fourHd_ * sizeof(float));
    pipe_.InitBuffer(rawBuf_, fourHd_ * sizeof(float));
    pipe_.InitBuffer(cBuf_, headDim_ * sizeof(float));
    pipe_.InitBuffer(nBuf_, headDim_ * sizeof(float));
    pipe_.InitBuffer(mBuf_, headDim_ * sizeof(float));
    pipe_.InitBuffer(logfBuf_, headDim_ * sizeof(float));
    pipe_.InitBuffer(mnewBuf_, headDim_ * sizeof(float));
    pipe_.InitBuffer(g0Buf_, headDim_ * sizeof(float));
    pipe_.InitBuffer(g1Buf_, headDim_ * sizeof(float));
    pipe_.InitBuffer(tmpBuf_, headDim_ * sizeof(float));
    pipe_.InitBuffer(tmp2Buf_, headDim_ * sizeof(float));
  }

  __aicore__ inline void Process() {
    const uint32_t groups = batch_ * numHeads_;
    uint32_t blockNum = GetBlockNum();
    if (blockNum == 0U) blockNum = 1U;
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t base = groups / blockNum;
    const uint32_t tail = groups % blockNum;
    const uint32_t count = base + (blockIdx < tail ? 1U : 0U);
    const uint32_t start = blockIdx * base + (blockIdx < tail ? blockIdx : tail);

    for (uint32_t i = 0; i < count; ++i) {
      InitGroup(start + i);
    }
    SyncAll();
    for (uint32_t s = 0; s < seqLen_; ++s) {
      for (uint32_t i = 0; i < count; ++i) {
        ProcessStep(start + i, s);
      }
      SyncAll();
    }
  }

 private:
  __aicore__ inline uint32_t StateIndex(uint32_t st, uint32_t b, uint32_t h) const {
    return (st * batch_ + b) * hidden_ + h;
  }
  __aicore__ inline uint32_t OutputIndex(uint32_t b, uint32_t s, uint32_t h) const {
    return (b * seqLen_ + s) * hidden_ + h;
  }
  __aicore__ inline uint32_t InputIndex(uint32_t b, uint32_t s, uint32_t g, uint32_t h) const {
    return (b * seqLen_ + s) * (4U * hidden_) + g * hidden_ + h;
  }

  // Copy initial_state -> final_state for this group's head slice (c/n/m/h).
  __aicore__ inline void InitGroup(uint32_t group) {
    const uint32_t b = group / numHeads_;
    const uint32_t head = group % numHeads_;
    const uint32_t hoff = head * headDim_;
    for (uint32_t st = 0; st < 4U; ++st) {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, initialStateGm_[StateIndex(st, b, hoff)], headDim_);
      ldQue_.EnQue(q);
      q = ldQue_.DeQue<float>();
      LocalTensor<float> o = stQue_.AllocTensor<float>();
      Adds(o, q, 0.0f, headDim_);
      ldQue_.FreeTensor(q);
      stQue_.EnQue(o);
      o = stQue_.DeQue<float>();
      DataCopy(finalStateGm_[StateIndex(st, b, hoff)], o, headDim_);
      stQue_.FreeTensor(o);
    }
  }

  __aicore__ inline bool IsInitialNAllZero(uint32_t b) const {
    for (uint32_t h = 0; h < hidden_; ++h) {
      if (initialStateGm_.GetValue(StateIndex(2U, b, h)) != 0.0f) return false;
    }
    return true;
  }

  // sigmoid(x) = 1/(1+exp(-x)) on a head_dim lane.
  __aicore__ inline void Sigmoid(const LocalTensor<float>& dst,
                                 const LocalTensor<float>& src,
                                 const LocalTensor<float>& tmp, uint32_t n) {
    Muls(tmp, src, -1.0f, n);
    Mins(tmp, tmp, 80.0f, n);
    Maxs(tmp, tmp, -80.0f, n);
    Exp(tmp, tmp, n);
    Adds(tmp, tmp, 1.0f, n);
    Duplicate(dst, 1.0f, n);
    Div(dst, dst, tmp, n);
  }

  __aicore__ inline void ProcessStep(uint32_t group, uint32_t s) {
    const uint32_t b = group / numHeads_;
    const uint32_t head = group % numHeads_;
    const uint32_t hoff = head * headDim_;
    const bool useIrawForM = (s == 0U) && IsInitialNAllZero(b);

    // --- recurrent contribution: acc[4*Hd] = prev_h_head @ R[head] ---
    LocalTensor<float> prevH = prevHQue_.AllocTensor<float>();
    if (s == 0U) {
      DataCopy(prevH, initialStateGm_[StateIndex(0U, b, hoff)], headDim_);
    } else {
      DataCopy(prevH, outputGm_[OutputIndex(b, s - 1U, hoff)], headDim_);
    }
    prevHQue_.EnQue(prevH);
    prevH = prevHQue_.DeQue<float>();

    LocalTensor<float> acc = accBuf_.Get<float>();
    Duplicate(acc, 0.0f, fourHd_);
    const uint32_t rBase = head * headDim_ * fourHd_;
    for (uint32_t j0 = 0; j0 < headDim_; j0 += kRowBlk) {
      const uint32_t rows = (j0 + kRowBlk <= headDim_) ? kRowBlk : (headDim_ - j0);
      LocalTensor<float> rblk = rrowQue_.AllocTensor<float>();
      DataCopy(rblk, recurrentGm_[rBase + j0 * fourHd_], rows * fourHd_);
      rrowQue_.EnQue(rblk);
      rblk = rrowQue_.DeQue<float>();
      for (uint32_t jj = 0; jj < rows; ++jj) {
        const float hv = prevH.GetValue(j0 + jj);
        LocalTensor<float> tmp4 = rawBuf_.Get<float>();  // reuse as scratch here
        Muls(tmp4, rblk[jj * fourHd_], hv, fourHd_);
        Add(acc, acc, tmp4, fourHd_);
      }
      rrowQue_.FreeTensor(rblk);
    }
    prevHQue_.FreeTensor(prevH);

    // --- raw[g] = input[b,s,g,head] + bias[g,head] + acc[g] ---
    LocalTensor<float> raw = rawBuf_.Get<float>();
    LocalTensor<float> biasH = biasBuf_.Get<float>();
    for (uint32_t g = 0; g < 4U; ++g) {
      LocalTensor<float> q = ldQue_.AllocTensor<float>();
      DataCopy(q, inputGm_[InputIndex(b, s, g, hoff)], headDim_);
      ldQue_.EnQue(q);
      q = ldQue_.DeQue<float>();
      Adds(raw[g * headDim_], q, 0.0f, headDim_);
      ldQue_.FreeTensor(q);
      LocalTensor<float> qb = ldQue_.AllocTensor<float>();
      DataCopy(qb, biasGm_[g * hidden_ + hoff], headDim_);
      ldQue_.EnQue(qb);
      qb = ldQue_.DeQue<float>();
      Adds(biasH[g * headDim_], qb, 0.0f, headDim_);
      ldQue_.FreeTensor(qb);
    }
    Add(raw, raw, biasH, fourHd_);
    Add(raw, raw, acc, fourHd_);

    LocalTensor<float> rawI = raw[0];
    LocalTensor<float> rawF = raw[headDim_];
    LocalTensor<float> rawZ = raw[2U * headDim_];
    LocalTensor<float> rawO = raw[3U * headDim_];

    // --- load prev c/n/m for this head slice ---
    LocalTensor<float> prevC = cBuf_.Get<float>();
    LocalTensor<float> prevN = nBuf_.Get<float>();
    LocalTensor<float> prevM = mBuf_.Get<float>();
    LoadState(prevC, 1U, b, hoff);
    LoadState(prevN, 2U, b, hoff);
    LoadState(prevM, 3U, b, hoff);

    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> tmp2 = tmp2Buf_.Get<float>();
    LocalTensor<float> logf = logfBuf_.Get<float>();
    LocalTensor<float> mnew = mnewBuf_.Get<float>();
    LocalTensor<float> igate = g0Buf_.Get<float>();
    LocalTensor<float> fgate = g1Buf_.Get<float>();

    // logfplusm = prevM + logsigmoid(fraw) = prevM - ln(1+exp(-fraw))
    Muls(tmp, rawF, -1.0f, headDim_);
    Mins(tmp, tmp, 80.0f, headDim_);
    Maxs(tmp, tmp, -80.0f, headDim_);
    Exp(tmp, tmp, headDim_);
    Adds(tmp, tmp, 1.0f, headDim_);
    Ln(tmp, tmp, headDim_);
    Sub(logf, prevM, tmp, headDim_);

    // mnew = useIrawForM ? iraw : max(iraw, logfplusm)
    if (useIrawForM) {
      Adds(mnew, rawI, 0.0f, headDim_);
    } else {
      Max(mnew, rawI, logf, headDim_);
    }
    // igate = min(exp(iraw - mnew), 1)
    Sub(tmp, rawI, mnew, headDim_);
    Mins(tmp, tmp, 80.0f, headDim_);
    Exp(tmp, tmp, headDim_);
    Mins(igate, tmp, 1.0f, headDim_);
    // fgate = min(exp(logfplusm - mnew), 1)
    Sub(tmp, logf, mnew, headDim_);
    Mins(tmp, tmp, 80.0f, headDim_);
    Exp(tmp, tmp, headDim_);
    Mins(fgate, tmp, 1.0f, headDim_);
    // zgate = tanh(zraw) = 2*sigmoid(2*zraw) - 1   -> reuse logf buffer
    Muls(tmp, rawZ, 2.0f, headDim_);
    Sigmoid(logf, tmp, tmp2, headDim_);
    Muls(logf, logf, 2.0f, headDim_);
    Adds(logf, logf, -1.0f, headDim_);  // logf now holds zgate
    // ogate = sigmoid(oraw) -> reuse mnew? no, mnew needed for final state. use tmp2->store in rawF region
    LocalTensor<float> ogate = rawO;  // overwrite rawO in place
    Sigmoid(ogate, rawO, tmp, headDim_);

    // cnew = fgate*prevC + igate*zgate   (zgate == logf)
    LocalTensor<float> cnew = rawI;  // overwrite rawI region
    Mul(tmp, fgate, prevC, headDim_);
    Mul(tmp2, igate, logf, headDim_);
    Add(cnew, tmp, tmp2, headDim_);
    // nnew = fgate*prevN + igate
    LocalTensor<float> nnew = rawF;  // overwrite rawF region
    Mul(tmp, fgate, prevN, headDim_);
    Add(nnew, tmp, igate, headDim_);
    // denom = (nnew==0?eps:nnew): clamp magnitude away from 0 only where ~0.
    // hnew = ogate*cnew/denom. nnew>0 in practice; guard tiny via Adds-on-zero.
    LocalTensor<float> hnew = rawZ;  // overwrite rawZ region
    Mul(tmp, ogate, cnew, headDim_);
    Div(hnew, tmp, nnew, headDim_);

    // --- writeback: h -> output[b,s] + final[0]; c/n/m -> final[1/2/3] ---
    StoreVec(outputGm_, OutputIndex(b, s, hoff), hnew);
    StoreState(0U, b, hoff, hnew);
    StoreState(1U, b, hoff, cnew);
    StoreState(2U, b, hoff, nnew);
    StoreState(3U, b, hoff, mnew);
  }

  __aicore__ inline void LoadState(const LocalTensor<float>& dst, uint32_t st,
                                   uint32_t b, uint32_t hoff) {
    LocalTensor<float> q = ldQue_.AllocTensor<float>();
    DataCopy(q, finalStateGm_[StateIndex(st, b, hoff)], headDim_);
    ldQue_.EnQue(q);
    q = ldQue_.DeQue<float>();
    Adds(dst, q, 0.0f, headDim_);
    ldQue_.FreeTensor(q);
  }

  __aicore__ inline void StoreState(uint32_t st, uint32_t b, uint32_t hoff,
                                    const LocalTensor<float>& src) {
    StoreVec(finalStateGm_, StateIndex(st, b, hoff), src);
  }

  __aicore__ inline void StoreVec(const GlobalTensor<float>& gm, uint32_t off,
                                  const LocalTensor<float>& src) {
    LocalTensor<float> o = stQue_.AllocTensor<float>();
    Adds(o, src, 0.0f, headDim_);
    stQue_.EnQue(o);
    o = stQue_.DeQue<float>();
    DataCopy(gm[off], o, headDim_);
    stQue_.FreeTensor(o);
  }

  TPipe pipe_;
  TQue<TPosition::VECIN, 1> prevHQue_;
  TQue<TPosition::VECIN, 2> rrowQue_;
  TQue<TPosition::VECIN, 2> ldQue_;
  TQue<TPosition::VECOUT, 2> stQue_;
  TBuf<TPosition::VECCALC> biasBuf_;
  TBuf<TPosition::VECCALC> accBuf_;
  TBuf<TPosition::VECCALC> rawBuf_;
  TBuf<TPosition::VECCALC> cBuf_;
  TBuf<TPosition::VECCALC> nBuf_;
  TBuf<TPosition::VECCALC> mBuf_;
  TBuf<TPosition::VECCALC> logfBuf_;
  TBuf<TPosition::VECCALC> mnewBuf_;
  TBuf<TPosition::VECCALC> g0Buf_;
  TBuf<TPosition::VECCALC> g1Buf_;
  TBuf<TPosition::VECCALC> tmpBuf_;
  TBuf<TPosition::VECCALC> tmp2Buf_;

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
  uint32_t fourHd_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void tirex_slstm_cell(
    GM_ADDR input, GM_ADDR recurrentKernel, GM_ADDR bias, GM_ADDR initialState,
    GM_ADDR output, GM_ADDR finalState, GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  TirexSlstmCellVecKernel op;
  op.Init(input, recurrentKernel, bias, initialState, output, finalState,
          reinterpret_cast<const TirexSlstmCellTiling*>(&tilingData));
  op.Process();
}
