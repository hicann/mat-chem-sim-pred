#include "kernel_operator.h"

using namespace AscendC;

namespace {

struct S6scanFusedTiling {
  uint32_t batch;
  uint32_t length;
  uint32_t in_size;
  uint32_t hidden;
};

class S6scanFusedKernel {
 public:
  __aicore__ inline S6scanFusedKernel() = default;

  __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
                              GM_ADDR output, const S6scanFusedTiling* t) {
    B_ = t->batch; L_ = t->length; IN_ = t->in_size; H_ = t->hidden;
    hAligned_ = ((H_ + 7) / 8) * 8;
    inAligned_ = ((IN_ + 7) / 8) * 8;
    wSize_ = (uint64_t)IN_ * 3 * H_;

    xGm_.SetGlobalBuffer((__gm__ float*)x, (uint64_t)B_ * L_ * IN_);
    wGm_.SetGlobalBuffer((__gm__ float*)weight, wSize_);
    bGm_.SetGlobalBuffer((__gm__ float*)bias, (uint64_t)4 * H_);
    outGm_.SetGlobalBuffer((__gm__ float*)output, (uint64_t)B_ * L_ * H_);

    pipe_.InitBuffer(ldQue_, 2, inAligned_ * sizeof(float));
    pipe_.InitBuffer(stQue_, 2, hAligned_ * sizeof(float));
    pipe_.InitBuffer(hBuf_, hAligned_ * sizeof(float));
    pipe_.InitBuffer(aBuf_, hAligned_ * sizeof(float));
    pipe_.InitBuffer(bBuf_, hAligned_ * sizeof(float));
    pipe_.InitBuffer(cBuf_, hAligned_ * sizeof(float));
    pipe_.InitBuffer(accBuf_, hAligned_ * sizeof(float));
    pipe_.InitBuffer(tmpBuf_, hAligned_ * sizeof(float));
    pipe_.InitBuffer(wBuf_, wSize_ * sizeof(float));
    pipe_.InitBuffer(biasBuf_, (uint64_t)4 * hAligned_ * sizeof(float));
  }

  __aicore__ inline void Process() {
    uint32_t blocks = GetBlockNum();
    uint32_t bid = GetBlockIdx();
    uint32_t perBlock = (B_ + blocks - 1) / blocks;
    uint32_t start = bid * perBlock;
    if (start >= B_) return;
    uint32_t count = ((start + perBlock) > B_) ? (B_ - start) : perBlock;

    LocalTensor<float> wLocal = wBuf_.Get<float>();
    LocalTensor<float> biasLocal = biasBuf_.Get<float>();
    uint64_t wTotal = wSize_;
    uint64_t copied = 0;
    while (copied < wTotal) {
      uint64_t chunk = wTotal - copied;
      if (chunk > 65504) chunk = 65504;
      uint64_t ca = ((chunk + 7) / 8) * 8;
      DataCopy(wLocal[copied], wGm_[copied], (uint32_t)ca);
      pipe_barrier(PIPE_MTE2);
      copied += chunk;
    }
    uint32_t biasTotal = 4 * hAligned_;
    DataCopy(biasLocal, bGm_[0], biasTotal);
    pipe_barrier(PIPE_MTE2);

    LocalTensor<float> hLocal = hBuf_.Get<float>();
    LocalTensor<float> aLocal = aBuf_.Get<float>();
    LocalTensor<float> bLocal = bBuf_.Get<float>();
    LocalTensor<float> cLocal = cBuf_.Get<float>();
    LocalTensor<float> accLocal = accBuf_.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();

    for (uint32_t idx = 0; idx < count; ++idx) {
      uint32_t b = start + idx;
      Duplicate(hLocal, 0.0f, hAligned_);
      pipe_barrier(PIPE_V);

      for (uint32_t s = 0; s < L_; ++s) {
        LocalTensor<float> xT = ldQue_.AllocTensor<float>();
        DataCopy(xT, xGm_[((uint64_t)b * L_ + s) * IN_], inAligned_);
        ldQue_.EnQue(xT);
        xT = ldQue_.DeQue<float>();

        // delta = softplus(Wd*x + bd) = log(1 + exp(Wd*x + bd))
        DataCopy(accLocal, biasLocal, hAligned_);
        pipe_barrier(PIPE_V);
        for (uint32_t j = 0; j < IN_; ++j) {
          float xval = xT.GetValue(j);
          Muls(tmpLocal, wLocal[(uint64_t)j * H_], xval, hAligned_);
          pipe_barrier(PIPE_V);
          Add(accLocal, accLocal, tmpLocal, hAligned_);
          pipe_barrier(PIPE_V);
        }
        // softplus: log(1+exp(x)) ~ use Exp then Adds 1 then Ln
        Exp(tmpLocal, accLocal, hAligned_);
        pipe_barrier(PIPE_V);
        Adds(tmpLocal, tmpLocal, 1.0f, hAligned_);
        pipe_barrier(PIPE_V);
        Ln(aLocal, tmpLocal, hAligned_);  // aLocal = delta (softplus result)
        pipe_barrier(PIPE_V);

        // A_disc = exp(-delta * a_param)
        // a_param stored at biasLocal[3*hAligned_]
        Mul(tmpLocal, aLocal, biasLocal[3 * hAligned_], hAligned_);
        pipe_barrier(PIPE_V);
        Muls(tmpLocal, tmpLocal, -1.0f, hAligned_);
        pipe_barrier(PIPE_V);
        Exp(aLocal, tmpLocal, hAligned_);  // aLocal = A_disc
        pipe_barrier(PIPE_V);

        // B = sigmoid(Wb*x + bb)
        DataCopy(accLocal, biasLocal[hAligned_], hAligned_);
        pipe_barrier(PIPE_V);
        uint64_t wb_off = (uint64_t)IN_ * H_;
        for (uint32_t j = 0; j < IN_; ++j) {
          float xval = xT.GetValue(j);
          Muls(tmpLocal, wLocal[wb_off + (uint64_t)j * H_], xval, hAligned_);
          pipe_barrier(PIPE_V);
          Add(accLocal, accLocal, tmpLocal, hAligned_);
          pipe_barrier(PIPE_V);
        }
        Sigmoid(bLocal, accLocal, hAligned_);
        pipe_barrier(PIPE_V);

        // x_proj = Wx*x + bx
        DataCopy(cLocal, biasLocal[2 * hAligned_], hAligned_);
        pipe_barrier(PIPE_V);
        uint64_t wx_off = (uint64_t)IN_ * 2 * H_;
        for (uint32_t j = 0; j < IN_; ++j) {
          float xval = xT.GetValue(j);
          Muls(tmpLocal, wLocal[wx_off + (uint64_t)j * H_], xval, hAligned_);
          pipe_barrier(PIPE_V);
          Add(cLocal, cLocal, tmpLocal, hAligned_);
          pipe_barrier(PIPE_V);
        }
        pipe_barrier(PIPE_V);

        ldQue_.FreeTensor(xT);

        // h = A_disc * h + B * x_proj
        Mul(hLocal, aLocal, hLocal, hAligned_);
        pipe_barrier(PIPE_V);
        Mul(tmpLocal, bLocal, cLocal, hAligned_);
        pipe_barrier(PIPE_V);
        Add(hLocal, hLocal, tmpLocal, hAligned_);
        pipe_barrier(PIPE_V);

        LocalTensor<float> stT = stQue_.AllocTensor<float>();
        DataCopy(stT, hLocal, hAligned_);
        stQue_.EnQue(stT);
        stT = stQue_.DeQue<float>();
        DataCopy(outGm_[((uint64_t)b * L_ + s) * H_], stT, hAligned_);
        stQue_.FreeTensor(stT);
        pipe_barrier(PIPE_MTE3);
      }
    }
  }

 private:
  TPipe pipe_;
  TQue<QuePosition::VECIN, 2> ldQue_;
  TQue<QuePosition::VECOUT, 2> stQue_;
  TBuf<TPosition::VECCALC> hBuf_, aBuf_, bBuf_, cBuf_, accBuf_, tmpBuf_;
  TBuf<TPosition::VECCALC> wBuf_, biasBuf_;
  GlobalTensor<float> xGm_, wGm_, bGm_, outGm_;
  uint32_t B_, L_, IN_, H_;
  uint32_t hAligned_, inAligned_;
  uint64_t wSize_;
};

}  // namespace

extern "C" __global__ __aicore__ void s6scan_fused(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR output, GM_ADDR workspace,
    GM_ADDR tiling) {
  (void)workspace;
  GET_TILING_DATA(tilingData, tiling);
  S6scanFusedKernel op;
  op.Init(x, weight, bias, output,
          reinterpret_cast<const S6scanFusedTiling*>(&tilingData));
  op.Process();
}
