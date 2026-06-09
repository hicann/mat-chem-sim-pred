#ifndef FNO_KERNEL_H
#define FNO_KERNEL_H

#include "kernel_operator.h"
#include "pde_math.h"
#include <cmath>

using namespace AscendC;

struct FnoTilingData {
    int32_t batchSize;
    int32_t gridDim;
    int32_t inChannels;
    int32_t outChannels;
    int32_t hiddenChannels;
    int32_t modes;
    int32_t tileSize;
    int32_t coreNum;
    int32_t totalWeightSize;
};

constexpr int32_t FNO_MAX_GRID = 1024;
constexpr int32_t FNO_MAX_CHANNELS = 64;
constexpr int32_t FNO_BLOCK_SIZE = 32;
constexpr int32_t FNO_MAX_CORES = 32;
// Uses pde_math::FNO_PI from pde_math.h

__aicore__ inline uint32_t FnoAlignUp(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

class FnoOp {
public:
    __aicore__ inline FnoOp() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weights, GM_ADDR output,
                                 GM_ADDR tiling) {
        __gm__ const FnoTilingData* td =
            reinterpret_cast<__gm__ const FnoTilingData*>(tiling);

        batchSize_ = td->batchSize;
        gridDim_ = td->gridDim;
        inChannels_ = td->inChannels;
        outChannels_ = td->outChannels;
        hiddenChannels_ = td->hiddenChannels;
        modes_ = td->modes;
        tileSize_ = td->tileSize;
        coreNum_ = td->coreNum;

        coreIdx_ = GetBlockIdx();

        int32_t gridsPerCore = (batchSize_ + coreNum_ - 1) / coreNum_;
        myStart_ = coreIdx_ * gridsPerCore;
        myEnd_ = myStart_ + gridsPerCore;
        if (myEnd_ > batchSize_) myEnd_ = batchSize_;
        if (myStart_ >= batchSize_) {
            myStart_ = 0;
            myEnd_ = 0;
        }
        myCount_ = myEnd_ - myStart_;

        input_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input));
        weights_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weights));
        output_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));

        uint32_t gridBufSize = FnoAlignUp(gridDim_ * FNO_MAX_CHANNELS * sizeof(float), FNO_BLOCK_SIZE);
        uint32_t freqBufSize = FnoAlignUp(FNO_MAX_GRID * FNO_MAX_CHANNELS * 2 * sizeof(float), FNO_BLOCK_SIZE);

        pipe_.InitBuffer(inputQueue_, 1, gridBufSize);
        pipe_.InitBuffer(freqQueue_, 1, freqBufSize);
        pipe_.InitBuffer(outputQueue_, 1, gridBufSize);

        ComputeWeightOffsets();
    }

    __aicore__ inline void ComputeWeightOffsets() {
        liftOffset_ = 0;
        int32_t liftSize = inChannels_ * hiddenChannels_ + hiddenChannels_;
        spectralOffset_ = liftOffset_ + liftSize;
        int32_t spectralSize = modes_ * hiddenChannels_ * hiddenChannels_ * 2;
        projectOffset_ = spectralOffset_ + spectralSize;
    }

    __aicore__ inline void LiftLayer(const float* x, float* h, int32_t batchIdx) {
        for (int32_t j = 0; j < hiddenChannels_; j++) {
            float sum = 0.0f;
            int32_t wBase = liftOffset_ + j * inChannels_;
            for (int32_t k = 0; k < inChannels_; k++) {
                int32_t idx = (batchIdx * gridDim_) * inChannels_ + k * gridDim_;
                sum += x[idx] * weights_gm_.GetValue(wBase + k);
            }
            float bias = weights_gm_.GetValue(liftOffset_ + inChannels_ * hiddenChannels_ + j);
            h[j * gridDim_] = sum + bias;
        }
    }

    __aicore__ inline void SimpleDFT1D(const float* signal, float* re, float* im,
                                         int32_t ch, int32_t n) {
        for (int32_t k = 0; k < n; k++) {
            float sumRe = 0.0f, sumIm = 0.0f;
            for (int32_t i = 0; i < n; i++) {
                float angle = -2.0f * pde_math::FNO_PI * k * i / n;
                float val = signal[ch * n + i];
                sumRe += val * pde_math::pde_cosf(angle);
                sumIm += val * pde_math::pde_sinf(angle);
            }
            re[k] = sumRe;
            im[k] = sumIm;
        }
    }

    __aicore__ inline void SimpleIDFT1D(const float* re, const float* im,
                                          float* signal, int32_t ch, int32_t n) {
        for (int32_t i = 0; i < n; i++) {
            float sumRe = 0.0f, sumIm = 0.0f;
            for (int32_t k = 0; k < n; k++) {
                float angle = 2.0f * pde_math::FNO_PI * k * i / n;
                sumRe += re[k] * pde_math::pde_cosf(angle) - im[k] * pde_math::pde_sinf(angle);
                sumIm += re[k] * pde_math::pde_sinf(angle) + im[k] * pde_math::pde_cosf(angle);
            }
            signal[ch * n + i] = sumRe / n;
        }
    }

    __aicore__ inline void SpectralMultiply(float* re, float* im, int32_t n) {
        int32_t wBase = spectralOffset_;
        int32_t m = (modes_ < n / 2 + 1) ? modes_ : n / 2 + 1;

        for (int32_t chOut = 0; chOut < hiddenChannels_; chOut++) {
            for (int32_t chIn = 0; chIn < hiddenChannels_; chIn++) {
                for (int32_t k = 0; k < m; k++) {
                    int32_t wIdx = wBase + (chOut * hiddenChannels_ + chIn) * modes_ * 2 + k * 2;
                    float Wre = weights_gm_.GetValue(wIdx);
                    float Wim = weights_gm_.GetValue(wIdx + 1);
                    float inRe = re[chIn * n + k];
                    float inIm = im[chIn * n + k];
                    re[chOut * n + k] += Wre * inRe - Wim * inIm;
                    im[chOut * n + k] += Wre * inIm + Wim * inRe;
                }
            }
        }
    }

    __aicore__ inline void ProjectLayer(const float* h, float* out, int32_t batchIdx) {
        int32_t wBase = projectOffset_;
        for (int32_t j = 0; j < outChannels_; j++) {
            float sum = 0.0f;
            int32_t wOff = wBase + j * hiddenChannels_;
            for (int32_t k = 0; k < hiddenChannels_; k++) {
                sum += h[k * gridDim_] * weights_gm_.GetValue(wOff + k);
            }
            float bias = weights_gm_.GetValue(wBase + hiddenChannels_ * outChannels_ + j);
            int32_t outIdx = (batchIdx * gridDim_) * outChannels_ + j * gridDim_;
            out[outIdx] = sum + bias;
        }
    }

    __aicore__ inline void Process() {
        if (myCount_ <= 0) return;

        LocalTensor<float> freqRe = freqQueue_.AllocTensor<float>();
        LocalTensor<float> freqIm = freqQueue_.AllocTensor<float>();
        LocalTensor<float> work = inputQueue_.AllocTensor<float>();

        for (int32_t b = 0; b < myCount_; b++) {
            int32_t batchIdx = myStart_ + b;

            float h[FNO_MAX_CHANNELS * FNO_MAX_GRID];
            for (int32_t c = 0; c < hiddenChannels_; c++)
                for (int32_t i = 0; i < gridDim_; i++)
                    h[c * gridDim_ + i] = 0.0f;

            for (int32_t c = 0; c < hiddenChannels_; c++) {
                float sum = 0.0f;
                int32_t wBase = liftOffset_ + c * inChannels_;
                for (int32_t k = 0; k < inChannels_; k++) {
                    float val = input_gm_.GetValue(
                        (batchIdx * gridDim_) * inChannels_ + k * gridDim_);
                    sum += val * weights_gm_.GetValue(wBase + k);
                }
                float bias = weights_gm_.GetValue(
                    liftOffset_ + inChannels_ * hiddenChannels_ + c);
                h[c * gridDim_] = sum + bias;
            }

            float fftRe[FNO_MAX_CHANNELS * FNO_MAX_GRID];
            float fftIm[FNO_MAX_CHANNELS * FNO_MAX_GRID];

            for (int32_t c = 0; c < hiddenChannels_; c++) {
                SimpleDFT1D(h, &fftRe[c * gridDim_], &fftIm[c * gridDim_], 0, gridDim_);
            }

            for (int32_t i = 0; i < hiddenChannels_ * gridDim_; i++) {
                fftRe[i] = 0.0f;
                fftIm[i] = 0.0f;
            }

            SpectralMultiply(fftRe, fftIm, gridDim_);

            for (int32_t c = 0; c < hiddenChannels_; c++) {
                SimpleIDFT1D(&fftRe[c * gridDim_], &fftIm[c * gridDim_],
                              h, 0, gridDim_);
            }

            for (int32_t c = 0; c < outChannels_; c++) {
                float sum = 0.0f;
                int32_t wBase = projectOffset_ + c * hiddenChannels_;
                for (int32_t k = 0; k < hiddenChannels_; k++) {
                    sum += h[k * gridDim_] * weights_gm_.GetValue(wBase + k);
                }
                float bias = weights_gm_.GetValue(
                    projectOffset_ + hiddenChannels_ * outChannels_ + c);
                output_gm_.SetValue(
                    (batchIdx * gridDim_) * outChannels_ + c * gridDim_, sum + bias);
            }
        }

        freqQueue_.FreeTensor(freqRe);
        freqQueue_.FreeTensor(freqIm);
        inputQueue_.FreeTensor(work);
    }

private:
    GlobalTensor<float> input_gm_;
    GlobalTensor<float> weights_gm_;
    GlobalTensor<float> output_gm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TQue<QuePosition::VECOUT, 1> freqQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;

    int32_t batchSize_;
    int32_t gridDim_;
    int32_t inChannels_;
    int32_t outChannels_;
    int32_t hiddenChannels_;
    int32_t modes_;
    int32_t tileSize_;
    int32_t coreNum_;
    int32_t coreIdx_;
    int32_t myStart_;
    int32_t myEnd_;
    int32_t myCount_;

    int32_t liftOffset_;
    int32_t spectralOffset_;
    int32_t projectOffset_;
};

#endif
