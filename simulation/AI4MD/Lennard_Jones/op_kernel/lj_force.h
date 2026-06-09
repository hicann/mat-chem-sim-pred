#ifndef LJ_FORCE_H
#define LJ_FORCE_H

#include "kernel_operator.h"

using namespace AscendC;

struct LJForceTilingData {
    int32_t numAtoms;
    int32_t tileSize;
    int32_t coreNum;
    float epsilon;
    float sigma;
    float cutoff;
    float cutoffSq;
    float sigma6;
    float sigma12;
    float eps4;
    float eps24;
};

constexpr int32_t LJ_BLOCK_SIZE = 32;
constexpr int32_t LJ_FLOAT_PER_BLOCK = 8;
constexpr int32_t LJ_MAX_CORES = 32;

__aicore__ inline uint32_t LjAlignUp(uint32_t x, uint32_t align) {
    return ((x + align - 1) / align) * align;
}

class LJForceOp {
public:
    __aicore__ inline LJForceOp() {}

    __aicore__ inline void Init(GM_ADDR positions, GM_ADDR forces, GM_ADDR energy,
                                 GM_ADDR tiling) {
        __gm__ const LJForceTilingData* tilingData =
            reinterpret_cast<__gm__ const LJForceTilingData*>(tiling);

        numAtoms_ = tilingData->numAtoms;
        tileSize_ = tilingData->tileSize;
        coreNum_ = tilingData->coreNum;
        cutoffSq_ = tilingData->cutoffSq;
        sigma6_ = tilingData->sigma6;
        sigma12_ = tilingData->sigma12;
        eps4_ = tilingData->eps4;
        eps24_ = tilingData->eps24;

        coreIdx_ = GetBlockIdx();

        int32_t atomsPerCore = (numAtoms_ + coreNum_ - 1) / coreNum_;
        myStart_ = coreIdx_ * atomsPerCore;
        myEnd_ = myStart_ + atomsPerCore;
        if (myEnd_ > numAtoms_) myEnd_ = numAtoms_;
        if (myStart_ >= numAtoms_) {
            myStart_ = 0;
            myEnd_ = 0;
        }

        myAtoms_ = myEnd_ - myStart_;

        positions_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(positions));
        forces_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(forces));
        energy_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(energy));

        uint32_t tileBufSize = LjAlignUp(tileSize_ * 3 * sizeof(float), LJ_BLOCK_SIZE);
        uint32_t forceSize = LjAlignUp(myAtoms_ * 3 * sizeof(float), LJ_BLOCK_SIZE);
        if (forceSize < LJ_BLOCK_SIZE) forceSize = LJ_BLOCK_SIZE;
        uint32_t energySize = LjAlignUp(LJ_FLOAT_PER_BLOCK * sizeof(float), LJ_BLOCK_SIZE);

        pipe_.InitBuffer(posQueue_, 1, tileBufSize);
        pipe_.InitBuffer(forceQueue_, 1, forceSize);
        pipe_.InitBuffer(energyQueue_, 1, energySize);
    }

    __aicore__ inline void Process() {
        if (myAtoms_ <= 0) return;

        int32_t numTiles = (numAtoms_ + tileSize_ - 1) / tileSize_;

        LocalTensor<float> forceLocal = forceQueue_.AllocTensor<float>();
        Duplicate(forceLocal, 0.0f, myAtoms_ * 3);

        LocalTensor<float> energyLocal = energyQueue_.AllocTensor<float>();
        Duplicate(energyLocal, 0.0f, LJ_FLOAT_PER_BLOCK);

        float totalEnergy = 0.0f;

        for (int32_t i = 0; i < myAtoms_; i++) {
            int32_t atomI = myStart_ + i;

            float xi = positions_gm_.GetValue(atomI * 3);
            float yi = positions_gm_.GetValue(atomI * 3 + 1);
            float zi = positions_gm_.GetValue(atomI * 3 + 2);

            float fx = 0.0f, fy = 0.0f, fz = 0.0f;

            for (int32_t t = 0; t < numTiles; t++) {
                int32_t jStart = t * tileSize_;
                int32_t jEnd = jStart + tileSize_;
                if (jEnd > numAtoms_) jEnd = numAtoms_;
                int32_t tileCount = jEnd - jStart;

                LocalTensor<float> tilePos = posQueue_.AllocTensor<float>();
                DataCopy(tilePos, positions_gm_[jStart * 3], tileCount * 3);
                posQueue_.EnQue(tilePos);
                tilePos = posQueue_.DeQue<float>();

                for (int32_t j = jStart; j < jEnd; j++) {
                    if (j == atomI) continue;
                    int32_t jj = j - jStart;

                    float dx = xi - tilePos.GetValue(jj * 3);
                    float dy = yi - tilePos.GetValue(jj * 3 + 1);
                    float dz = zi - tilePos.GetValue(jj * 3 + 2);
                    float r2 = dx * dx + dy * dy + dz * dz;

                    if (r2 < cutoffSq_ && r2 > 1e-10f) {
                        float r2inv = 1.0f / r2;
                        float r6inv = r2inv * r2inv * r2inv;
                        float s6r6 = sigma6_ * r6inv;
                        float s12r12 = s6r6 * s6r6;

                        if (atomI < j) {
                            totalEnergy += eps4_ * (s12r12 - s6r6);
                        }

                        float fscalar = eps24_ * r2inv * (2.0f * s12r12 - s6r6);
                        fx += fscalar * dx;
                        fy += fscalar * dy;
                        fz += fscalar * dz;
                    }
                }

                posQueue_.FreeTensor(tilePos);
            }

            int32_t idx = i * 3;
            forceLocal.SetValue(idx, fx);
            forceLocal.SetValue(idx + 1, fy);
            forceLocal.SetValue(idx + 2, fz);
        }

        energyLocal.SetValue(0, totalEnergy);
        pipe_barrier(PIPE_ALL);

        forceQueue_.EnQue(forceLocal);
        LocalTensor<float> forceOut = forceQueue_.DeQue<float>();
        DataCopy(forces_gm_[myStart_ * 3], forceOut, myAtoms_ * 3);

        energyQueue_.EnQue(energyLocal);
        LocalTensor<float> energyOut = energyQueue_.DeQue<float>();
        DataCopy(energy_gm_[coreIdx_ * LJ_FLOAT_PER_BLOCK], energyOut, LJ_FLOAT_PER_BLOCK);

        pipe_barrier(PIPE_ALL);

        forceQueue_.FreeTensor(forceOut);
        energyQueue_.FreeTensor(energyOut);
    }

private:
    GlobalTensor<float> positions_gm_;
    GlobalTensor<float> forces_gm_;
    GlobalTensor<float> energy_gm_;

    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> posQueue_;
    TQue<QuePosition::VECOUT, 1> forceQueue_;
    TQue<QuePosition::VECOUT, 1> energyQueue_;

    int32_t numAtoms_;
    int32_t tileSize_;
    int32_t coreNum_;
    int32_t coreIdx_;
    int32_t myStart_;
    int32_t myEnd_;
    int32_t myAtoms_;
    float cutoffSq_;
    float sigma6_;
    float sigma12_;
    float eps4_;
    float eps24_;
};

#endif
