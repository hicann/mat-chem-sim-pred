/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lj_force.h
 * \brief Lennard-Jones 力场融合算子 Kernel 实现
 */

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

        forceStride_ = LjAlignUp(atomsPerCore * 3, LJ_FLOAT_PER_BLOCK);

        positions_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(positions));
        forces_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(forces));
        energy_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(energy));

        uint32_t posSize = LjAlignUp(numAtoms_ * 3 * sizeof(float), LJ_BLOCK_SIZE);
        uint32_t forceSize = forceStride_ * sizeof(float);
        if (forceSize < LJ_BLOCK_SIZE) forceSize = LJ_BLOCK_SIZE;
        uint32_t energySize = LjAlignUp(LJ_FLOAT_PER_BLOCK * sizeof(float), LJ_BLOCK_SIZE);

        pipe_.InitBuffer(posQueue_, 1, posSize);
        pipe_.InitBuffer(forceQueue_, 1, forceSize);
        pipe_.InitBuffer(energyQueue_, 1, energySize);
    }

    __aicore__ inline void Process() {
        if (myStart_ >= myEnd_) return;

        int32_t posElements = LjAlignUp(numAtoms_ * 3, LJ_FLOAT_PER_BLOCK);

        LocalTensor<float> posLocal = posQueue_.AllocTensor<float>();
        DataCopy(posLocal, positions_gm_, posElements);
        posQueue_.EnQue(posLocal);
        LocalTensor<float> pos = posQueue_.DeQue<float>();

        LocalTensor<float> forceLocal = forceQueue_.AllocTensor<float>();
        Duplicate(forceLocal, 0.0f, forceStride_);

        LocalTensor<float> energyLocal = energyQueue_.AllocTensor<float>();
        Duplicate(energyLocal, 0.0f, LJ_FLOAT_PER_BLOCK);

        pipe_barrier(PIPE_ALL);

        float totalEnergy = 0.0f;

        for (int32_t i = myStart_; i < myEnd_; i++) {
            float xi = pos.GetValue(i * 3);
            float yi = pos.GetValue(i * 3 + 1);
            float zi = pos.GetValue(i * 3 + 2);

            float fx = 0.0f, fy = 0.0f, fz = 0.0f;

            for (int32_t j = 0; j < numAtoms_; j++) {
                if (j == i) continue;

                float dx = xi - pos.GetValue(j * 3);
                float dy = yi - pos.GetValue(j * 3 + 1);
                float dz = zi - pos.GetValue(j * 3 + 2);
                float r2 = dx * dx + dy * dy + dz * dz;

                if (r2 < cutoffSq_ && r2 > 1e-10f) {
                    float r2inv = 1.0f / r2;
                    float r6inv = r2inv * r2inv * r2inv;
                    float s6r6 = sigma6_ * r6inv;
                    float s12r12 = s6r6 * s6r6;

                    if (i < j) {
                        totalEnergy += eps4_ * (s12r12 - s6r6);
                    }

                    float fscalar = eps24_ * r2inv * (2.0f * s12r12 - s6r6);
                    fx += fscalar * dx;
                    fy += fscalar * dy;
                    fz += fscalar * dz;
                }
            }

            int32_t idx = (i - myStart_) * 3;
            forceLocal.SetValue(idx, fx);
            forceLocal.SetValue(idx + 1, fy);
            forceLocal.SetValue(idx + 2, fz);
        }

        energyLocal.SetValue(0, totalEnergy);
        pipe_barrier(PIPE_ALL);

        forceQueue_.EnQue(forceLocal);
        LocalTensor<float> forceOut = forceQueue_.DeQue<float>();
        DataCopy(forces_gm_[coreIdx_ * forceStride_], forceOut, forceStride_);

        energyQueue_.EnQue(energyLocal);
        LocalTensor<float> energyOut = energyQueue_.DeQue<float>();
        DataCopy(energy_gm_[coreIdx_ * LJ_FLOAT_PER_BLOCK], energyOut, LJ_FLOAT_PER_BLOCK);

        pipe_barrier(PIPE_ALL);

        posQueue_.FreeTensor(pos);
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
    int32_t forceStride_;
    float cutoffSq_;
    float sigma6_;
    float sigma12_;
    float eps4_;
    float eps24_;
};

#endif  // LJ_FORCE_H
