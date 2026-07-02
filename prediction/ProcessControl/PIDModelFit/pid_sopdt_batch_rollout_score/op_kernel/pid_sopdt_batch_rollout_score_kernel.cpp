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

struct LocalTilingData {
    uint32_t batch;
    uint32_t candidates;
    uint32_t sim_steps;
    uint32_t candidate_start;
    uint32_t candidate_count;
    uint32_t tile_index;
    uint32_t tile_count;
    uint32_t core_num;
    uint32_t reserved;
    float sample_interval;
    float settle_band;
    float overshoot_weight;
    float settling_weight;
    float control_weight;
};

struct FinalTilingData {
    uint32_t batch;
    uint32_t candidates;
    uint32_t core_num;
    uint32_t reserved;
};

constexpr uint32_t kResultCount = 8U;
constexpr uint32_t kMaxDelayBuffer = 32U;
constexpr uint32_t kLoopsPerTaskUnit = 1U;
constexpr float kLargeScore = 3.4028234663852886e30f;
constexpr float kEpsilon = 1.0e-6f;

// Vector lane width for candidate-axis SIMD inside the rollout. The step loop is a
// serial time recurrence, so each timestep issues a chain of dependent vector ops
// that cannot be pipelined across steps; with a narrow lane the loop is bound by
// per-instruction issue/latency rather than throughput. Widening the lane amortises
// that fixed latency over more candidates per instruction (fewer instructions for
// the same work), turning the kernel from latency-bound into throughput-bound. 768
// is the largest lane that keeps the full 8 state vectors + 17-block scratch + the
// 32-slot delay ring (delay spec 0..31) + I/O queues within the 192 KB UB budget.
constexpr uint32_t kLane = 768U;
// Compute scratch blocks (each kLane floats) kept in the framework calc buffer.
constexpr uint32_t kCalcNamedBlocks = 17U;
// Total calc blocks = named scratch + delay ring(kMaxDelayBuffer).
constexpr uint32_t kCalcTotalBlocks = kCalcNamedBlocks + kMaxDelayBuffer;
// GM->UB gain blocks (kp, ki, kd) routed through the in queue.
constexpr uint32_t kGainBlocks = 3U;
// UB->GM result planes (score, iae, ise, maxos, settling) routed through the out queue.
constexpr uint32_t kResultBlocks = 5U;
// Single-buffered queues: framework inserts the MTE2<->V<->MTE3 sync, no manual flags.
constexpr int32_t kQueueDepth = 1;

__aicore__ inline float AbsF(float value)
{
    return value >= 0.0f ? value : -value;
}

__aicore__ inline float MaxF(float lhs, float rhs)
{
    return lhs > rhs ? lhs : rhs;
}

__aicore__ inline float MinF(float lhs, float rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline float ClampF(float value, float low, float high)
{
    return MinF(MaxF(value, low), high);
}

__aicore__ inline bool IsInvalid(float value)
{
    return value != value || AbsF(value) > 1.0e20f;
}

__aicore__ inline void LoopRange(uint32_t batch, uint32_t core_num, uint32_t core_idx, uint32_t& begin, uint32_t& end)
{
    const uint32_t task_units = (batch + kLoopsPerTaskUnit - 1U) / kLoopsPerTaskUnit;
    const uint32_t units_per_core = (task_units + core_num - 1U) / core_num;
    const uint32_t start_unit = core_idx * units_per_core;
    uint32_t end_unit = start_unit + units_per_core;
    if (end_unit > task_units) {
        end_unit = task_units;
    }
    begin = start_unit * kLoopsPerTaskUnit;
    end = end_unit * kLoopsPerTaskUnit;
    if (end > batch) {
        end = batch;
    }
}

}  // namespace

extern "C" __global__ __aicore__ void pid_sopdt_batch_rollout_score_local_kernel(
    GM_ADDR a1, GM_ADDR a2, GM_ADDR b, GM_ADDR delay, GM_ADDR y0, GM_ADDR sp, GM_ADDR kp, GM_ADDR ki, GM_ADDR kd,
    GM_ADDR partial_result, GM_ADDR partial_idx, GM_ADDR tiling)
{
    const __gm__ LocalTilingData* tiling_data = reinterpret_cast<const __gm__ LocalTilingData*>(tiling);
    const uint32_t batch = tiling_data->batch;
    const uint32_t candidates = tiling_data->candidates;
    const uint32_t sim_steps = tiling_data->sim_steps;
    const uint32_t candidate_start = tiling_data->candidate_start;
    const uint32_t candidate_count = tiling_data->candidate_count;
    const uint32_t tile_index = tiling_data->tile_index;
    const uint32_t tile_count = tiling_data->tile_count;
    const uint32_t core_num = tiling_data->core_num;
    const float sample_interval = tiling_data->sample_interval;
    const float settle_band = tiling_data->settle_band;
    const float overshoot_weight = tiling_data->overshoot_weight;
    const float settling_weight = tiling_data->settling_weight;
    const float control_weight = tiling_data->control_weight;
    const uint32_t core_idx = GetBlockIdx();

    if (batch == 0U || candidates == 0U || sim_steps == 0U || candidate_count == 0U || tile_count == 0U ||
        core_num == 0U) {
        return;
    }
    // Mix-mode op: the vector intrinsics below only run on the AIV subcore.
    // Running on AIC would execute illegal vector ops, so bail out there.
    if ASCEND_IS_AIC {
        return;
    }

    GlobalTensor<float> a1_gm;
    GlobalTensor<float> a2_gm;
    GlobalTensor<float> b_gm;
    GlobalTensor<int32_t> delay_gm;
    GlobalTensor<float> y0_gm;
    GlobalTensor<float> sp_gm;
    GlobalTensor<float> kp_gm;
    GlobalTensor<float> ki_gm;
    GlobalTensor<float> kd_gm;
    GlobalTensor<float> partial_result_gm;
    GlobalTensor<int32_t> partial_idx_gm;
    a1_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a1));
    a2_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a2));
    b_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
    delay_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(delay));
    y0_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y0));
    sp_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sp));
    kp_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kp));
    ki_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ki));
    kd_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kd));
    partial_result_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partial_result));
    partial_idx_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(partial_idx));

    (void)tile_index;
    (void)partial_idx_gm;

    // Framework-managed buffers: the in/out TQues own the GM<->UB synchronisation so
    // there are no hand-written SetFlag/WaitFlag flags (the previous source of the
    // mix-mode AIV sync race). Pure compute scratch lives in a plain calc TBuf because
    // it never crosses a pipe boundary (all vector ops execute in order on the V pipe).
    TPipe pipe;
    TQue<TPosition::VECIN, kQueueDepth> in_queue;
    TQue<TPosition::VECOUT, kQueueDepth> out_queue;
    TBuf<TPosition::VECCALC> calc_buf;
    pipe.InitBuffer(in_queue, kQueueDepth, kGainBlocks * kLane * sizeof(float));
    pipe.InitBuffer(out_queue, kQueueDepth, kResultBlocks * kLane * sizeof(float));
    pipe.InitBuffer(calc_buf, kCalcTotalBlocks * kLane * sizeof(float));
    LocalTensor<float> base = calc_buf.Get<float>();

    LocalTensor<float> y = base[0U * kLane];
    LocalTensor<float> integral = base[1U * kLane];
    LocalTensor<float> prev_error = base[2U * kLane];
    LocalTensor<float> iae = base[3U * kLane];
    LocalTensor<float> ise = base[4U * kLane];
    LocalTensor<float> maxos = base[5U * kLane];
    LocalTensor<float> settling = base[6U * kLane];
    LocalTensor<float> ce = base[7U * kLane];
    LocalTensor<float> error = base[8U * kLane];
    LocalTensor<float> deriv = base[9U * kLane];
    LocalTensor<float> control = base[10U * kLane];
    LocalTensor<float> y_prev = base[11U * kLane];  // second-order history state y[k-1]
    LocalTensor<float> ae = base[12U * kLane];
    LocalTensor<float> tmp = base[13U * kLane];
    LocalTensor<float> ytmp = base[14U * kLane];
    LocalTensor<float> dtmp = base[15U * kLane];
    LocalTensor<float> tval = base[16U * kLane];
    LocalTensor<float> ring = base[kCalcNamedBlocks * kLane];  // [kMaxDelayBuffer][kLane]

    const float dt = sample_interval;
    const float inv_dt = 1.0f / MaxF(sample_interval, kEpsilon);

    uint32_t begin = 0U;
    uint32_t end = 0U;
    LoopRange(batch, core_num, core_idx, begin, end);

    const uint64_t plane_stride = ((static_cast<uint64_t>(batch) * candidates + 7U) / 8U) * 8U;

    for (uint32_t loop = begin; loop < end; ++loop) {
        const float model_a1 = a1_gm.GetValue(loop);
        const float model_a2 = a2_gm.GetValue(loop);
        const float model_b = b_gm.GetValue(loop);
        int32_t delay_value = delay_gm.GetValue(loop);
        if (delay_value < 0) {
            delay_value = 0;
        }
        if (delay_value >= static_cast<int32_t>(kMaxDelayBuffer)) {
            delay_value = static_cast<int32_t>(kMaxDelayBuffer - 1U);
        }
        const uint32_t delay_len = static_cast<uint32_t>(delay_value + 1);
        const float initial_y = y0_gm.GetValue(loop);
        const float target = sp_gm.GetValue(loop);

        for (uint32_t sub = 0U; sub < candidate_count; sub += kLane) {
            uint32_t cnt = candidate_count - sub;
            if (cnt > kLane) {
                cnt = kLane;
            }

            const uint32_t cand0 = candidate_start + sub;

            // Byte-exact DMA so a candidate count that is not a multiple of the
            // 32-byte access block (e.g. a partial final tile) still copies its
            // whole tail instead of truncating to the aligned part.
            DataCopyExtParams gain_params;
            gain_params.blockCount = 1U;
            gain_params.blockLen = cnt * sizeof(float);
            gain_params.srcStride = 0U;
            gain_params.dstStride = 0U;
            gain_params.rsv = 0U;
            const DataCopyPadExtParams<float> gain_pad{false, 0U, 0U, 0.0f};

            // Copy-in this sub-tile of candidate gains (GM -> UB) through the in queue;
            // EnQue/DeQue makes the framework insert the MTE2->V dependency.
            LocalTensor<float> gains = in_queue.AllocTensor<float>();
            DataCopyPad(gains[0U * kLane], kp_gm[cand0], gain_params, gain_pad);
            DataCopyPad(gains[1U * kLane], ki_gm[cand0], gain_params, gain_pad);
            DataCopyPad(gains[2U * kLane], kd_gm[cand0], gain_params, gain_pad);
            in_queue.EnQue(gains);
            gains = in_queue.DeQue<float>();
            LocalTensor<float> kpv = gains[0U * kLane];
            LocalTensor<float> kiv = gains[1U * kLane];
            LocalTensor<float> kdv = gains[2U * kLane];

            // Initialise rollout state vectors.
            Duplicate(y, initial_y, cnt);
            Duplicate(y_prev, initial_y, cnt);
            Duplicate(integral, 0.0f, cnt);
            Duplicate(prev_error, target - initial_y, cnt);
            Duplicate(iae, 0.0f, cnt);
            Duplicate(ise, 0.0f, cnt);
            Duplicate(maxos, 0.0f, cnt);
            Duplicate(settling, 0.0f, cnt);
            Duplicate(ce, 0.0f, cnt);
            Duplicate(error, target - initial_y, cnt);
            Duplicate(ring, 0.0f, kMaxDelayBuffer * kLane);

            uint32_t head = 0U;
            float time_value = 0.0f;

            for (uint32_t step = 0U; step < sim_steps; ++step) {
                // `error` already holds e[k] = target - y[k]: initialised before the loop
                // and then carried from the previous step's response error computed below,
                // so the redundant top-of-loop (target - y) recompute is dropped.
                // integral += error * dt
                Muls(tmp, error, dt, cnt);
                Add(integral, integral, tmp, cnt);
                // derivative = (error - prev_error) * inv_dt
                Sub(deriv, error, prev_error, cnt);
                Muls(deriv, deriv, inv_dt, cnt);
                // prev_error = error  (save e[k] before `error` is advanced to e[k+1])
                Adds(prev_error, error, 0.0f, cnt);
                // control = kp*error + ki*integral + kd*deriv
                Mul(control, kpv, error, cnt);
                Mul(tmp, kiv, integral, cnt);
                Add(control, control, tmp, cnt);
                Mul(tmp, kdv, deriv, cnt);
                Add(control, control, tmp, cnt);
                // clamp(control, -10, 10)
                Maxs(control, control, -10.0f, cnt);
                Mins(control, control, 10.0f, cnt);

                // delayed control = ring[head]; then ring[head] = control
                LocalTensor<float> slot = ring[head * kLane];
                Muls(dtmp, slot, model_b, cnt);   // model_b * delayed_control
                Adds(slot, control, 0.0f, cnt);   // store current control into ring slot
                head = (head + 1U) % delay_len;

                // SOPDT 2nd order: y[k+1] = a1*y[k] + a2*y[k-1] + b*u[k-delay].
                // Summation order matches the CPU reference ((a1*y + a2*y_prev) + b*u)
                // so the long-horizon recurrence stays float-aligned with it.
                Muls(ytmp, y, model_a1, cnt);       // a1 * y[k]
                Muls(tmp, y_prev, model_a2, cnt);   // a2 * y[k-1]
                Add(ytmp, ytmp, tmp, cnt);          // a1*y[k] + a2*y[k-1]
                Add(ytmp, ytmp, dtmp, cnt);         // + b * delayed_control -> y[k+1]
                Adds(y_prev, y, 0.0f, cnt);         // shift history: y[k-1] <- y[k]
                Adds(y, ytmp, 0.0f, cnt);           // y[k] <- y[k+1]

                // response error e[k+1] = target - y; reused as the next step's `error`.
                Muls(error, y, -1.0f, cnt);
                Adds(error, error, target, cnt);
                // abs_error
                Abs(ae, error, cnt);
                // iae += |e| * dt  (fused multiply-accumulate; metric sum, not in feedback)
                Axpy(iae, ae, dt, cnt);
                // ise += e*e * dt
                Mul(tmp, error, error, cnt);
                Axpy(ise, tmp, dt, cnt);
                // max_overshoot = max(max_overshoot, y - target)
                Adds(tmp, y, -target, cnt);
                Max(maxos, maxos, tmp, cnt);
                // control_energy += control*control * dt
                Mul(tmp, control, control, cnt);
                Axpy(ce, tmp, dt, cnt);
                // settling = (|e| > band) ? (time+dt) : settling, accumulated as
                // settling = max(settling, (ae>band) ? tval : 0) since tval increases monotonically.
                // q = min(relu(ae-band)*HUGE, tval): HUGE for ae>band -> tval, 0 otherwise.
                Duplicate(tval, time_value + dt, cnt);
                Adds(tmp, ae, -settle_band, cnt);   // p = ae - band
                Maxs(tmp, tmp, 0.0f, cnt);          // relu(p)
                Muls(tmp, tmp, 1.0e15f, cnt);       // saturate positive p to huge
                Min(tmp, tmp, tval, cnt);           // q = min(huge, tval)
                Max(settling, settling, tmp, cnt);  // settling = max(settling, q)
                time_value += dt;
            }

            // Gains are fully consumed; release the in-queue buffer (framework tracks WAR).
            in_queue.FreeTensor(gains);

            // overshoot = max(max_overshoot, 0)
            Maxs(maxos, maxos, 0.0f, cnt);

            // Assemble per-candidate score + metrics into the out-queue tensor.
            LocalTensor<float> result = out_queue.AllocTensor<float>();
            LocalTensor<float> out_score = result[0U * kLane];
            LocalTensor<float> out_iae = result[1U * kLane];
            LocalTensor<float> out_ise = result[2U * kLane];
            LocalTensor<float> out_maxos = result[3U * kLane];
            LocalTensor<float> out_settling = result[4U * kLane];

            // score = iae + ow*overshoot + sw*settling + cw*ce
            Muls(out_score, maxos, overshoot_weight, cnt);
            Add(out_score, out_score, iae, cnt);
            Muls(tmp, settling, settling_weight, cnt);
            Add(out_score, out_score, tmp, cnt);
            Muls(tmp, ce, control_weight, cnt);
            Add(out_score, out_score, tmp, cnt);
            Adds(out_iae, iae, 0.0f, cnt);
            Adds(out_ise, ise, 0.0f, cnt);
            Adds(out_maxos, maxos, 0.0f, cnt);
            Adds(out_settling, settling, 0.0f, cnt);

            out_queue.EnQue(result);
            result = out_queue.DeQue<float>();

            // Publish per-candidate score + metrics to GM planes (UB -> GM).
            // Byte-exact DMA writes only the owned candidates, so a partial tail
            // tile is stored fully and adjacent cores never share a written block.
            DataCopyExtParams store_params;
            store_params.blockCount = 1U;
            store_params.blockLen = cnt * sizeof(float);
            store_params.srcStride = 0U;
            store_params.dstStride = 0U;
            store_params.rsv = 0U;
            const uint64_t off = static_cast<uint64_t>(loop) * candidates + cand0;
            DataCopyPad(partial_result_gm[0U * plane_stride + off], result[0U * kLane], store_params);
            DataCopyPad(partial_result_gm[1U * plane_stride + off], result[1U * kLane], store_params);
            DataCopyPad(partial_result_gm[2U * plane_stride + off], result[2U * kLane], store_params);
            DataCopyPad(partial_result_gm[3U * plane_stride + off], result[3U * kLane], store_params);
            DataCopyPad(partial_result_gm[4U * plane_stride + off], result[4U * kLane], store_params);
            out_queue.FreeTensor(result);
        }
    }
}

extern "C" __global__ __aicore__ void pid_sopdt_batch_rollout_score_final_kernel(
    GM_ADDR kp, GM_ADDR ki, GM_ADDR kd, GM_ADDR metrics, GM_ADDR best_result, GM_ADDR best_idx, GM_ADDR tiling)
{
    const __gm__ FinalTilingData* tiling_data = reinterpret_cast<const __gm__ FinalTilingData*>(tiling);
    const uint32_t batch = tiling_data->batch;
    const uint32_t candidates = tiling_data->candidates;
    const uint32_t core_num = tiling_data->core_num;
    const uint32_t core_idx = GetBlockIdx();

    if (batch == 0U || candidates == 0U || core_num == 0U) {
        return;
    }
    if ASCEND_IS_AIC {
        return;
    }

    GlobalTensor<float> kp_gm;
    GlobalTensor<float> ki_gm;
    GlobalTensor<float> kd_gm;
    GlobalTensor<float> metrics_gm;
    GlobalTensor<float> best_result_gm;
    GlobalTensor<int32_t> best_idx_gm;
    kp_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kp));
    ki_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ki));
    kd_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(kd));
    metrics_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(metrics));
    best_result_gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(best_result));
    best_idx_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(best_idx));

    const uint64_t plane_stride = ((static_cast<uint64_t>(batch) * candidates + 7U) / 8U) * 8U;

    uint32_t begin = 0U;
    uint32_t end = 0U;
    LoopRange(batch, core_num, core_idx, begin, end);
    if (begin >= end) {
        return;
    }

    // Each core writes a disjoint slice of best_result/best_idx. Scalar SetValue
    // to GM does a cached read-modify-write of the surrounding cache line, so two
    // cores whose outputs share a line race and lose updates. Stage every loop's
    // result in UB and publish it with a byte-exact DMA (DataCopyPad), which writes
    // only the owned bytes and is coherent across cores.
    TPipe pipe;
    TBuf<TPosition::VECCALC> res_buf;
    TBuf<TPosition::VECCALC> idx_buf;
    pipe.InitBuffer(res_buf, 32U);  // kResultCount floats (32 bytes)
    pipe.InitBuffer(idx_buf, 32U);  // one int32 (min UB block)
    LocalTensor<float> res_ub = res_buf.Get<float>();
    LocalTensor<int32_t> idx_ub = idx_buf.Get<int32_t>();

    const TEventID ev_s_mte3 = pipe.AllocEventID<HardEvent::S_MTE3>();
    const TEventID ev_mte3_s = pipe.AllocEventID<HardEvent::MTE3_S>();

    DataCopyExtParams res_params;
    res_params.blockCount = 1U;
    res_params.blockLen = kResultCount * sizeof(float);
    res_params.srcStride = 0U;
    res_params.dstStride = 0U;
    res_params.rsv = 0U;
    DataCopyExtParams idx_params;
    idx_params.blockCount = 1U;
    idx_params.blockLen = sizeof(int32_t);
    idx_params.srcStride = 0U;
    idx_params.dstStride = 0U;
    idx_params.rsv = 0U;

    for (uint32_t loop = begin; loop < end; ++loop) {
        const uint64_t row = static_cast<uint64_t>(loop) * candidates;
        float best_score = kLargeScore;
        int32_t best_candidate = 0;
        for (uint32_t c = 0U; c < candidates; ++c) {
            float sc = metrics_gm.GetValue(0U * plane_stride + row + c);
            if (IsInvalid(sc)) {
                sc = kLargeScore;
            }
            if (sc < best_score) {
                best_score = sc;
                best_candidate = static_cast<int32_t>(c);
            }
        }

        const uint64_t bi = row + static_cast<uint64_t>(best_candidate);
        res_ub.SetValue(0U, best_score);
        res_ub.SetValue(1U, kp_gm.GetValue(static_cast<uint32_t>(best_candidate)));
        res_ub.SetValue(2U, ki_gm.GetValue(static_cast<uint32_t>(best_candidate)));
        res_ub.SetValue(3U, kd_gm.GetValue(static_cast<uint32_t>(best_candidate)));
        res_ub.SetValue(4U, metrics_gm.GetValue(1U * plane_stride + bi));
        res_ub.SetValue(5U, metrics_gm.GetValue(2U * plane_stride + bi));
        res_ub.SetValue(6U, metrics_gm.GetValue(3U * plane_stride + bi));
        res_ub.SetValue(7U, metrics_gm.GetValue(4U * plane_stride + bi));
        idx_ub.SetValue(0U, best_candidate);

        SetFlag<HardEvent::S_MTE3>(ev_s_mte3);
        WaitFlag<HardEvent::S_MTE3>(ev_s_mte3);

        DataCopyPad(best_result_gm[static_cast<uint64_t>(loop) * kResultCount], res_ub, res_params);
        DataCopyPad(best_idx_gm[loop], idx_ub, idx_params);

        SetFlag<HardEvent::MTE3_S>(ev_mte3_s);
        WaitFlag<HardEvent::MTE3_S>(ev_mte3_s);
    }

    pipe.ReleaseEventID<HardEvent::S_MTE3>(ev_s_mte3);
    pipe.ReleaseEventID<HardEvent::MTE3_S>(ev_mte3_s);
}
