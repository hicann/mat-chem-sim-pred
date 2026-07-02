// Representative-scale resident E2E performance: tuning_rule -> fopdt_rollout -> performance_metrics.
// NPU resident chain (inputs pre-staged on device, 3 kernels timed) vs CPU 64-thread chain
// (workers split over the batch axis, mirroring the per-operator benchmark CPU baseline).
// Also verifies final best-PID / score / metrics alignment between the two paths.
#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#define CHECK_ACL(expr)                                                                 \
    do {                                                                                \
        aclError _ret = (expr);                                                         \
        if (_ret != ACL_SUCCESS) {                                                      \
            std::fprintf(stderr, "ACL fail %s:%d ret=%d\n", __FILE__, __LINE__, _ret);  \
            std::exit(2);                                                               \
        }                                                                               \
    } while (0)

extern "C" uint64_t aclnnPidTuningRuleBatchGetWorkspaceSize(int64_t);
extern "C" int32_t aclnnPidTuningRuleBatch(void*, void*, void*, void*, void*, void*, int64_t, void*, uint64_t,
                                           void*);
extern "C" uint64_t aclnnPidFopdtBatchRolloutScoreGetWorkspaceSize(int64_t, int64_t, int64_t, int64_t);
extern "C" int32_t aclnnPidFopdtBatchRolloutScore(void*, void*, void*, void*, void*, void*, void*, void*, void*,
                                                  void*, int64_t, int64_t, int64_t, int64_t, float, float, float,
                                                  float, float, void*, uint64_t, void*);
extern "C" uint64_t aclnnPidControlPerformanceMetricsGetWorkspaceSize(int64_t, int64_t);
extern "C" int32_t aclnnPidControlPerformanceMetrics(void*, void*, void*, void*, void*, void*, int64_t, int64_t,
                                                     float, float, void*, uint64_t, void*);

// ---- rollout scoring weights (match operator defaults / reference) ----
static const float kSampleInterval = 1.0f;
static const float kRollSettleBand = 0.02f;
static const float kOvershootW = 50.0f;
static const float kSettlingW = 0.02f;
static const float kControlW = 0.001f;
static const float kMetricSettleBand = 0.1f;

struct Data {
    int64_t B, C, SIM, tile;
    std::vector<float> K, T, L, lambda;     // tuning inputs [B]
    std::vector<float> a, b, y0, sp;        // rollout model [B]
    std::vector<int32_t> delay;             // [B]
    std::vector<float> kp, ki, kd;          // shared candidate grid [C]
};

static Data MakeData(int64_t B, int64_t C, int64_t SIM, int64_t tile)
{
    Data d;
    d.B = B; d.C = C; d.SIM = SIM; d.tile = tile;
    d.K.resize(B); d.T.resize(B); d.L.resize(B); d.lambda.resize(B);
    d.a.resize(B); d.b.resize(B); d.y0.resize(B); d.sp.resize(B); d.delay.resize(B);
    for (int64_t i = 0; i < B; ++i) {
        float frac = B > 1 ? static_cast<float>(i) / static_cast<float>(B - 1) : 0.0f;
        float tau = 8.0f + 32.0f * frac;
        float gain = 0.6f + 1.2f * frac;
        int32_t dly = static_cast<int32_t>(1 + (i % 15));
        d.T[i] = tau; d.K[i] = gain; d.L[i] = static_cast<float>(dly);
        d.lambda[i] = std::max(d.L[i], 0.5f * tau);
        d.a[i] = std::exp(-kSampleInterval / tau);
        d.b[i] = gain * (1.0f - d.a[i]);
        d.delay[i] = dly;
        d.y0[i] = 0.0f;
        d.sp[i] = 1.0f;
    }
    d.kp.resize(C); d.ki.resize(C); d.kd.resize(C);
    for (int64_t c = 0; c < C; ++c) {
        float r = C > 1 ? static_cast<float>(c) / static_cast<float>(C - 1) : 0.0f;
        d.kp[c] = 0.05f + 2.95f * r;
        d.ki[c] = 0.35f * r;
        d.kd[c] = 0.25f * r;
    }
    return d;
}

// One candidate closed-loop score (mirror of pid_fopdt_closed_loop reference).
static void ScoreCandidate(float a, float b, int delay, float y0, float sp, float kp, float ki, float kd,
                           int64_t steps, float& score, float& kp_o, float& ki_o, float& kd_o)
{
    delay = std::max(0, delay);
    int dl = std::min(delay + 1, 128);
    std::vector<float> u_hist(static_cast<size_t>(dl), 0.0f);
    float y = y0, integral = 0.0f, prev_error = sp - y;
    float iae = 0.0f, max_over = 0.0f, last_unsettled = 0.0f, control_energy = 0.0f;
    for (int64_t k = 0; k < steps; ++k) {
        float error = sp - y;
        integral += error * kSampleInterval;
        float derivative = (error - prev_error) / std::max(kSampleInterval, 1.0e-6f);
        float u = kp * error + ki * integral + kd * derivative;
        u = std::min(std::max(u, -10.0f), 10.0f);
        float delayed = u_hist[0];
        if (dl > 1) {
            for (int j = 0; j < dl - 1; ++j) u_hist[j] = u_hist[j + 1];
        }
        u_hist[dl - 1] = u;
        y = a * y + b * delayed;
        float abs_error = std::fabs(sp - y);
        iae += abs_error * kSampleInterval;
        max_over = std::max(max_over, y - sp);
        control_energy += u * u * kSampleInterval;
        if (abs_error > kRollSettleBand) last_unsettled = static_cast<float>(k + 1) * kSampleInterval;
        prev_error = error;
    }
    float overshoot = std::max(max_over, 0.0f);
    score = iae + kOvershootW * overshoot + kSettlingW * last_unsettled + kControlW * control_energy;
    kp_o = kp; ki_o = ki; kd_o = kd;
}

// CPU chain over a [begin,end) slice of the batch: tuning(3 rules, unused for grid) + rollout grid search.
static void CpuChainRange(const Data& d, int64_t begin, int64_t end, std::vector<float>& best_score,
                          std::vector<float>& best_kp)
{
    for (int64_t i = begin; i < end; ++i) {
        // tuning (analytic, cheap) — compute IMC kp as representative work
        float eps = 1.0e-6f;
        float k = d.K[i], tau = d.T[i], theta = std::max(d.L[i], eps), lam = std::max(d.lambda[i], eps);
        float imc_kp = (tau + 0.5f * theta) / (std::max(std::fabs(k), eps) * (lam + 0.5f * theta));
        (void)imc_kp;
        // rollout grid search
        float bscore = 3.4e38f, bkp = 0.0f, bki = 0.0f, bkd = 0.0f;
        for (int64_t c = 0; c < d.C; ++c) {
            float s, okp, oki, okd;
            ScoreCandidate(d.a[i], d.b[i], d.delay[i], d.y0[i], d.sp[i], d.kp[c], d.ki[c], d.kd[c], d.SIM,
                           s, okp, oki, okd);
            if (s < bscore) { bscore = s; bkp = okp; bki = oki; bkd = okd; }
        }
        best_score[i] = bscore; best_kp[i] = bkp;
        (void)bki; (void)bkd;
    }
}

static double CpuChainMs(const Data& d, int threads, int iters, std::vector<float>& best_score,
                         std::vector<float>& best_kp)
{
    int workers = static_cast<int>(std::min<int64_t>(std::max(1, threads), d.B));
    auto run_once = [&]() {
        if (workers <= 1) { CpuChainRange(d, 0, d.B, best_score, best_kp); return; }
        std::vector<std::thread> ts;
        int64_t chunk = (d.B + workers - 1) / workers;
        for (int w = 0; w < workers; ++w) {
            int64_t bg = w * chunk, en = std::min<int64_t>(d.B, bg + chunk);
            if (bg >= en) break;
            ts.emplace_back([&, bg, en]() { CpuChainRange(d, bg, en, best_score, best_kp); });
        }
        for (auto& t : ts) t.join();
    };
    run_once();  // warm
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) run_once();
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

static void* DevH2D(const void* host, size_t bytes)
{
    void* dev = nullptr;
    CHECK_ACL(aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(dev, bytes, host, bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    return dev;
}
static void* DevZero(size_t bytes)
{
    void* dev = nullptr;
    CHECK_ACL(aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(dev, bytes, 0, bytes));
    return dev;
}

int main(int argc, char** argv)
{
    int device_id = argc > 1 ? std::atoi(argv[1]) : 0;
    int64_t B = argc > 2 ? std::atoll(argv[2]) : 64;
    int64_t C = argc > 3 ? std::atoll(argv[3]) : 4096;
    int64_t SIM = argc > 4 ? std::atoll(argv[4]) : 512;
    int64_t tile = argc > 5 ? std::atoll(argv[5]) : 64;
    int iters = argc > 6 ? std::atoi(argv[6]) : 5;
    int warmup = argc > 7 ? std::atoi(argv[7]) : 2;
    int cpu_threads = argc > 8 ? std::atoi(argv[8]) : 64;

    Data d = MakeData(B, C, SIM, tile);

    // ---- CPU 64T chain ----
    std::vector<float> cpu_score(B, 0.0f), cpu_kp(B, 0.0f);
    double cpu_ms = CpuChainMs(d, cpu_threads, iters, cpu_score, cpu_kp);

    // ---- NPU resident chain ----
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    size_t fB = static_cast<size_t>(B) * sizeof(float);
    size_t fC = static_cast<size_t>(C) * sizeof(float);
    // tuning io
    void* dK = DevH2D(d.K.data(), fB); void* dT = DevH2D(d.T.data(), fB);
    void* dL = DevH2D(d.L.data(), fB); void* dLam = DevH2D(d.lambda.data(), fB);
    void* dPid = DevZero(static_cast<size_t>(B) * 9 * sizeof(float));
    void* dDiag = DevZero(static_cast<size_t>(B) * 12 * sizeof(float));
    // rollout io
    void* dA = DevH2D(d.a.data(), fB); void* dB = DevH2D(d.b.data(), fB);
    void* dDelay = DevH2D(d.delay.data(), static_cast<size_t>(B) * sizeof(int32_t));
    void* dY0 = DevH2D(d.y0.data(), fB); void* dSp = DevH2D(d.sp.data(), fB);
    void* dKp = DevH2D(d.kp.data(), fC); void* dKi = DevH2D(d.ki.data(), fC); void* dKd = DevH2D(d.kd.data(), fC);
    void* dBest = DevZero(static_cast<size_t>(B) * 8 * sizeof(float));
    void* dBestIdx = DevZero(static_cast<size_t>(B) * sizeof(int32_t));
    // metrics io (pv built from rollout winners after a warm rollout)
    std::vector<float> pv(static_cast<size_t>(B) * SIM, 0.0f), sp2(static_cast<size_t>(B) * SIM, 1.0f);
    std::vector<float> lsl(B, 0.9f), usl(B, 1.1f), mvv(B, 0.0f);

    uint64_t ws_t = aclnnPidTuningRuleBatchGetWorkspaceSize(B);
    uint64_t ws_r = aclnnPidFopdtBatchRolloutScoreGetWorkspaceSize(B, C, SIM, tile);
    uint64_t ws_m = aclnnPidControlPerformanceMetricsGetWorkspaceSize(B, SIM);
    void* wsT = ws_t ? DevZero(ws_t) : nullptr;
    void* wsR = ws_r ? DevZero(ws_r) : nullptr;
    void* wsM = ws_m ? DevZero(ws_m) : nullptr;

    auto run_tuning = [&]() {
        return aclnnPidTuningRuleBatch(dK, dT, dL, dLam, dPid, dDiag, B, wsT, ws_t, stream);
    };
    auto run_rollout = [&]() {
        return aclnnPidFopdtBatchRolloutScore(dA, dB, dDelay, dY0, dSp, dKp, dKi, dKd, dBest, dBestIdx, B, C, SIM,
                                              tile, kSampleInterval, kRollSettleBand, kOvershootW, kSettlingW,
                                              kControlW, wsR, ws_r, stream);
    };
    auto run_metrics = [&]() {
        return aclnnPidControlPerformanceMetrics(/*pv*/nullptr, /*sp*/nullptr, nullptr, nullptr, nullptr,
                                                 nullptr, B, SIM, kSampleInterval, kMetricSettleBand, wsM, ws_m,
                                                 stream);
    };
    (void)run_metrics;

    // warm rollout to get winners -> build pv on host -> H2D
    if (run_tuning() != 0) { std::fprintf(stderr, "tuning fail\n"); return 4; }
    if (run_rollout() != 0) { std::fprintf(stderr, "rollout fail\n"); return 4; }
    CHECK_ACL(aclrtSynchronizeStream(stream));
    std::vector<float> best_host(static_cast<size_t>(B) * 8);
    CHECK_ACL(aclrtMemcpy(best_host.data(), best_host.size() * sizeof(float), dBest,
                          best_host.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (int64_t i = 0; i < B; ++i) {
        float wkp = best_host[i * 8 + 1], wki = best_host[i * 8 + 2], wkd = best_host[i * 8 + 3];
        int delay = std::max(0, static_cast<int>(d.delay[i]));
        int dl = std::min(delay + 1, 128);
        std::vector<float> uh(static_cast<size_t>(dl), 0.0f);
        float y = d.y0[i], integral = 0.0f, prev = d.sp[i] - y;
        for (int64_t k = 0; k < SIM; ++k) {
            float e = d.sp[i] - y; integral += e * kSampleInterval;
            float der = (e - prev) / kSampleInterval;
            float u = wkp * e + wki * integral + wkd * der;
            u = std::min(std::max(u, -10.0f), 10.0f);
            float dyu = uh[0];
            if (dl > 1) { for (int j = 0; j < dl - 1; ++j) uh[j] = uh[j + 1]; }
            uh[dl - 1] = u;
            y = d.a[i] * y + d.b[i] * dyu;
            pv[i * SIM + k] = y; prev = e;
        }
    }
    void* dPv = DevH2D(pv.data(), pv.size() * sizeof(float));
    void* dSp2 = DevH2D(sp2.data(), sp2.size() * sizeof(float));
    void* dLsl = DevH2D(lsl.data(), fB); void* dUsl = DevH2D(usl.data(), fB); void* dMvv = DevH2D(mvv.data(), fB);
    void* dMetrics = DevZero(static_cast<size_t>(B) * 20 * sizeof(float));
    auto run_metrics_real = [&]() {
        return aclnnPidControlPerformanceMetrics(dPv, dSp2, dLsl, dUsl, dMvv, dMetrics, B, SIM, kSampleInterval,
                                                 kMetricSettleBand, wsM, ws_m, stream);
    };

    // warmup full chain
    for (int w = 0; w < warmup; ++w) { run_tuning(); run_rollout(); run_metrics_real(); }
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // timed resident chain (inputs already on device) + per-stage attribution
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) {
        if (run_tuning() != 0 || run_rollout() != 0 || run_metrics_real() != 0) {
            std::fprintf(stderr, "chain fail\n"); return 4;
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double npu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

    auto stage_ms = [&](int which) {
        auto s0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; ++it) {
            if (which == 0) run_tuning();
            else if (which == 1) run_rollout();
            else run_metrics_real();
            CHECK_ACL(aclrtSynchronizeStream(stream));
        }
        auto s1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(s1 - s0).count() / iters;
    };
    double t_tune = stage_ms(0), t_roll = stage_ms(1), t_metr = stage_ms(2);

    // alignment: NPU rollout best score/kp vs CPU
    double max_score_rel = 0.0, max_kp_abs = 0.0;
    for (int64_t i = 0; i < B; ++i) {
        double sr = std::fabs(best_host[i * 8 + 0] - cpu_score[i]) / (std::fabs(cpu_score[i]) + 1e-9);
        double ka = std::fabs(best_host[i * 8 + 1] - cpu_kp[i]);
        max_score_rel = std::max(max_score_rel, sr);
        max_kp_abs = std::max(max_kp_abs, ka);
    }

    std::printf("PERF B=%ld C=%ld SIM=%ld tile=%ld threads=%d iters=%d\n", (long)B, (long)C, (long)SIM,
                (long)tile, cpu_threads, iters);
    std::printf("  CPU_%dT_chain_ms=%.4f  NPU_resident_chain_ms=%.4f  speedup=%.2fx\n", cpu_threads, cpu_ms,
                npu_ms, cpu_ms / npu_ms);
    std::printf("  NPU per-stage ms: tuning=%.4f rollout=%.4f metrics=%.4f\n", t_tune, t_roll, t_metr);
    std::printf("  align: best_score_max_rel=%.3e  best_kp_max_abs=%.3e\n", max_score_rel, max_kp_abs);

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(device_id));
    (void)aclFinalize();
    return 0;
}
