// FOPDT full-chain E2E NPU runner.
// Dispatches one operator stage per invocation, reading/writing raw float32/int32 .bin files.
// Links the 4 prebuilt operator host .so libraries; aclnn entrypoints declared below.
#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CHECK_ACL(expr)                                                            \
    do {                                                                           \
        aclError _ret = (expr);                                                    \
        if (_ret != ACL_SUCCESS) {                                                 \
            std::fprintf(stderr, "ACL fail %s:%d ret=%d\n", __FILE__, __LINE__, _ret); \
            std::exit(2);                                                          \
        }                                                                          \
    } while (0)

extern "C" uint64_t aclnnPidFopdtBasisGemmFitGetWorkspaceSize(int64_t, int64_t);
extern "C" int32_t aclnnPidFopdtBasisGemmFit(void*, void*, void*, void*, void*, void*, int64_t, int64_t,
                                             void*, uint64_t, void*);
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

static std::vector<char> ReadAll(const std::string& path)
{
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(3);
    }
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<char> buf(static_cast<size_t>(n));
    if (n > 0 && std::fread(buf.data(), 1, static_cast<size_t>(n), f) != static_cast<size_t>(n)) {
        std::fprintf(stderr, "short read %s\n", path.c_str());
        std::exit(3);
    }
    std::fclose(f);
    return buf;
}

static void WriteAll(const std::string& path, const void* data, size_t bytes)
{
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(3);
    }
    std::fwrite(data, 1, bytes, f);
    std::fclose(f);
}

static void* DevFromFile(const std::string& path, size_t* bytes_out)
{
    std::vector<char> host = ReadAll(path);
    void* dev = nullptr;
    CHECK_ACL(aclrtMalloc(&dev, host.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(dev, host.size(), host.data(), host.size(), ACL_MEMCPY_HOST_TO_DEVICE));
    if (bytes_out != nullptr) {
        *bytes_out = host.size();
    }
    return dev;
}

static void* DevOut(size_t bytes)
{
    void* dev = nullptr;
    CHECK_ACL(aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(dev, bytes, 0, bytes));
    return dev;
}

static void Dump(const std::string& path, void* dev, size_t bytes)
{
    std::vector<char> host(bytes);
    CHECK_ACL(aclrtMemcpy(host.data(), bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteAll(path, host.data(), bytes);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <stage> <dir> [dims...]\n", argv[0]);
        return 1;
    }
    std::string stage = argv[1];
    std::string dir = argv[2];
    auto P = [&](const char* name) { return dir + "/" + name; };

    int32_t device_id = std::getenv("E2E_DEVICE") ? std::atoi(std::getenv("E2E_DEVICE")) : 0;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    if (stage == "fit") {
        int64_t B = std::atoll(argv[3]);
        int64_t C = std::atoll(argv[4]);
        void* dot = DevFromFile(P("dot.bin"), nullptr);
        void* bn = DevFromFile(P("basis_norm.bin"), nullptr);
        void* ye = DevFromFile(P("y_energy.bin"), nullptr);
        void* bs = DevOut(static_cast<size_t>(B) * sizeof(float));
        void* bk = DevOut(static_cast<size_t>(B) * sizeof(float));
        void* bi = DevOut(static_cast<size_t>(B) * sizeof(int32_t));
        uint64_t wss = aclnnPidFopdtBasisGemmFitGetWorkspaceSize(B, C);
        void* ws = wss > 0 ? DevOut(wss) : nullptr;
        int32_t ret = aclnnPidFopdtBasisGemmFit(dot, bn, ye, bs, bk, bi, B, C, ws, wss, stream);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        if (ret != 0) { std::fprintf(stderr, "fit ret=%d\n", ret); return 4; }
        Dump(P("fit_best_sse.bin"), bs, static_cast<size_t>(B) * sizeof(float));
        Dump(P("fit_best_k.bin"), bk, static_cast<size_t>(B) * sizeof(float));
        Dump(P("fit_best_idx.bin"), bi, static_cast<size_t>(B) * sizeof(int32_t));
    } else if (stage == "tuning") {
        int64_t B = std::atoll(argv[3]);
        void* pg = DevFromFile(P("tune_K.bin"), nullptr);
        void* tc = DevFromFile(P("tune_T.bin"), nullptr);
        void* dt = DevFromFile(P("tune_L.bin"), nullptr);
        void* lv = DevFromFile(P("tune_lambda.bin"), nullptr);
        void* pid = DevOut(static_cast<size_t>(B) * 3 * 3 * sizeof(float));
        void* diag = DevOut(static_cast<size_t>(B) * 3 * 4 * sizeof(float));
        uint64_t wss = aclnnPidTuningRuleBatchGetWorkspaceSize(B);
        void* ws = wss > 0 ? DevOut(wss) : nullptr;
        int32_t ret = aclnnPidTuningRuleBatch(pg, tc, dt, lv, pid, diag, B, ws, wss, stream);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        if (ret != 0) { std::fprintf(stderr, "tuning ret=%d\n", ret); return 4; }
        Dump(P("tune_pid.bin"), pid, static_cast<size_t>(B) * 3 * 3 * sizeof(float));
        Dump(P("tune_diag.bin"), diag, static_cast<size_t>(B) * 3 * 4 * sizeof(float));
    } else if (stage == "rollout") {
        int64_t B = std::atoll(argv[3]);
        int64_t C = std::atoll(argv[4]);
        int64_t S = std::atoll(argv[5]);
        int64_t tile = std::atoll(argv[6]);
        void* a = DevFromFile(P("roll_a.bin"), nullptr);
        void* b = DevFromFile(P("roll_b.bin"), nullptr);
        void* delay = DevFromFile(P("roll_delay.bin"), nullptr);
        void* y0 = DevFromFile(P("roll_y0.bin"), nullptr);
        void* sp = DevFromFile(P("roll_sp.bin"), nullptr);
        void* kp = DevFromFile(P("roll_kp.bin"), nullptr);
        void* ki = DevFromFile(P("roll_ki.bin"), nullptr);
        void* kd = DevFromFile(P("roll_kd.bin"), nullptr);
        void* br = DevOut(static_cast<size_t>(B) * 8 * sizeof(float));
        void* bi = DevOut(static_cast<size_t>(B) * sizeof(int32_t));
        uint64_t wss = aclnnPidFopdtBatchRolloutScoreGetWorkspaceSize(B, C, S, tile);
        void* ws = wss > 0 ? DevOut(wss) : nullptr;
        int32_t ret = aclnnPidFopdtBatchRolloutScore(a, b, delay, y0, sp, kp, ki, kd, br, bi, B, C, S, tile,
                                                     1.0f, 0.02f, 50.0f, 0.02f, 0.001f, ws, wss, stream);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        if (ret != 0) { std::fprintf(stderr, "rollout ret=%d\n", ret); return 4; }
        Dump(P("roll_best_result.bin"), br, static_cast<size_t>(B) * 8 * sizeof(float));
        Dump(P("roll_best_idx.bin"), bi, static_cast<size_t>(B) * sizeof(int32_t));
    } else if (stage == "metrics") {
        int64_t B = std::atoll(argv[3]);
        int64_t N = std::atoll(argv[4]);
        void* pv = DevFromFile(P("perf_pv.bin"), nullptr);
        void* sp = DevFromFile(P("perf_sp.bin"), nullptr);
        void* lsl = DevFromFile(P("perf_lsl.bin"), nullptr);
        void* usl = DevFromFile(P("perf_usl.bin"), nullptr);
        void* mvv = DevFromFile(P("perf_mvv.bin"), nullptr);
        void* metrics = DevOut(static_cast<size_t>(B) * 20 * sizeof(float));
        uint64_t wss = aclnnPidControlPerformanceMetricsGetWorkspaceSize(B, N);
        void* ws = wss > 0 ? DevOut(wss) : nullptr;
        int32_t ret = aclnnPidControlPerformanceMetrics(pv, sp, lsl, usl, mvv, metrics, B, N, 1.0f, 0.1f, ws, wss,
                                                        stream);
        CHECK_ACL(aclrtSynchronizeStream(stream));
        if (ret != 0) { std::fprintf(stderr, "metrics ret=%d\n", ret); return 4; }
        Dump(P("perf_metrics.bin"), metrics, static_cast<size_t>(B) * 20 * sizeof(float));
    } else {
        std::fprintf(stderr, "unknown stage %s\n", stage.c_str());
        return 1;
    }

    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(device_id));
    (void)aclFinalize();
    std::fprintf(stderr, "stage %s OK\n", stage.c_str());
    return 0;
}
