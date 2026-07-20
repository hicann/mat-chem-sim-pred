// CPU oracle + robustness harness for the BatchSpdInvFp32 Ascend C kernel.
//
// This reproduces the kernel's exact fp32 LDL^T + triangular-inverse arithmetic
// (batch_spd_inv_fp32_ascendc.cpp) on the host so the operator semantics can be
// validated WITHOUT a CANN/NPU build (node202), and so a robustness fix can be
// designed against the same math the kernel runs.
//
// It compares against an independent fp64 Gauss-Jordan inverse (different
// algorithm) on well-conditioned SPD inputs, and probes near-singular /
// rank-deficient Gram matrices to show where the current (unguarded) pivot
// reciprocal 1/dj blows up to Inf/NaN -- the gap a "stable" op must close.
//
// Build (WSL):  g++ -O2 -std=c++17 batch_spd_inv_cpu_ref.cpp -o bspd_ref && ./bspd_ref

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

// ---- faithful copy of the kernel scalar math (fp32), current/unguarded ----
// guardEps <= 0  -> exactly the shipped kernel (1/dj, no guard)
// guardEps >  0  -> clamp pivot to max(dj, guardEps*scale): the proposed fix.
static void ldlt_inv_fp32(const float* g, float* gi, uint32_t M, float guardScaledEps) {
  std::vector<float> L(M * M, 0.0f), Minv(M * M, 0.0f), d(M, 0.0f), di(M, 0.0f);

  // pivot clamp floor (relative to the matrix scale = max |diag|)
  float scale = 0.0f;
  for (uint32_t j = 0; j < M; ++j) scale = std::fmax(scale, std::fabs(g[j * M + j]));
  if (scale <= 0.0f) scale = 1.0f;
  const float floorPivot = guardScaledEps > 0.0f ? guardScaledEps * scale : 0.0f;

  for (uint32_t j = 0; j < M; ++j) {
    float dj = g[j * M + j];
    for (uint32_t k = 0; k < j; ++k) {
      float ljk = L[j * M + k];
      dj -= ljk * ljk * d[k];
    }
    if (guardScaledEps > 0.0f && dj < floorPivot) dj = floorPivot;  // <-- fix
    d[j] = dj;
    float invdj = 1.0f / dj;
    L[j * M + j] = 1.0f;
    for (uint32_t i = j + 1; i < M; ++i) {
      float s = g[i * M + j];
      for (uint32_t k = 0; k < j; ++k) s -= L[i * M + k] * d[k] * L[j * M + k];
      L[i * M + j] = s * invdj;
    }
  }
  for (uint32_t k = 0; k < M; ++k) di[k] = 1.0f / d[k];

  for (uint32_t j = 0; j < M; ++j) {
    Minv[j * M + j] = 1.0f;
    for (uint32_t i = j + 1; i < M; ++i) {
      float s = 0.0f;
      for (uint32_t k = j; k < i; ++k) s -= L[i * M + k] * Minv[k * M + j];
      Minv[i * M + j] = s;
    }
  }
  for (uint32_t i = 0; i < M; ++i) {
    for (uint32_t j = i; j < M; ++j) {
      float s = 0.0f;
      for (uint32_t k = j; k < M; ++k)
        s += Minv[k * M + i] * di[k] * Minv[k * M + j];
      gi[i * M + j] = s;
      if (j != i) gi[j * M + i] = s;
    }
  }
}

// ---- independent fp64 Gauss-Jordan inverse with partial pivoting (oracle) ----
static bool gauss_inv_fp64(const float* g, std::vector<double>& inv, uint32_t M) {
  std::vector<double> a(M * M);
  inv.assign(M * M, 0.0);
  for (uint32_t i = 0; i < M; ++i) { a[i * M + i] = a[i * M + i]; }
  for (uint32_t i = 0; i < M * M; ++i) a[i] = (double)g[i];
  for (uint32_t i = 0; i < M; ++i) inv[i * M + i] = 1.0;
  for (uint32_t col = 0; col < M; ++col) {
    uint32_t piv = col;
    double best = std::fabs(a[col * M + col]);
    for (uint32_t r = col + 1; r < M; ++r) {
      double v = std::fabs(a[r * M + col]);
      if (v > best) { best = v; piv = r; }
    }
    if (best < 1e-300) return false;  // singular
    if (piv != col)
      for (uint32_t c = 0; c < M; ++c) {
        std::swap(a[col * M + c], a[piv * M + c]);
        std::swap(inv[col * M + c], inv[piv * M + c]);
      }
    double pv = a[col * M + col];
    for (uint32_t c = 0; c < M; ++c) { a[col * M + c] /= pv; inv[col * M + c] /= pv; }
    for (uint32_t r = 0; r < M; ++r) {
      if (r == col) continue;
      double f = a[r * M + col];
      for (uint32_t c = 0; c < M; ++c) {
        a[r * M + c] -= f * a[col * M + c];
        inv[r * M + c] -= f * inv[col * M + c];
      }
    }
  }
  return true;
}

// residual ||g * gi - I||_max  (fp64 accumulation)
static double residual_identity(const float* g, const float* gi, uint32_t M) {
  double worst = 0.0;
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < M; ++j) {
      double s = 0.0;
      for (uint32_t k = 0; k < M; ++k) s += (double)g[i * M + k] * (double)gi[k * M + j];
      double target = (i == j) ? 1.0 : 0.0;
      worst = std::fmax(worst, std::fabs(s - target));
    }
  return worst;
}

static bool has_nonfinite(const float* v, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) if (!std::isfinite(v[i])) return true;
  return false;
}

// Build SPD G = A A^T / E (+ optional ridge). rankDeficient: make 2 rows equal.
static void make_gram(std::vector<float>& g, uint32_t M, uint32_t E, double ridge,
                      bool rankDeficient, std::mt19937& rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> A(M * E);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t e = 0; e < E; ++e) A[i * E + e] = nd(rng);
  if (rankDeficient && M >= 2)
    for (uint32_t e = 0; e < E; ++e) A[1 * E + e] = A[0 * E + e];  // row1 = row0
  g.assign(M * M, 0.0f);
  for (uint32_t i = 0; i < M; ++i)
    for (uint32_t j = 0; j < M; ++j) {
      double s = 0.0;
      for (uint32_t e = 0; e < E; ++e) s += A[i * E + e] * A[j * E + e];
      s /= (double)E;
      if (i == j) s += ridge;
      g[i * M + j] = (float)s;
    }
}

int main() {
  const float GUARD = 1e-6f;  // proposed pivot floor (relative to matrix scale)
  std::mt19937 rng(12345);

  printf("=== A. well-conditioned SPD: kernel(unguarded) vs fp64 Gauss oracle ===\n");
  printf("%4s %3s %5s | %14s %14s %14s\n", "B", "m", "E",
         "maxdiff_oracle", "resid_kernel", "resid_guard");
  double worstDiff = 0.0, worstResid = 0.0;
  for (uint32_t m : {3u, 5u, 6u, 7u, 16u, 32u, 64u}) {
    uint32_t E = m * 16;
    double diffAcc = 0.0, residK = 0.0, residG = 0.0;
    const uint32_t B = 32;
    for (uint32_t b = 0; b < B; ++b) {
      std::vector<float> g; make_gram(g, m, E, 1e-3, false, rng);
      std::vector<float> giK(m * m), giG(m * m);
      ldlt_inv_fp32(g.data(), giK.data(), m, 0.0f);    // shipped kernel
      ldlt_inv_fp32(g.data(), giG.data(), m, GUARD);   // proposed fix
      std::vector<double> oracle; gauss_inv_fp64(g.data(), oracle, m);
      for (uint32_t i = 0; i < m * m; ++i)
        diffAcc = std::fmax(diffAcc, std::fabs((double)giK[i] - oracle[i]));
      residK = std::fmax(residK, residual_identity(g.data(), giK.data(), m));
      residG = std::fmax(residG, residual_identity(g.data(), giG.data(), m));
    }
    printf("%4u %3u %5u | %14.3e %14.3e %14.3e\n", B, m, E, diffAcc, residK, residG);
    worstDiff = std::fmax(worstDiff, diffAcc);
    worstResid = std::fmax(worstResid, residK);
  }
  printf("  worst maxdiff vs oracle: %.3e (gate 1e-5)\n", worstDiff);
  printf("  worst kernel residual  : %.3e (gate 1e-5)\n\n", worstResid);

  printf("=== B. near-singular / rank-deficient Gram: blow-up vs guard ===\n");
  printf("%-22s | %12s %12s | %12s %12s\n", "case", "kern_nonfin", "kern_resid",
         "guard_nonfin", "guard_resid");
  struct Case { const char* name; uint32_t m, E; double ridge; bool rank; };
  Case cases[] = {
    {"tiny ridge 1e-7", 5, 80, 1e-7, false},
    {"ridge 0",         5, 80, 0.0,  false},
    {"rank-deficient",  6, 96, 0.0,  true},
    {"rank-def+1e-9",   6, 96, 1e-9, true},
    {"m=3 collinear",   3, 48, 0.0,  true},
  };
  for (auto& c : cases) {
    std::vector<float> g; make_gram(g, c.m, c.E, c.ridge, c.rank, rng);
    std::vector<float> giK(c.m * c.m), giG(c.m * c.m);
    ldlt_inv_fp32(g.data(), giK.data(), c.m, 0.0f);
    ldlt_inv_fp32(g.data(), giG.data(), c.m, GUARD);
    bool nfK = has_nonfinite(giK.data(), c.m * c.m);
    bool nfG = has_nonfinite(giG.data(), c.m * c.m);
    double rK = nfK ? INFINITY : residual_identity(g.data(), giK.data(), c.m);
    double rG = nfG ? INFINITY : residual_identity(g.data(), giG.data(), c.m);
    printf("%-22s | %12s %12.3e | %12s %12.3e\n", c.name,
           nfK ? "YES" : "no", rK, nfG ? "YES" : "no", rG);
  }
  printf("\n(guard = clamp pivot to %.0e * max|diag|; emulates an adaptive ridge)\n", GUARD);
  return 0;
}
