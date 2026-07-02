#!/usr/bin/env python3
"""FOPDT full-chain E2E: fit -> tuning_rule -> fopdt_rollout -> performance_metrics.

CPU golden uses the operators' own reference modules. The NPU path runs the real
operators (aclnn) via e2e_runner. Host "glue" between stages (best_idx->(T,L),
lambda choice, rule->candidate reshape, winner->pv simulation) is identical for
both paths, so each stage comparison isolates the operator itself.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

COMMON = os.environ.get(
    "PID_COMMON",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"),
)
sys.path.insert(0, COMMON)

import pid_basis_gemm_reference as fitref
import pid_tuning_rule_batch_reference as tuneref
import pid_fopdt_closed_loop_reference as rollref
import pid_control_performance_reference as perfref

RUNNER = os.environ["E2E_RUNNER"]
WORK = os.environ.get("E2E_WORK", "/tmp/e2e_work")
os.makedirs(WORK, exist_ok=True)

# ---- problem size (minimal but real) ----
B = 8            # loops identified/tuned in batch
N_FIT = 256      # samples per identification window
M = 65           # candidate models in the fit grid
SIM = 512        # closed-loop rollout / pv horizon
DT = 1.0


def w(name, arr, dtype):
    np.asarray(arr, dtype=dtype).tofile(os.path.join(WORK, name))


def r(name, dtype, count):
    return np.fromfile(os.path.join(WORK, name), dtype=dtype, count=count)


def run_stage(stage, *dims):
    env = dict(os.environ)
    cmd = [RUNNER, stage, WORK] + [str(d) for d in dims]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[runner {stage}] FAILED rc={res.returncode}\n{res.stdout}\n{res.stderr}")
        sys.exit(5)
    print(f"[runner {stage}] {res.stderr.strip().splitlines()[-1] if res.stderr.strip() else 'ok'}")


def errs(npu, cpu):
    npu = np.asarray(npu, dtype=np.float64).ravel()
    cpu = np.asarray(cpu, dtype=np.float64).ravel()
    abs_err = np.abs(npu - cpu)
    rel = abs_err / (np.abs(cpu) + 1e-9)
    return float(abs_err.max()), float(rel.max())


def fopdt_grid_TL(c):
    """Reconstruct (T, L_time) for FOPDT candidate index c (mirror of fit reference)."""
    truth_index = M // 2
    truth_t1, truth_l = 18.0, 4.0
    a = (c % 17) - 8.0
    d = (c // 17) % 13
    t1 = max(1.0, truth_t1 * (1.0 + 0.03 * a))
    delay = max(0.0, truth_l + d - 6.0)
    if c == truth_index:
        t1, delay = truth_t1, truth_l
    return float(t1), float(delay)


report = {"config": {"B": B, "N_FIT": N_FIT, "M": M, "SIM": SIM}, "stages": {}}

# ============================================================
# Stage 1: FOPDT basis-GEMM fit  (identification)
# ============================================================
case = fitref.make_basis_gemm_case("FOPDT", batch=B, n=N_FIT, candidates=M)
dot = (case.y_centered @ case.basis_t).astype(np.float32)
cpu_sse, cpu_k, cpu_idx = fitref.reduce_best(dot, case.basis_norm, case.y_energy)

w("dot.bin", dot, np.float32)
w("basis_norm.bin", case.basis_norm, np.float32)
w("y_energy.bin", case.y_energy, np.float32)
run_stage("fit", B, M)
npu_sse = r("fit_best_sse.bin", np.float32, B)
npu_k = r("fit_best_k.bin", np.float32, B)
npu_idx = r("fit_best_idx.bin", np.int32, B)

ae_k, re_k = errs(npu_k, cpu_k)
idx_diff = int(np.sum(npu_idx != cpu_idx))
report["stages"]["1_fit"] = {
    "best_k_max_abs": ae_k, "best_k_max_rel": re_k, "best_idx_diff": idx_diff,
    "cpu_truth_index": int(M // 2), "npu_idx": npu_idx.tolist(),
}

# host glue: identified model per loop (use NPU fit outputs to drive the chain)
ident_K = npu_k.astype(np.float32)
ident_T = np.array([fopdt_grid_TL(int(i))[0] for i in npu_idx], dtype=np.float32)
ident_L = np.array([fopdt_grid_TL(int(i))[1] for i in npu_idx], dtype=np.float32)

# ============================================================
# Stage 2: tuning_rule_batch  (model -> PID rules)
# ============================================================
ident_lambda = np.maximum(ident_L, 0.5 * ident_T).astype(np.float32)
cpu_tune = tuneref.tuning_rule_batch(ident_K, ident_T, ident_L, ident_lambda)
cpu_pid = cpu_tune.pid_params      # [B,3,3]
cpu_diag = cpu_tune.diagnostics    # [B,3,4]

w("tune_K.bin", ident_K, np.float32)
w("tune_T.bin", ident_T, np.float32)
w("tune_L.bin", ident_L, np.float32)
w("tune_lambda.bin", ident_lambda, np.float32)
run_stage("tuning", B)
npu_pid = r("tune_pid.bin", np.float32, B * 9).reshape(B, 3, 3)
npu_diag = r("tune_diag.bin", np.float32, B * 12).reshape(B, 3, 4)

ae_pid, re_pid = errs(npu_pid, cpu_pid)
ae_diag, re_diag = errs(npu_diag, cpu_diag)
report["stages"]["2_tuning"] = {
    "pid_max_abs": ae_pid, "pid_max_rel": re_pid,
    "diag_max_abs": ae_diag, "diag_max_rel": re_diag,
}

# ============================================================
# Stage 3: fopdt_batch_rollout_score  (simulate 3 rule-candidates -> best)
# per loop: batch=1, candidates=3 (ZN/IMC/CC).  rollout shares candidates across
# its batch axis, so each loop is its own batch=1 invocation.
# ============================================================
roll_rows_npu = np.zeros((B, 8), dtype=np.float32)
roll_idx_npu = np.zeros(B, dtype=np.int32)
roll_rows_cpu = np.zeros((B, 8), dtype=np.float32)
roll_idx_cpu = np.zeros(B, dtype=np.int32)

for b in range(B):
    a_coef = np.array([np.exp(-DT / ident_T[b])], dtype=np.float32)
    b_coef = np.array([ident_K[b] * (1.0 - a_coef[0])], dtype=np.float32)
    delay = np.array([int(round(ident_L[b] / DT))], dtype=np.int32)
    y0 = np.array([0.0], dtype=np.float32)
    sp = np.array([1.0], dtype=np.float32)
    kp = npu_pid[b, :, 0].astype(np.float32).copy()
    ki = npu_pid[b, :, 1].astype(np.float32).copy()
    kd = npu_pid[b, :, 2].astype(np.float32).copy()

    # CPU golden via reference (same model + candidates)
    rc = rollref.FopdtClosedLoopCase(a_coef, b_coef, delay, y0, sp, kp, ki, kd, DT, SIM)
    cres, cidx = rollref.fopdt_closed_loop_score(rc)
    roll_rows_cpu[b] = cres[0]
    roll_idx_cpu[b] = cidx[0]

    # NPU
    w("roll_a.bin", a_coef, np.float32)
    w("roll_b.bin", b_coef, np.float32)
    w("roll_delay.bin", delay, np.int32)
    w("roll_y0.bin", y0, np.float32)
    w("roll_sp.bin", sp, np.float32)
    w("roll_kp.bin", kp, np.float32)
    w("roll_ki.bin", ki, np.float32)
    w("roll_kd.bin", kd, np.float32)
    run_stage("rollout", 1, 3, SIM, 3)
    roll_rows_npu[b] = r("roll_best_result.bin", np.float32, 8)
    roll_idx_npu[b] = r("roll_best_idx.bin", np.int32, 1)[0]

ae_roll, re_roll = errs(roll_rows_npu[:, 0], roll_rows_cpu[:, 0])  # best_score
ae_pidsel, re_pidsel = errs(roll_rows_npu[:, 1:4], roll_rows_cpu[:, 1:4])  # best kp/ki/kd
report["stages"]["3_rollout"] = {
    "best_score_max_abs": ae_roll, "best_score_max_rel": re_roll,
    "best_pid_max_abs": ae_pidsel, "best_pid_max_rel": re_pidsel,
    "best_idx_diff": int(np.sum(roll_idx_npu != roll_idx_cpu)),
    "npu_winning_rule": roll_idx_npu.tolist(), "cpu_winning_rule": roll_idx_cpu.tolist(),
}

# host glue: simulate winning controller closed-loop -> pv trajectory
RULE = ("ZN", "IMC", "CC")


def simulate_pv(a, b, delay, y0, sp, kp, ki, kd, steps):
    delay = max(0, int(delay))
    dl = min(delay + 1, 128)
    u_hist = [0.0] * dl
    y = float(y0)
    integral = 0.0
    prev_error = float(sp) - y
    pv = np.zeros(steps, dtype=np.float32)
    for k in range(steps):
        error = float(sp) - y
        integral += error * DT
        derivative = (error - prev_error) / max(DT, 1e-6)
        u = kp * error + ki * integral + kd * derivative
        u = min(max(u, -10.0), 10.0)
        delayed = u_hist[0]
        if dl > 1:
            u_hist[:-1] = u_hist[1:]
        u_hist[-1] = u
        y = a * y + b * delayed
        pv[k] = y
        prev_error = error
    return pv


pv = np.zeros((B, SIM), dtype=np.float32)
for b in range(B):
    a_coef = float(np.exp(-DT / ident_T[b]))
    b_coef = float(ident_K[b] * (1.0 - a_coef))
    delay = int(round(ident_L[b] / DT))
    win_kp, win_ki, win_kd = roll_rows_npu[b, 1], roll_rows_npu[b, 2], roll_rows_npu[b, 3]
    pv[b] = simulate_pv(a_coef, b_coef, delay, 0.0, 1.0, win_kp, win_ki, win_kd, SIM)

# ============================================================
# Stage 4: control_performance_metrics  (winning controller pv -> 20 metrics)
# ============================================================
sp_arr = np.ones((B, SIM), dtype=np.float32)
lsl = np.full(B, 0.9, dtype=np.float32)
usl = np.full(B, 1.1, dtype=np.float32)
mvv = np.full(B, 0.0, dtype=np.float32)
cpu_metrics = perfref.control_performance_metrics(pv, sp_arr, lsl, usl, mvv,
                                                  sample_interval=1.0, settle_band=0.1)

w("perf_pv.bin", pv, np.float32)
w("perf_sp.bin", sp_arr, np.float32)
w("perf_lsl.bin", lsl, np.float32)
w("perf_usl.bin", usl, np.float32)
w("perf_mvv.bin", mvv, np.float32)
run_stage("metrics", B, SIM)
npu_metrics = r("perf_metrics.bin", np.float32, B * 20).reshape(B, 20)

ae_m, re_m = errs(npu_metrics, cpu_metrics)
report["stages"]["4_metrics"] = {"metrics_max_abs": ae_m, "metrics_max_rel": re_m}

# ============================================================
# Report
# ============================================================
print("\n================ FOPDT FULL-CHAIN E2E (NPU vs CPU golden) ================")
print(f"config: B={B} loops, fit N={N_FIT} M={M} candidates, rollout/pv SIM={SIM}\n")
print("Stage 1  fit       : best_k abs=%.3e rel=%.3e | best_idx_diff=%d (truth_index=%d)"
      % (report["stages"]["1_fit"]["best_k_max_abs"], report["stages"]["1_fit"]["best_k_max_rel"],
         report["stages"]["1_fit"]["best_idx_diff"], M // 2))
print("Stage 2  tuning    : pid abs=%.3e rel=%.3e | diag abs=%.3e rel=%.3e"
      % (ae_pid, re_pid, ae_diag, re_diag))
print("Stage 3  rollout   : best_score abs=%.3e rel=%.3e | best_pid abs=%.3e rel=%.3e | rule_diff=%d"
      % (ae_roll, re_roll, ae_pidsel, re_pidsel, report["stages"]["3_rollout"]["best_idx_diff"]))
print("Stage 4  metrics   : metrics abs=%.3e rel=%.3e (20 indicators x %d loops)"
      % (ae_m, re_m, B))
print("\nidentified model (NPU fit): K=%s" % np.round(ident_K, 4).tolist())
print("                           T=%s L=%s" % (ident_T.tolist(), ident_L.tolist()))
print("winning rule per loop (NPU): %s"
      % [RULE[i] for i in roll_idx_npu.tolist()])
print("final best PID per loop (NPU): kp=%s" % np.round(roll_rows_npu[:, 1], 4).tolist())
print("==========================================================================")

with open(os.path.join(WORK, "e2e_report.json"), "w") as f:
    json.dump(report, f, indent=2)
print("report json -> %s/e2e_report.json" % WORK)
