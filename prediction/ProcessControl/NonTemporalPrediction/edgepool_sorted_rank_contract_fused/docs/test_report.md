# Test report

- Clean Ascend910B1-compatible build and Ascend910B3 execution passed.
- ACLNN C++ smoke: passed.
- Python binding smoke: passed.
- NumPy oracle: descending matching, self-loop, singleton, bad permutation, and
  out-of-range endpoint tests passed.
- Guarded dispatch: validated NPU custom path and fallback behavior passed.
- Trained PROTEINS B16/B32/B128 A/B: cluster, pooled feature, new edge index,
  batch, resident logits, and predictions exact.
- Strong resident full-model baseline: `5.371/7.178/14.099 ms`; custom:
  `4.654/5.643/9.904 ms`, or `13.34%/21.39%/29.75%` lower latency.
- Same-host best CPU: `63.712/151.547/413.883 ms`; custom dynamic-graph
  host-to-host: `4.989/5.943/10.082 ms`, or `12.77x/25.50x/41.05x` faster.
- Two current streams, independent output pointers, 500 calls, and zero
  post-cache memory growth passed. Undersized workspace is rejected by CTest.
- 2026-08-16 custom OPP: FP32 TorchAir `fullgraph=True` passed with exact
  fixed-capacity cluster, selected-edge, score, and device-count outputs.

The current release is FP32 inference-only. Host output memory is caller-owned,
and every in-flight invocation must use distinct outputs. Returning dynamic
valid views requires a count copy to the host.

The local checkpoint reaches 74.89% CPU test accuracy. Measured B128 task
accuracy is 74.22% on CPU and custom NPU with 100% prediction agreement. The
full 223-graph NPU task sweep remains a submission-readiness item.
