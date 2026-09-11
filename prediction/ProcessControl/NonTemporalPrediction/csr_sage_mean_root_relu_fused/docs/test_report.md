# Test Report

- Clean Ascend910B3 Release build: passed.
- NumPy deterministic, random/empty-row, dispatch, and Host contract: 4 passed.
- ACL smoke output: `[3,3,2,0]`, matching the CPU reference.
- Official PyG SAGEConv checkpoint SHA256:
  `fa6ff9cc61484d51dda7b2b69d87183835aaa84f06c8ee80b18266a05c8fe047`.
- Cora validation/test accuracy: `0.7220/0.7120`.
- Five model shapes: maximum error `4.77e-7`, task metric unchanged.

The benchmark includes both learned SAGEConv projections and the classifier in
model E2E timing and does not truncate Cora edges.
