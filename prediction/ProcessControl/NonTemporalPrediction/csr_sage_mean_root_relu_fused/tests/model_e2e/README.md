# Model E2E Reproduction

`benchmark_pyg_sage_e2e.py` instantiates maintained PyG `SAGEConv`, trains or
loads the Cora checkpoint, and compares official PyG, a correct resident
torch_npu rewrite, and the custom operator. It evaluates the Cora test mask and
includes both projections plus the classifier in every E2E timing.

`benchmark_shape_dispatch.py` validates the small-graph boundary. Immutable raw
outputs are `cora_pyg_sageconv_formal_20260801.json` and
`csr_sage_dispatch_20260801.json`.
