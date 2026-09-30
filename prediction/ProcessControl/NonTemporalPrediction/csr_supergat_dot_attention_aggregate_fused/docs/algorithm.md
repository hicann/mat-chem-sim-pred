# Algorithm

For each destination node `i`, head `h`, and CSR neighbor `j`, the operator
computes `s_ijh = LeakyReLU(dot(z_ih,z_jh)/sqrt(C))`, normalizes `s` across the
CSR row with a max-subtracted softmax, and emits `sum_j alpha_ijh*z_jh`.

One AI Core processes destination rows independently. The row maximum, sum,
and weighted accumulation stay local, removing edge score, softmax, and message
tensors from global memory. Empty rows produce zeros. The mathematical scope is
the inference message stage; SuperGAT's training-only negative-edge objective
remains in PyG.
