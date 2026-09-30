# Algorithm

For target node `i` and source `j`, the operator computes

`logit_ij = beta * dot(normalized[i], normalized[j])`,

applies a numerically stable softmax within target `i`'s CSR row, and returns
`sum_j softmax(logit_ij) * features[j]`.

One AIV work item owns one target row. Scores are accumulated in Unified
Buffer, shifted by the row maximum, exponentiated, normalized, and used for the
channel reduction. This removes edge score/value materialization and repeated
full-graph indexing while retaining the exact AGNN formula.
