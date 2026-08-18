<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# Algorithm

For each batch item, begin at point zero. After selecting a center, update the
minimum squared distance of every point to any selected center and choose the
first point with the greatest minimum distance. This preserves deterministic
tie handling without materializing a pairwise distance matrix.
