# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
# 工具链由顶层 CMakeLists.txt 经 cmake/op_prelude.cmake + cmake/operator.cmake 提供；
# 本文件只声明"本算子是哪种形态"，形态定义在 cmake/operator.cmake 里，
# 不 include 单元目录外的路径、不做 project 声明、不探测 CANN（CodeCheck 要求）。
matchem_add_kernel_host_opdef(gemnet_quadruplet_geometry_fused)
