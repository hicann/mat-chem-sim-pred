# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

add_executable(test_aclnn_farthest_point_sampling_fused test_aclnn_farthest_point_sampling_fused.cpp)
target_include_directories(test_aclnn_farthest_point_sampling_fused PRIVATE ../op_host ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_directories(test_aclnn_farthest_point_sampling_fused PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_aclnn_farthest_point_sampling_fused PRIVATE farthest_point_sampling_fused_host ascendcl)
