# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
add_executable(test_aclnn_sort_aggregation_topk_fused
    test_aclnn_sort_aggregation_topk_fused.cpp)
target_include_directories(test_aclnn_sort_aggregation_topk_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../op_host)
target_link_directories(test_aclnn_sort_aggregation_topk_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_aclnn_sort_aggregation_topk_fused PRIVATE
    sort_aggregation_topk_fused_host ascendcl)
add_test(NAME sort_aggregation_topk_smoke
    COMMAND test_aclnn_sort_aggregation_topk_fused 0)
