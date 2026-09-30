# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
add_executable(test_aclnn_edgepool_sorted_rank_contract_fused
    test_aclnn_edgepool_sorted_rank_contract_fused.cpp)
target_link_libraries(test_aclnn_edgepool_sorted_rank_contract_fused
    edgepool_sorted_rank_contract_fused_host ascendcl)
add_test(NAME edgepool_sorted_rank_contract_smoke
    COMMAND test_aclnn_edgepool_sorted_rank_contract_fused)
set_tests_properties(edgepool_sorted_rank_contract_smoke PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}")
