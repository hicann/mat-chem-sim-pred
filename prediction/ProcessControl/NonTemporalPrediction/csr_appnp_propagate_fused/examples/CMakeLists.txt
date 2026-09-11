add_executable(test_aclnn_csr_appnp_propagate_fused
    test_aclnn_csr_appnp_propagate_fused.cpp)
target_include_directories(test_aclnn_csr_appnp_propagate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../op_host)
target_link_directories(test_aclnn_csr_appnp_propagate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(test_aclnn_csr_appnp_propagate_fused PRIVATE
    csr_appnp_propagate_fused_host ascendcl)
add_test(NAME csr_appnp_propagate_smoke
    COMMAND test_aclnn_csr_appnp_propagate_fused 0)
