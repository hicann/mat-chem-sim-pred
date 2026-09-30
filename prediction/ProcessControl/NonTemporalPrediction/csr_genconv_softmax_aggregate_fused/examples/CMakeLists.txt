add_executable(test_aclnn_csr_genconv_softmax_aggregate_fused
    test_aclnn_csr_genconv_softmax_aggregate_fused.cpp)
target_include_directories(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include ../op_host)
target_link_directories(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64)
set(ASCEND_DRIVER_LIBS "")
if(EXISTS "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so")
    target_link_directories(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
        /usr/local/Ascend/driver/lib64/driver)
    target_link_options(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
        "-Wl,-rpath-link,/usr/local/Ascend/driver/lib64/driver")
    list(APPEND ASCEND_DRIVER_LIBS ascend_hal)
endif()
target_link_libraries(test_aclnn_csr_genconv_softmax_aggregate_fused PRIVATE
    csr_genconv_softmax_aggregate_fused_host ascendcl ${ASCEND_DRIVER_LIBS})
add_test(NAME csr_genconv_softmax_aggregate_smoke
    COMMAND test_aclnn_csr_genconv_softmax_aggregate_fused 0)
set_tests_properties(csr_genconv_softmax_aggregate_smoke PROPERTIES
    ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:${CMAKE_BINARY_DIR}/lib:${ASCEND_CANN_PACKAGE_PATH}/lib64:$ENV{LD_LIBRARY_PATH}")
