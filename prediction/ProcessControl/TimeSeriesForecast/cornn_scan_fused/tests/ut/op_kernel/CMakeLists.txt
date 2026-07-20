if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(cornn_scan_fused "ascend910B1" "" )
endif()
