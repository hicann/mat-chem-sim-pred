if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(autocorr_fused_aggregate "ascend910B1" "" )
endif()
