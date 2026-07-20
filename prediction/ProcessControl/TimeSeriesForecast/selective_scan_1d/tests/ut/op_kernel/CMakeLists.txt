if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(selective_scan_1d "ascend910B1" "" )
endif()
