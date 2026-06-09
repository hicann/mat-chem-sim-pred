if ((UT_TEST_ALL OR OP_KERNEL_UT) AND NOT UT_DONE)
    AddOpTestCase(mesh_graph_net "ascend910B1" "" )
endif()
