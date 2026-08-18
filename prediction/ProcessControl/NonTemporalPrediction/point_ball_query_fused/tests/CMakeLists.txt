# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME point_ball_query_fused_reference COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test_point_ball_query_fused.py -q)
