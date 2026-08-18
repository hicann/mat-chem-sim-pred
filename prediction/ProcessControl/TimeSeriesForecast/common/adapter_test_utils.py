# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Shared test doubles and dynamic adapter loading for operator tests."""

import importlib.util
from pathlib import Path


def load_adapter(operator_root, module_name):
    path = Path(operator_root) / "integration" / "adapter.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load adapter from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Device:
    type = "npu"


class FakeTensor:
    def __init__(self, shape, dtype="torch.float32", contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.device = Device()
        self._contiguous = contiguous

    def is_contiguous(self):
        return self._contiguous
