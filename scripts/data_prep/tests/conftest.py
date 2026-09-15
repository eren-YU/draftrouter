# -*- coding: utf-8 -*-
"""pytest conftest:把 scripts/data_prep 加入 sys.path,使单测可直接 import 各模块。

本地 Windows 无 GPU、无 datasets/transformers/huggingface_hub;被测模块的重依赖全部
延迟导入(函数体内 import),纯逻辑部分(core/field_regex)不触发任何网络或重依赖。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
