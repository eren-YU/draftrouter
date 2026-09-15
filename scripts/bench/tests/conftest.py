"""本地单测公共夹具:把 scripts/bench 目录加进 sys.path(本地无 vllm 也能跑)。"""

import os
import sys

BENCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)
