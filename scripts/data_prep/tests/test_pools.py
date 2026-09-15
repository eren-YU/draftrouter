# -*- coding: utf-8 -*-
"""四池分配:零交集、确定性、分层、规模;与 mock 记录一起测(不触网)。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest  # noqa: E402

from core import POOL_SIZES, assert_disjoint, assign_pools  # noqa: E402


def _make_records(n_per_scene=200, scenes=("chinese", "code", "rag")):
    records = []
    for s in scenes:
        for i in range(n_per_scene):
            records.append({"sample_id": f"{s}-{i:04d}", "scene": s})
    return records


def test_four_pools_disjoint_and_sizes():
    records = _make_records()
    pools = assign_pools(records, seed=0, sizes=POOL_SIZES)
    assert set(pools) == {"tuning", "report", "data_valid", "router_train"}
    for pool in ("tuning", "report"):
        assert len(pools[pool]) == 50
    assert len(pools["data_valid"]) == 200
    assert len(pools["router_train"]) == 200
    # 两两零交集
    conflicts = assert_disjoint(pools)
    assert conflicts == []
    names = sorted(pools)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (set(pools[names[i]]) & set(pools[names[j]]))


def test_pool_assignment_deterministic():
    records = _make_records()
    a = assign_pools(records, seed=0)
    b = assign_pools(records, seed=0)
    c = assign_pools(records, seed=1)
    assert a == b, "同 seed 必须逐 ID 一致"
    assert a != c, "不同 seed 应产生不同划分(抽样随机性)"


def test_stratified_by_scene():
    records = _make_records()
    pools = assign_pools(records, seed=0)
    for pool in ("tuning", "data_valid"):
        scene_of = {r["sample_id"]: r["scene"] for r in records}
        counts = {}
        for i in pools[pool]:
            counts[scene_of[i]] = counts.get(scene_of[i], 0) + 1
        # 三场景均有代表,且配额接近均分(各场景可用量相同)
        assert set(counts) == {"chinese", "code", "rag"}
        for n in counts.values():
            assert abs(n - len(pools[pool]) / 3) <= 2


def test_overlap_raises():
    bad = {"a": ["x", "y"], "b": ["y", "z"]}
    with pytest.raises(AssertionError):
        assert_disjoint(bad)


def test_router_train_independent_pool():
    records = _make_records(n_per_scene=100)  # 总量恰好够前三池 + router_train
    pools = assign_pools(records, seed=0)
    used = set(pools["tuning"]) | set(pools["report"]) | set(pools["data_valid"])
    assert not (set(pools["router_train"]) & used), "router-train 必须与前三池完全独立"
