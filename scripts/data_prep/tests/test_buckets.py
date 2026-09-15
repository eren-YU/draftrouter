"""三分位桶与分位数:全局冻结口径、桶指派、分位数插值(纯逻辑)。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest  # noqa: E402
from core import assign_bucket, compute_tertile_boundaries, freeze_buckets, quantiles  # noqa: E402


def test_quantiles_known_values():
    assert quantiles(range(1, 101)) == {"p10": 10.9, "p50": 50.5, "p90": 90.1, "p99": 99.01}
    assert quantiles([5]) == {"p10": 5.0, "p50": 5.0, "p90": 5.0, "p99": 5.0}
    with pytest.raises(ValueError):
        quantiles([])


def test_tertile_boundaries():
    b = compute_tertile_boundaries(range(1, 100))  # 1..99,三分位 ≈ 33.67 / 66.33
    assert len(b) == 2
    assert b[0] < b[1]
    assert b[0] == pytest.approx(33 + 2 / 3)
    assert b[1] == pytest.approx(66 + 1 / 3)


def test_assign_bucket():
    b = [100.0, 200.0]
    assert assign_bucket(50, b) == "short"
    assert assign_bucket(100, b) == "short"   # 边界含在左桶(<=)
    assert assign_bucket(150, b) == "mid"
    assert assign_bucket(200.1, b) == "long"
    with pytest.raises(ValueError):
        assign_bucket(1, [1.0])


def test_freeze_buckets_merges_scenes_globally():
    # C4:边界在 tuning+report 全量(三场景合并)上计算,不是分场景
    fb = freeze_buckets({"chinese": [10, 20, 30], "code": [40, 50, 60], "rag": [70, 80, 90]})
    assert fb["boundaries"] == [pytest.approx(36.67, abs=0.01),
                                pytest.approx(63.33, abs=0.01)]
    assert fb["version"] == 1
