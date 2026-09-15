"""四池构造(C12):tuning 50 / report 50 / data-valid 200 / router-train 独立池。

【云端执行方式】
    ssh autodl
    /root/miniconda3/envs/draftrouter/bin/python scripts/data_prep/build_pools.py
(纯逻辑,不需要网络加速;只需 ALCE/repobench 数据已下载到 /root/autodl-tmp/data/)

- 分层抽样(按场景:中文/代码/RAG),seed=0 确定性;
- 样本 ID 落盘 data/pools/*.ids.json;脚本内 assert 两两零交集并打印报告;
- router-train 池 P0 只落盘 ID(默认 200),不构造样本正文。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_records import build_base_records  # noqa: E402
from common import data_root, save_json  # noqa: E402
from core import POOL_SIZES, assert_disjoint, assign_pools  # noqa: E402


def build_and_write_pools(records: list[dict], root: Path | None = None,
                          seed: int = 0, sizes: dict | None = None) -> dict[str, list[str]]:
    """四池分配 + ID 落盘;返回 {pool: [sample_id]}。"""
    pools = assign_pools(records, seed=seed, sizes=sizes)
    assert_disjoint(pools)  # 双保险:落盘前再断言一次
    root = (root or data_root()) / "pools"
    for pool, ids in pools.items():
        save_json({
            "pool": pool,
            "size": len(ids),
            "seed": seed,
            "sample_ids": ids,
            "scene_counts": _scene_counts(records, ids),
        }, root / f"{pool}.ids.json")
    return pools


def _scene_counts(records: list[dict], ids: list[str]) -> dict[str, int]:
    scene_of = {r["sample_id"]: r["scene"] for r in records}
    counts: dict[str, int] = {}
    for i in ids:
        counts[scene_of[i]] = counts.get(scene_of[i], 0) + 1
    return counts


def main() -> None:
    records = build_base_records()
    print(f"[pools] 基础记录 {len(records)} 条,场景分布: "
          f"{ {s: sum(1 for r in records if r['scene'] == s) for s in ('chinese', 'code', 'rag')} }")
    pools = build_and_write_pools(records, sizes=POOL_SIZES)
    for pool, ids in pools.items():
        print(f"[pools] {pool}: {len(ids)} 条 → data/pools/{pool}.ids.json")
    print("[pools] 两两零交集断言通过。")


if __name__ == "__main__":
    main()
