"""长度分布与三分位桶(C4):Qwen3 tokenizer 分词,按场景 p10/p50/p90/p99 + 全局三分位桶。

【云端执行方式】
    ssh autodl
    /root/miniconda3/envs/draftrouter/bin/python scripts/data_prep/length_stats.py
(需要 transformers + 本地 Qwen3-8B 缓存:export HF_HUB_OFFLINE=1;不需要网络加速)

- 桶边界只在 tuning+report 池样本的自然长度上全局计算(三场景合并),一次计算后冻结
  落盘 data/length_buckets.json;后续 build_samples / 分析脚本只读此文件,不得重算。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_records import build_base_records, load_pools, qwen_tokenize  # noqa: E402
from common import data_root, save_json  # noqa: E402
from core import freeze_buckets, quantiles  # noqa: E402


def compute_length_stats(records: list[dict], tokenize) -> dict:
    """按场景统计自然长度分位数;并返回每条记录的长度(供桶冻结与超长剔除)。"""
    by_scene: dict[str, list[float]] = {}
    for r in records:
        n = len(tokenize(r["question"]))
        n += sum(len(tokenize("\n" + p)) for p in r["passages"])
        r["token_len"] = n
        by_scene.setdefault(r["scene"], []).append(n)
    return {scene: quantiles(vals) for scene, vals in sorted(by_scene.items())}


def freeze_global_buckets(records: list[dict], pools: dict[str, list[str]]) -> dict:
    """C4 口径:边界 = tuning+report 两池样本(三场景合并)自然长度的全局三分位。"""
    length_of = {r["sample_id"]: r.get("token_len") for r in records}
    calib_ids = pools.get("tuning", []) + pools.get("report", [])
    calib_lengths = [length_of[i] for i in calib_ids if length_of[i] is not None]
    if len(calib_lengths) < 10:
        raise RuntimeError(f"tuning+report 池长度样本不足({len(calib_lengths)}),先跑 build_pools")
    return freeze_buckets({"merged": calib_lengths})


def main() -> None:
    root = data_root()
    records = build_base_records(root, tokenize=qwen_tokenize)
    pools = load_pools(root)
    if not pools:
        print("[length] 未找到 data/pools/*.ids.json,先执行 build_pools.py", file=sys.stderr)
        sys.exit(2)

    per_scene = compute_length_stats(records, qwen_tokenize)
    buckets = freeze_global_buckets(records, pools)

    save_json(per_scene, root / "length_stats.json")
    save_json(buckets, root / "length_buckets.json")
    print("[length] 按场景分位数:")
    print(json.dumps(per_scene, ensure_ascii=False, indent=2))
    print(f"[length] 冻结的全局三分位边界: {buckets['boundaries']} → data/length_buckets.json")


if __name__ == "__main__":
    main()
