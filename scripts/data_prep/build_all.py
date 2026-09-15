# -*- coding: utf-8 -*-
"""WP2 主入口:下载 → 构样 → 四池 → 四字段筛查 → 长度统计,每步产物落盘并打印汇总。

【云端执行方式】
    ssh autodl
    # 第 1 步(下载)需要网络加速;后续步骤不需要:
    source /etc/network_turbo && export HF_HUB_DISABLE_XET=1 \
        && /root/miniconda3/envs/draftrouter/bin/python scripts/data_prep/build_all.py --skip-after download
    # 或一次性全跑(下载也需加速):
    source /etc/network_turbo && export HF_HUB_DISABLE_XET=1 \
        && /root/miniconda3/envs/draftrouter/bin/python scripts/data_prep/build_all.py

执行顺序与产物:
    1. download_data  → data/data_manifest.json
    2. build_pools    → data/pools/*.ids.json(两两零交集,seed=0)
    3. length_stats   → data/length_stats.json + data/length_buckets.json(冻结)
    4. build_samples  → data/samples/*.jsonl + build_log.json
    5. field_screen   → data/field_screen_data_valid.json(每类字段 >=60% 才达标)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    ap = argparse.ArgumentParser(description="WP2 数据链路一体化构建")
    ap.add_argument("--skip-after", choices=["download", "pools", "length", "samples", "screen"],
                    default=None, help="执行到指定步骤后停止(用于分开开关加速)")
    ap.add_argument("--limit-per-ds", type=int, default=None,
                    help="每个数据集最多加载 N 条(云端试跑用);省略=全量")
    args = ap.parse_args()
    stop = args.skip_after
    summary: dict = {}

    # 1. 下载(需网络加速)
    import download_data
    summary["download"] = download_data.run()
    print("[all] download 完成:", json.dumps(summary["download"], ensure_ascii=False))
    if summary["download"]["failed"]:
        print(f"[all] 警告:{len(summary['download']['failed'])} 个数据源下载失败,后续步骤可能缺场景。")
    if stop == "download":
        _finish(summary)

    # 2. 四池(纯逻辑)
    import build_pools
    from base_records import build_base_records
    from common import data_root
    from core import POOL_SIZES
    records = build_base_records(data_root(), limit_per_ds=args.limit_per_ds)
    pools = build_pools.build_and_write_pools(records, sizes=POOL_SIZES)
    summary["pools"] = {k: len(v) for k, v in pools.items()}
    print("[all] pools 完成:", json.dumps(summary["pools"], ensure_ascii=False))
    if stop == "pools":
        _finish(summary)

    # 3. 长度统计 + 桶冻结(需 Qwen3 tokenizer)
    import length_stats
    from base_records import qwen_tokenize
    length_stats.compute_length_stats(records, qwen_tokenize)
    buckets = length_stats.freeze_global_buckets(records, pools)
    from common import save_json
    save_json({s: length_stats.quantiles(
        [r.get("token_len", 0) for r in records if r["scene"] == s]) for s in
        sorted({r["scene"] for r in records})}, data_root() / "length_stats.json")
    save_json(buckets, data_root() / "length_buckets.json")
    summary["length"] = {"per_scene": json.load(open(data_root() / "length_stats.json", encoding="utf-8")),
                         "bucket_boundaries": buckets["boundaries"]}
    print("[all] length 完成,边界:", buckets["boundaries"])
    if stop == "length":
        _finish(summary)

    # 4. 样本构造
    import build_samples
    samples, insufficients = build_samples.build_samples(records, qwen_tokenize, buckets["boundaries"])
    from common import write_jsonl
    by_scene: dict[str, list] = {}
    for s in samples:
        by_scene.setdefault(s["scene"], []).append(s)
    for scene, items in sorted(by_scene.items()):
        write_jsonl(items, data_root() / "samples" / f"{scene}.jsonl")
    save_json({"insufficient_or_dropped": insufficients},
              data_root() / "samples" / "build_log.json")
    summary["samples"] = {k: len(v) for k, v in by_scene.items()}
    summary["samples_dropped"] = len(insufficients)
    print("[all] samples 完成:", json.dumps(summary["samples"], ensure_ascii=False))
    if stop == "samples":
        _finish(summary)

    # 5. 四字段筛查(G4 数据侧门槛)
    import field_screen
    valid_ids = set(pools.get("data_valid", []))
    report = field_screen.screen_records([r for r in records if r["sample_id"] in valid_ids],
                                         tokenize=qwen_tokenize)
    save_json(report, data_root() / "field_screen_data_valid.json")
    summary["field_screen"] = {"coverage": report["coverage"], "pass": report["pass"]}
    _finish(summary)


def _finish(summary: dict) -> None:
    print("\n===== WP2 构建汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    sys.exit(0)


if __name__ == "__main__":
    main()
