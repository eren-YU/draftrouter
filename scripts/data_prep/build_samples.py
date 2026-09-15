# -*- coding: utf-8 -*-
"""样本构造(WP2 第 3/7 条):RAG short/long 扩长 + 输出形态配对 → data/samples/*.jsonl。

【云端执行方式】
    ssh autodl
    export HF_HUB_OFFLINE=1   # tokenizer 读本地缓存;不需要网络加速
    /root/miniconda3/envs/draftrouter/bin/python scripts/data_prep/build_samples.py

- 同一批 ALCE 原生样本 = short 档;long 档按原始段落顺序拼接,Qwen3 tokenizer 计长,
  落入 [7500,8192] 为主判档;段落不足的样本记日志(core.build_long_input.insufficient);
- 同一批 RAG long 样本配 verbatim / non-verbatim 两套指令;中文 JSON vs 自然段落配对;
- 每条样本含 sample_id/scene/dataset/length_bucket/input_text/output_mode;
- 长度桶边界读 data/length_buckets.json(C4 已冻结);不存在时现场在 tuning+report 池
  上计算并落盘(等价于 length_stats 的口径,保证单独跑本脚本也能闭环)。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_records import (build_base_records, load_pools,  # noqa: E402
                          qwen_tokenize)
from common import data_root, load_json, save_json, write_jsonl  # noqa: E402
from core import (MAX_INPUT_TOKENS, RAG_LONG_HI, RAG_LONG_LO,  # noqa: E402
                  assign_bucket, build_long_input, expand_output_modes, make_sample)
from field_regex import detect_fields  # noqa: F401 —— 中文段落场景指令内嵌四字段协议


def _bucket_for(scene: str, token_len: int, boundaries: list[float]) -> str:
    """长度桶:中文/代码按自然长度三分位;RAG 用 short/long 主判档命名。"""
    if scene == "rag":
        if token_len >= RAG_LONG_LO:
            return "long"
        return "short"
    return assign_bucket(token_len, boundaries)


def build_samples(records: list[dict], tokenize, boundaries: list[float]) -> tuple[list, list]:
    """由基础记录构造最终样本;返回 (samples, insufficients)。

    - RAG:每条基础记录产 short(原生)+ long(扩长)两组正文,再各配 output_mode;
      long 超出 8192(C5)剔除并记录;不足 7500 标 insufficient 记日志;
      short/long 共享同一 sample_id 前缀(rag-alce-XXXXXX),WP3 按前缀配对。
    - 中文:question+篇章为输入,配 json/paragraph 两种形态;
    - 代码:prompt 原生,单一形态。
    """
    samples: list[dict] = []
    insufficients: list[dict] = []
    for r in records:
        scene = r["scene"]
        if scene == "rag":
            # short:原生(question + 全部原始段落)
            short_text = r["question"] + "\n" + "\n".join(r["passages"])
            short_len = len(tokenize(short_text))
            if short_len > MAX_INPUT_TOKENS:
                insufficients.append({"sample_id": r["sample_id"], "variant": "short",
                                      "reason": f"overlong {short_len} > {MAX_INPUT_TOKENS}"})
            else:
                b = _bucket_for(scene, short_len, boundaries)
                for style, fmt in expand_output_modes(scene):
                    samples.append(make_sample(r["sample_id"], scene, r["dataset"], b,
                                               short_text, style, fmt))
            # long:按原始段落顺序扩长
            built = build_long_input(r["question"], r["passages"], tokenize)
            if built["insufficient"]:
                insufficients.append({"sample_id": r["sample_id"], "variant": "long",
                                      "reason": f"insufficient {built['token_len']} < {RAG_LONG_LO}",
                                      "passages_used": built["passages_used"]})
                continue
            long_id = r["sample_id"] + "-long"
            for style, fmt in expand_output_modes(scene):
                samples.append(make_sample(long_id, scene, r["dataset"], "long",
                                           built["input_text"], style, fmt))
        else:
            text = r["question"] + "\n" + "\n".join(r["passages"])
            n = len(tokenize(text))
            if n > MAX_INPUT_TOKENS:
                insufficients.append({"sample_id": r["sample_id"], "variant": "native",
                                      "reason": f"overlong {n} > {MAX_INPUT_TOKENS}"})
                continue
            b = _bucket_for(scene, n, boundaries)
            for style, fmt in expand_output_modes(scene):
                samples.append(make_sample(r["sample_id"], scene, r["dataset"], b,
                                           text, style, fmt))
    return samples, insufficients


def main() -> None:
    root = data_root()
    records = build_base_records(root)
    if not records:
        print("[samples] 基础记录为空:先执行 download_data.py", file=sys.stderr)
        sys.exit(2)

    # 长度桶边界(C4 冻结值;缺失时现场按 tuning+report 池计算,口径与 length_stats 一致)
    buckets_path = root / "length_buckets.json"
    if buckets_path.exists():
        boundaries = load_json(buckets_path)["boundaries"]
    else:
        from length_stats import compute_length_stats, freeze_global_buckets
        pools = load_pools(root)
        if not pools:
            print("[samples] 无 length_buckets.json 且无 pools,先跑 build_pools.py", file=sys.stderr)
            sys.exit(2)
        compute_length_stats(records, qwen_tokenize)
        boundaries = freeze_global_buckets(records, pools)["boundaries"]
        save_json({"boundaries": boundaries,
                   "rule": "computed on demand by build_samples (same as length_stats)",
                   "version": 1}, buckets_path)
    print(f"[samples] 长度桶边界: {boundaries}")

    samples, insufficients = build_samples(records, qwen_tokenize, boundaries)

    # 按场景落盘 jsonl
    by_scene: dict[str, list] = {}
    for s in samples:
        by_scene.setdefault(s["scene"], []).append(s)
    for scene, items in sorted(by_scene.items()):
        path = root / "samples" / f"{scene}.jsonl"
        write_jsonl(items, path)
        print(f"[samples] {scene}: {len(items)} 条 → data/samples/{scene}.jsonl")

    # 长构建日志(段落不足/超长剔除)
    save_json({"insufficient_or_dropped": insufficients}, root / "samples" / "build_log.json")
    n_ins = sum(1 for x in insufficients if "insufficient" in x["reason"])
    n_drop = len(insufficients) - n_ins
    print(f"[samples] RAG long 段落不足 {n_ins} 条、超长剔除 {n_drop} 条,详见 build_log.json")
    print(f"[samples] 合计 {len(samples)} 条样本。")


if __name__ == "__main__":
    main()
