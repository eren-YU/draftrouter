#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单组离线 bench CLI 入口(WP3 / C1)。

用法示例(云端):
  /root/miniconda3/envs/draftrouter/bin/python scripts/bench/run_group.py \
      --spec-config A --dataset dureader --length-bucket long[7500,8192] \
      --bs 1 --output-mode fixed512 --data-jsonl data/tuning_zh.jsonl \
      --out-json results/p0-smoke/A_dureader_long_bs1.json --seed 0

约定:
- 每个 spec 配置新建 LLM 实例(本脚本每次调用只构造一个 LLM,
  函数内不缓存、不跨 spec 配置复用,靠调用结构强制);
- 引擎启动后先跑 WARMUP_REQUESTS 条预热并丢弃;
- 组顺序用 seed 打乱并记录。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# 保证直接以脚本路径运行时能 import 同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import EngineConfig, apply_no_think, build_sampling_params, chat_template_kwargs
from configs import resolve_spec_config
from prompts import TEMPLATE_VERSION, render
from runner import VALID_BS, WARMUP_REQUESTS, run_one_group, shuffle_group_order
from schema import build_result, fill_metrics, write_result

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_group")


def load_samples(jsonl_path: str) -> list[dict]:
    """读样本 jsonl:至少含 prompt 字段与 sample_id;保留 text/question 供模板。"""
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "prompt" not in rec:
                raise KeyError(f"{jsonl_path} 第 {i+1} 行缺少 prompt 字段")
            if "sample_id" not in rec:
                raise KeyError(f"{jsonl_path} 第 {i+1} 行缺少 sample_id 字段")
            samples.append(rec)
    if not samples:
        raise ValueError(f"{jsonl_path} 无样本")
    return samples


def build_prompts(samples: list[dict], template: str | None) -> tuple[list[str], list[str]]:
    """渲染 prompt;返回 (prompt 列表, 对齐的 sample_id 列表)。

    样本带现成 prompt 字段则直接用(可再经 /no_think 后缀关闭 thinking);
    给了 template 且样本带 text 字段则按模板重渲染。
    """
    prompts, ids = [], []
    for rec in samples:
        text = rec.get("text")
        if template and text is not None:
            prompt = render(template, text, rec.get("question"))
        else:
            prompt = rec["prompt"]
        # raw completion 路径统一追加 /no_think 关闭 thinking(C9);
        # chat 模板路径由 chat_template_kwargs() 处理(当前 runner 走 raw prompt)
        prompt = apply_no_think(prompt)
        prompts.append(prompt)
        ids.append(str(rec["sample_id"]))
    return prompts, ids


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="DraftRouter P0 离线 bench 单组 runner")
    p.add_argument("--spec-config", required=True,
                   help="A / B_0.6B_K5 / C_K5_L5 等(configs.py 枚举)")
    p.add_argument("--dataset", required=True, help="数据集标识,如 dureader")
    p.add_argument("--length-bucket", default=None,
                   help="长度档,如 native / short / long[7500,8192]")
    p.add_argument("--bs", type=int, required=True, choices=list(VALID_BS))
    p.add_argument("--output-mode", default="fixed512",
                   choices=["fixed512", "natural-stop"])
    p.add_argument("--template", default=None,
                   help="zh_json/zh_paragraph/rag_verbatim/rag_nonverbatim;"
                        "缺省用样本自带 prompt 字段")
    p.add_argument("--greedy", action="store_true",
                   help="greedy 附加对照(temperature=0);主口径不用")
    p.add_argument("--data-jsonl", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-model", default=None,
                   help="覆盖默认目标模型(默认 Qwen/Qwen3-8B)")
    args = p.parse_args(argv)

    spec = resolve_spec_config(args.spec_config)
    samples = load_samples(args.data_jsonl)
    prompts, sample_ids = build_prompts(samples, args.template)
    # 组顺序:固定 seed 打乱并记录(样本与 prompt 同步重排)
    order = list(range(len(prompts)))
    order = shuffle_group_order(order, seed=args.seed)
    prompts = [prompts[i] for i in order]
    sample_ids = [sample_ids[i] for i in order]

    # --- 引擎构造:每个 spec 配置新建 LLM,绝不复用(调用结构强制) ---
    from vllm import LLM  # 延迟 import:本地无 vllm 时仅 --help 可用
    engine_cfg = EngineConfig(
        model=args.target_model or spec.target_model,
        spec_kwargs=spec.llm_kwargs_patch(), seed=args.seed)
    logger.info("构造 LLM: %s", engine_cfg.to_llm_kwargs())
    llm = LLM(**engine_cfg.to_llm_kwargs())

    sampling_params = build_sampling_params(args.output_mode, greedy=args.greedy,
                                            seed=args.seed)
    # 预热:每引擎启动后丢弃前 3 条请求(用组内 prompt 复用即可,输出不使用)
    warm_prompts = (prompts * ((WARMUP_REQUESTS // len(prompts)) + 1))[:WARMUP_REQUESTS]
    llm.generate(warm_prompts, sampling_params)  # 丢弃
    logger.info("预热完成,已丢弃 %d 条请求", len(warm_prompts))

    metrics = run_one_group(llm, prompts, sampling_params, bs=args.bs)

    # chat_template_kwargs 仅在走 chat 接口时使用;记录在配置里供核对(C9)
    result = build_result(
        config={
            "spec_config": spec.config_id,
            "kind": spec.kind,
            "engine": engine_cfg.to_llm_kwargs(),
            "sampling": {"temperature": 0.0 if args.greedy else 1.0,
                         "top_p": 1.0 if args.greedy else 0.95,
                         "greedy": args.greedy},
            "bs": args.bs,
            "output_mode": args.output_mode,
            "template": args.template,
            "template_version": TEMPLATE_VERSION,
            "chat_template_kwargs": chat_template_kwargs(),
            "thinking_suffix_applied": True,
            "warmup_requests": WARMUP_REQUESTS,
        },
        seed=args.seed,
        dataset=args.dataset,
        length_bucket=args.length_bucket,
        sample_ids=sample_ids,
    )
    result["group_order"] = order  # 记录打乱后的执行顺序(原索引)
    fill_metrics(result, metrics)
    write_result(args.out_json, result)
    logger.info("结果已写入 %s", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
