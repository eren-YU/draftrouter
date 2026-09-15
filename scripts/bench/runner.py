#!/usr/bin/env python
"""离线 bench runner(WP3 / C1-C2)。

- LLM.generate 离线 batch,BS∈{1,4,8}(BS = 离线 batch size,非客户端并发)
- 每请求从 RequestOutput.metrics(arrival_time / first_token_time /
  finished_time)计算 TTFT / TPOT / e2e / 输出 token/s
- batch 级 achieved running batch 与 preemption 次数:尽力从 vLLM 0.29 的
  引擎 metrics 接口提取,失败记 None,不编造 API
  (TODO(WP0): 0.29 离线 LLM 是否暴露 get_metrics / Prometheus 指标需冒烟核实)

所有取数用 getattr 探测 + None 兜底;纯计算部分均为可单测的独立函数。
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

VALID_BS = (1, 4, 8)
WARMUP_REQUESTS = 3  # 每引擎启动后丢弃的前 3 条请求


# ---------------------------------------------------------------------------
# 每请求指标(纯计算,单测覆盖)
# ---------------------------------------------------------------------------

def compute_request_metrics(metrics: Any, num_output_tokens: int) -> dict:
    """从 RequestOutput.metrics(或 fake)计算请求级指标。

    探测字段:arrival_time / first_token_time / finished_time;
    任一缺失记 None。字段名与 0.29 实际定义不符时以 None 落盘并告警。
    """
    arrival = getattr(metrics, "arrival_time", None)
    first_tok = getattr(metrics, "first_token_time", None)
    finished = getattr(metrics, "finished_time", None)

    missing = [n for n, v in (("arrival_time", arrival),
                              ("first_token_time", first_tok),
                              ("finished_time", finished)) if v is None]
    if missing:
        logger.warning("RequestOutput.metrics 缺少时间字段: %s", ", ".join(missing))

    ttft = e2e = tpot = out_tok_per_s = None
    if None not in (arrival, first_tok):
        ttft = float(first_tok) - float(arrival)
    if None not in (arrival, finished):
        e2e = float(finished) - float(arrival)
    if None not in (first_tok, finished) and num_output_tokens >= 2:
        # TPOT = 解码阶段每 token 时间;(n-1) 个 decode 步,首 token 不算
        tpot = (float(finished) - float(first_tok)) / (num_output_tokens - 1)
        if tpot > 0:
            out_tok_per_s = (num_output_tokens - 1) / tpot
    return {
        "num_output_tokens": num_output_tokens,
        "ttft_s": ttft,
        "tpot_s": tpot,
        "e2e_s": e2e,
        "output_tokens_per_s": out_tok_per_s,
    }


def extract_num_output_tokens(output: Any) -> int | None:
    """从 RequestOutput(或 fake)取输出 token 数,尽力探测。"""
    n = getattr(output, "num_output_tokens", None)  # 旧版字段
    if isinstance(n, int):
        return n
    outs = getattr(output, "outputs", None)
    if outs:
        tok_ids = getattr(outs[0], "token_ids", None)
        if tok_ids is not None:
            return len(tok_ids)
    return None


def summarize_requests(req_metrics: list[dict]) -> dict:
    """batch 级汇总(median/p95,不含 None 值)。"""
    def med(vals: list[float]) -> float | None:
        if not vals:
            return None
        s = sorted(vals)
        return s[len(s) // 2] if len(s) % 2 else (s[len(s)//2 - 1] + s[len(s)//2]) / 2

    def pick(key: str) -> list[float]:
        return [m[key] for m in req_metrics if m.get(key) is not None]

    return {
        "n_requests": len(req_metrics),
        "median_ttft_s": med(pick("ttft_s")),
        "median_tpot_s": med(pick("tpot_s")),
        "median_e2e_s": med(pick("e2e_s")),
        "median_output_tokens_per_s": med(pick("output_tokens_per_s")),
        "median_num_output_tokens": med(pick("num_output_tokens")),
    }


# ---------------------------------------------------------------------------
# batch 级统计:achieved running batch / preemption(尽力提取,失败记 None)
# ---------------------------------------------------------------------------

# Prometheus 文本格式里关心的指标名;0.29 实际名称待 WP0 核实
_RUNNING_BATCH_PATTERNS = (
    re.compile(r"^vllm:num_requests_running(\s|\{).*$", re.M),
)
_PREEMPTION_PATTERNS = (
    re.compile(r"^vllm:num_preemptions(_total)?(\s|\{).*$", re.M),
)
_VALUE_RE = re.compile(r"([0-9.eE+\-]+)\s*$")


def _last_value(metrics_text: str, patterns) -> float | None:
    """从 Prometheus 文本指标里取某指标最后一个采样值;取不到返回 None。"""
    best = None
    for pat in patterns:
        for m in pat.finditer(metrics_text):
            vm = _VALUE_RE.search(m.group(0))
            if vm:
                try:
                    best = float(vm.group(1))
                except ValueError:
                    continue
    return best


def extract_batch_stats(llm: Any) -> dict:
    """尽力提取 achieved running batch(末值/最大值)与 preemption 次数。

    路径:llm.llm_engine.get_metrics() 的 Prometheus 文本(若 0.29 提供);
    任一步失败记 None 并告警,不编造。
    """
    out = {"achieved_running_batch": None, "num_preemptions": None}
    engine = getattr(llm, "llm_engine", None)
    get_metrics = getattr(engine, "get_metrics", None)
    if not callable(get_metrics):
        logger.warning("离线引擎无 get_metrics 接口,achieved batch / "
                       "preemption 记 null(TODO WP0 核实取数路径)")
        return out
    try:
        text = get_metrics()
    except Exception as exc:  # noqa: BLE001 —— 探测失败必须兜底
        logger.warning("get_metrics() 调用失败: %s", exc)
        return out
    running = _last_value(text, _RUNNING_BATCH_PATTERNS)
    preempt = _last_value(text, _PREEMPTION_PATTERNS)
    if running is not None:
        out["achieved_running_batch"] = running
    if preempt is not None:
        out["num_preemptions"] = preempt
    return out


# ---------------------------------------------------------------------------
# 组顺序与预热
# ---------------------------------------------------------------------------

def shuffle_group_order(items: list, seed: int = 0) -> list:
    """固定 seed 打乱组执行顺序;同 seed 同序(单测覆盖)。"""
    order = list(items)
    random.Random(seed).shuffle(order)
    return order


def warmup(llm: Any, prompts: list[str], sampling_params: Any,
           n: int = WARMUP_REQUESTS) -> list:
    """每引擎启动后丢弃前 n 条请求(返回被丢弃的输出,调用方不得使用)。"""
    if not prompts:
        return []
    take = prompts[:n]
    return llm.generate(take, sampling_params)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_one_group(llm: Any, prompts: list[str], sampling_params: Any,
                  bs: int) -> dict:
    """跑一个组(已定序的一批 prompt),返回请求级+batch 级指标。

    注意:llm 必须是本 spec 配置新建的实例(不跨 spec 配置复用,
    该约束由 run_group.py 的调用结构强制)。
    """
    if bs not in VALID_BS:
        raise ValueError(f"BS 必须属于 {VALID_BS}, got {bs}")
    t0 = time.perf_counter()
    # 离线 batch:一次 generate 提交 bs 条请求(vLLM 内部调度 running batch)
    outputs = llm.generate(prompts, sampling_params)
    wall_s = time.perf_counter() - t0

    req_metrics = []
    spec_parts = []
    for out in outputs:
        n = extract_num_output_tokens(out)
        req_metrics.append(compute_request_metrics(getattr(out, "metrics", None),
                                                   n if n is not None else 0))
        from spec_stats import spec_stats_from_request_output
        s = spec_stats_from_request_output(out)
        if s is not None:
            spec_parts.append(s)
    result = summarize_requests(req_metrics)
    result.update(extract_batch_stats(llm))
    if spec_parts:
        from spec_stats import aggregate_spec_stats
        spec = aggregate_spec_stats(spec_parts)
    else:
        from spec_stats import SpecStats
        spec = SpecStats()  # plain(A)配置:四字段全 null 属预期
    result["spec_stats"] = spec.to_dict()
    result["bs"] = bs
    result["wall_time_s"] = wall_s
    result["requests"] = req_metrics
    return result
