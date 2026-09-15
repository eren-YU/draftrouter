#!/usr/bin/env python
"""spec-decode 统计提取(WP3 / C10)。

口径:四字段 = num_drafts / num_draft_tokens / num_accepted_tokens /
num_accepted_tokens_per_pos;接受率分母固定 num_draft_tokens。

取数路径(按优先级):
1. SpecDecodingStats 字段快照:对引擎对象做 getattr 探测,不硬编码猜测
   以外的路径;字段不存在时记录 None 并告警。
2. vLLM 引擎日志行解析兜底:parse_spec_log_line 可单测。

TODO(WP0 冒烟):vLLM 0.29 离线路径下 SpecDecodingStats 的实际挂载点
(如 llm.llm_engine.spec_decoding_stats / scheduler 输出)需以 5 条 prompt
实测确认后回填 _SPEC_STAT_ATTRS;日志行的确切格式同样以冒烟样例为准。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

SPEC_FIELDS = (
    "num_drafts",
    "num_draft_tokens",
    "num_accepted_tokens",
    "num_accepted_tokens_per_pos",
)

# getattr 探测路径(按序尝试;全部命中失败则返回 None)
# TODO(WP0): 以 0.29 实测挂载点回填本列表,当前仅为常见候选。
_SPEC_STAT_ATTRS = (
    "spec_decoding_stats",
    "llm_engine.spec_decoding_stats",
    "engine_core.spec_decoding_stats",
    "speculator.spec_decoding_stats",
)


@dataclass
class SpecStats:
    """一次统计快照。字段缺失记 None;接受率分母固定 num_draft_tokens。"""

    num_drafts: int | None = None
    num_draft_tokens: int | None = None
    num_accepted_tokens: int | None = None
    num_accepted_tokens_per_pos: list[int] | None = None

    @property
    def acceptance_rate(self) -> float | None:
        """接受率 = num_accepted_tokens / num_draft_tokens(分母固定)。"""
        if self.num_accepted_tokens is None or not self.num_draft_tokens:
            return None
        return self.num_accepted_tokens / self.num_draft_tokens

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in SPEC_FIELDS}
        d["acceptance_rate"] = self.acceptance_rate
        return d


def _warn_missing(missing: list[str]) -> None:
    """字段缺失告警(不静默、不编造)。"""
    logger.warning("SpecDecodingStats 缺少字段: %s —— 以 null 记录, "
                   "请回查 WP0 冒烟取数路径", ", ".join(missing))


def snapshot_from_object(stats_obj: Any) -> SpecStats:
    """从 SpecDecodingStats(或任意有四字段的)对象提取快照。

    用 getattr 探测字段是否存在;不存在记 None 并告警。
    本函数纯逻辑,可单测(传 fake 对象)。
    """
    stats = SpecStats()
    missing: list[str] = []
    for name in SPEC_FIELDS:
        val = getattr(stats_obj, name, None)
        if val is None:
            missing.append(name)
            continue
        if name == "num_accepted_tokens_per_pos":
            try:
                stats.num_accepted_tokens_per_pos = [int(x) for x in val]
            except (TypeError, ValueError):
                missing.append(name)
                continue
        else:
            try:
                setattr(stats, name, int(val))
            except (TypeError, ValueError):
                missing.append(name)
    if missing:
        _warn_missing(missing)
    return stats


def spec_stats_from_request_output(ro: Any) -> SpecStats | None:
    """vLLM 0.29 实测路径(WP0 冒烟确认,2026-09-15):

    CompletionOutput.spec_decode_metrics = RequestSpecDecodeMetrics{
        num_spec_tokens, histogram(len=k+1), num_draft_tokens,
        per_step_accepted, per_step_drafted(detailed 档)}。

    四字段映射:
      num_drafts = sum(histogram)(= verify 步数)
      num_draft_tokens = num_draft_tokens
      num_accepted_tokens = sum_j j*histogram[j]
      num_accepted_tokens_per_pos[p] = sum_{j>p} histogram[j]
    引擎构造需 per_request_spec_decode_metrics='detailed'(config.py 已强制)。
    非 spec 配置(A)或引擎未开启该开关时返回 None。
    """
    outs = getattr(ro, "outputs", None) or []
    m = getattr(outs[0], "spec_decode_metrics", None) if outs else None
    if m is None:
        return None
    hist = list(getattr(m, "histogram", None) or [])
    if not hist:
        return None
    stats = SpecStats()
    stats.num_drafts = sum(hist)
    stats.num_draft_tokens = int(getattr(m, "num_draft_tokens", 0) or 0)
    stats.num_accepted_tokens = sum(j * c for j, c in enumerate(hist))
    stats.num_accepted_tokens_per_pos = [
        sum(hist[p + 1:]) for p in range(len(hist) - 1)]
    return stats


def aggregate_spec_stats(parts: list["SpecStats"]) -> SpecStats:
    """逐请求 SpecStats 求和(整批口径)。None/全 None 项跳过。"""
    agg = SpecStats()
    valid = [p for p in parts if p.num_draft_tokens is not None]
    if not valid:
        _warn_missing(["全部请求均无 spec 统计(plain 配置或开关未生效?)"])
        return agg
    agg.num_drafts = sum(p.num_drafts or 0 for p in valid)
    agg.num_draft_tokens = sum(p.num_draft_tokens for p in valid)
    agg.num_accepted_tokens = sum(p.num_accepted_tokens or 0 for p in valid)
    pos_lists = [p.num_accepted_tokens_per_pos for p in valid
                 if p.num_accepted_tokens_per_pos]
    if pos_lists:
        width = min(len(x) for x in pos_lists)
        agg.num_accepted_tokens_per_pos = [
            sum(x[i] for x in pos_lists) for i in range(width)]
    return agg


def snapshot_spec_stats(llm_or_engine: Any) -> SpecStats:
    """对 vllm.LLM / 引擎对象按 _SPEC_STAT_ATTRS 探测统计对象并提取。

    探测不到任何统计对象时返回全 None 的 SpecStats 并告警。
    """
    for path in _SPEC_STAT_ATTRS:
        obj: Any = llm_or_engine
        ok = True
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and obj is not None:
            stats = snapshot_from_object(obj)
            if stats.num_draft_tokens is not None:
                return stats
    _warn_missing(["SpecDecodingStats 对象(全部探测路径未命中)"])
    return SpecStats()


# ---------------------------------------------------------------------------
# 日志解析兜底(纯函数,单测覆盖)
# ---------------------------------------------------------------------------

# 数字字段:key 后允许任意分隔(=、:、空格),取整数
_INT_FIELD_PATTERNS = {
    name: re.compile(rf"{name}\s*[=:]\s*(\d+)", re.IGNORECASE)
    for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
}
# per-pos 列表:[1, 0, 2] 或 (1, 0, 2) 等
_PER_POS_PATTERN = re.compile(
    r"num_accepted_tokens_per_pos\s*[=:]\s*[\[(]\s*([\d\s,]+?)\s*[\])]",
    re.IGNORECASE)


def parse_spec_log_line(line: str) -> SpecStats | None:
    """解析一行 vLLM spec 统计日志;不含四字段中任一则返回 None。

    兼容 num_drafts=123 / num_drafts: 123 / Num_Drafts = 123 等写法;
    per-pos 支持空格/逗号分隔的列表。TODO(WP0): 0.29 实际日志措辞与
    本正则的差异以冒烟样例为准,必要时收紧正则。
    """
    if "spec" not in line.lower():
        return None
    stats = SpecStats()
    matched = False
    for name, pat in _INT_FIELD_PATTERNS.items():
        m = pat.search(line)
        if m:
            setattr(stats, name, int(m.group(1)))
            matched = True
    m = _PER_POS_PATTERN.search(line)
    if m:
        stats.num_accepted_tokens_per_pos = [
            int(x) for x in re.split(r"[,\s]+", m.group(1).strip()) if x]
        matched = True
    if not matched:
        return None
    missing = [k for k in SPEC_FIELDS if getattr(stats, k) is None]
    if missing:
        _warn_missing(missing)
    return stats
