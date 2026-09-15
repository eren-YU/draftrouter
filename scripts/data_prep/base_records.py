# -*- coding: utf-8 -*-
"""基础记录构造:从已下载数据 → 统一的基础记录列表(scene/sample_id/input_text/passages)。

供 build_pools / length_stats / field_screen / build_samples 共用,保证各步骤看到同一份记录。
Qwen3 tokenizer 延迟导入(transformers),本地单测不触发。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_root, load_json  # noqa: E402
from loaders import (load_alce_tar, load_longbench_zip,  # noqa: E402
                     load_repobench_records)

# Qwen3 tokenizer 只在云端加载;get_tokenizer 是唯一入口
_TOKENIZER = None


def get_tokenizer(name: str = "Qwen/Qwen3-8B"):
    """延迟加载 Qwen3 tokenizer(HF_HUB_OFFLINE=1 时读本地缓存)。"""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer  # 延迟导入

        _TOKENIZER = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER


def qwen_tokenize(text: str) -> list:
    """默认计长函数:Qwen3 tokenizer 编码(不开 add_special_tokens 以稳定口径)。"""
    return get_tokenizer().encode(text, add_special_tokens=False)


def _alce_tar_path(root: Path) -> Path | None:
    p = root / "alce_data.tar"
    return p if p.exists() else None


def build_base_records(root: Path | None = None, limit_per_ds: int | None = None,
                       tokenize=None) -> list[dict]:
    """构造基础记录。tokenize 为 None 时不计长(池构造只需 ID+scene)。"""
    root = root or data_root()
    records: list[dict] = []

    # RAG:ALCE tar(short 档原生样本;long 档在 build_samples 里按 passages 扩)
    tar = _alce_tar_path(root)
    if tar:
        for i, rec in enumerate(load_alce_tar(tar, limit=limit_per_ds)):
            records.append({
                "sample_id": f"rag-alce-{i:06d}",
                "scene": "rag",
                "dataset": "alce-hotpotqa",
                "question": rec["question"],
                "passages": rec["passages"],
                "source_file": rec.get("source_file", ""),
            })

    # 中文:D2 fallback 优先级 = DuReader BOS tar(若有)→ LongBench dureader zip
    lb_zip = root / "longbench" / "dureader.zip"
    if lb_zip.exists():
        for i, rec in enumerate(load_longbench_zip(lb_zip, limit=limit_per_ds)):
            records.append({
                "sample_id": f"chinese-dureader-{i:06d}",
                "scene": "chinese",
                "dataset": "longbench-dureader" if "longbench" in str(lb_zip) else "dureader-robust",
                "question": rec["question"],
                "passages": rec["passages"],  # 中文篇章:整篇为单 passage
                "source_file": rec.get("source_file", ""),
            })

    # 代码:repobench-c(python/java)
    for config in ("python", "java"):
        try:
            recs = load_repobench_records(root, config, limit=limit_per_ds)
        except Exception as e:  # noqa: BLE001 —— 数据缺失不阻断其他场景
            print(f"[base] repobench-c/{config} 加载失败: {e}", file=sys.stderr)
            continue
        for i, rec in enumerate(recs):
            records.append({
                "sample_id": f"code-repobench-{config}-{i:06d}",
                "scene": "code",
                "dataset": f"repobench-c-{config}",
                "question": rec["question"],
                "passages": [],
                "source_file": rec.get("source_file", ""),
            })

    if tokenize is not None:
        for r in records:
            # 原生长度:question + 全部原始段落(RAG short 档即此长度)
            n = len(tokenize(r["question"]))
            n += sum(len(tokenize("\n" + p)) for p in r["passages"])
            r["token_len"] = n
    return records


def load_pools(root: Path | None = None) -> dict[str, list[str]]:
    """读取 data/pools/*.ids.json。"""
    root = root or data_root()
    pools_dir = root / "pools"
    pools = {}
    for p in sorted(pools_dir.glob("*.ids.json")):
        pools[p.name.replace(".ids.json", "")] = load_json(p)["sample_ids"]
    return pools
