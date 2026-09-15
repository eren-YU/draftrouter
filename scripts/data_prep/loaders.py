# -*- coding: utf-8 -*-
"""各数据集的加载器:全部延迟导入重依赖(datasets/tarfile),字段解析宽容 + 记录异常。

统一输出「基础记录」:{sample_id, scene, dataset, input_text, passages, meta}
- sample_id 规则 {scene}-{dataset}-{序号},确定性(按文件内出现顺序编号);
- passages 只在 RAG 场景携带(原始检索段落顺序,供扩长);
- 解析不出有效内容的文件跳过并计入 skipped,不中断。
"""

from __future__ import annotations

import json
import tarfile
import zipfile
from pathlib import Path

SCENE_DATASET = {
    "rag": "alce-hotpotqa",
    "chinese": "dureader",
    "code": "repobench-c",
}


def _first_key(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def parse_alce_record(obj: dict) -> dict | None:
    """ALCE jsonl 记录:question + docs:[{title,text}]。"""
    q = _first_key(obj, ["question", "query", "prompt"])
    docs = _first_key(obj, ["docs", "documents", "evidences"]) or []
    passages = []
    for d in docs:
        if isinstance(d, dict):
            t = str(d.get("text", "")).strip()
            title = str(d.get("title", "")).strip()
            if t:
                passages.append((title + "\n" + t).strip())
        elif isinstance(d, str) and d.strip():
            passages.append(d.strip())
    if not q or not passages:
        return None
    return {"question": str(q).strip(), "passages": passages}


def load_alce_tar(tar_path: Path, limit: int | None = None) -> list[dict]:
    """解包 ALCE data.tar 中的 jsonl,返回解析后的记录。limit 用于云端快速试跑。"""
    out = []
    with tarfile.open(tar_path, "r:*") as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".jsonl")):
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            for line in f.read().decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = parse_alce_record(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if rec:
                    rec["source_file"] = m.name
                    out.append(rec)
                    if limit and len(out) >= limit:
                        return out
    return out


def load_longbench_zip(zip_path: Path, subset: str = "dureader",
                       limit: int | None = None) -> list[dict]:
    """D2 fallback:LongBench 数据 zip 内 {subset}.jsonl / {subset}/,字段 context+input。"""
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if not (name.endswith(".jsonl") and base.startswith(subset)):
                continue
            for line in zf.read(name).decode("utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ctx = str(obj.get("context", "")).strip()
                q = str(obj.get("input", obj.get("question", ""))).strip()
                if not ctx:
                    continue
                out.append({"question": q or "总结以下文档。", "passages": [ctx],
                            "source_file": name})
                if limit and len(out) >= limit:
                    return out
    return out


def load_hf_split(repo_id: str, config: str | None, split: str,
                  limit: int | None = None) -> list[dict]:
    """datasets.load_dataset 流式读取(HF 数据集统一入口;云端使用,延迟导入)。"""
    import datasets  # 延迟导入

    ds = datasets.load_dataset(repo_id, config, split=split, streaming=True)
    return list(ds.take(limit)) if limit else list(ds)


def load_repobench_records(root: Path, config: str, limit: int | None = None) -> list[dict]:
    """repobench-c:行 dict 含 prompt/code_file 等字段;取 prompt 作为输入。"""
    from huggingface_hub import snapshot_download  # 延迟导入
    import datasets  # noqa: F401

    path = snapshot_download(repo_id="tianyang/repobench-c", repo_type="dataset",
                             cache_dir=str(root / "hf-cache"),
                             endpoint=None)
    # repobench-c 仓内为 parquet/json 数据文件,直接用 datasets 从本地目录加载
    ds = datasets.load_dataset(str(path), config, split="completion",
                               streaming=True, trust_remote_code=False)
    out = []
    for obj in (ds.take(limit) if limit else ds):
        prompt = _first_key(obj, ["prompt", "code", "input"])
        if not prompt:
            continue
        out.append({"question": str(prompt), "passages": [],
                    "source_file": f"repobench-c/{config}"})
    return out
