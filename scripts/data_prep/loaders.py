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
    """ALCE 记录:question + docs:[{title,text}](2026-09-15 云端实测:
    tar 内为 eli5/asqa/qampari 的 eval json 数组,每元素含 question/answer/claims/docs;
    tar 内不含 hotpotqa,hotpotqa 仅作短答案对照数据集另行下载)。"""
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


def load_alce_tar(tar_path: Path, limit: int | None = None,
                  subsets: tuple[str, ...] = ("eli5", "asqa")) -> list[dict]:
    """解包 ALCE data.tar:成员为 {subset}_eval_*.json(JSON 数组)。
    只取 subsets 前缀匹配的文件,按成员名排序保证确定性;limit 为总条数上限。"""
    out: list[dict] = []
    with tarfile.open(tar_path, "r:*") as tf:
        members = sorted(m.name for m in tf.getmembers()
                         if m.isfile() and m.name.endswith(".json"))
        for name in members:
            base = Path(name).name
            if not base.startswith(subsets):
                continue
            subset = next(s for s in subsets if base.startswith(s))
            f = tf.extractfile(dict((m.name, m) for m in tf.getmembers())[name])
            try:
                arr = json.loads(f.read().decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            for obj in arr if isinstance(arr, list) else []:
                rec = parse_alce_record(obj)
                if rec:
                    rec["source_file"] = name
                    rec["subset"] = subset
                    out.append(rec)
                    if limit and len(out) >= limit:
                        return out
    return out


def load_longbench_zip(zip_path: Path, subset: str = "dureader",
                       limit: int | None = None) -> list[dict]:
    """LongBench data.zip(注意是整包 data.zip):内含各子集 jsonl,
    dureader.jsonl 每行为 context(整篇文档)+ input(问题)。"""
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


# ---------------------------------------------------------------------------
# DuReader-robust(BOS 直链 tar.gz;2026-09-15 云端实测格式)

def dureader_records(obj: dict) -> list[dict]:
    """解析 DuReader-robust 的 JSON 结构(SQuAD 风格,云端实测):
    {"data": [{"paragraphs": [{"context": 篇章, "qas": [{"question": 问题, ...}]}]}]}
    每个段落一个样本:整篇 context 作为单 passage,question 取 qas[0].question。
    纯函数(不含文件 I/O),便于单测。"""
    out: list[dict] = []
    for article in obj.get("data", []):
        for para in article.get("paragraphs", []):
            ctx = str(para.get("context", "")).strip()
            qas = para.get("qas") or []
            q = str(qas[0].get("question", "")).strip() if qas else ""
            if not ctx:
                continue
            out.append({"question": q or "总结以下文档。", "passages": [ctx]})
    return out


def load_dureader_tar(tar_path: Path, split: str = "dev",
                      limit: int | None = None) -> list[dict]:
    """解包 dureader_robust-data.tar.gz 中 {split}.json 并解析。"""
    member_name = f"dureader_robust-data/{split}.json"
    with tarfile.open(tar_path, "r:*") as tf:
        names = {m.name: m for m in tf.getmembers() if m.isfile()}
        member = names.get(member_name)
        if member is None:  # 兼容去掉顶层目录的打包
            member = next((m for n, m in names.items()
                           if n.endswith(f"/{split}.json") or n == f"{split}.json"), None)
        if member is None:
            raise FileNotFoundError(f"{tar_path} 内找不到 {split}.json")
        f = tf.extractfile(member)
        obj = json.loads(f.read().decode("utf-8"))
    recs = dureader_records(obj)
    return recs[:limit] if limit else recs


# ---------------------------------------------------------------------------
# repobench-c parquet(官方 refs/convert/parquet 分支;2026-09-15 云端实测)

def repobench_row_to_record(obj: dict) -> dict | None:
    """repobench parquet 行 → 样本;输入列取 prompt(云端实测列名:
    repo_name/file_path/context/import_statement/code/prompt/next_line)。纯函数。"""
    prompt = obj.get("prompt")
    if not prompt:
        return None
    return {"question": str(prompt), "passages": []}


def load_parquet_records(parquet_path: Path, limit: int | None = None) -> list[dict]:
    """读 repobench parquet 为记录列表;pyarrow 延迟导入(本地单测不装)。"""
    import pyarrow.parquet as pq  # 延迟导入

    table = pq.read_table(str(parquet_path))
    cols = {name: table.column(name).to_pylist() for name in table.column_names}
    out = []
    n = len(next(iter(cols.values()))) if cols else 0
    for i in range(n):
        row = {k: v[i] for k, v in cols.items()}
        rec = repobench_row_to_record(row)
        if rec:
            out.append(rec)
            if limit and len(out) >= limit:
                break
    return out


def load_hf_split(repo_id: str, config: str | None, split: str,
                  limit: int | None = None) -> list[dict]:
    """datasets.load_dataset 流式读取(HF 数据集统一入口;云端使用,延迟导入)。"""
    import datasets  # 延迟导入

    ds = datasets.load_dataset(repo_id, config, split=split, streaming=True)
    return list(ds.take(limit)) if limit else list(ds)
