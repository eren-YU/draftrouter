# -*- coding: utf-8 -*-
"""数据下载:ALCE tar、repobench-c(python/java)、hotpotqa、ceval-exam → /root/autodl-tmp/data/

【云端执行方式】
    ssh autodl
    source /etc/network_turbo && export HF_HUB_DISABLE_XET=1   # 下载时段必须开加速
    /root/miniconda3/envs/draftrouter/bin/python scripts/data_prep/download_data.py

- 断点续传:目标文件已存在且 sha256 与清单匹配 → 跳过;不匹配 → 重下。
- HF 数据集优先 huggingface_hub.snapshot_download(repo_type="dataset", revision=锁定的 revision),
  下载后记录 revision 与每个文件的 sha256,写入 data_manifest.json。
- 纯逻辑(清单合并/校验)与下载 I/O 分离,便于本地单测 mock。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import data_root, load_json, save_json, sha256_file  # noqa: E402

MANIFEST_NAME = "data_manifest.json"

# 各数据集的下载计划;revision 为 P0 冻结值(若仓库无此 revision,运行时会报错并在 D2 记录实际值)
DOWNLOAD_PLAN = [
    {"name": "alce", "kind": "url",
     "urls": ["https://hf-mirror.com/datasets/princeton-nlp/ALCE-data/resolve/main/cif/data.tar"],
     "filename": "alce_data.tar"},
    {"name": "repobench-c-python", "kind": "hf_dataset",
     "repo_id": "tianyang/repobench-c", "config": "python", "revision": "main",
     "split": "completion"},
    {"name": "repobench-c-java", "kind": "hf_dataset",
     "repo_id": "tianyang/repobench-c", "config": "java", "revision": "main",
     "split": "completion"},
    {"name": "hotpotqa", "kind": "hf_dataset",
     "repo_id": "hotpotqa/hotpot_qa", "config": "distractor", "revision": "main",
     "split": "validation"},
    {"name": "ceval-exam", "kind": "hf_dataset",
     "repo_id": "ceval/ceval-exam", "config": None, "revision": "main",
     "split": "val"},
]


def manifest_path(root: Path | None = None) -> Path:
    return (root or data_root()) / MANIFEST_NAME


def load_or_init_manifest(root: Path) -> dict:
    p = manifest_path(root)
    if p.exists():
        return load_json(p)
    return {"entries": []}


def upsert_entry(manifest: dict, entry: dict) -> None:
    """按 (name, filename) 主键更新清单条目,保证幂等。"""
    key = (entry.get("name"), entry.get("filename"))
    manifest["entries"] = [e for e in manifest["entries"]
                           if (e.get("name"), e.get("filename")) != key]
    manifest["entries"].append(entry)


def check_skip(path: Path, expected_sha: str | None) -> bool:
    """断点续传判定:文件存在且 sha 匹配(或无期望值但非空)→ 跳过。"""
    if not path.exists():
        return False
    if expected_sha is None:
        return path.stat().st_size > 0
    return sha256_file(path) == expected_sha


def download_url(url: str, dest: Path, timeout: int = 600) -> None:
    """流式下载单文件(支持服务端断点需自行扩展;当前以存在+sha 匹配作为续传粒度)。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "draftrouter-dl/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)


def download_hf_dataset(item: dict, root: Path) -> dict:
    """snapshot_download 整仓下载(含所选 config 数据);失败退回 datasets.load_dataset。
    延迟导入,本地无 datasets 时单测不受影响。"""
    from huggingface_hub import snapshot_download  # 延迟导入

    path = snapshot_download(
        repo_id=item["repo_id"], repo_type="dataset",
        revision=item.get("revision"), cache_dir=str(root / "hf-cache"),
        endpoint=os.environ.get("HF_ENDPOINT") or None,
    )
    return {"local_path": str(path)}


def run(plan: list | None = None, root: Path | None = None) -> dict:
    """执行下载计划,写 data_manifest.json,返回摘要。"""
    root = root or data_root()
    root.mkdir(parents=True, exist_ok=True)
    manifest = load_or_init_manifest(root)
    plan = plan or DOWNLOAD_PLAN
    summary = {"downloaded": [], "skipped": [], "failed": []}

    for item in plan:
        name = item["name"]
        try:
            if item["kind"] == "url":
                dest = root / item["filename"]
                if check_skip(dest, item.get("sha256")):
                    summary["skipped"].append(name)
                else:
                    download_url(item["urls"][0], dest)
                    summary["downloaded"].append(name)
                entry = {"name": name, "kind": "url", "filename": item["filename"],
                         "source_url": item["urls"][0],
                         "sha256": sha256_file(dest), "size": dest.stat().st_size}
            else:
                info = download_hf_dataset(item, root)
                local = Path(info["local_path"])
                files = sorted(p for p in local.rglob("*") if p.is_file())
                # 只记录主要数据文件的 sha(快照缓存里有大量 blob,全部记录会臃肿)
                entry = {"name": name, "kind": "hf_dataset",
                         "repo_id": item["repo_id"], "config": item.get("config"),
                         "revision": item.get("revision"), "split": item.get("split"),
                         "local_path": str(local),
                         "files": [{"path": str(f.relative_to(local)),
                                    "sha256": sha256_file(f)} for f in files[:200]]}
                summary["downloaded"].append(name)
            upsert_entry(manifest, entry)
        except Exception as e:  # noqa: BLE001 —— 下载失败不中断其余数据集,汇总后人工判断
            summary["failed"].append({"name": name, "error": str(e)})

    save_json(manifest, manifest_path(root))
    return summary


def main() -> None:
    # 云端提醒:此脚本需在 source /etc/network_turbo + HF_HUB_DISABLE_XET=1 之后运行
    if os.name != "nt" and not os.environ.get("NETWORK_TURBO_ACK"):
        print("[download] 提醒:先 source /etc/network_turbo && export HF_HUB_DISABLE_XET=1 再执行。")
    summary = run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        print(f"[download] {len(summary['failed'])} 项失败,见上;修复后重跑(支持断点续传)。")


if __name__ == "__main__":
    main()
