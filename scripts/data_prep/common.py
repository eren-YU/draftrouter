"""WP2 数据构造公共工具:路径、sha256、JSON/JSONL 落盘。

【云端执行方式】
    ssh autodl
    source /etc/network_turbo          # 仅下载时段需要;纯本地构样不开加速
    export HF_HUB_DISABLE_XET=1        # 绕开 Xet 协议(匿名 token 会 401)
    export HF_ENDPOINT=https://hf-mirror.com   # 可选,镜像加速
    cd /root/autodl-tmp/draftrouter && git pull
    /root/miniconda3/envs/draftrouter/bin/python scripts/data_prep/build_all.py

所有产物默认落 /root/autodl-tmp/data/(本地调试用环境变量 DRAFTROUTER_DATA_DIR 覆盖)。
数据文件不进 git(见 .gitignore / AGENTS.md)。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# 云端数据盘路径;本地测试/调试时用 DRAFTROUTER_DATA_DIR 覆盖
DEFAULT_DATA_ROOT = "/root/autodl-tmp/data"


def data_root() -> Path:
    """返回数据根目录(可被环境变量覆盖)。"""
    return Path(os.environ.get("DRAFTROUTER_DATA_DIR", DEFAULT_DATA_ROOT))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """流式计算文件 sha256(大文件不整读内存)。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def save_json(obj, path: str | Path) -> None:
    """确定性落盘 JSON:sort_keys + 固定缩进,保证同输入同字节。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(records, path: str | Path) -> None:
    """逐行写 JSONL(UTF-8,无 ASCII 转义)。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
