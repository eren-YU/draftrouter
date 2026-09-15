"""链路探测:只输出 URL 与可达性 JSON,不下载大文件。

【云端执行方式】
    ssh autodl
    source /etc/network_turbo && export HF_HUB_DISABLE_XET=1   # 探测外网链路,需开加速
    export HF_ENDPOINT=https://hf-mirror.com   # 默认即此,可用环境变量覆盖
    /root/miniconda3/envs/draftrouter/bin/python scripts/data_prep/probe_links.py

输出:data/probe/links.json,内含
- DuReader_robust 加载脚本里提取的 BOS 原始数据 URL(datasets>=3 不支持脚本加载,需 curl 直测);
- ALCE 数据 tar(princeton-nlp/ALCE-data,约 451MB)URL;
- THUDM/LongBench dureader 子集(备用源,D2 fallback)信息。
每个 URL 附 HEAD/Range 探测的 status/content-length,供人工与 D2 决策。
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import data_root, save_json  # noqa: E402

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# 已知候选(与 HF resolve 布局一致);探测时逐个验证
CANDIDATES = {
    # ALCE 数据 tar:官方 download_data.sh 用的 ALCE-data.tar(约 451MB;2026-09-15 实测)
    "alce_tar": [
        f"{HF_ENDPOINT}/datasets/princeton-nlp/ALCE-data/resolve/main/ALCE-data.tar",
        f"{HF_ENDPOINT}/datasets/princeton-nlp/ALCE-data/resolve/main/cif/data.tar",
        f"{HF_ENDPOINT}/datasets/princeton-nlp/ALCE-data/resolve/main/data.tar",
    ],
    # LongBench 整包 data.zip(实测 200;内含 dureader.jsonl,D2 fallback 源)
    "longbench_dureader": [
        f"{HF_ENDPOINT}/datasets/THUDM/LongBench/resolve/main/data.zip",
        f"{HF_ENDPOINT}/datasets/THUDM/LongBench/resolve/main/data/dureader.zip",
    ],
    # DuReader_robust 仓库文件(加载脚本本身,从中提取 BOS URL)
    "dureader_robust_repo": [
        f"{HF_ENDPOINT}/datasets/PaddlePaddle/dureader_robust/resolve/main/dureader_robust.py",
        f"{HF_ENDPOINT}/datasets/PaddlePaddle/dureader_robust/resolve/main/dureader_robust-1.0.0.py",
    ],
}

# BOS 原始数据直链候选(首条为 2026-09-15 云端实测 200 的主源)
BOS_URLS = [
    "https://bj.bcebos.com/paddlenlp/datasets/dureader_robust-data.tar.gz",
    "https://dataset-bj.cdn.bcebos.com/dureader/data/dureader_robust-data.tar.gz",
    "https://dataset-bj.cdn.bcebos.com/dureader/dureader-robust-data.tar.gz",
]


def probe_url(url: str, timeout: int = 15) -> dict:
    """HEAD(失败退到 Range GET)探测,只取状态与长度,不下载内容。"""
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "draftrouter-probe/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"url": url, "ok": True, "status": resp.status,
                    "content_length": resp.headers.get("Content-Length")}
    except Exception as e:
        # 有些 CDN 不支持 HEAD,退到 1 字节 Range GET
        try:
            req2 = urllib.request.Request(url, headers={"User-Agent": "draftrouter-probe/0.1",
                                                        "Range": "bytes=0-0"})
            with urllib.request.urlopen(req2, timeout=timeout) as resp:
                return {"url": url, "ok": True, "status": resp.status, "method": "range-get",
                        "content_length": resp.headers.get("Content-Range")}
        except Exception as e2:
            return {"url": url, "ok": False, "error": f"HEAD:{e}; RANGE:{e2}"}


def extract_bos_urls(script_text: str) -> list[str]:
    """从 DuReader 加载脚本源码里提取 http(s) 数据 URL。"""
    return sorted(set(re.findall(r"https?://[^\s\"'<>]+", script_text)))


def probe_dureader_script() -> dict:
    """拉 dureader_robust 加载脚本,提取其中的数据 URL。"""
    out: dict = {"candidates": [], "extracted_urls": []}
    for url in CANDIDATES["dureader_robust_repo"]:
        r = probe_url(url)
        out["candidates"].append(r)
        if r.get("ok"):
            try:
                with urllib.request.urlopen(urllib.request.Request(url), timeout=30) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
                extracted = extract_bos_urls(text)
                # 过滤掉 github.com/huggingface.co 等非数据 URL
                out["extracted_urls"] = [u for u in extracted
                                         if "bcebos" in u or "bos" in u or ".baidu" in u]
            except Exception as e:
                out["extract_urls_error"] = str(e)
            break
    return out


def main() -> None:
    report: dict = {"hf_endpoint": HF_ENDPOINT, "sections": {}}
    sections = report["sections"]

    sections["dureader_robust_script"] = probe_dureader_script()

    # BOS 直链候选 + 提取到的 URL,统一 curl 可达性实测
    probe_targets = list(BOS_URLS)
    probe_targets += sections["dureader_robust_script"].get("extracted_urls", [])
    sections["dureader_bos_probe"] = [probe_url(u) for u in dict.fromkeys(probe_targets)]

    for key in ("alce_tar", "longbench_dureader"):
        sections[key] = [probe_url(u) for u in CANDIDATES[key]]

    out_path = data_root() / "probe" / "links.json"
    save_json(report, out_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[probe] 落盘 {out_path}")
    # 汇总一行:每个候选组是否至少一条可达
    for key, entries in sections.items():
        if isinstance(entries, list):
            print(f"[probe] {key}: {'OK' if any(e.get('ok') for e in entries) else 'FAIL'}")


if __name__ == "__main__":
    main()
