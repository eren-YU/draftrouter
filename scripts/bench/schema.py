#!/usr/bin/env python
"""结果 JSON schema 与落盘(WP3 / C2、C11)。

要求字段(全量列出;取不到的记 null,不省略、不编造):
配置、seed、数据集与长度档、样本 ID 列表、样本数、vLLM 版本/commit、
镜像 digest、GPU 型号/驱动、完整命令行、BS、achieved running batch、
preemption、prefix_caching 开关、thinking 开关、output_mode、wall time、
实例运行时长。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from config import (
    ENABLE_PREFIX_CACHING,
    ENABLE_THINKING,
    read_image_digest,
    read_instance_uptime_seconds,
)

# 结果 JSON 必含的顶层键(单测校验完整性)
REQUIRED_KEYS = (
    "schema_version", "config", "seed", "dataset", "length_bucket",
    "sample_ids", "num_samples", "vllm_version", "vllm_commit",
    "image_digest", "gpu_name", "gpu_driver", "command_line",
    "bs", "achieved_running_batch", "num_preemptions",
    "prefix_caching", "thinking_enabled", "output_mode",
    "template_version", "wall_time_s", "instance_uptime_s",
    "group_order_seed", "spec_config", "metrics",
)

SCHEMA_VERSION = "p0-result-v1"


def get_vllm_version() -> str | None:
    """延迟 import vllm 取版本;本地无 vllm 记 None。"""
    try:
        import vllm
        return getattr(vllm, "__version__", None)
    except ImportError:
        return None


def get_vllm_commit() -> str | None:
    """vLLM commit:getattr 探测,无则 None(WP0 云端用 pip freeze 回填)。"""
    try:
        import vllm
        return getattr(vllm, "__commit__", None) or getattr(vllm, "commit", None)
    except ImportError:
        return None


def get_gpu_info() -> tuple[str | None, str | None]:
    """GPU 型号与驱动版本:torch/nvidia-smi 尽力探测,失败记 None。"""
    name = driver = None
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=10)
        if proc.returncode == 0 and proc.stdout.strip():
            driver = proc.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return name, driver


def build_result(*, config: dict, seed: int, dataset: str,
                 length_bucket: str | None, sample_ids: list[str],
                 command_line: str | None = None, **extra: Any) -> dict:
    """组装结果 JSON;环境侧信息在此填,metrics 侧由 runner 传入 extra。"""
    gpu_name, gpu_driver = get_gpu_info()
    result = {
        "schema_version": SCHEMA_VERSION,
        "config": config,                      # 引擎+采样配置快照
        "seed": seed,
        "dataset": dataset,
        "length_bucket": length_bucket,        # 如 native / short / long[7500,8192]
        "sample_ids": list(sample_ids),
        "num_samples": len(sample_ids),
        "vllm_version": get_vllm_version(),
        "vllm_commit": get_vllm_commit(),
        "image_digest": read_image_digest(),
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "command_line": command_line if command_line is not None
                        else " ".join(sys.argv),
        "bs": config.get("bs"),
        "achieved_running_batch": None,        # 由 runner 指标回填
        "num_preemptions": None,
        "prefix_caching": ENABLE_PREFIX_CACHING,
        "thinking_enabled": ENABLE_THINKING,
        "output_mode": config.get("output_mode"),
        "template_version": config.get("template_version"),
        "wall_time_s": None,
        "instance_uptime_s": read_instance_uptime_seconds(),
        "group_order_seed": seed,
        "spec_config": config.get("spec_config"),
        "metrics": {},
    }
    result.update(extra)
    return result


def fill_metrics(result: dict, metrics: dict) -> dict:
    """把 runner 的组指标回填进结果(achieved batch / preemption / wall)。"""
    result["metrics"] = {k: v for k, v in metrics.items() if k != "requests"}
    result["metrics"]["requests"] = metrics.get("requests", [])
    result["achieved_running_batch"] = metrics.get("achieved_running_batch")
    result["num_preemptions"] = metrics.get("num_preemptions")
    result["wall_time_s"] = metrics.get("wall_time_s")
    return result


def validate_result(result: dict) -> None:
    """校验必含键齐全;缺失抛 KeyError(交付前自证)。"""
    missing = [k for k in REQUIRED_KEYS if k not in result]
    if missing:
        raise KeyError(f"结果 JSON 缺少必含键: {missing}")


def write_result(path: str, result: dict) -> None:
    """validate 后写 JSON(ensure_ascii=False,中文可读)。"""
    validate_result(result)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
