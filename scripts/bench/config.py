#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""WP3 bench 全局配置口径(依据 docs/P0-action-plan.md §0 C5-C9)。

注意:本模块只在函数体内延迟 import vllm,保证本地 Windows 无 GPU/无 vllm
时单测可以跑通(单测只 import 本模块与纯逻辑部分)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 引擎级配置(C5/C8/C9)
# ---------------------------------------------------------------------------

MAX_MODEL_LEN = 10240          # 8192 输入 + 2048 输出余量(C5)
ENABLE_PREFIX_CACHING = False  # C8:统一关闭,防止组间预热污染 TTFT
ENABLE_THINKING = False        # C9:thinking 统一关闭
THINKING_SUFFIX = "/no_think"  # raw completion 路径的 thinking 关闭后缀
GPU_MEMORY_UTILIZATION = 0.95  # 与 WP1 显存探针口径一致
DTYPE = "auto"

# ---------------------------------------------------------------------------
# 采样配置(baseline §6 / C6)
# ---------------------------------------------------------------------------

TEMPERATURE = 1.0
TOP_P = 0.95
DEFAULT_SEED = 0

# 输出两档(C6)
OUTPUT_MODES = ("fixed512", "natural-stop")


@dataclass
class EngineConfig:
    """LLM 构造参数快照;每个 spec 配置新建一个 LLM 实例时使用。"""

    model: str
    spec_kwargs: dict = field(default_factory=dict)  # speculative 配置(A 为空)
    max_model_len: int = MAX_MODEL_LEN
    enable_prefix_caching: bool = ENABLE_PREFIX_CACHING
    gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION
    dtype: str = DTYPE
    seed: int = DEFAULT_SEED

    def to_llm_kwargs(self) -> dict:
        """展开为 vllm.LLM(**kwargs) 的关键字参数。

        vLLM 0.29 的 speculative 相关签名以 import 后的实际签名为准,
        spec_kwargs 的键名由 configs.py 按探测结果给出(见该文件 TODO)。
        """
        kwargs = dict(
            model=self.model,
            max_model_len=self.max_model_len,
            enable_prefix_caching=self.enable_prefix_caching,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype=self.dtype,
            seed=self.seed,
        )
        kwargs.update(self.spec_kwargs)
        return kwargs


def build_sampling_params(output_mode: str, greedy: bool = False,
                           seed: int = DEFAULT_SEED):
    """构造 SamplingParams(延迟 import,单测不触发)。

    fixed512:ignore_eos=True,  max_tokens=512  —— 主口径/门槛(C6)
    natural-stop:ignore_eos=False, max_tokens=2048 —— 次要对照
    greedy 为附加对照选项(temperature=0);主口径 temperature=1.0/top_p=0.95/seed。
    """
    if output_mode not in OUTPUT_MODES:
        raise ValueError(f"未知 output_mode: {output_mode!r}, 可选 {OUTPUT_MODES}")
    from vllm import SamplingParams  # 延迟导入:本地无 vllm 时不触发

    if greedy:
        temperature, top_p, use_seed = 0.0, 1.0, None
    else:
        temperature, top_p, use_seed = TEMPERATURE, TOP_P, seed
    return SamplingParams(
        ignore_eos=(output_mode == "fixed512"),
        max_tokens=512 if output_mode == "fixed512" else 2048,
        temperature=temperature,
        top_p=top_p,
        seed=use_seed,
    )


def chat_template_kwargs() -> dict:
    """chat completions 路径关闭 thinking(C9)。离线 chat 用。"""
    return {"enable_thinking": ENABLE_THINKING}


def apply_no_think(prompt: str) -> str:
    """raw completion 路径关闭 thinking(C9):追加 /no_think 后缀。"""
    if ENABLE_THINKING:
        return prompt
    return prompt.rstrip() + f" {THINKING_SUFFIX}"


# ---------------------------------------------------------------------------
# 环境信息读取(结果 JSON 用;云端取真值,本地/缺失记 None,不编造)
# ---------------------------------------------------------------------------

def read_instance_uptime_seconds() -> float | None:
    """实例运行时长:Linux 读 /proc/uptime;Windows 本地填 None。
    可用环境变量 DRAFTROUTER_INSTANCE_UPTIME_S 覆盖。"""
    env_val = os.environ.get("DRAFTROUTER_INSTANCE_UPTIME_S")
    if env_val is not None:
        try:
            return float(env_val)
        except ValueError:
            return None
    try:
        with open("/proc/uptime", "r", encoding="ascii") as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def read_image_digest() -> str | None:
    """镜像 digest:从环境变量读取,取不到记 null(由 WP0 在云端注入)。"""
    return os.environ.get("DRAFTROUTER_IMAGE_DIGEST")
