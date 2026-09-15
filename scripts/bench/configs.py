#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""spec 配置枚举(WP3 / C7):A=plain;B=固定草稿;C=ngram。

vLLM 0.29 构造方式(以 vllm.LLM 实际签名为准,WP0 冒烟验证后回填):
- A(plain): 不传任何 speculative 参数。
- B(固定草稿模型): vLLM >= 0.4.x 的离线接口为
      LLM(model=target, speculative_model=draft, num_speculative_tokens=K)
  TODO(WP0): 0.29 中该参数名是否更名(如 speculative_config dict)需 import
  vllm.LLM 后用 inspect.signature 核实;核实前不写死其他变体。
- C(ngram): 0.29 走统一 method 入口:
      LLM(model=target, method="ngram",
          num_speculative_tokens=K, prompt_lookup_max=L, prompt_lookup_min=?)
  TODO(WP0): prompt_lookup_min 是否必填/默认值待冒烟确认;当前不传,
  依赖 0.29 默认(prompt_lookup_min 若为 None 引擎应自行取 1 或 2)。
"""

from __future__ import annotations

from dataclasses import dataclass

# P0 草稿池(C7:主线只用 0.6B / 1.7B)
TARGET_MODEL = "Qwen/Qwen3-8B"
DRAFT_MODELS = {
    "0.6B": "Qwen/Qwen3-0.6B",
    "1.7B": "Qwen/Qwen3-1.7B",
}
DRAFT_KS = (3, 5, 7)
NGRAM_KS = (3, 5, 7)
NGRAM_LOOKUP_MAXS = (3, 5)


@dataclass(frozen=True)
class SpecConfig:
    """一个 spec 配置的完整描述;kind 决定 LLM 构造路径。"""

    config_id: str           # 如 "A" / "B_0.6B_K5" / "C_K5_L5"
    kind: str                # "plain" | "draft" | "ngram"
    target_model: str = TARGET_MODEL
    draft_model: str | None = None
    num_speculative_tokens: int | None = None
    prompt_lookup_max: int | None = None

    def llm_kwargs_patch(self) -> dict:
        """返回追加到 EngineConfig 的 speculative 关键字参数。

        键名以 vllm.LLM 0.29 签名为准(见文件头 TODO,WP0 冒烟后回填)。
        """
        if self.kind == "plain":
            return {}
        if self.kind == "draft":
            return {
                "speculative_model": self.draft_model,
                "num_speculative_tokens": self.num_speculative_tokens,
            }
        if self.kind == "ngram":
            return {
                "method": "ngram",
                "num_speculative_tokens": self.num_speculative_tokens,
                "prompt_lookup_max": self.prompt_lookup_max,
            }
        raise ValueError(f"未知 spec kind: {self.kind!r}")


def enumerate_spec_configs() -> list[SpecConfig]:
    """P0 全部 spec 配置(WP3 只做枚举与构造,不在此启动引擎)。"""
    out = [SpecConfig(config_id="A", kind="plain")]
    for draft_name, draft_model in DRAFT_MODELS.items():
        for k in DRAFT_KS:
            out.append(SpecConfig(
                config_id=f"B_{draft_name}_K{k}", kind="draft",
                draft_model=draft_model, num_speculative_tokens=k))
    for k in NGRAM_KS:
        for lmax in NGRAM_LOOKUP_MAXS:
            out.append(SpecConfig(
                config_id=f"C_K{k}_L{lmax}", kind="ngram",
                num_speculative_tokens=k, prompt_lookup_max=lmax))
    return out


def resolve_spec_config(config_id: str) -> SpecConfig:
    """按 CLI 传入的 config_id 解析;未知 id 报错并列出合法值。"""
    for cfg in enumerate_spec_configs():
        if cfg.config_id == config_id:
            return cfg
    legal = ", ".join(c.config_id for c in enumerate_spec_configs())
    raise ValueError(f"未知 --spec-config {config_id!r};合法值: {legal}")
