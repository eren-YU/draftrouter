#!/usr/bin/env python
"""spec 配置枚举(WP3 / C7):A=plain;B=固定草稿;C=ngram。

vLLM 0.29 构造方式(2026-09-15 云端 inspect.signature 实测回填):
- A(plain): 不传任何 speculative 参数。
- B(固定草稿模型)/ C(ngram)统一走 speculative_config dict:
      LLM(model=target,
          speculative_config={
              "method": "draft_model" | "ngram",
              "model": <draft, 仅 B>,
              "num_speculative_tokens": K,
              "prompt_lookup_max": L, "prompt_lookup_min": L,  # 仅 C;ngram 二者必填其一
          },
          per_request_spec_decode_metrics="detailed")   # C10 离线 per-request spec 统计
  注:0.29 移除了 speculative_model/num_speculative_tokens 顶层参数
  (EngineArgs 实测 TypeError);ngram 默认 lookup_min=max=5,我们显式传。
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
        """返回追加到 vllm.LLM 的 speculative 关键字参数(0.29 实测口径)。"""
        if self.kind == "plain":
            return {}
        cfg: dict = {"method": "draft_model" if self.kind == "draft" else "ngram",
                     "num_speculative_tokens": self.num_speculative_tokens}
        if self.kind == "draft":
            cfg["model"] = self.draft_model
        if self.kind == "ngram":
            cfg["prompt_lookup_max"] = self.prompt_lookup_max
            cfg["prompt_lookup_min"] = self.prompt_lookup_max
        return {"speculative_config": cfg}


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
