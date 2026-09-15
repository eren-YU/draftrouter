# -*- coding: utf-8 -*-
"""配置/模板/spec 枚举单测(全部纯逻辑,不 import vllm)。"""

import pytest

import config
import prompts
from configs import enumerate_spec_configs, resolve_spec_config


# ---------------------------------------------------------------- config.py

def test_output_modes_sampling_shape():
    # 不 import vllm,只校验分支与非法值
    with pytest.raises(ValueError):
        config.build_sampling_params("bogus")


def test_no_think_suffix():
    p = config.apply_no_think("你好。")
    assert p.endswith(config.THINKING_SUFFIX)
    assert "/no_think" in p


def test_engine_config_prefix_caching_off():
    cfg = config.EngineConfig(model="m")
    kw = cfg.to_llm_kwargs()
    assert kw["enable_prefix_caching"] is False       # C8
    assert kw["max_model_len"] == 10240               # C5
    assert kw["seed"] == 0


def test_chat_template_kwargs_disables_thinking():
    assert config.chat_template_kwargs() == {"enable_thinking": False}  # C9


# --------------------------------------------------------------- prompts.py

def test_template_version_constant():
    assert prompts.TEMPLATE_VERSION  # 版本常量非空
    assert set(prompts.TEMPLATES) == {
        "zh_json", "zh_paragraph", "rag_verbatim", "rag_nonverbatim"}


def test_render_four_templates():
    for t in ("zh_json", "zh_paragraph"):
        out = prompts.render(t, "某公司发布产品")
        assert isinstance(out, str) and "某公司发布产品" in out
    for t in ("rag_verbatim", "rag_nonverbatim"):
        out = prompts.render(t, "文档", question="谁发布的?")
        assert "谁发布的?" in out and "文档" in out
    # rag 模板缺 question 必须报错
    with pytest.raises(ValueError):
        prompts.render("rag_verbatim", "文档")
    with pytest.raises(ValueError):
        prompts.render("unknown", "x")


# ---------------------------------------------------------------- configs.py

def test_spec_config_enumeration():
    cfgs = {c.config_id: c for c in enumerate_spec_configs()}
    # A plain
    assert cfgs["A"].kind == "plain"
    assert cfgs["A"].llm_kwargs_patch() == {}
    # B:0.6B/1.7B × K{3,5,7}
    for draft in ("0.6B", "1.7B"):
        for k in (3, 5, 7):
            c = cfgs[f"B_{draft}_K{k}"]
            assert c.kind == "draft"
            patch = c.llm_kwargs_patch()
            assert patch["speculative_model"].endswith(draft) or draft in patch["speculative_model"]
            assert patch["num_speculative_tokens"] == k
    # C:K{3,5,7} × lookup_max{3,5} = 6 个
    n_c = sum(1 for cid in cfgs if cid.startswith("C_"))
    assert n_c == 6
    c = cfgs["C_K5_L5"]
    patch = c.llm_kwargs_patch()
    assert patch["method"] == "ngram"
    assert patch["num_speculative_tokens"] == 5
    assert patch["prompt_lookup_max"] == 5
    # 总数 = 1 + 6 + 6
    assert len(cfgs) == 13


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_spec_config("Z_bogus")
    assert resolve_spec_config("A").kind == "plain"
