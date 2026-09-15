"""RAG 扩长拼接逻辑(fake tokenizer)+ 样本 schema + 输出形态配对(不触网、无重依赖)。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest  # noqa: E402
from core import (  # noqa: E402
    RAG_LONG_HI,
    RAG_LONG_LO,
    SAMPLE_FIELDS,
    assign_pools,
    build_long_input,
    expand_output_modes,
    make_output_mode,
    make_sample,
    validate_sample,
)


# fake tokenizer:按字符计数(中文友好、确定性);"1"×n 段落长度可控
def fake_tokenize(text: str) -> list:
    return list(text)


def _passage(n_chars: int, tag: str) -> str:
    return tag + "字" * (n_chars - len(tag))


def test_build_long_input_reaches_target_window():
    passages = [_passage(2000, f"P{i}") for i in range(10)]
    built = build_long_input("问题?", passages, fake_tokenize,
                             lo=7500, hi=8192)
    assert RAG_LONG_LO <= built["token_len"] <= RAG_LONG_HI
    assert built["passages_used"] == 4  # 每段 2000 token,4 段 8000 落窗
    assert not built["insufficient"]
    assert built["input_text"].startswith("问题?\n")
    # 段落保持原始顺序
    for i in range(built["passages_used"] - 1):
        first, second = f"P{i}", f"P{i + 1}"
        assert built["input_text"].index(first) < built["input_text"].index(second)


def test_build_long_input_insufficient_logged():
    passages = [_passage(100, f"P{i}") for i in range(3)]  # 总量远不足 7500
    built = build_long_input("问题?", passages, fake_tokenize, lo=7500, hi=8192)
    assert built["insufficient"]
    assert built["passages_used"] == 3
    assert built["token_len"] < RAG_LONG_LO


def test_build_long_input_respects_upper_bound():
    passages = [_passage(3000, f"P{i}") for i in range(10)]
    built = build_long_input("问题?", passages, fake_tokenize, lo=7500, hi=8192)
    assert built["token_len"] <= RAG_LONG_HI  # 永不超上限(段落完整追加,越界即停)
    assert built["passages_used"] == 2  # 3(问题)+2×3001=6005,再追加会到 9006>8192


def test_output_mode_encoding_and_parsing():
    assert make_output_mode("non-verbatim", "json") == "non-verbatim-json"
    s = make_sample("rag-alce-000000-long", "rag", "alce-hotpotqa", "long",
                    "正文", "non-verbatim", "paragraph")
    assert s["output_mode"] == "non-verbatim-paragraph"
    with pytest.raises(ValueError):
        make_output_mode("json", "verbatim")


def test_sample_schema_validation():
    s = make_sample("chinese-dureader-000000", "chinese", "longbench-dureader",
                    "mid", "文档正文", "verbatim", "json")
    validate_sample(s)
    assert SAMPLE_FIELDS <= set(s)
    bad = dict(s, scene="english")
    with pytest.raises(AssertionError):
        validate_sample(bad)
    bad2 = dict(s, output_mode="verbatim-audio")
    with pytest.raises(AssertionError):
        validate_sample(bad2)
    bad3 = dict(s, length_bucket="huge")
    with pytest.raises(AssertionError):
        validate_sample(bad3)
    bad4 = dict(s, input_text="")
    with pytest.raises(AssertionError):
        validate_sample(bad4)


def test_expand_output_modes_pairing():
    # RAG:verbatim / non-verbatim 配对;中文:json / paragraph 配对;代码:单一
    assert expand_output_modes("rag") == [("verbatim", "paragraph"),
                                          ("non-verbatim", "paragraph")]
    assert expand_output_modes("chinese") == [("verbatim", "json"),
                                              ("verbatim", "paragraph")]
    assert expand_output_modes("code") == [("verbatim", "paragraph")]


def test_jsonl_roundtrip(tmp_path):
    from common import read_jsonl, write_jsonl
    recs = [make_sample(f"rag-{i}", "rag", "alce", "long", f"文本{i}",
                        "verbatim", "paragraph") for i in range(3)]
    p = tmp_path / "samples.jsonl"
    write_jsonl(recs, p)
    assert read_jsonl(p) == recs


def test_pools_and_samples_end_to_end_consistency():
    """样本前缀 ID 进四池,扩长后的 -long 变体不进入池(池只含基础 ID)。"""
    records = [{"sample_id": f"rag-{i:04d}", "scene": "rag"} for i in range(60)]
    records += [{"sample_id": f"chinese-{i:04d}", "scene": "chinese"} for i in range(60)]
    pools = assign_pools(records, seed=0)
    for ids in pools.values():
        for i in ids:
            assert not i.endswith("-long"), "池 ID 必须是基础记录 ID,不含变体后缀"
