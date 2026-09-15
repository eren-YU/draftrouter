# -*- coding: utf-8 -*-
"""spec 统计单测:字段快照探测、日志正则解析、接受率分母口径。"""

from spec_stats import (SpecStats, parse_spec_log_line, snapshot_from_object,
                        snapshot_spec_stats)


class FakeSpecStats:
    """模拟 SpecDecodingStats 字段快照(与 vLLM 同名字段)。"""

    def __init__(self, drafts=10, draft_tokens=50, accepted=30, per_pos=(10, 8, 6)):
        self.num_drafts = drafts
        self.num_draft_tokens = draft_tokens
        self.num_accepted_tokens = accepted
        self.num_accepted_tokens_per_pos = list(per_pos)


def test_snapshot_from_object_full():
    s = snapshot_from_object(FakeSpecStats())
    assert s.num_drafts == 10
    assert s.num_draft_tokens == 50
    assert s.num_accepted_tokens == 30
    assert s.num_accepted_tokens_per_pos == [10, 8, 6]
    # 接受率分母固定 num_draft_tokens:30/50
    assert s.acceptance_rate == 0.6


class PartialSpecStats:
    num_drafts = 5
    # 其余字段缺失


def test_snapshot_partial_fields_warn_none():
    s = snapshot_from_object(PartialSpecStats())
    assert s.num_drafts == 5
    assert s.num_draft_tokens is None
    assert s.num_accepted_tokens is None
    assert s.acceptance_rate is None  # 分母缺失 => None,不猜


def test_log_line_equals_format():
    line = ("INFO ... Speculative decoding stats: num_drafts=12, "
            "num_draft_tokens=36, num_accepted_tokens=24, "
            "num_accepted_tokens_per_pos=[12, 8, 4]")
    s = parse_spec_log_line(line)
    assert s is not None
    assert (s.num_drafts, s.num_draft_tokens, s.num_accepted_tokens) == (12, 36, 24)
    assert s.num_accepted_tokens_per_pos == [12, 8, 4]
    assert s.acceptance_rate == pytest_approx(24 / 36)


def pytest_approx(x):
    import pytest
    return pytest.approx(x)


def test_log_line_colon_format_case_insensitive():
    line = "spec stats Num_Draft_Tokens : 100 | Num_Accepted_Tokens: 55 | num_drafts = 50"
    s = parse_spec_log_line(line)
    assert s.num_draft_tokens == 100
    assert s.num_accepted_tokens == 55
    assert s.num_drafts == 50
    assert s.num_accepted_tokens_per_pos is None  # 该行没有,记 None


def test_log_line_tuple_per_pos():
    line = "spec num_accepted_tokens_per_pos=(3, 2, 1) num_draft_tokens=9 num_accepted_tokens=6 num_drafts=3"
    s = parse_spec_log_line(line)
    assert s.num_accepted_tokens_per_pos == [3, 2, 1]


def test_log_line_non_spec_returns_none():
    assert parse_spec_log_line("INFO application startup complete") is None


def test_spec_line_without_fields_returns_none():
    # 含 spec 但无四字段数字 => None(不误配)
    assert parse_spec_log_line("speculative decoding enabled") is None


def test_snapshot_probe_miss_returns_all_none():
    # 引擎对象无任何探测路径 => 全 None,不抛
    class FakeLLM:
        pass
    s = snapshot_spec_stats(FakeLLM())
    assert isinstance(s, SpecStats)
    assert s.num_drafts is None and s.num_draft_tokens is None
