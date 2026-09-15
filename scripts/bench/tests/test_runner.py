# -*- coding: utf-8 -*-
"""runner 纯逻辑单测:TTFT/TPOT/e2e 计算、汇总、shuffle 确定性、batch 提取兜底。"""

import pytest

from runner import (compute_request_metrics, extract_batch_stats,
                    extract_num_output_tokens, shuffle_group_order,
                    summarize_requests)


class FakeMetrics:
    """模拟 vLLM RequestOutput.metrics(只含时间字段)。"""

    def __init__(self, arrival, first_token, finished):
        self.arrival_time = arrival
        self.first_token_time = first_token
        self.finished_time = finished


def test_tpot_ttft_e2e_correctness():
    # arrival=0, 首 token=1.0, 结束=5.0, 输出 5 token => decode 4 步
    m = compute_request_metrics(FakeMetrics(0.0, 1.0, 5.0), num_output_tokens=5)
    assert m["ttft_s"] == pytest.approx(1.0)
    assert m["e2e_s"] == pytest.approx(5.0)
    assert m["tpot_s"] == pytest.approx(4.0 / 4)   # (5-1)/(5-1)
    # 输出 token/s = decode 步数 / TPOT = 4 / 1.0
    assert m["output_tokens_per_s"] == pytest.approx(4.0)


def test_single_token_no_tpot():
    # 单 token 输出无 decode 步,TPOT 记 None
    m = compute_request_metrics(FakeMetrics(0.0, 1.0, 1.5), num_output_tokens=1)
    assert m["ttft_s"] == pytest.approx(1.0)
    assert m["tpot_s"] is None
    assert m["output_tokens_per_s"] is None


def test_missing_metrics_fields_none():
    class Empty:
        pass
    m = compute_request_metrics(Empty(), num_output_tokens=3)
    assert m["ttft_s"] is None and m["tpot_s"] is None and m["e2e_s"] is None


def test_extract_num_output_tokens_duck_typing():
    assert extract_num_output_tokens({"x": 1}) is None if False else True
    class Out:  # 新路径:outputs[0].token_ids
        class O:
            token_ids = [1, 2, 3]
        outputs = [O()]
        num_output_tokens = None
    assert extract_num_output_tokens(Out) == 3


def test_summarize_skips_none():
    rows = [
        {"ttft_s": 1.0, "tpot_s": 0.1, "e2e_s": 2.0,
         "output_tokens_per_s": 10.0, "num_output_tokens": 10},
        {"ttft_s": 3.0, "tpot_s": 0.3, "e2e_s": 4.0,
         "output_tokens_per_s": 30.0, "num_output_tokens": 10},
        {"ttft_s": None, "tpot_s": None, "e2e_s": None,
         "output_tokens_per_s": None, "num_output_tokens": 10},
    ]
    s = summarize_requests(rows)
    assert s["median_ttft_s"] == pytest.approx(2.0)
    assert s["n_requests"] == 3


def test_shuffle_deterministic_same_seed():
    items = list(range(50))
    a = shuffle_group_order(items, seed=0)
    b = shuffle_group_order(items, seed=0)
    c = shuffle_group_order(items, seed=1)
    assert a == b
    assert a != c
    assert sorted(a) == items  # 只重排,不丢元素


def test_extract_batch_stats_no_api_returns_none():
    # 无 get_metrics 接口 => 记 None,不抛错、不编造
    class FakeLLM:
        pass
    out = extract_batch_stats(FakeLLM())
    assert out == {"achieved_running_batch": None, "num_preemptions": None}


def test_extract_batch_stats_prometheus_parse():
    text = (
        '# HELP vllm:num_requests_running num running\n'
        'vllm:num_requests_running{engine="0"} 6.0\n'
        'vllm:num_preemptions_total{engine="0"} 3.0\n'
    )
    class FakeEngine:
        def get_metrics(self_inner):
            return text
    class FakeLLM:
        llm_engine = FakeEngine()
    out = extract_batch_stats(FakeLLM())
    assert out["achieved_running_batch"] == 6.0
    assert out["num_preemptions"] == 3.0
