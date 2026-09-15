# -*- coding: utf-8 -*-
"""schema 单测:必含键完整性、null 兜底、写盘回读。"""

import json

import pytest

import schema
from schema import build_result, fill_metrics, validate_result, write_result


def test_required_keys_complete():
    result = build_result(
        config={"bs": 1, "output_mode": "fixed512",
                "template_version": "v1", "spec_config": "A"},
        seed=0, dataset="dureader", length_bucket="native",
        sample_ids=["s1", "s2"])
    validate_result(result)  # 不抛即通过
    for key in ("vllm_version", "image_digest", "gpu_name", "wall_time_s"):
        assert key in result


def test_env_fields_none_without_gpu(monkeypatch):
    # 本地无 vllm/GPU/镜像 digest:对应字段必须是 None,不允许编造
    monkeypatch.delenv("DRAFTROUTER_IMAGE_DIGEST", raising=False)
    result = build_result(config={}, seed=0, dataset="d", length_bucket=None,
                          sample_ids=[])
    assert result["vllm_version"] is None or isinstance(result["vllm_version"], str)
    assert result["image_digest"] is None
    assert result["num_samples"] == 0


def test_fill_metrics_roundtrip():
    result = build_result(config={"bs": 8}, seed=0, dataset="d",
                          length_bucket=None, sample_ids=["a"])
    metrics = {"bs": 8, "wall_time_s": 12.5, "achieved_running_batch": 3.0,
               "num_preemptions": 1, "requests": [{"ttft_s": 0.1}],
               "median_ttft_s": 0.1}
    fill_metrics(result, metrics)
    assert result["achieved_running_batch"] == 3.0
    assert result["num_preemptions"] == 1
    assert result["wall_time_s"] == 12.5
    assert result["metrics"]["requests"] == [{"ttft_s": 0.1}]


def test_write_result_rejects_incomplete(tmp_path):
    with pytest.raises(KeyError):
        write_result(str(tmp_path / "bad.json"), {"foo": 1})


def test_write_result_roundtrip(tmp_path):
    result = build_result(config={"bs": 4, "output_mode": "fixed512",
                                  "template_version": "v1", "spec_config": "A"},
                          seed=0, dataset="ceval", length_bucket="native",
                          sample_ids=["x1"])
    path = tmp_path / "out.json"
    write_result(str(path), result)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    validate_result(loaded)
    assert loaded["seed"] == 0 and loaded["dataset"] == "ceval"


def test_uptime_reader_handles_windows(monkeypatch):
    # /proc/uptime 不存在 => None(Windows 本地口径)
    monkeypatch.delenv("DRAFTROUTER_INSTANCE_UPTIME_S", raising=False)
    import config
    orig_open = open

    def fake_open(path, *a, **k):
        if str(path) == "/proc/uptime":
            raise OSError("no such file")
        return orig_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    assert config.read_instance_uptime_seconds() is None
    # 环境变量覆盖口径
    monkeypatch.setenv("DRAFTROUTER_INSTANCE_UPTIME_S", "123.5")
    assert config.read_instance_uptime_seconds() == 123.5
