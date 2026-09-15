"""数据源解析单测:DuReader-robust tar/json 与 repobench parquet 纯函数(全 mock,不触网)。"""

import io
import json
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest  # noqa: E402
from loaders import (  # noqa: E402
    dureader_records,
    load_alce_tar,
    load_dureader_tar,
    load_longbench_zip,
    repobench_row_to_record,
)

# ---------------------------------------------------------------------------
# DuReader-robust(SQuAD 风格;2026-09-15 云端实测格式)

def _dureader_obj():
    return {"data": [{
        "paragraphs": [
            {"context": "2023年华为公司发布了新产品,售价5999元。",
             "qas": [{"question": "哪年发布的?", "id": "a", "answers": []}]},
            {"context": "第二篇文章讲的是金额与时间。",
             "qas": [{"question": "讲了什么?", "id": "b", "answers": []}]},
            {"context": "",  # 空 context 应被跳过
             "qas": [{"question": "空文档?", "id": "c", "answers": []}]},
        ],
    }]}


def test_dureader_records_structure():
    recs = dureader_records(_dureader_obj())
    assert len(recs) == 2  # 空 context 跳过
    assert recs[0]["question"] == "哪年发布的?"
    assert recs[0]["passages"] == ["2023年华为公司发布了新产品,售价5999元。"]
    # 整篇作为单 passage(供四字段筛查与四池)
    assert len(recs[1]["passages"]) == 1


def test_dureader_records_empty_and_missing_qas():
    assert dureader_records({"data": []}) == []
    obj = {"data": [{"paragraphs": [{"context": "只有篇章没有问题。"}]}]}
    recs = dureader_records(obj)
    assert recs[0]["question"] == "总结以下文档。"  # 无 qas 时给默认问题


def test_load_dureader_tar(tmp_path):
    tar_path = tmp_path / "dureader_robust-data.tar.gz"
    payload = json.dumps(_dureader_obj(), ensure_ascii=False).encode("utf-8")
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo("dureader_robust-data/dev.json")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))
    recs = load_dureader_tar(tar_path, split="dev", limit=1)
    assert len(recs) == 1
    assert recs[0]["passages"][0].startswith("2023年华为公司")
    with pytest.raises(FileNotFoundError):
        load_dureader_tar(tar_path, split="nope")


# ---------------------------------------------------------------------------
# repobench-c parquet(列名 2026-09-15 云端实测:prompt 列存在)

def test_repobench_row_to_record():
    row = {"repo_name": "a/b", "file_path": "x.py", "context": "c",
           "import_statement": "import os", "code": "code", "prompt": "PROMPT",
           "next_line": "nl"}
    rec = repobench_row_to_record(row)
    assert rec == {"question": "PROMPT", "passages": []}
    assert repobench_row_to_record({"prompt": None}) is None
    assert repobench_row_to_record({}) is None


# ---------------------------------------------------------------------------
# ALCE tar(2026-09-15 云端实测:成员为 eli5/asqa 等 eval json 数组,无 hotpotqa)

def test_load_alce_tar_json_array(tmp_path):
    tar_path = tmp_path / "alce_data.tar"
    eli5 = [{"question": f"eli5 问题{i}?",
             "answer": "答案",
             "docs": [{"title": f"t{i}", "text": "2023年某公司发布产品售价1亿元。"}]}
            for i in range(3)]
    asqa = [{"question": "asqa 问题?",
             "docs": [{"title": "t", "text": "文档"}]},
            {"question": "无 docs 应跳过", "docs": []}]
    with tarfile.open(tar_path, "w") as tf:
        for name, arr in (("ALCE-data/eli5_eval_bm25_top100.json", eli5),
                          ("ALCE-data/asqa_eval_gtr_top100.json", asqa)):
            payload = json.dumps(arr, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    recs = load_alce_tar(tar_path)
    assert len(recs) == 4  # eli5 3 条 + asqa 1 条(无 docs 跳过)
    assert all(r["passages"] and r["question"] for r in recs)
    assert {r["subset"] for r in recs} == {"eli5", "asqa"}
    limited = load_alce_tar(tar_path, limit=2)
    assert len(limited) == 2
    assert limited[0]["subset"] == "asqa"  # 成员名排序,asqa 在前;limit 跨子集截断
    assert limited[1]["subset"] == "eli5"


# ---------------------------------------------------------------------------
# LongBench data.zip fallback(data.zip 内各子集 jsonl:context+input)

def test_load_longbench_zip(tmp_path):
    import zipfile
    zp = tmp_path / "longbench_data.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        zf.writestr("dureader.jsonl",
                    json.dumps({"input": "总结一下", "context": "2024年,某银行发布了产品。"},
                               ensure_ascii=False) + "\n" +
                    json.dumps({"context": "只有文档没有问题。"}, ensure_ascii=False))
    recs = load_longbench_zip(zp, limit=10)
    assert len(recs) == 2
    assert recs[0]["question"] == "总结一下"
    assert recs[1]["question"] == "总结以下文档。"  # 缺 input 用默认问题
