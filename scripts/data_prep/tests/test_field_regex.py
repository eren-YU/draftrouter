"""四字段中文正则:命中/不命中样例(正则集中在 field_regex.py,调规则只动那里)。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest  # noqa: E402
from field_regex import (  # noqa: E402
    FIELD_COVERAGE_THRESHOLD,
    FIELD_NAMES,
    detect_fields,
    field_coverage,
)

HIT_TEXT = ("2023年5月12日,华为公司在深圳举办发布会,正式发布了Mate 60系列产品,"
            "售价5999元,由华为技术有限公司与招商银行联合推出,首销当日销售额突破1亿元。")

MISS_TEXT = "这是一段没有任何实体信息的普通描述,讲述了天气与心情。"


def test_hit_text_all_four_fields():
    hits = detect_fields(HIT_TEXT)
    for name in FIELD_NAMES:
        assert hits[name], f"{name} 应命中,实际 {hits}"


def test_amount_variants():
    hits = detect_fields("预算为3.5万元,总投资12亿元,营收增长到200万美元。")["amount"]
    assert hits, "万元/亿元/美元 形态应命中金额"


def test_amount_not_percent():
    hits = detect_fields("增长率达到15.5%,排名第3名。")["amount"]
    assert not any("%" in h for h in hits), "百分比不应作为金额命中"


def test_time_variants():
    for text in ("1998年发生了一件大事。", "上世纪90年代开始流行。", "近日,相关部门发文。"):
        assert detect_fields(text)["time"], f"{text} 应命中时间"


def test_company_variants():
    for text in ("腾讯科技(深圳)有限公司成立。", "中国工商银行发布公告。", "Apple Inc was founded."):
        assert detect_fields(text)["company"], f"{text} 应命中公司名"


def test_product_quoted_with_release_context():
    text = "发布会上,「星舰X1」正式发布,引发关注。"
    assert detect_fields(text)["product"], "引号产品名+发布上下文应命中"
    # 无发布上下文的引号词不应命中
    assert not detect_fields("他读了「红楼梦」这本书。")["product"], "引号书名不应误报"


def test_miss_text_no_fields():
    hits = detect_fields(MISS_TEXT)
    for name in FIELD_NAMES:
        assert not hits[name], f"{name} 不应命中,实际 {hits[name]}"


def test_field_coverage_threshold_semantics():
    docs = [HIT_TEXT, MISS_TEXT]
    cov = field_coverage(docs)
    assert set(cov) == set(FIELD_NAMES)
    for v in cov.values():
        assert 0.0 <= v <= 1.0
    assert cov["company"] == pytest.approx(0.5)  # 两篇中一篇命中
    assert FIELD_COVERAGE_THRESHOLD == 0.60      # P0 冻结硬门槛
