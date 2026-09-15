# -*- coding: utf-8 -*-
r"""四字段(公司名/金额/时间/产品名)中文启发式正则,集中一个模块便于调。

【已知局限】(写给 reviewer / P6)
- 公司名:只覆盖「X公司/集团/银行/大学/研究院」等显式后缀与「XX股份/控股」形态;
  外企音译名(如"微软")、无后缀简称、英文公司名会漏;括号别名(如"(华为)")算命中但可能误收普通括号词。
- 金额:覆盖 元/万元/亿元/美元/万/亿/人民币/美金;纯数字+英文货币符号($100)只部分覆盖;
  百分比、"第X名"中的数字不算金额(特意排除),但"3 万元预算"与"跑了3万米"无法区分,会误报。
- 时间:覆盖 \d{4}年 / X月X日 / 上世纪X年代 / 明年/去年 等相对词;仅"周三/本月"这类弱时间词
  单独不算命中(避免把"本产品"误判),精确性优先于召回。
- 产品名:纯启发式 = 引号(「」“”《》)内名词 + 发布/推出/上市/上线 上下文,或 "X 产品/X 系列/X 手机";
  引号内可能抓到书名/活动名;上下文词窗只看紧邻 12 字,跨句引用会漏。
- 以上正则只为数据侧预筛(G4 出现率统计)服务,不是信息抽取器;字段值提取质量不用于任何结论。

正则必须集中在这里,改阈值/加规则只动本文件,保证确定性可复现。
"""

from __future__ import annotations

import re

# 每类字段一组正则;任一命中即视为该文档含此字段
FIELD_PATTERNS: dict[str, list[re.Pattern]] = {
    # 公司名:显式中文组织后缀
    "company": [
        re.compile(r"[\u4e00-\u9fa5A-Za-z0-9·]{1,20}(?:股份有限公司|有限责任公司|有限公司|集团公司|集团|公司)"),
        re.compile(r"[\u4e00-\u9fa5A-Za-z0-9·]{1,20}(?:银行|证券|保险|基金会|研究院|研究所)"),
        re.compile(r"[\u4e00-\u9fa5A-Za-z0-9·]{1,20}(?:集团|公司|大学)"),
        re.compile(r"[A-Za-z][A-Za-z0-9 &·]{1,30}(?:Inc|Corp|Ltd|LLC|GmbH)\b"),
    ],
    # 金额:数字 + 货币单位;排除纯百分比/序号
    "amount": [
        re.compile(r"(?:人民币|美金|美元)?\s*[¥￥$]?\s*\d+(?:\.\d+)?\s*(?:万亿|亿|万)?(?:元|美元|人民币|美金)"),
        re.compile(r"\d+(?:\.\d+)?\s*(?:亿元|万元|亿|万)(?:人民币)?"),
        re.compile(r"[¥￥$€£]\s*\d[\d,\.]*"),
    ],
    # 时间:年月日 / 年代 / 相对时间词
    "time": [
        re.compile(r"\d{2,4}\s*年(?:度)?"),
        re.compile(r"\d{1,2}\s*月\d{1,2}\s*日"),
        re.compile(r"(?:19|20)\d{2}\s*年代"),
        re.compile(r"(?:上世纪|公元)\s*\d{1,2}\s*世纪"),
        re.compile(r"(?:去年|今年|明年|近日|日前|昨日|今日|本周|上周)(?![\u4e00-\u9fa5])"),
    ],
    # 产品名:引号内名词 + 发布类上下文;"X产品/X系列/X机型"形态
    "product": [
        re.compile(r"[「「“\"'《][\u4e00-\u9fa5A-Za-z0-9·\-\s]{1,30}[」”\"'》][^。]{0,12}?(?:发布|推出|上市|上线|亮相|开售|发售)"),
        re.compile(r"(?:发布|推出|上市|上线|亮相|开售|发售)(?:了)?[「「“\"'《][\u4e00-\u9fa5A-Za-z0-9·\-\s]{1,30}[」”\"'》]"),
        re.compile(r"[\u4e00-\u9fa5A-Za-z0-9·\-]{1,20}(?:系列|机型|型号|产品线)(?:的)?(?:新)?(?:产品|手机|设备|芯片|模型)?"),
        re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,15}\s*(?:\d+(?:\.\d+)?)?\s*(?:Pro|Max|Plus|Ultra|SE)\b"),
    ],
}

FIELD_NAMES = ["company", "amount", "time", "product"]


def detect_fields(text: str) -> dict[str, list[str]]:
    """返回 {字段名: [命中片段...]};片段去重保序。"""
    hits: dict[str, list[str]] = {}
    for name in FIELD_NAMES:
        found: list[str] = []
        for pat in FIELD_PATTERNS[name]:
            for m in pat.finditer(text):
                s = m.group(0).strip()
                if s and s not in found:
                    found.append(s)
        hits[name] = found
    return hits


def field_coverage(docs: list[str]) -> dict[str, float]:
    """每类字段的出现率 = 含该字段的文档数 / 总文档数。G4 门槛:每类 >= 0.60。"""
    n = len(docs)
    if n == 0:
        return {name: 0.0 for name in FIELD_NAMES}
    counts = {name: 0 for name in FIELD_NAMES}
    for d in docs:
        hits = detect_fields(d)
        for name in FIELD_NAMES:
            if hits[name]:
                counts[name] += 1
    return {name: counts[name] / n for name in FIELD_NAMES}


# G4 数据侧硬门槛(P0 冻结为 0.60,不做事后校准;见 P0-action-plan WP2 第 5 条)
FIELD_COVERAGE_THRESHOLD = 0.60
