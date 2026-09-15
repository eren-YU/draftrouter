#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""四套版本化 Prompt 模板(WP3 / baseline §6)。

- zh_json:      中文任务,要求 JSON 四字段输出(公司名/金额/时间/产品名)
- zh_paragraph: 中文任务,自然段落输出(与 zh_json 配对)
- rag_verbatim: RAG 逐字引用(与 non-verbatim 配对,G6 控制组用后者)
- rag_nonverbatim: RAG 禁止逐字引用,要求同长度改写/理由输出

模板一经冻结不得改写语义;调整必须递增 TEMPLATE_VERSION 并记录在结果 JSON。
"""

from __future__ import annotations

# 模板版本:语义/措辞变更时递增(日期-序号制)
TEMPLATE_VERSION = "2026-09-15-v1"

TEMPLATES = ("zh_json", "zh_paragraph", "rag_verbatim", "rag_nonverbatim")

# zh_json 四字段口径(与 WP2 数据侧四字段筛查一致)
FOUR_FIELDS = ("公司名", "金额", "时间", "产品名")

_JSON_INSTRUCTION = (
    "请从下面的文本中抽取信息,并以 JSON 对象输出,包含四个字段:"
    '"company"(公司名)、"amount"(金额)、"time"(时间)、"product"(产品名)。'
    '某个字段在文本中没有出现时,对应值填 null。只输出 JSON,不要输出其他内容。'
)
_PARAGRAPH_INSTRUCTION = (
    "请阅读下面的文本,用一段连贯的中文自然语言(不要使用 JSON、不要使用列表)"
    "概述其中涉及的公司名、金额、时间与产品名。没有提到的信息不要提及。"
)
_RAG_VERBATIM_INSTRUCTION = (
    "请仅依据下面提供的参考文档回答问题。答案中的关键信息必须逐字引用参考文档原文,"
    "不得改写、不得概括,不得使用文档以外的知识。"
)
_RAG_NONVERBATIM_INSTRUCTION = (
    "请仅依据下面提供的参考文档回答问题。禁止逐字复制参考文档原文:"
    "请用与原文长度相近的改写表述关键信息,并简要给出依据理由。"
)


def render(template: str, text: str, question: str | None = None) -> str:
    """渲染模板。zh_* 只用 text;rag_* 需要 text(拼接文档)与 question。"""
    if template == "zh_json":
        return _JSON_INSTRUCTION + "\n\n文本:\n" + text
    if template == "zh_paragraph":
        return _PARAGRAPH_INSTRUCTION + "\n\n文本:\n" + text
    if template == "rag_verbatim":
        if question is None:
            raise ValueError("rag_verbatim 模板需要 question 参数")
        return (_RAG_VERBATIM_INSTRUCTION + "\n\n参考文档:\n" + text
                + "\n\n问题:" + question)
    if template == "rag_nonverbatim":
        if question is None:
            raise ValueError("rag_nonverbatim 模板需要 question 参数")
        return (_RAG_NONVERBATIM_INSTRUCTION + "\n\n参考文档:\n" + text
                + "\n\n问题:" + question)
    raise ValueError(f"未知模板: {template!r}, 可选 {TEMPLATES}")
