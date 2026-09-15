"""WP2 核心纯逻辑:RAG 扩长、三分位长度桶、四池分层抽样、输出形态配对、样本 schema。

本模块不含网络/GPU 依赖,tokenizer 以参数注入(duck-typing:有 encode(text)->list 即可),
本地单测用 fake tokenizer 验证;Qwen3 tokenizer 只在云端脚本里延迟导入。
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable

# ---------------------------------------------------------------------------
# C3 长度档口径:RAG long 主判档 = [7500, 8192](Qwen3 tokenizer)
RAG_LONG_LO = 7500
RAG_LONG_HI = 8192
MAX_INPUT_TOKENS = 8192  # C5:主矩阵只纳入 tokenizer 长度 <= 8192 的样本

SCENES = ("chinese", "code", "rag")

OUTPUT_STYLES = ("verbatim", "non-verbatim")      # RAG 引用形态
OUTPUT_FORMATS = ("json", "paragraph")             # 中文结构化形态

# 样本落盘 schema(P0 冻结);WP3 runner 按此读取
SAMPLE_FIELDS = {"sample_id", "scene", "dataset", "length_bucket",
                 "input_text", "output_mode"}


# ---------------------------------------------------------------------------
# 长度计算(tokenizer 注入)

def token_len(text: str, tokenize: Callable[[str], list]) -> int:
    return len(tokenize(text))


def build_long_input(question: str, passages: list[str],
                     tokenize: Callable[[str], list],
                     lo: int = RAG_LONG_LO, hi: int = RAG_LONG_HI) -> dict:
    """RAG 扩长:按原始检索段落顺序拼接多篇文档,直到 token 数落入 [lo, hi]。

    - short 档基准 = 原生样本(不做任何拼接);
    - 不足 lo 时按序追加段落;段落用尽仍不足 → 标记 insufficient=True(调用方记日志);
    - 超过 hi 时截到最后一个不越界的段落(保持段落完整,不做段内截断);
    - 分段累计计长与整文计长存在分词合并漂移,故拼接完成后对整文复测:
      仍超 hi 则回退段落重测,直到 <= hi(保证 C5 上限硬约束)。
    返回 {passages_used, token_len, insufficient, input_text},token_len 为整文实测值。
    """
    def _join(used: list[str]) -> str:
        return question + "\n" + "\n".join(used)

    used: list[str] = []
    n_tokens = token_len(question, tokenize)
    for p in passages:
        cand = n_tokens + token_len("\n" + p, tokenize)
        if cand > hi and used:  # 首段就超 hi 也收下(有文档总比空好),否则停
            break
        used.append(p)
        n_tokens = cand
        if n_tokens >= lo:
            break

    input_text = _join(used)
    n_full = token_len(input_text, tokenize)  # 整文复测(消除合并漂移)
    while used and n_full > hi:               # 漂移导致超上限 → 回退段落
        used.pop()
        input_text = _join(used)
        n_full = token_len(input_text, tokenize)
    insufficient = n_full < lo
    return {"passages_used": len(used), "token_len": n_full,
            "insufficient": insufficient, "input_text": input_text}


# ---------------------------------------------------------------------------
# C4 三分位长度桶(边界在 tuning+report 池上全局计算并冻结)

def _percentile(sorted_vals: list[float], q: float) -> float:
    """线性插值分位数(与 numpy default percentile 一致的定义)。"""
    if not sorted_vals:
        raise ValueError("empty values")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo_i = int(pos)
    hi_i = min(lo_i + 1, len(sorted_vals) - 1)
    frac = pos - lo_i
    return sorted_vals[lo_i] * (1 - frac) + sorted_vals[hi_i] * frac


def quantiles(values: Iterable[float], qs=(0.10, 0.50, 0.90, 0.99)) -> dict[str, float]:
    vals = sorted(float(v) for v in values)
    return {f"p{int(q * 100)}": round(_percentile(vals, q), 2) for q in qs}


def compute_tertile_boundaries(lengths: Iterable[float]) -> list[float]:
    """全局三分位边界 [t1, t2];桶:short<=t1 < mid<=t2 < long。"""
    vals = sorted(float(v) for v in lengths)
    return [_percentile(vals, 1 / 3), _percentile(vals, 2 / 3)]


def assign_bucket(length: float, boundaries: list[float]) -> str:
    if len(boundaries) != 2:
        raise ValueError("expect 2 boundaries")
    if length <= boundaries[0]:
        return "short"
    if length <= boundaries[1]:
        return "mid"
    return "long"


def freeze_buckets(lengths_by_scene: dict[str, list[float]]) -> dict:
    """C4:边界取三场景自然长度合并后的全局三分位,一次计算后冻结落盘。"""
    merged = [v for vals in lengths_by_scene.values() for v in vals]
    b = compute_tertile_boundaries(merged)
    return {"boundaries": [round(b[0], 2), round(b[1], 2)],
            "rule": "global tertile over tuning+report pools, all scenes merged",
            "version": 1}


# ---------------------------------------------------------------------------
# C12 四池分层抽样:tuning 50 / report 50 / data-valid 200 / router-train(独立)

POOL_SIZES = {"tuning": 50, "report": 50, "data_valid": 200, "router_train": 200}

# 池的场景策略:tuning/report 按各场景可用量比例分层;
# data-valid 专用于四字段数据侧筛查(G4 要求 200 条中文篇章),固定全中文;
# router-train 从剩余样本补足(任意场景)。
POOL_SCENE_POLICY = {"data_valid": ("chinese",)}

# 分层配额按场景占比(实现按各场景可用量比例分配,余数给排在前面的场景)
_SCENE_ORDER = ("chinese", "code", "rag")


def _quota(total: int, counts: dict[str, int]) -> dict[str, int]:
    """按可用样本量比例给各场景分配配额,余数依 _SCENE_ORDER 顺序补齐。"""
    total_avail = sum(counts.get(s, 0) for s in _SCENE_ORDER)
    quota: dict[str, int] = {}
    assigned = 0
    for s in _SCENE_ORDER:
        avail = counts.get(s, 0)
        if total_avail == 0:
            quota[s] = 0
            continue
        q = total * avail // total_avail
        q = min(q, avail)
        quota[s] = q
        assigned += q
    for s in _SCENE_ORDER:  # 余数(受 avail 截断产生)按序补给仍有余量的场景
        if assigned >= total:
            break
        avail_room = counts.get(s, 0) - quota[s]
        if avail_room > 0:
            add = min(avail_room, total - assigned)
            quota[s] += add
            assigned += add
    return quota


def assign_pools(records: list[dict], seed: int = 0,
                 sizes: dict[str, int] | None = None) -> dict[str, list[str]]:
    """确定性四池分配:按场景分层、每场景内用 seed=0 的 RNG 抽样。

    records: [{sample_id, scene, ...}](scene ∈ SCENES;同场景内按 sample_id 排序后抽样,保证确定性)。
    返回 {pool_name: [sample_id...]};四个池两两零交集(不满足则 AssertionError)。
    router-train 是独立可扩展池:P0 只落盘 ID,不与前三池共享任何样本。
    """
    sizes = sizes or POOL_SIZES
    by_scene: dict[str, list[str]] = {s: [] for s in _SCENE_ORDER}
    for r in records:
        scene = r["scene"]
        if scene not in by_scene:
            raise ValueError(f"unknown scene: {scene}")
        by_scene[scene].append(r["sample_id"])
    for s in by_scene:
        by_scene[s].sort()

    pools: dict[str, list[str]] = {}
    remaining = {s: list(ids) for s, ids in by_scene.items()}
    # 前三池依次抽取;router-train 从剩余中补足
    for pool in ("tuning", "report", "data_valid"):
        rng = random.Random(seed)
        allowed = POOL_SCENE_POLICY.get(pool, _SCENE_ORDER)
        quota = _quota(sizes[pool], {s: len(v) for s, v in remaining.items()
                                     if s in allowed})
        picked: list[str] = []
        for s in _SCENE_ORDER:
            if s not in allowed:
                continue
            take = min(quota[s], len(remaining[s]))
            for _ in range(take):
                # 逐个 pop(采样前先洗牌,洗牌序列由 seed 决定,确定性)
                idx = rng.randrange(len(remaining[s]))
                picked.append(remaining[s].pop(idx))
        picked.sort()
        pools[pool] = picked
    # router-train:剩余样本(任意场景)中取 sizes["router_train"] 个
    rest = [i for s in _SCENE_ORDER for i in remaining[s]]
    rng = random.Random(seed)
    picked = []
    for _ in range(min(sizes["router_train"], len(rest))):
        picked.append(rest.pop(rng.randrange(len(rest))))
    pools["router_train"] = sorted(picked)

    assert_disjoint(pools)
    return pools


def assert_disjoint(pools: dict[str, list[str]]) -> list[str]:
    """两两零交集断言;返回冲突描述列表(空即通过)。内部再 assert 保证硬失败。"""
    names = sorted(pools)
    conflicts: list[str] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = set(pools[names[i]]) & set(pools[names[j]])
            if overlap:
                conflicts.append(f"{names[i]} ∩ {names[j]} = {sorted(overlap)[:10]} ...")
    assert not conflicts, "pool overlap: " + "; ".join(conflicts)
    return conflicts


# ---------------------------------------------------------------------------
# 输出形态配对(C6/WP2 第 7 条):同一批样本 × {verbatim,non-verbatim} × {json,paragraph}

# 模板版本化;WP3 runner 落盘时引用此版本号
TEMPLATE_VERSION = "v1"

RAG_INSTRUCTIONS = {
    "verbatim": {
        "json": "根据上文,抽取与问题相关的信息,输出 JSON(键:company/amount/time/product)。"
                "每个字段值必须逐字引用上文原文,不得改写、不得概括;上文没有的信息留空字符串。",
        "paragraph": "根据上文,用自然段落回答问题。回答中涉及上文事实的表述必须逐字引用原文,"
                     "不得改写;并在末尾以「原文引用:」标注逐字引用的片段。",
    },
    "non-verbatim": {
        "json": "根据上文,抽取与问题相关的信息,输出 JSON(键:company/amount/time/product)。"
                "禁止逐字复制上文任何连续 8 字以上片段:所有字段值必须用自己的话同长度改写,"
                "保持信息量不变;上文没有的信息留空字符串。",
        "paragraph": "根据上文,用自然段落回答问题。禁止逐字复制上文任何连续 8 字以上片段:"
                     "所有事实必须用自己的话同长度改写,并附一句理由说明改写依据。",
    },
}

CHINESE_INSTRUCTIONS = {
    "json": "阅读以下中文文档,抽取四类字段输出 JSON(键:company/amount/time/product):"
            "公司名、金额、时间、产品名。字段值必须逐字取自文档原文,不得改写;缺失字段留空字符串。",
    "paragraph": "阅读以下中文文档,用自然段落总结其中的公司、金额、时间与产品信息,"
                 "涉及原文事实的表述必须逐字引用。",
}

CODE_INSTRUCTIONS = {
    "paragraph": "阅读以下代码,续写缺失的代码行,只输出代码,不输出解释。",
}


def make_output_mode(style: str, fmt: str) -> str:
    """output_mode 编码:{verbatim|non-verbatim}×{json|paragraph},如 verbatim-json。"""
    if style not in OUTPUT_STYLES or fmt not in OUTPUT_FORMATS:
        raise ValueError(f"bad style/format: {style}/{fmt}")
    return f"{style}-{fmt}"


def make_sample(sample_id: str, scene: str, dataset: str, length_bucket: str,
                input_text: str, style: str, fmt: str) -> dict:
    """构造一条样本记录;schema 见 SAMPLE_FIELDS(另附 template_version 供 WP3 记录)。"""
    rec = {
        "sample_id": sample_id,
        "scene": scene,
        "dataset": dataset,
        "length_bucket": length_bucket,
        "input_text": input_text,
        "output_mode": make_output_mode(style, fmt),
        "template_version": TEMPLATE_VERSION,
    }
    validate_sample(rec)
    return rec


def validate_sample(rec: dict) -> None:
    """样本 schema 校验:必填字段、场景与 output_mode 合法性。"""
    missing = SAMPLE_FIELDS - set(rec)
    assert not missing, f"sample missing fields: {missing}"
    assert rec["scene"] in SCENES, f"bad scene: {rec['scene']}"
    style, _, fmt = rec["output_mode"].rpartition("-")  # non-verbatim 含连字符,必须从右切
    assert style in OUTPUT_STYLES and fmt in OUTPUT_FORMATS, \
        f"bad output_mode: {rec['output_mode']}"
    assert rec["length_bucket"] in ("short", "mid", "long"), \
        f"bad length_bucket: {rec['length_bucket']}"
    assert isinstance(rec["input_text"], str) and rec["input_text"], "empty input_text"


def expand_output_modes(scene: str) -> list[tuple[str, str]]:
    """给定场景,返回应配对的 (style, fmt) 组合:
    - rag:verbatim/non-verbatim 两套 × paragraph(引用型输出);
    - chinese:json/paragraph 两套(逐字引用);
    - code:单一 paragraph。"""
    if scene == "rag":
        return [("verbatim", "paragraph"), ("non-verbatim", "paragraph")]
    if scene == "chinese":
        # 中文 JSON vs 自然段落配对;两者都要求逐字,风格档固定 verbatim
        return [("verbatim", "json"), ("verbatim", "paragraph")]
    return [("verbatim", "paragraph")]
