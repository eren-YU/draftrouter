"""四字段数据侧预筛(G4 数据侧):中文篇章正则筛查 + data-valid 池覆盖率统计。

【云端执行方式】
    /root/miniconda3/envs/draftrouter/bin/python scripts/data_prep/field_screen.py
(纯逻辑,不需要网络)

- 对 data-valid 池中的中文场景文档跑 detect_fields(正则集中在 field_regex.py);
- 输出每类字段出现率;门槛 = 每类 >= 60%(FIELD_COVERAGE_THRESHOLD,P0 冻结硬门槛);
- 任一类不达标 → 退出码 3(D2 触发换源决策);达标 → 退出码 0。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from base_records import build_base_records, load_pools  # noqa: E402
from common import data_root, save_json  # noqa: E402
from core import MAX_INPUT_TOKENS  # noqa: E402
from field_regex import (  # noqa: E402
    FIELD_COVERAGE_THRESHOLD,
    FIELD_NAMES,
    detect_fields,
    field_coverage,
)


def screen_records(records: list[dict], tokenize=None) -> dict:
    """对中文记录做四字段检测,返回统计报告(含 C5 超长剔除记录)。"""
    docs = [r for r in records if r["scene"] == "chinese"]
    report: dict = {"n_chinese": len(docs), "dropped_overlong": [],
                    "per_doc": [], "coverage": {}, "threshold": FIELD_COVERAGE_THRESHOLD}
    texts = []
    for r in docs:
        full_text = r["question"] + "\n" + "\n".join(r["passages"])
        if tokenize is not None:
            n = len(tokenize(full_text))
            if n > MAX_INPUT_TOKENS:  # C5:主矩阵只纳入 <=8192,超长剔除并记录
                report["dropped_overlong"].append({"sample_id": r["sample_id"], "token_len": n})
                continue
        hits = detect_fields(full_text)
        report["per_doc"].append({"sample_id": r["sample_id"],
                                  "fields": {k: len(v) for k, v in hits.items()}})
        texts.append(full_text)
    report["coverage"] = field_coverage(texts)
    report["pass"] = all(report["coverage"][k] >= FIELD_COVERAGE_THRESHOLD for k in FIELD_NAMES)
    return report


def main() -> None:
    root = data_root()
    records = build_base_records(root)
    pools = load_pools(root)
    valid_ids = set(pools.get("data_valid", []))
    if not valid_ids:
        print("[screen] data_valid 池为空,先执行 build_pools.py", file=sys.stderr)
        sys.exit(2)
    subset = [r for r in records if r["sample_id"] in valid_ids]
    report = screen_records(subset)

    save_json(report, root / "field_screen_data_valid.json")
    print("[screen] data-valid 中文文档四字段出现率:")
    print(json.dumps(report["coverage"], ensure_ascii=False, indent=2))
    if report["dropped_overlong"]:
        print(f"[screen] 剔除超长样本 {len(report['dropped_overlong'])} 条(>8192,已记录)")
    if report["pass"]:
        print(f"[screen] 达标(每类 >= {FIELD_COVERAGE_THRESHOLD:.0%})。")
    else:
        print(f"[screen] 未达标(门槛 {FIELD_COVERAGE_THRESHOLD:.0%})→ 触发 D2 换源评估。")
        sys.exit(3)


if __name__ == "__main__":
    main()
