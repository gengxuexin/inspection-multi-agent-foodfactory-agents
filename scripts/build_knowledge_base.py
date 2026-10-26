"""构建你自己的条款库（从 HACCP 计划 / SSOP / GMP 导出）。

用法：
    python scripts/build_knowledge_base.py --export          # 导出内置模板为 JSON 模板
    python scripts/build_knowledge_base.py --validate         # 校验现有 knowledge/clauses.json
    python scripts/build_knowledge_base.py --stats            # 查看条款库统计

导入自有条款库时，元数据 zone / process / is_ccp / hazard_type 必须由质量部人工校验，
不能全靠模型抽取——这是硬过滤正确性的前提。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import KNOWLEDGE_DIR  # noqa: E402
from app.rag.knowledge import ClauseLibrary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="导出内置模板为 knowledge/clauses_template.json")
    ap.add_argument("--validate", action="store_true", help="校验 knowledge/clauses.json")
    ap.add_argument("--stats", action="store_true", help="统计条款库")
    args = ap.parse_args()

    lib = ClauseLibrary.load()

    if args.export:
        out = KNOWLEDGE_DIR / "clauses_template.json"
        lib.dump_to_json(out)
        print(f"已导出模板：{out}（{len(lib.clauses)} 条）")
        print("请替换为企业自有条款，并人工校验元数据后另存为 knowledge/clauses.json")
        return

    if args.validate:
        problems = lib.validate()
        if problems:
            print(f"发现 {len(problems)} 个元数据问题：")
            for p in problems:
                print(f"  - {p}")
        else:
            print("条款库元数据校验通过。")
        return

    # 默认：统计
    effective = lib.latest_by_id()
    print(f"条款总数（去重前）：{len(lib.clauses)}")
    print(f"生效版本条款数    ：{len(effective)}")
    print("\n按来源：")
    for k, v in Counter(c.source for c in effective).most_common():
        print(f"  {k:<16} {v}")
    print("\n按危害类型：")
    for k, v in Counter(c.hazard_type for c in effective).most_common():
        print(f"  {k:<16} {v}")
    print("\nCCP 条款：")
    for c in effective:
        if c.is_ccp:
            print(f"  {c.clause_id:<10} {c.question_category:<24} CL={c.critical_limit}")
    print("\n按问题类别：")
    for k, v in Counter(c.question_category for c in effective).most_common():
        print(f"  {k:<28} {v}")


if __name__ == "__main__":
    main()
