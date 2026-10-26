"""离线评测入口：一次性跑完三套评测并输出报告。

运行： python scripts/run_eval.py
输出： data/eval_report.json + 终端摘要
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import DATA_DIR  # noqa: E402
from app.eval.evaluate import (  # noqa: E402
    evaluate_grading,
    evaluate_mapping,
    grid_search_thresholds,
    write_report,
)
from app.eval.verify_eval import run_verify_eval  # noqa: E402


def main() -> None:
    report: dict = {}

    print("=" * 78)
    print("  1/3  条款映射评测（Top-1 / Top-3 / 错误预填率）")
    print("=" * 78)
    m = evaluate_mapping()
    report["mapping"] = {k: v for k, v in m.items() if k != "cases"}
    print(f"  样本数            : {m['n']}")
    print(f"  Top-1 准确率      : {m['top1_acc']:.2%}")
    print(f"  Top-3 命中率      : {m['top3_acc']:.2%}")
    print(f"  预填率            : {m['prefill_rate']:.2%}")
    print(f"  人工选条款率      : {m['human_select_rate']:.2%}")
    print(f"  无可靠匹配(弃答)率: {m['no_match_rate']:.2%}")
    print(f"  错误预填率        : {m['wrong_prefill_rate']:.2%}   ← 优先压低")
    print(f"  当前阈值          : high={m['thresholds']['high']} mid={m['thresholds']['mid']}")

    print("\n  错配样本明细：")
    for c in m["cases"]:
        if not c["hit3"]:
            print(f"      期望 {c['gold']}  实际 {c['top1']} (conf={c['top1_conf']})  「{c['raw_text']}」")

    print("\n  阈值网格搜索（在可接受错误预填率下尽量提高预填率）：")
    grid = grid_search_thresholds()
    ok = [g for g in grid if g["wrong_prefill_rate"] <= 0.06]
    best = max(ok or grid, key=lambda g: (g["prefill_rate"], -g["wrong_prefill_rate"]))
    report["mapping_threshold_grid"] = grid
    report["mapping_recommended"] = best
    print("      high   mid    top1    top3   prefill  human   nomatch  wrong_prefill")
    for g in sorted(grid, key=lambda x: (x["thresholds"]["high"], x["thresholds"]["mid"])):
        print(f"      {g['thresholds']['high']:.2f}   {g['thresholds']['mid']:.2f}   "
              f"{g['top1_acc']:.2%}   {g['top3_acc']:.2%}   {g['prefill_rate']:.2%}   "
              f"{g['human_select_rate']:.2%}   {g['no_match_rate']:.2%}    {g['wrong_prefill_rate']:.2%}")
    print(f"      → 推荐阈值 high={best['thresholds']['high']} mid={best['thresholds']['mid']} "
          f"(错误预填率 {best['wrong_prefill_rate']:.2%})")

    print("\n" + "=" * 78)
    print("  2/3  风险分级评测（A级召回优先）")
    print("=" * 78)
    g = evaluate_grading()
    report["grading"] = {k: v for k, v in g.items() if k != "cases"}
    print(f"  样本数              : {g['n']}")
    print(f"  A级召回率           : {g['A_recall']:.2%}   ← 最重要，目标接近 100%")
    print(f"  A级精确率           : {g['A_precision']:.2%}")
    print(f"  整体一致率          : {g['overall_agreement']:.2%}")
    print(f"  人工确认负担        : {g['human_confirm_rate']:.2%}")
    print(f"  LLM分支占比         : {g['llm_branch_rate']:.2%}   （过高说明规则表覆盖不足）")
    print(f"  过度升级率          : {g['over_escalation_rate']:.2%}")
    print("\n  分级不一致样本：")
    for c in g["cases"]:
        if c["gold"] != c["pred"]:
            print(f"      gold={c['gold']} pred={c['pred']}  命中规则={c['hits']}  「{c['text']}」")

    print("\n" + "=" * 78)
    print("  3/3  证据验证评测（误杀率优先控制）")
    print("=" * 78)
    v = run_verify_eval(n_each=30)
    report["evidence"] = v
    print(f"  样本数（每类30）    : {v['n']}")
    print(f"  打回精确率          : {v['reject_precision']}")
    print(f"  打回召回率          : {v['reject_recall']}")
    print(f"  真整改误杀率        : {v['false_reject_rate']:.2%}   ← 目标 < 5%")
    print(f"  存疑率              : {v['uncertain_rate']:.2%}")
    print(f"  自动通过率          : {v['auto_pass_rate']:.2%}")
    print("  分类别打回情况：")
    for k, s in v["by_kind"].items():
        print(f"      {k:<10} 共 {s['total']:>3} 条，打回 {s['rejected']:>3} 条")

    path = write_report(str(DATA_DIR / "eval_report.json"), report)
    print(f"\n报告已写入：{path}")


if __name__ == "__main__":
    main()
