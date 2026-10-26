"""离线评测集与指标（上线前必做）。

三套评测：
1. 条款映射：Top-1 / Top-3 准确率、人工改选率、无可靠匹配率、**错误预填率**
2. 风险分级：A级召回/精确率、整体一致率、人工确认负担、LLM分支占比
3. 证据验证：打回精确率/召回率、**误杀率**、存疑率、自动通过率

调参原则：
- 条款映射优先压低 wrong_prefill_rate（预填错条款比不预填更糟，会误导分级）；
- 分级 A级召回优先（宁可多升级人工确认，不可漏判A）；
- 证据验证 false_reject_rate 优先控制（误杀会摧毁现场对系统的信任）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Optional

from app.agents.clause_map_agent import ClauseMapAgent
from app.agents.risk_grade_agent import RiskGradeAgent
from app.core.config import settings
from app.core.schemas import CheckVerdict, MapMode, RiskLevel, VerifyRoute

# --------------------------------------------------------------------------
# 1) 条款映射评测集（人工标注 gold 条款）
# --------------------------------------------------------------------------
@dataclass
class MappingCase:
    raw_text: str
    gold_clause_id: str
    note: str = ""


MAPPING_CASES: list[MappingCase] = [
    MappingCase("前处理间地面有积水，排水口不干净", "SSOP-4.3", "清洁不彻底"),
    MappingCase("加工间地面积水严重，班后未清洁", "SSOP-4.3"),
    MappingCase("冷库温度记录缺失，昨天有两段没有数据", "CCP-01", "CCP温度记录"),
    MappingCase("冷库温度断链，中心温度超标", "CCP-01"),
    MappingCase("发酵间pH超出限值", "CCP-02"),
    MappingCase("金检间标准片校验不过，探测到金属异物", "CCP-03"),
    MappingCase("灌装间金检出金属异物，已停线", "CCP-03"),
    MappingCase("操作工未戴发网进入灌装间", "SSOP-2.1"),
    MappingCase("有人在更衣间不洗手直接进车间", "SSOP-2.1"),
    MappingCase("加工间员工手套破损未更换", "SSOP-2.1"),
    MappingCase("CIP消毒浓度不足，接触面可能残留", "SSOP-6.2"),
    MappingCase("捕鼠器破损未修复", "SSOP-8.4"),
    MappingCase("粘鼠板失效，虫害防控屏障失效", "SSOP-8.4"),
    MappingCase("发酵罐参数漂移，校准已过期", "GMP-7.2"),
    MappingCase("烘焙炉温度参数偏离设定值", "GMP-7.2"),
    MappingCase("标签过敏原信息缺失，缺少麸质标示", "AL-2.5", "过敏原"),
    MappingCase("换线后未做过敏原清洗，存在交叉接触风险", "AL-2.5"),
    MappingCase("标签批次号打印错误但可追溯", "AL-2.1"),
    MappingCase("刀具破损碎片可能掉入产品", "SSOP-5.6"),
    MappingCase("包装容器破损，存在异物风险", "SSOP-5.6"),
    MappingCase("仓库供应商到货包装变形", "SUP-3.4"),
    MappingCase("车间照明灯坏了一个", "SSOP-11.2"),
    MappingCase("更衣间门锁损坏", "SSOP-11.2"),
    MappingCase("生产记录台账漏填未记录", "SSOP-9.1"),
    MappingCase("记录表填错了，需要划改签名", "SSOP-9.1"),
    MappingCase("台面上有残留，桌面不洁", "SSOP-4.3"),
    MappingCase("垃圾桶未盖", "SSOP-4.3"),
]


def evaluate_mapping(
    agent: Optional[ClauseMapAgent] = None,
    cases: Optional[list[MappingCase]] = None,
    high_th: Optional[float] = None,
    mid_th: Optional[float] = None,
) -> dict[str, Any]:
    agent = agent or ClauseMapAgent()
    cases = cases or MAPPING_CASES
    high_th = high_th if high_th is not None else settings.clause_map.high_threshold
    mid_th = mid_th if mid_th is not None else settings.clause_map.mid_threshold
    cfg = agent.cfg

    top1 = top3 = prefill = human_select = no_match = 0
    wrong_prefill = 0
    rescored: list[dict[str, Any]] = []

    for c in cases:
        saved_high, saved_mid = cfg.high_threshold, cfg.mid_threshold
        cfg_high, cfg_mid = high_th, mid_th
        agent.cfg = type(cfg)(**{**cfg.__dict__, "high_threshold": cfg_high, "mid_threshold": cfg_mid})
        res = agent.map_clause(c.raw_text and f"INS-EVAL" or "INS-EVAL", c.raw_text)
        agent.cfg = cfg

        ids = [x.clause_id for x in res.candidates]
        hit1 = bool(ids) and ids[0] == c.gold_clause_id
        hit3 = c.gold_clause_id in ids
        top1 += int(hit1)
        top3 += int(hit3)

        if res.mode == MapMode.PREFILL:
            prefill += 1
            if not hit1:
                wrong_prefill += 1
        elif res.mode == MapMode.HUMAN_SELECT:
            human_select += 1
        else:
            no_match += 1

        rescored.append({
            "raw_text": c.raw_text, "gold": c.gold_clause_id, "mode": res.mode.value,
            "top1": ids[0] if ids else None,
            "top1_conf": res.candidates[0].confidence if res.candidates else 0.0,
            "hit1": hit1, "hit3": hit3, "ids": ids,
        })

    n = len(cases)
    return {
        "n": n,
        "top1_acc": round(top1 / n, 4),
        "top3_acc": round(top3 / n, 4),
        "prefill_rate": round(prefill / n, 4),
        "human_select_rate": round(human_select / n, 4),
        "no_match_rate": round(no_match / n, 4),
        "wrong_prefill_rate": round(wrong_prefill / n, 4),
        "thresholds": {"high": high_th, "mid": mid_th},
        "cases": rescored,
    }


def grid_search_thresholds(cases: Optional[list[MappingCase]] = None) -> list[dict[str, Any]]:
    """网格搜索高/中阈值：在可接受的错误预填率下尽量提高预填比例。"""
    agent = ClauseMapAgent()
    results = []
    for hi, mid in product([0.70, 0.75, 0.80, 0.85], [0.45, 0.50, 0.55, 0.60]):
        if hi <= mid:
            continue
        m = evaluate_mapping(agent, cases, hi, mid)
        m.pop("cases", None)
        results.append(m)
    return results


# --------------------------------------------------------------------------
# 2) 风险分级评测集
# --------------------------------------------------------------------------
@dataclass
class GradeCase:
    raw_text: str
    gold_level: RiskLevel
    clause_id: str = ""
    is_ccp: bool = False
    context: dict[str, Any] = field(default_factory=dict)
    note: str = ""


GRADE_CASES: list[GradeCase] = [
    GradeCase("冷库温度记录缺失，未记录两段数据", RiskLevel.B, "CCP-01", True),
    GradeCase("冷库中心温度超出关键限值已接触产品", RiskLevel.A, "CCP-01", True,
              {"monitoring_out_of_cl": True}),
    GradeCase("金检探测到金属异物，已停线隔离", RiskLevel.A, "CCP-03", True),
    GradeCase("标签过敏原缺失，存在交叉接触风险", RiskLevel.A, "AL-2.5", False),
    GradeCase("环境采样检出致病菌阳性", RiskLevel.A, "SSOP-4.3", False),
    GradeCase("刀具破损碎片可能进入产品流", RiskLevel.A, "SSOP-5.6", False),
    GradeCase("冷链断链，温度超出允许范围", RiskLevel.A, "CCP-01", True),
    GradeCase("CIP消毒浓度不足，接触面有残留", RiskLevel.A, "SSOP-6.2", False),
    GradeCase("同区域同类问题已出现3例，属批量偏差", RiskLevel.A, "SSOP-4.3", False,
              {"recent_same_issue_count": 3}),
    GradeCase("已发货批次发现问题，需追溯", RiskLevel.A, "SSOP-9.1", False),
    GradeCase("地面积水，排水口不洁", RiskLevel.B, "SSOP-4.3"),
    GradeCase("未戴发网进入灌装间", RiskLevel.B, "SSOP-2.1"),
    GradeCase("标签批次号错误但可隔离追溯", RiskLevel.B, "AL-2.1"),
    GradeCase("捕鼠器异常未修复", RiskLevel.B, "SSOP-8.4"),
    GradeCase("发酵罐参数漂移未超CL", RiskLevel.B, "GMP-7.2"),
    GradeCase("记录缺失需补证", RiskLevel.B, "SSOP-9.1"),
    GradeCase("该问题30天内再次出现", RiskLevel.B, "SSOP-4.3", False, {"same_issue_in_30d": 1}),
    GradeCase("供应商到货包装变形", RiskLevel.B, "SUP-3.4"),
    GradeCase("记录填写笔误当场更正", RiskLevel.C, "SSOP-9.1"),
    GradeCase("工具容器摆放不规范", RiskLevel.C, "SSOP-5.6"),
    GradeCase("单个包装轻微变形可立即更换", RiskLevel.C, "SSOP-5.6"),
    GradeCase("车间照明灯坏了一个", RiskLevel.C, "SSOP-11.2"),
]


def evaluate_grading(
    agent: Optional[RiskGradeAgent] = None, cases: Optional[list[GradeCase]] = None
) -> dict[str, Any]:
    agent = agent or RiskGradeAgent()
    cases = cases or GRADE_CASES
    a_tp = a_fp = a_fn = agree = need_human = llm_branch = 0
    over_up = 0
    rows = []

    for c in cases:
        res = agent.grade_issue("INS-EVAL", c.raw_text, c.clause_id, "", c.is_ccp,
                                context=c.context)
        pred = res.level
        gold = c.gold_level
        if pred == gold:
            agree += 1
        if gold == RiskLevel.A:
            if pred == RiskLevel.A:
                a_tp += 1
            else:
                a_fn += 1
        elif pred == RiskLevel.A:
            a_fp += 1
        if pred == RiskLevel.A and gold != RiskLevel.A:
            over_up += 1
        need_human += int(res.need_human)
        llm_branch += int(res.source == "llm_suggestion")
        rows.append({"text": c.raw_text, "gold": gold.value, "pred": pred.value,
                     "source": res.source, "hits": [h.rule_id for h in res.hits],
                     "need_human": res.need_human})

    n = len(cases)
    return {
        "n": n,
        "A_recall": round(a_tp / (a_tp + a_fn), 4) if (a_tp + a_fn) else None,
        "A_precision": round(a_tp / (a_tp + a_fp), 4) if (a_tp + a_fp) else None,
        "overall_agreement": round(agree / n, 4),
        "human_confirm_rate": round(need_human / n, 4),
        "llm_branch_rate": round(llm_branch / n, 4),
        "over_escalation_rate": round(over_up / n, 4),
        "cases": rows,
    }


# --------------------------------------------------------------------------
# 3) 证据验证评测集：四类样本，每类 ≥30 条（按类型批量生成）
# --------------------------------------------------------------------------
@dataclass
class VerifyCase:
    kind: str  # genuine | fake | reused | incomplete
    expected: VerifyRoute  # 期望分流
    reject_expected: bool


def build_verify_cases(n_each: int = 30) -> list[VerifyCase]:
    cases: list[VerifyCase] = []
    for _ in range(n_each):
        cases.append(VerifyCase("genuine", VerifyRoute.AUTO_PASS, False))
        cases.append(VerifyCase("fake", VerifyRoute.REJECT, True))
        cases.append(VerifyCase("reused", VerifyRoute.REJECT, True))
        cases.append(VerifyCase("incomplete", VerifyRoute.REJECT, True))
    return cases


# 汇总函数
def summarize_verify(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    rejects = [r for r in results if r["route"] == VerifyRoute.REJECT.value]
    tp = sum(1 for r in rejects if r["reject_expected"])
    fp = len(rejects) - tp
    actual_bad = [r for r in results if r["reject_expected"]]
    fn = sum(1 for r in actual_bad if r["route"] != VerifyRoute.REJECT.value)
    genuine = [r for r in results if r["kind"] == "genuine"]
    false_reject = sum(1 for r in genuine if r["route"] == VerifyRoute.REJECT.value)
    uncertain = sum(1 for r in results if r["route"] in
                    (VerifyRoute.HUMAN_REVIEW.value, VerifyRoute.HUMAN_ONSITE.value))
    auto = sum(1 for r in results if r["route"] == VerifyRoute.AUTO_PASS.value)
    return {
        "n": n,
        "reject_precision": round(tp / len(rejects), 4) if rejects else None,
        "reject_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "false_reject_rate": round(false_reject / len(genuine), 4) if genuine else None,
        "uncertain_rate": round(uncertain / n, 4) if n else None,
        "auto_pass_rate": round(auto / n, 4) if n else None,
        "false_rejects": fp,
        "missed": fn,
    }


def write_report(path: str, data: dict[str, Any]) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    return path
