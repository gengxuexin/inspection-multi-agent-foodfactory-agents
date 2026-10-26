"""条款级知识库。

对应《Agent1：条款映射离线评测》第 1 节：
- **一个条款一个 chunk**，不做固定长度切分；长条款额外生成摘要块，召回时原文块优先；
- 元数据 zone / process / is_ccp / hazard_type 是硬过滤依据，导入时校验；
- 保留版本与生效日期，检索时只取 effective_date <= 当前日期的最新版本。

本模块提供内置的最小 HACCP/SSOP/GMP 条款库（可直接运行），
同时支持从 JSON 文件导入企业自有条款库。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Optional

from app.core.config import KNOWLEDGE_DIR


@dataclass
class Clause:
    clause_id: str
    text: str
    source: str
    version: str
    effective_date: str
    zone: list[str] = field(default_factory=list)
    process: list[str] = field(default_factory=list)
    is_ccp: bool = False
    hazard_type: str = ""
    question_category: str = ""
    critical_limit: str = ""
    summary: str = ""

    def is_effective(self, today: Optional[str] = None) -> bool:
        today = today or date.today().isoformat()
        return self.effective_date <= today

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# 内置最小条款库（覆盖设计稿中出现的全部问题类型）
# --------------------------------------------------------------------------
BUILTIN_CLAUSES: list[Clause] = [
    Clause(
        clause_id="CCP-01",
        text="冷藏工序CCP：产品中心温度须控制在0~4℃，每2小时监测并记录；"
             "超出关键限值CL=4℃时须立即隔离产品、启动纠偏并记录。",
        source="HACCP计划", version="v3.2", effective_date="2025-01-01",
        zone=["冷库", "灌装间"], process=["冷却", "冷链"], is_ccp=True,
        hazard_type="生物", question_category="温度记录缺失/断链",
        critical_limit="中心温度≤4℃",
    ),
    Clause(
        clause_id="CCP-02",
        text="发酵工序CCP：pH须控制在3.8~4.2范围，每批监测；pH超出CL=4.2须隔离并评估。",
        source="HACCP计划", version="v3.2", effective_date="2025-01-01",
        zone=["发酵间"], process=["发酵"], is_ccp=True,
        hazard_type="生物", question_category="发酵pH超限",
        critical_limit="pH≤4.2",
    ),
    Clause(
        clause_id="CCP-03",
        text="金属探测CCP：Fe1.5mm/NonFe2.0mm/SUS2.5mm，每2小时用标准片校验，"
             "探测到异物须停线并隔离自上次校验以来全部产品。",
        source="HACCP计划", version="v3.2", effective_date="2025-01-01",
        zone=["金检间", "灌装间"], process=["金检"], is_ccp=True,
        hazard_type="物理", question_category="异物风险（工具/包装破损）",
        critical_limit="标准片可检出",
    ),
    Clause(
        clause_id="SSOP-4.3",
        text="生产结束后应彻底清洁地面及排水口，不得有积水；"
             "排水口每日班前检查并清洁，记录留存。",
        source="SSOP", version="v3.1", effective_date="2025-03-01",
        zone=["前处理间", "加工间", "灌装间"], process=["清洗"],
        hazard_type="卫生", question_category="清洁不彻底/积水",
    ),
    Clause(
        clause_id="SSOP-2.1",
        text="进入生产区域须穿戴清洁工作服、口罩、发网并洗手消毒；"
             "手套破损须立即更换，不得复用。",
        source="SSOP", version="v3.1", effective_date="2025-03-01",
        zone=["更衣间", "加工间", "灌装间"], process=["转运"],
        hazard_type="卫生", question_category="人员卫生/着装",
    ),
    Clause(
        clause_id="SSOP-6.2",
        text="食品接触面使用后须按CIP程序清洗、消毒并验证残留；"
             "消毒浓度或时间不足不得进入生产。",
        source="SSOP", version="v3.1", effective_date="2025-03-01",
        zone=["加工间", "灌装间"], process=["清洗"], is_ccp=False,
        hazard_type="生物", question_category="清洁消毒失效",
    ),
    Clause(
        clause_id="SSOP-8.4",
        text="捕鼠器、粘鼠板、风幕机须每周检查并记录，设施破损须24小时内修复。",
        source="SSOP", version="v3.1", effective_date="2025-03-01",
        zone=["仓库", "加工间", "内包材库"], process=["入库"],
        hazard_type="生物", question_category="虫害设施异常",
    ),
    Clause(
        clause_id="SSOP-9.1",
        text="生产记录须当场如实填写，不得补记、代签、空项；"
             "笔误须划改并签名，保持追溯链完整。",
        source="SSOP", version="v3.1", effective_date="2025-03-01",
        zone=["加工间", "灌装间"], process=["转运"],
        hazard_type="记录", question_category="记录缺失/笔误",
    ),
    Clause(
        clause_id="SSOP-5.6",
        text="工具、容器定置存放，不得混用；破损工具须立即报废并留存报废记录。",
        source="SSOP", version="v3.1", effective_date="2025-03-01",
        zone=["加工间", "前处理间"], process=["切割", "清洗"],
        hazard_type="物理", question_category="异物风险（工具/包装破损）",
    ),
    Clause(
        clause_id="AL-2.1",
        text="标签须与配方一致，完整标注配料与过敏原信息；"
             "含麸质、乳制品、花生等致敏成分须显著标示。标签错误须隔离追溯。",
        source="标签法规", version="v2.0", effective_date="2025-01-01",
        zone=["灌装间", "外包材库"], process=["灌装"],
        hazard_type="过敏原", question_category="标签标识错误",
    ),
    Clause(
        clause_id="GMP-7.2",
        text="生产设备参数须在设定值范围内；预防性维护不得逾期，"
             "校准证书须在有效期内并留存。",
        source="GMP", version="v4.0", effective_date="2025-01-01",
        zone=["加工间", "灌装间"], process=["发酵", "烘焙"],
        hazard_type="化学", question_category="设备参数漂移",
    ),
    Clause(
        clause_id="SUP-3.4",
        text="供应商到货须经验收合格方可入库；包装变形、标签轻微不符但未影响"
             "产品安全的，须记录并反馈供应商。",
        source="供应商验收标准", version="v1.8", effective_date="2025-02-01",
        zone=["仓库", "外包材库"], process=["入库"],
        hazard_type="化学", question_category="供应商到货偏差",
    ),
    Clause(
        clause_id="AL-2.5",
        text="过敏原管理：生产与换线须执行过敏原清洗；未申报过敏原或交叉接触风险"
             "须立即停线评估。",
        source="标签法规", version="v2.0", effective_date="2025-01-01",
        zone=["加工间", "灌装间"], process=["清洗", "灌装"],
        hazard_type="过敏原", question_category="过敏原交叉污染",
    ),
    Clause(
        clause_id="SSOP-11.2",
        text="车间照明、门窗、地面设施缺陷须及时报修；轻微缺陷不影响卫生与产品的，"
             "登记并纳入月度维护。",
        source="SSOP", version="v3.1", effective_date="2025-03-01",
        zone=["加工间", "更衣间", "仓库"], process=["转运"],
        hazard_type="卫生", question_category="设施轻微缺陷",
    ),
    # 过期版本样本：用于验证"只检索 effective_date <= 今天"的逻辑
    Clause(
        clause_id="SSOP-4.3",
        text="（旧版，已作废）生产结束后清洁地面，允许少量积水。",
        source="SSOP", version="v2.0", effective_date="2023-01-01",
        zone=["前处理间"], process=["清洗"],
        hazard_type="卫生", question_category="清洁不彻底/积水",
    ),
]


class ClauseLibrary:
    """条款库。导入时做元数据校验（zone/process/is_ccp 必须人工确认过）。"""

    def __init__(self, clauses: Optional[list[Clause]] = None) -> None:
        self._clauses: list[Clause] = list(clauses or BUILTIN_CLAUSES)

    # -- 加载 ------------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ClauseLibrary":
        path = path or (KNOWLEDGE_DIR / "clauses.json")
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            clauses = [Clause(**item) for item in data]
            return cls(clauses)
        return cls()

    def dump_to_json(self, path: Optional[Path] = None) -> Path:
        path = path or (KNOWLEDGE_DIR / "clauses.json")
        path.write_text(
            json.dumps([c.to_dict() for c in self._clauses], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    # -- 查询 ------------------------------------------------------------
    @property
    def clauses(self) -> list[Clause]:
        return list(self._clauses)

    def effective(self, today: Optional[str] = None) -> list[Clause]:
        return [c for c in self._clauses if c.is_effective(today)]

    def latest_by_id(self, today: Optional[str] = None) -> list[Clause]:
        """同一 clause_id 只保留最新生效版本。"""
        best: dict[str, Clause] = {}
        for c in self.effective(today):
            cur = best.get(c.clause_id)
            if cur is None or c.effective_date > cur.effective_date:
                best[c.clause_id] = c
        return list(best.values())

    def get(self, clause_id: str, today: Optional[str] = None) -> Optional[Clause]:
        cands = [c for c in self.effective(today) if c.clause_id == clause_id]
        if not cands:
            return None
        return max(cands, key=lambda c: c.effective_date)

    def validate(self) -> list[str]:
        """导入质量校验，返回问题清单。"""
        problems: list[str] = []
        seen: set[tuple[str, str]] = set()
        for c in self._clauses:
            if not c.zone:
                problems.append(f"{c.clause_id}@{c.version} 缺少 zone 元数据")
            if not c.process:
                problems.append(f"{c.clause_id}@{c.version} 缺少 process 元数据")
            if not c.hazard_type:
                problems.append(f"{c.clause_id}@{c.version} 缺少 hazard_type 元数据")
            key = (c.clause_id, c.version)
            if key in seen:
                problems.append(f"重复条款版本：{key}")
            seen.add(key)
        return problems
