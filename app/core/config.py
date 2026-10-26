"""全局配置与关键参数清单。

设计文档《最小系统规格说明》第六节要求业务方拍板的参数集中在此，
默认值取自《Agent1/2/3》三份设计稿的"关键参数清单"，全部可用环境变量覆盖。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# app/core/config.py → 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.getenv("FQA_DATA_DIR", BASE_DIR / "data"))
KNOWLEDGE_DIR = Path(os.getenv("FQA_KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
EVIDENCE_DIR = DATA_DIR / "evidence"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = Path(os.getenv("FQA_DB_PATH", DATA_DIR / "fqa.db"))

for _p in (DATA_DIR, KNOWLEDGE_DIR, EVIDENCE_DIR, UPLOAD_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class ClauseMapConfig:
    """条款映射 Agent 参数。

    默认 high/mid 取自 `scripts/run_eval.py` 的阈值网格搜索结果
    （在错误预填率 3.7% 不变的前提下，预填率由 37% 提升到 52%）。
    接入真实向量库与 cross-encoder 后必须重跑网格搜索。
    """

    high_threshold: float = _f("FQA_MAP_HIGH_TH", 0.70)
    mid_threshold: float = _f("FQA_MAP_MID_TH", 0.45)
    recall_k: int = _i("FQA_MAP_RECALL_K", 10)
    rerank_top_n: int = _i("FQA_MAP_RERANK_N", 5)
    rrf_k: int = _i("FQA_MAP_RRF_K", 60)
    min_candidates: int = _i("FQA_MAP_MIN_CANDIDATES", 20)
    agent_version: str = "clause-map-v0.3"


@dataclass(frozen=True)
class GradeConfig:
    """风险分级 Agent 参数。"""

    llm_confidence_threshold: float = _f("FQA_GRADE_LLM_CONF", 0.70)
    agent_version: str = "risk-grade-v0.3"


@dataclass(frozen=True)
class EvidenceConfig:
    """证据验证 Agent 参数。"""

    image_reject_threshold: float = _f("FQA_EV_IMG_REJECT", 0.95)
    image_uncertain_low: float = _f("FQA_EV_IMG_UNC_LO", 0.85)
    doc_reject_threshold: float = _f("FQA_EV_DOC_REJECT", 0.90)
    vlm_pass_threshold: float = _f("FQA_EV_VLM_PASS", 0.80)
    consecutive_reject_escalate: int = _i("FQA_EV_REJECT_ESC", 2)
    consecutive_reject_notify: int = _i("FQA_EV_REJECT_NOTIFY", 3)
    agent_version: str = "evidence-verify-v0.3"


@dataclass(frozen=True)
class ClockConfig:
    """三级响应时钟（Agent5 跟踪升级）。"""

    # 等级 -> (首次响应小时, 整改完成工作日, 升级路径)
    first_response_hours: dict = field(
        default_factory=lambda: {"A": 0.5, "B": 2, "C": 24}
    )
    fix_deadline_hours: dict = field(
        default_factory=lambda: {"A": 24, "B": 72, "C": 168}
    )
    escalation_path: dict = field(
        default_factory=lambda: {
            "A": ["班组长", "车间主任", "厂长/质量总监"],
            "B": ["班组长", "车间主任", "质量经理"],
            "C": ["班组长", "车间主任"],
        }
    )
    human_task_timeout_hours: int = _i("FQA_HUMAN_TASK_TIMEOUT", 24)


@dataclass(frozen=True)
class RecheckConfig:
    """复查抽验比例（不预先通知现场）。"""

    sampling: dict = field(default_factory=lambda: {"A": 1.0, "B": 0.4, "C": 0.15})
    recurrence_window_days: int = _i("FQA_RECUR_WINDOW", 30)


@dataclass(frozen=True)
class Settings:
    clause_map: ClauseMapConfig = field(default_factory=ClauseMapConfig)
    grade: GradeConfig = field(default_factory=GradeConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    clock: ClockConfig = field(default_factory=ClockConfig)
    recheck: RecheckConfig = field(default_factory=RecheckConfig)
    llm_provider: str = os.getenv("FQA_LLM_PROVIDER", "mock")  # mock | openai
    llm_model: str = os.getenv("FQA_LLM_MODEL", "gpt-4o-mini")
    llm_api_key: str = os.getenv("FQA_LLM_API_KEY", "")
    llm_base_url: str = os.getenv("FQA_LLM_BASE_URL", "https://api.openai.com/v1")
    embedding_dim: int = _i("FQA_EMBED_DIM", 512)

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "mock" and bool(self.llm_api_key)


settings = Settings()
