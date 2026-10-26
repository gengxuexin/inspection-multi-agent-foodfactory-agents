"""条款映射 Agent（Agent 2）。

流程对应设计稿：要素抽取 → 元数据硬过滤 → 向量+BM25双路召回 → RRF融合
→ rerank → 多信号置信度 → 三档分流（prefill / human_select / no_reliable_match）。

硬约束：低置信度不硬猜；命中 is_ccp=True 的条款，**等级确认强制人工**。
"""
from __future__ import annotations

from typing import Optional

from app.core.config import settings
from app.core.llm import extract_query_elements
from app.core.schemas import (
    ClauseCandidate,
    ClauseMapResult,
    MapMode,
    QueryElements,
)
from app.rag.knowledge import ClauseLibrary
from app.rag.retriever import ClauseRetriever, RetrievedClause

_retriever: Optional[ClauseRetriever] = None


def get_retriever() -> ClauseRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ClauseRetriever(ClauseLibrary.load())
    return _retriever


class ClauseMapAgent:
    """条款映射 Agent：只输出结构化候选与分流结果，不做最终条款确认。"""

    def __init__(self, retriever: Optional[ClauseRetriever] = None) -> None:
        self.retriever = retriever or get_retriever()
        self.cfg = settings.clause_map

    def map_clause(self, issue_id: str, raw_text: str, top_k: Optional[int] = None) -> ClauseMapResult:
        top_k = top_k or self.cfg.recall_k

        # Step 1 查询要素抽取（失败字段留空，不阻断）
        q: QueryElements = extract_query_elements(raw_text)

        # Step 2 元数据硬过滤 + 逐层放宽
        pool, relaxed = self.retriever.meta_filter(
            zone=q.zone, process=q.process, hazard=q.hazard_guess
        )
        # 抽取失败 → 直接全库双路召回，标注要素缺失
        if not (q.zone or q.process or q.keywords):
            pool = self.retriever.clauses
            relaxed = ["elements_missing"]

        # Step 3 双路召回
        sem_query = f"{q.phenomenon} {' '.join(q.keywords)}".strip() or raw_text
        kw_query = " ".join(q.keywords) or sem_query
        vec_hits = self.retriever.vector_search(sem_query, pool, top_k)
        bm25_hits = self.retriever.bm25_search(kw_query, pool, top_k)

        # Step 4 RRF 融合
        fused = self.retriever.rrf_fuse([vec_hits, bm25_hits], k=self.cfg.rrf_k, top_k=top_k)

        # Step 5 rerank
        ranked: list[RetrievedClause] = self.retriever.rerank(raw_text, fused, self.cfg.rerank_top_n)

        # Step 6 置信度合成 + 分流
        pct = self.retriever.percentiles(fused)
        candidates: list[ClauseCandidate] = []
        for r in ranked:
            conf = self.retriever.confidence(r, raw_text, q.zone, q.process, q.hazard_guess, pct)
            candidates.append(
                ClauseCandidate(
                    clause_id=r.clause.clause_id,
                    source=r.clause.source,
                    version=r.clause.version,
                    effective_date=r.clause.effective_date,
                    text_snippet=r.clause.text,
                    confidence=conf,
                    reason=self.retriever.explain(r),
                    zone=r.clause.zone,
                    process=r.clause.process,
                    is_ccp=r.clause.is_ccp,
                    hazard_type=r.clause.hazard_type,
                    retrieval_trace={
                        "vec_rank": r.vec_rank,
                        "bm25_rank": r.bm25_rank,
                        "vec_score": r.vec_score,
                        "bm25_score": r.bm25_score,
                        "rrf_score": r.rrf_score,
                        "rerank_score": r.rerank_score,
                        "rank_meta": r.rank_meta,
                    },
                )
            )

        mode, needs_human, human_reason = self._route(candidates)
        return ClauseMapResult(
            issue_id=issue_id,
            mode=mode,
            candidates=candidates[:3],
            needs_human=needs_human,
            needs_human_reason=human_reason,
            relaxed_filter_layers=relaxed,
            query_elements=q,
            retrieval_trace={
                "pool_size": len(pool),
                "vec_hits": len(vec_hits),
                "bm25_hits": len(bm25_hits),
                "fused": len(fused),
                "reranked": len(ranked),
                "thresholds": {"high": self.cfg.high_threshold, "mid": self.cfg.mid_threshold},
                "extraction_ok": q.extraction_ok,
            },
            agent_version=self.cfg.agent_version,
        )

    # -- 三档分流 -------------------------------------------------------
    def _route(self, candidates: list[ClauseCandidate]) -> tuple[MapMode, bool, str]:
        if not candidates:
            return (
                MapMode.NO_RELIABLE_MATCH,
                True,
                "未召回任何条款，请人工选择或标记新增条款需求",
            )
        top = candidates[0]
        # 命中 CCP 条款：不额外加分，但强制人工确认
        if top.confidence >= self.cfg.high_threshold:
            mode = MapMode.PREFILL
            reason = f"置信度 {top.confidence} ≥ 高阈值 {self.cfg.high_threshold}，已预填但允许人工改"
        elif top.confidence >= self.cfg.mid_threshold:
            mode = MapMode.HUMAN_SELECT
            reason = f"置信度 {top.confidence} 处于中区间，展示 Top-{len(candidates)} 且必须人工选"
        else:
            mode = MapMode.NO_RELIABLE_MATCH
            reason = f"最高置信度仅 {top.confidence}，低于中阈值 {self.cfg.mid_threshold}，不预填"

        if top.is_ccp:
            needs_human = True
            reason += f"；命中 CCP 条款 {top.clause_id}，等级确认强制人工"
        else:
            needs_human = mode != MapMode.PREFILL
        return mode, needs_human, reason

    # -- 人工确认回流（样本库） ----------------------------------------
    @staticmethod
    def build_sample(
        result: ClauseMapResult,
        raw_text: str,
        selected_clause_id: str,
        confirmed_by: str,
    ) -> dict:
        """人工选择结果回流样本库。**只入库，不在线自动训练。**"""
        top1 = result.candidates[0].clause_id if result.candidates else None
        return {
            "issue_id": result.issue_id,
            "raw_text": raw_text,
            "query_elements": result.query_elements.model_dump() if result.query_elements else {},
            "candidates": [c.model_dump() for c in result.candidates],
            "selected_clause_id": selected_clause_id,
            "original_top1": top1,
            "top1_was_correct": top1 == selected_clause_id,
            "mode": result.mode.value,
            "confirmed_by": confirmed_by,
        }
