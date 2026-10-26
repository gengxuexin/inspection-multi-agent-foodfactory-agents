"""条款级混合检索流水线。

严格对应《Agent1》第 2 节主流程：
    元数据硬过滤 → 向量+BM25双路召回 → RRF融合 → rerank → 多信号置信度 → 三档分流

说明：本模块不依赖任何外部向量数据库。向量用可复现的 hashing-trigram 投影实现，
目的是让整条流水线在离线环境可运行、可测试；替换为 ChromaDB/FAISS 时
只需替换 `vector_search` 内部实现，其余逻辑（过滤/融合/重排/置信度）完全不变。
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional

from app.core.config import settings
from app.rag.knowledge import Clause, ClauseLibrary

_CJK = re.compile(r"[一-鿿]")


def tokenize(text: str) -> list[str]:
    """中文按字 + 双字组切分，英文数字按词切分。"""
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    cjk = _CJK.findall(text)
    tokens += cjk
    tokens += [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return tokens


def embed(text: str, dim: int = 512) -> list[float]:
    """可复现的 hashing 投影向量（L2 归一化）。"""
    vec = [0.0] * dim
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
        vec[(h // dim) % dim] += 0.5  # 二次散列，降低碰撞影响
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class RetrievedClause:
    clause: Clause
    vec_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0
    vec_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    rank_meta: dict = field(default_factory=dict)


class BM25Index:
    """标准 BM25（k1=1.5, b=0.75）。"""

    def __init__(self, docs: dict[str, list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = docs
        self.doc_len = {k: len(v) for k, v in docs.items()}
        self.avgdl = (sum(self.doc_len.values()) / len(self.doc_len)) if self.doc_len else 0.0
        self.tf = {k: Counter(v) for k, v in docs.items()}
        n = len(docs)
        self.idf: dict[str, float] = {}
        df = Counter()
        for v in docs.values():
            df.update(set(v))
        for term, freq in df.items():
            self.idf[term] = math.log(1 + (n - freq + 0.5) / (freq + 0.5))

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        tf, dl = self.tf[doc_id], self.doc_len[doc_id]
        s = 0.0
        for t in query_tokens:
            f = tf.get(t, 0)
            if not f:
                continue
            s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            )
        return s


class ClauseRetriever:
    """条款检索器。线程安全：索引在构造后只读。"""

    def __init__(self, library: Optional[ClauseLibrary] = None) -> None:
        self.library = library or ClauseLibrary.load()
        self._build()

    def _build(self) -> None:
        clauses = self.library.latest_by_id()
        self.clauses = clauses
        self.by_id = {c.clause_id: c for c in clauses}
        self.doc_tokens = {
            c.clause_id: tokenize(
                f"{c.text} {c.summary} {c.question_category} {' '.join(c.zone)} "
                f"{' '.join(c.process)} {c.hazard_type}"
            )
            for c in clauses
        }
        self.bm25 = BM25Index(self.doc_tokens)
        self.vectors = {c.clause_id: embed(c.text) for c in clauses}
        dim = settings.embedding_dim
        self.query_cache: dict[str, list[float]] = {}

    # -- Step 2 元数据硬过滤 --------------------------------------------
    def meta_filter(
        self,
        zone: str = "",
        process: str = "",
        hazard: str = "",
        min_candidates: Optional[int] = None,
    ) -> tuple[list[Clause], list[str]]:
        """先按 zone→process→hazard 逐层放宽，并记录放宽了哪一层。"""
        min_candidates = min_candidates or settings.clause_map.min_candidates

        def apply(pool: list[Clause], z: str, p: str, h: str) -> list[Clause]:
            out = pool
            if z:
                out = [c for c in out if z in c.zone]
            if p:
                out = [c for c in out if p in c.process]
            if h:
                out = [c for c in out if c.hazard_type == h]
            return out

        relaxed: list[str] = []
        layers = [("zone", zone), ("process", process), ("hazard", hazard)]
        pool = self.clauses
        for name, val in layers:
            if not val:
                continue
            cand = apply(pool, zone, process, hazard)
            if len(cand) >= min_candidates:
                pool = cand
                continue
            # 放宽当前层
            if name == "zone":
                cand = apply(pool, "", process, hazard)
            elif name == "process":
                cand = apply(pool, zone, "", hazard)
            else:
                cand = apply(pool, zone, process, "")
            if len(cand) > len(pool):
                relaxed.append(name)
                pool = cand
            if len(cand) < min_candidates and name == "hazard":
                if len(pool) < min_candidates:
                    relaxed.append("hazard")
                    pool = self.clauses
        return pool, relaxed

    # -- Step 3 双路召回 -------------------------------------------------
    def _qv(self, text: str) -> list[float]:
        key = text[:200]
        if key not in self.query_cache:
            self.query_cache[key] = embed(text)
        return self.query_cache[key]

    def vector_search(self, query: str, pool: Iterable[Clause], k: int) -> list[RetrievedClause]:
        qv = self._qv(query)
        scored = [
            RetrievedClause(clause=c, vec_score=round(cosine(qv, self.vectors[c.clause_id]), 4))
            for c in pool
        ]
        scored.sort(key=lambda r: -r.vec_score)
        top = scored[:k]
        for i, r in enumerate(top, 1):
            r.vec_rank = i
        return top

    def bm25_search(self, query: str, pool: Iterable[Clause], k: int) -> list[RetrievedClause]:
        q_tokens = tokenize(query)
        scored = [
            RetrievedClause(clause=c, bm25_score=round(self.bm25.score(q_tokens, c.clause_id), 4))
            for c in pool
        ]
        scored = [r for r in scored if r.bm25_score > 0]
        scored.sort(key=lambda r: -r.bm25_score)
        top = scored[:k]
        for i, r in enumerate(top, 1):
            r.bm25_rank = i
        return top

    # -- Step 4 RRF 融合 -------------------------------------------------
    @staticmethod
    def rrf_fuse(lists: list[list[RetrievedClause]], k: int = 60, top_k: int = 10) -> list[RetrievedClause]:
        """Reciprocal Rank Fusion：score = Σ 1/(k + rank)。"""
        merged: dict[str, RetrievedClause] = {}
        for lst in lists:
            for r in lst:
                hit = merged.get(r.clause.clause_id)
                if hit is None:
                    hit = r
                    merged[r.clause.clause_id] = hit
                else:
                    hit.vec_score = max(hit.vec_score, r.vec_score)
                    hit.bm25_score = max(hit.bm25_score, r.bm25_score)
                    hit.vec_rank = hit.vec_rank or r.vec_rank
                    hit.bm25_rank = hit.bm25_rank or r.bm25_rank
        for hit in merged.values():
            s = 0.0
            if hit.vec_rank:
                s += 1.0 / (k + hit.vec_rank)
            if hit.bm25_rank:
                s += 1.0 / (k + hit.bm25_rank)
            hit.rrf_score = round(s, 6)
        fused = sorted(merged.values(), key=lambda r: -r.rrf_score)
        return fused[:top_k]

    # -- Step 5 rerank ---------------------------------------------------
    def rerank(self, query: str, docs: list[RetrievedClause], top_n: int) -> list[RetrievedClause]:
        """轻量交叉特征重排：查询-条款词面重合 + 类别匹配 + 危害匹配 + RRF 先验。

        替换为 cross-encoder 时只需覆盖本方法，签名保持一致。
        """
        q_tokens = set(tokenize(query))
        for r in docs:
            c = r.clause
            c_tokens = set(self.doc_tokens[c.clause_id])
            overlap = len(q_tokens & c_tokens) / (len(q_tokens) or 1)
            cat_hit = 1.0 if any(k in query for k in tokenize(c.question_category)) else 0.0
            hz_hit = 1.0 if c.hazard_type and c.hazard_type in query else 0.0
            zone_hit = 1.0 if c.zone and any(z in query for z in c.zone) else 0.0
            proc_hit = 1.0 if c.process and any(p in query for p in c.process) else 0.0
            r.rerank_score = round(
                0.45 * overlap + 0.2 * cat_hit + 0.15 * hz_hit + 0.1 * zone_hit + 0.1 * proc_hit,
                4,
            )
            r.rank_meta = {
                "overlap": round(overlap, 4),
                "cat_hit": cat_hit,
                "hazard_hit": hz_hit,
                "zone_hit": zone_hit,
                "process_hit": proc_hit,
            }
        return sorted(docs, key=lambda r: -r.rerank_score)[:top_n]

    # -- Step 6 置信度合成 ----------------------------------------------
    @staticmethod
    def confidence(
        r: RetrievedClause,
        query: str,
        zone: str = "",
        process: str = "",
        hazard: str = "",
        percentiles: Optional[dict[str, float]] = None,
    ) -> float:
        """多信号合成：rerank归一化 + 双路一致性 + 元数据匹配。

        不用单一向量分数——单分数无法区分"检索偏了"和"条款库缺项"。
        """
        base = max(0.0, min(1.0, r.rerank_score))
        p70 = (percentiles or {}).get("p70", 0.5)

        # 双路一致性：向量和BM25都靠前 → 语义与关键词同时命中
        if r.vec_rank and r.bm25_rank and r.vec_rank <= p70 and r.bm25_rank <= p70:
            base += 0.10
        # 元数据匹配
        c = r.clause
        if zone and zone in c.zone:
            base += 0.05
        if process and process in c.process:
            base += 0.05
        if hazard and hazard == c.hazard_type:
            base += 0.05
        # RRF 先验做轻微兜底
        base += min(0.03, r.rrf_score * 0.5)
        return round(min(base, 1.0), 4)

    @staticmethod
    def percentiles(docs: list[RetrievedClause]) -> dict[str, float]:
        if not docs:
            return {"p70": 0.5}
        ranks = [r.vec_rank for r in docs if r.vec_rank] + [
            r.bm25_rank for r in docs if r.bm25_rank
        ]
        if not ranks:
            return {"p70": 0.5}
        ranks.sort()
        return {"p70": max(1, int(len(ranks) * 0.7))}

    @staticmethod
    def explain(r: RetrievedClause) -> str:
        m = r.rank_meta
        parts = []
        if m.get("cat_hit"):
            parts.append("问题类别匹配")
        if m.get("hazard_hit"):
            parts.append(f"危害类型一致({r.clause.hazard_type})")
        if m.get("zone_hit"):
            parts.append("适用区域一致")
        if m.get("process_hit"):
            parts.append("适用工序一致")
        if r.vec_rank and r.bm25_rank:
            parts.append(f"双路召回一致(向量#{r.vec_rank}/BM25#{r.bm25_rank})")
        elif r.vec_rank:
            parts.append(f"向量召回#{r.vec_rank}")
        return "，".join(parts) or "词面部分重合"
