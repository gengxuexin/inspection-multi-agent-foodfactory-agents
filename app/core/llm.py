"""LLM 适配层。

设计原则（贯穿三个 Agent）：
- LLM 只负责**抽取、检索理解、语义核验、理由生成**；
- **不做**分级判定、闸门放行、状态转换；
- 输出必须结构化，不合规则降级为规则分支 + 人工复核，**不自动放行**。

本模块提供 mock provider（默认，无需 API key 即可跑通全流程），
以及 openai-compatible provider（配置 FQA_LLM_PROVIDER=openai + FQA_LLM_API_KEY 后启用）。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from app.core.config import settings
from app.core.schemas import QueryElements, VlmJudgement

logger = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """LLM 不可用。调用方必须降级到规则分支 + 人工复核。"""


def _chat(prompt: str, system: str = "", max_tokens: int = 800) -> str:
    """调用 LLM。mock provider 走本地启发式实现。"""
    if not settings.llm_enabled:
        raise LLMUnavailable("LLM 未配置，走规则分支")
    try:
        import httpx  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise LLMUnavailable("httpx 未安装") from exc

    payload = {
        "model": settings.llm_model,
        "messages": (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}]
        ),
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {settings.llm_api_key}"}
    try:
        r = httpx.post(
            f"{settings.llm_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as exc:  # pragma: no cover - 网络异常统一降级
        logger.warning("LLM 调用失败，降级：%s", exc)
        raise LLMUnavailable(str(exc)) from exc


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("LLM 输出不含 JSON 对象")
    return json.loads(m.group(0))


# --------------------------------------------------------------------------
# 查询要素抽取（条款映射 Agent Step 1）
# --------------------------------------------------------------------------
_ZONE_WORDS = ["前处理间", "加工间", "灌装间", "内包材库", "外包材库", "冷库", "发酵间",
               "烘焙间", "罐装区", "金检间", "仓库", "更衣间", "洗手间"]
_PROCESS_WORDS = ["清洗", "切割", "腌制", "腌制发酵", "烘焙", "灌装", "封口", "金检",
                 "发酵", "冷却", "入库", "转运"]
_HAZARD_WORDS = {
    "生物": ["细菌", "菌", "霉", "虫害", "鼠", "污染", "微生物", "腐败", "发霉"],
    "化学": ["清洁剂", "残留", "消毒剂", "浓度", "化学"],
    "物理": ["异物", "金属", "玻璃", "碎片", "破损", "硬塑料"],
    "过敏原": ["过敏原", "交叉接触", "标签", "麸质", "花生", "乳制品"],
    "卫生": ["积水", "清洁", "灰尘", "台面", "地面积水", "排水"],
    "记录": ["记录", "台账", "签字", "漏填", "缺失"],
}


def extract_query_elements(issue_text: str, draft: Any = None) -> QueryElements:
    """从口语化巡检描述抽取查询要素。抽取失败字段留空，不阻断流程。"""
    text = issue_text or ""

    # 规则先行的启发式抽取：mock 环境下也能稳定复现，且与 LLM 抽取走同一出口
    zone = next((z for z in _ZONE_WORDS if z in text), "")
    process = next((p for p in _PROCESS_WORDS if p in text), "")
    hazard = ""
    for hz, words in _HAZARD_WORDS.items():
        if any(w in text for w in words):
            hazard = hz
            break
    phenomenon = re.sub(r"^(发现|存在|有|发现存在)+", "", text).strip() or text
    keywords = sorted({w for w in re.split(r"[\s，,。、；;：:]+", text) if 2 <= len(w) <= 12})

    if settings.llm_enabled:
        schema_hint = (
            '{"phenomenon":"","object":"","zone":"","process":"",'
            '"deviation_type":"","hazard_guess":"","keywords":[]}'
        )
        prompt = (
            f"把下面这句食品工厂巡检口语描述抽取为结构化查询要素，只输出 JSON，"
            f"字段严格如下：{schema_hint}\n\n描述：{text}"
        )
        try:
            data = _parse_json(_chat(prompt, system="你是食品巡检信息抽取器，只输出JSON。"))
            return QueryElements(
                phenomenon=data.get("phenomenon") or phenomenon,
                object=data.get("object", ""),
                zone=data.get("zone") or zone,
                process=data.get("process") or process,
                deviation_type=data.get("deviation_type", ""),
                hazard_guess=data.get("hazard_guess") or hazard,
                keywords=data.get("keywords") or keywords,
                extraction_ok=True,
            )
        except (LLMUnavailable, ValueError) as exc:
            logger.warning("要素抽取降级为启发式：%s", exc)

    return QueryElements(
        phenomenon=phenomenon,
        object="",
        zone=zone,
        process=process,
        deviation_type="",
        hazard_guess=hazard,
        keywords=keywords[:12],
        extraction_ok=bool(zone or process or keywords),
    )


# --------------------------------------------------------------------------
# 分级建议（LLM 只能建议，不能单独定 A）
# --------------------------------------------------------------------------
_LEVEL_KEYWORDS = {
    "A": ["致病菌", "异物", "过敏原", "断链", "超限", "检出", "阳性", "已发货", "已售"],
    "B": ["缺失", "不彻底", "积水", "漂移", "不规范", "未戴", "异常", "复发", "包装变形"],
    "C": ["笔误", "摆放", "变形", "照明", "门窗", "轻微", "定置"],
}


def suggest_level(issue_text: str, clause_text: str, clause_id: str) -> tuple[str, str, float]:
    """LLM 边界建议分支。返回 (level, reason, confidence)。

    调用方**必须**执行：若返回 A，强制 need_human=True——LLM 不能单独定 A。
    """
    fallback_level = "C"
    for lv in ("A", "B", "C"):
        if any(k in issue_text for k in _LEVEL_KEYWORDS[lv]):
            fallback_level = lv
            break
    fallback_reason = (
        f"规则表未命中。现象描述中出现的风险信号指向 {fallback_level} 级，"
        f"需结合条款原文「{clause_id}」人工判定。"
    )

    if settings.llm_enabled:
        prompt = (
            "你是食品安全风险分级助手。规则表未命中，需要你给出**建议等级**。"
            '只输出 JSON：{"level":"A|B|C","reason":"","confidence":0.0,"clause_ref":""}。\n'
            "注意：这是建议，不是最终判定，系统会强制人工确认。\n\n"
            f"现象：{issue_text}\n条款原文（{clause_id}）：{clause_text[:1200]}"
        )
        try:
            data = _parse_json(_chat(prompt, system="你只输出JSON，不要多余文字。"))
            lv = str(data.get("level", "C")).upper()
            if lv not in ("A", "B", "C"):
                lv = "C"
            return lv, str(data.get("reason", "")), float(data.get("confidence", 0.5))
        except (LLMUnavailable, ValueError) as exc:
            logger.warning("分级建议降级为启发式：%s", exc)

    return fallback_level, fallback_reason, 0.45


# --------------------------------------------------------------------------
# VLM 语义核验（证据验证 Agent 校验4）
# --------------------------------------------------------------------------
def vlm_verify(
    image_path: str,
    requirement: str,
    category: str = "",
) -> tuple[VlmJudgement, str]:
    """判断证据内容是否真的对应整改项。返回 (judgement, provider)。

    VLM 只做**辅助判断**：只对明确不符自动打回，存疑一律转人工。
    VLM 不可用时降级为"规则校验 + 人工复核"，不自动通过。
    """
    mock = _mock_vlm(image_path, requirement)
    if not settings.llm_enabled:
        return mock, "rules_only"

    prompt = (
        "你是整改证据核验助手。判断证据图片内容是否与整改要求一致。\n"
        '只输出 JSON：{"verdict":"match|mismatch|uncertain","confidence":0.0,'
        '"reason":"","evidence_refs":[]}。\n'
        f"整改要求：{requirement}\n问题类别：{category}\n图片：{image_path}"
    )
    try:
        data = _parse_json(_chat(prompt, system="你只输出JSON。", max_tokens=400))
        j = VlmJudgement(
            verdict=data.get("verdict", "uncertain"),
            confidence=float(data.get("confidence", 0.0)),
            reason=str(data.get("reason", "")),
            evidence_refs=list(data.get("evidence_refs", [])),
        )
        return j, "vlm"
    except (LLMUnavailable, ValueError) as exc:
        logger.warning("VLM 不可用，降级规则校验：%s", exc)
        return VlmJudgement(
            verdict="uncertain", confidence=0.0, reason=f"VLM不可用：{exc}"
        ), "vlm_unavailable"


_NEG_WORDS = ["积水", "脏", "污", "破损", "锈", "有杂物", "未清理"]
_POS_WORDS = ["干燥", "整洁", "干净", "已修复", "正常", "清晰", "完好"]


def _mock_vlm(image_path: str, requirement: str) -> VlmJudgement:
    """离线语义核验：结合文件名标注做可解释的启发式判定。

    约定：测试数据可在文件名中带 _ok / _bad 标注，本函数据此产出确定性结论，
    真实部署时该函数被真实 VLM 替换。
    """
    name = (image_path or "").lower()
    if any(k in name for k in ("_bad", "_mismatch", "_wrong")):
        return VlmJudgement(
            verdict="mismatch",
            confidence=0.92,
            reason=f"证据内容与整改要求「{requirement[:30]}」明显不符",
            evidence_refs=["画面内容与整改项不匹配"],
        )
    if any(k in name for k in ("_ok", "_match", "_fixed")):
        return VlmJudgement(
            verdict="match",
            confidence=0.88,
            reason=f"证据内容与整改要求「{requirement[:30]}」一致",
            evidence_refs=["画面显示整改项已消除"],
        )
    if any(w in name for w in _NEG_WORDS):
        return VlmJudgement(verdict="mismatch", confidence=0.8, reason="画面存在未消除的缺陷")
    if any(w in name for w in _POS_WORDS):
        return VlmJudgement(verdict="match", confidence=0.8, reason="画面显示状态正常")
    return VlmJudgement(
        verdict="uncertain",
        confidence=0.4,
        reason="画面无法可靠判断整改项是否完成",
    )
