"""证据指纹与感知哈希：拦截旧图/旧文档复用（校验3 的技术底座）。

- sha256：文件级完全一致检测；
- pHash：8x8 DCT 感知哈希，容忍压缩/轻微裁剪；
- 文本 SimHash：文档类证据的内容相似度。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.rag.retriever import tokenize


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    if not p.exists():
        return ""
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hamming(a: str, b: str) -> int:
    if not a or not b or len(a) != len(b):
        return 64
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def phash_file(path: str | Path) -> str:
    """8x8 DCT 感知哈希。依赖 Pillow；不可用时降级为文件哈希前缀。"""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        from PIL import Image  # type: ignore
        import math
    except ImportError:  # pragma: no cover
        return sha256_file(path)[:16]
    try:
        img = Image.open(p).convert("L").resize((32, 32))
    except Exception:
        return ""
    px = list(img.getdata())
    # 二维 DCT-II
    out: list[float] = []
    N = 32
    cos = [[math.cos((2 * x + 1) * u * math.pi / (2 * N)) for x in range(N)] for u in range(N)]
    for v in range(8):
        for u in range(8):
            s = 0.0
            for y in range(N):
                for x in range(N):
                    s += px[y * N + x] * cos[u][x] * cos[v][y]
            out.append(s)
    vals = out[1:]  # 去掉直流分量
    med = sorted(vals)[len(vals) // 2]
    bits = "".join("1" if v > med else "0" for v in vals[:63])
    return f"{int(bits, 2):016x}"


def phash_similarity(a: str, b: str) -> float:
    """汉明距离转相似度：距离 0 → 1.0，距离 ≥16 → 0。"""
    if not a or not b:
        return 0.0
    d = hamming(a, b)
    return max(0.0, 1.0 - d / 16.0)


def doc_text_similarity(a: str, b: str) -> float:
    """基于 token 集合的 Jaccard 相似度（文档复用检测）。"""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass
class EvidenceFingerprint:
    sha256: str
    phash: str
    text: str = ""


def fingerprint(path: str | Path, text: str = "") -> EvidenceFingerprint:
    return EvidenceFingerprint(
        sha256=sha256_file(path),
        phash=phash_file(path),
        text=text or _read_text(path),
    )


def _read_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    if p.suffix.lower() in (".txt", ".md", ".csv", ".json"):
        try:
            return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""
    return ""
