"""把 store + bm25 接起來，對外暴露 capture / retrieve（自動評分與 benchmark 會呼叫）。"""
from __future__ import annotations
import hashlib
import os
import time
from pathlib import Path

from .store import JsonStore
from .bm25 import bm25_search

_store_path = os.environ.get("PI_MEMORY_PATH") or str(Path.home() / ".pi-memory.json")
_store: JsonStore | None = None

# ── 任務二：檢索路徑開關（確定性外殼）──────────────────────────────────
# 預設純 BM25：公開＋隱藏單元測試一律走確定性 BM25，行為不變。
# 設 PI_RETRIEVAL=hybrid 才切換到 hybrid_search（BM25 + 本地 embedding）。
#   PI_HYBRID_ALPHA  融合權重 alpha（預設 0.5）：final = alpha*BM25 + (1-alpha)*embedding
#   PI_HYBRID_MODEL  指定 sentence-transformers 模型名（預設多語言模型，跨語言題友善）
_DEFAULT_HYBRID_ALPHA = 0.5

def _get_store() -> JsonStore:
    global _store
    if _store is None:
        _store = JsonStore(_store_path)
    return _store


def set_memory_path(path: str) -> None:
    """測試 / bridge 用來指定記憶檔位置（會重新載入）。"""
    global _store_path, _store
    _store_path = path
    _store = JsonStore(_store_path)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """粗估 token 數，約 = 字元數 / 4。"""
    return max(1, len(text) // 4)


def make_observation(summary: str, session_id: str = "s", tool_name: str = "remember",
                     tags: list[str] | None = None) -> dict:
    return {
        "id": sha256(summary),
        "sessionId": session_id,
        "timestamp": int(time.time() * 1000),
        "toolName": tool_name,
        "summary": summary,
        "tags": tags or [],
    }


def capture(obs: dict) -> None:
    """(a)(b) Capture + Store。"""
    _get_store().add(obs)

def _hybrid_alpha() -> float:
    """讀取並夾限 PI_HYBRID_ALPHA 到 [0, 1]；非法值退回預設。"""
    raw = os.environ.get("PI_HYBRID_ALPHA")
    if raw is None:
        return _DEFAULT_HYBRID_ALPHA
    try:
        alpha = float(raw)
    except ValueError:
        return _DEFAULT_HYBRID_ALPHA
    return min(1.0, max(0.0, alpha))

def retrieve(query: str, k: int) -> list[dict]:
    """(c) Retrieve：找出與 query 最相關的前 K 筆 Observation。
 
    預設用確定性 BM25（供單元測試 / 隱藏測試）。
    設 PI_RETRIEVAL=hybrid 時改走 hybrid_search（BM25 + 本地 embedding），
    用於修補純 BM25 接不到的語義 / 跨語言題；hybrid 不可用時內部會自動退回純 BM25。
    """
    all_obs = _get_store().all()
    docs = [{"id": o["id"], "text": " ".join([o["summary"], *o.get("tags", [])])} for o in all_obs]
 
    mode = (os.environ.get("PI_RETRIEVAL") or "bm25").strip().lower()
    if mode == "hybrid":
        from .bm25 import hybrid_search, DEFAULT_HYBRID_MODEL
        model_name = os.environ.get("PI_HYBRID_MODEL") or DEFAULT_HYBRID_MODEL
        ranked = hybrid_search(query, docs, k, alpha=_hybrid_alpha(), model_name=model_name)
    else:
        ranked = bm25_search(query, docs, k)
 
    by_id = {o["id"]: o for o in all_obs}
    return [by_id[r["id"]] for r in ranked if r["id"] in by_id]

def build_injection(query: str, token_budget: int = 2000, k: int = 8) -> str:
    """(d) Inject：把檢索結果在 token 預算內組成一段文字。"""
    hits = retrieve(query, k)
    header = "[記憶 - 來自過去的 session]"
    lines, used = [], estimate_tokens(header)
    for h in hits:
        line = f"- {h['summary']}"
        cost = estimate_tokens(line)
        if used + cost > token_budget:
            break
        lines.append(line)
        used += cost
    return "\n".join([header, *lines]) if lines else ""
