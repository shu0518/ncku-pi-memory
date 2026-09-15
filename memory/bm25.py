"""BM25-lite：給每筆文件對查詢打相關度分數，回傳排序後的前 K 筆。整個作業的核心。
tokenize() 已給你；bm25_search() 的計分要你填。"""
from __future__ import annotations
import hashlib
import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """小寫化後，取出英數字詞與單個 CJK 字元。不做 stemming。（已提供）"""
    return _TOKEN_RE.findall(text.lower())


def bm25_search(
    query: str,
    docs: list[dict],
    k: int = 8,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    """標準 BM25 排序（確定性，不呼叫任何模型）。
 
    docs = [{"id": str, "text": str}, ...]；回傳 [{"id", "score"}, ...]（高到低，前 k 筆）。
 
      score(q,d) = Σ_qi IDF(qi) * (tf*(k1+1)) / (tf + k1*(1 - b + b*|d|/avgdl))
      IDF(qi)    = ln( (N - n + 0.5)/(n + 0.5) + 1 )
      tf = qi 在 d 出現次數 | |d| = d 詞數 | avgdl = 平均詞數 | N = 文件數 | n = 含 qi 的文件數
    """
    n_docs = len(docs)
    if n_docs == 0:
        return []
 
    # 1) 斷詞、記每篇長度與詞頻、算 avgdl
    doc_tokens: list[list[str]] = [tokenize(d["text"]) for d in docs]
    doc_counts: list[Counter] = [Counter(toks) for toks in doc_tokens]
    doc_lens: list[int] = [len(toks) for toks in doc_tokens]
    total_len = sum(doc_lens)
    avgdl = total_len / n_docs if n_docs else 0.0
    # 全空語料時避免除以 0；此時 tf 必為 0，分數仍為 0。
    if avgdl == 0:
        avgdl = 1.0
 
    # 2) 每個詞的 document frequency（df）
    df: Counter = Counter()
    for counts in doc_counts:
        for term in counts:
            df[term] += 1
 
    # 查詢斷詞；依公式 Σ_qi 逐一累加（保留重複出現的查詢詞）
    q_terms = tokenize(query)
 
    # 預先算好出現在查詢中的詞的 IDF
    idf: dict[str, float] = {}
    for term in set(q_terms):
        n = df.get(term, 0)
        idf[term] = math.log((n_docs - n + 0.5) / (n + 0.5) + 1.0)
 
    # 3) 對每篇文件，加總查詢每個詞的 BM25 貢獻（依輸入順序計算，確保同分穩定）
    results: list[dict] = []
    for i, d in enumerate(docs):
        counts = doc_counts[i]
        dl = doc_lens[i]
        denom_norm = k1 * (1.0 - b + b * (dl / avgdl))
        score = 0.0
        for term in q_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            score += idf[term] * (tf * (k1 + 1.0)) / (tf + denom_norm)
        results.append({"id": d["id"], "score": score})
 
    # 4) 依分數高到低排序；同分維持原始順序（sorted 為穩定排序）
    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    return ranked[:k]

# ──────────────────────────────────────────────────────────────────────────
# 任務二：Hybrid retrieval（BM25 + 本地 embedding）——獨立函式，不污染純 BM25 路徑。
# bm25_search() 維持標準、確定性的 BM25，供公開與隱藏測試驗證。
# hybrid_search() 為選用路徑：若本地沒有 sentence-transformers，優雅退回純 BM25。
#
# 預設使用多語言模型，使「查英文 / 記憶寫中文」之類的跨語言題也接得到；
# 想換回較輕的英文模型，傳 model_name="all-MiniLM-L6-v2"（或設 PI_HYBRID_MODEL）即可。
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_HYBRID_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
 
_EMBED_MODELS: dict = {}          # model_name -> SentenceTransformer（惰性、可多模型快取）
_DOC_EMB_CACHE: dict = {}         # model_name -> (fingerprint, doc_embeddings)
 
 
def _get_embed_model(model_name: str = DEFAULT_HYBRID_MODEL):
    """惰性載入本地 embedding 模型（依模型名快取）；失敗則回傳 None（退回純 BM25）。"""
    if model_name in _EMBED_MODELS:
        return _EMBED_MODELS[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        _EMBED_MODELS[model_name] = None
        return None
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        model = None
    _EMBED_MODELS[model_name] = model
    return model
 
def _doc_fingerprint(docs: list[dict]) -> str:
    """以 (id, text) 串接的雜湊代表這批文件。文件集合不變時 embedding 即可重用。"""
    h = hashlib.sha256()
    for d in docs:
        h.update(str(d["id"]).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(d["text"]).encode("utf-8"))
        h.update(b"\x01")
    return h.hexdigest()
 
 
def _get_doc_embeddings(model, model_name: str, docs: list[dict]):
    """取得（並快取）整批文件的 normalized embedding。
 
    benchmark 會對同一批文件跑數十個 query；若每次都重新 encode 整個語料會很慢，
    因此以文件指紋為 key 快取，文件集合不變時直接重用。
    """
    fp = _doc_fingerprint(docs)
    cached = _DOC_EMB_CACHE.get(model_name)
    if cached is not None and cached[0] == fp:
        return cached[1]
    doc_texts = [d["text"] for d in docs]
    doc_emb = model.encode(doc_texts, normalize_embeddings=True)
    _DOC_EMB_CACHE[model_name] = (fp, doc_emb)
    return doc_emb

def _min_max_norm(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]
 
 
def hybrid_search(
    query: str,
    docs: list[dict],
    k: int = 8,
    alpha: float = 0.5,
    k1: float = 1.5,
    b: float = 0.75,
    model_name: str = DEFAULT_HYBRID_MODEL,
) -> list[dict]:
    """Hybrid：alpha*BM25 + (1-alpha)*embedding 餘弦相似度（皆 min-max 正規化後融合）。
 
    用於修補純 BM25 接不到的語義/跨語言題。若無 embedding 模型可用，退回純 BM25。
    這條路徑非確定性核心，不供單元測試驗證。
    """
    if not docs:
        return []
 
    # 先取全量 BM25 分數（k = 全部），保留 id->score 對照
    bm25_full = bm25_search(query, docs, k=len(docs), k1=k1, b=b)
    bm25_by_id = {r["id"]: r["score"] for r in bm25_full}
 
    model = _get_embed_model(model_name)
    if model is None:
        return bm25_full[:k]
 
    try:
        doc_emb = _get_doc_embeddings(model, model_name, docs)
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        cos = (doc_emb @ q_emb).tolist()
    except Exception:
        return bm25_full[:k]
 
    ids = [d["id"] for d in docs]
    bm25_scores = [bm25_by_id.get(i, 0.0) for i in ids]
    bm25_n = _min_max_norm(bm25_scores)
    cos_n = _min_max_norm(list(cos))
 
    fused = [
        {"id": ids[i], "score": alpha * bm25_n[i] + (1.0 - alpha) * cos_n[i]}
        for i in range(len(ids))
    ]
    ranked = sorted(fused, key=lambda r: r["score"], reverse=True)
    return ranked[:k]
