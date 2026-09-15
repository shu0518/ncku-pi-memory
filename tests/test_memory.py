"""公開單元測試（隱藏測試格式相同，只換資料）。對應題目 §(c) 三筆 pnpm 範例。"""
import os
import tempfile
import pytest

from memory.bm25 import tokenize, bm25_search

DOCS = [
    {"id": "d1", "text": "this project uses pnpm test"},
    {"id": "d2", "text": "the project readme is in docs"},
    {"id": "d3", "text": "run pnpm test before commit"},
]


def test_tokenize_basic():
    assert tokenize("Run pnpm Test!") == ["run", "pnpm", "test"]


def test_tokenize_cjk():
    # 中日韓字元各自成一個 token
    assert tokenize("資料庫 schema") == ["資", "料", "庫", "schema"]


def test_bm25_ranking_example():
    # 查詢 "pnpm test"：D1、D3 應進前 2，D2 不該進前 2
    top = [s["id"] for s in bm25_search("pnpm test", DOCS, 2)]
    assert "d1" in top and "d3" in top
    assert "d2" not in top


def test_bm25_irrelevant_zero():
    allres = bm25_search("pnpm test", DOCS, 3)
    d2 = next(s for s in allres if s["id"] == "d2")
    assert d2["score"] == pytest.approx(0.0, abs=1e-6)


def test_bm25_idf_rare_term_wins():
    docs = [{"id": "a", "text": "the the the pnpm"}, {"id": "b", "text": "the the the the"}]
    top = bm25_search("the pnpm", docs, 1)[0]
    assert top["id"] == "a"  # 命中罕見字 pnpm 者勝出


@pytest.fixture
def tmp_store(monkeypatch):
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    f.close()
    from memory import core
    core.set_memory_path(f.name)
    yield f.name
    os.unlink(f.name)


def test_capture_retrieve(tmp_store):
    from memory.core import capture, retrieve, make_observation
    capture(make_observation("this project uses pnpm"))
    capture(make_observation("deploy with docker compose"))
    hits = retrieve("how to run pnpm", 1)
    assert "pnpm" in hits[0]["summary"]


def test_dedup(tmp_store):
    from memory.core import capture, retrieve, make_observation
    capture(make_observation("use pnpm"))
    capture(make_observation("use pnpm"))
    assert len(retrieve("pnpm", 10)) == 1


def test_persistence(tmp_store):
    from memory import core
    from memory.core import capture, retrieve, make_observation
    capture(make_observation("use pnpm not npm"))
    core.set_memory_path(tmp_store)  # 模擬重啟：重新載入
    assert len(retrieve("pnpm", 5)) == 1


def test_injection_budget(tmp_store):
    from memory.core import capture, build_injection, make_observation
    for i in range(50):
        capture(make_observation(f"pnpm fact number {i} about the build system"))
    out = build_injection("pnpm build", token_budget=80)
    assert len(out) < 80 * 4 + 50
    assert "[記憶" in out


# ── edge cases（審查建議補強）──

def test_bm25_empty_docs():
    # 空語料不應 crash，回傳空 list
    assert bm25_search("anything", [], 5) == []


def test_bm25_empty_query():
    # 空 query：所有文件分數為 0，不應 crash
    res = bm25_search("", DOCS, 3)
    assert all(s["score"] == pytest.approx(0.0, abs=1e-9) for s in res)


def test_bm25_stable_order_on_ties():
    # 完全相同的文件（同分）應維持原始輸入順序，確保跨實作可重現
    docs = [
        {"id": "a", "text": "alpha beta"},
        {"id": "b", "text": "alpha beta"},
        {"id": "c", "text": "alpha beta"},
    ]
    order = [s["id"] for s in bm25_search("alpha", docs, 3)]
    assert order == ["a", "b", "c"]


def test_store_ignores_non_list_json(tmp_store):
    # JSON 存在但不是 list（例如 {}）時，應視為空而非 crash
    import json
    with open(tmp_store, "w", encoding="utf-8") as f:
        json.dump({"not": "a list"}, f)
    from memory import core
    core.set_memory_path(tmp_store)
    from memory.core import retrieve
    assert retrieve("anything", 5) == []
