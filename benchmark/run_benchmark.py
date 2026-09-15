#!/usr/bin/env python3
"""
記憶檢索 Benchmark（study tool）。

把 corpus 全部 capture 進你的記憶系統，對每個 query 呼叫你的 retrieve()，
與標準答案（relevant_ids）比對，算出 Recall@k、MRR、nDCG@k。

用法：
    python benchmark/run_benchmark.py            # 預設 k=5
    python benchmark/run_benchmark.py --k 3
    python benchmark/run_benchmark.py --k 5 --per-query   # 顯示每題細節

這支程式碼不評分，是給你自己迭代用的：改進 BM25 或加上混合排序（任務二）後，
回來跑一次，看分數有沒有變好。Verifiability Mindset 的實際練習。
"""
from __future__ import annotations
import argparse
import json
import math
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids, relevant, k):
    gains = [1.0 if rid in relevant else 0.0 for rid in ranked_ids[:k]]
    ideal = [1.0] * min(len(relevant), k)
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--per-query", action="store_true")
    ap.add_argument("--corpus", default="corpus.jsonl",
                    help="語料檔名（預設 corpus.jsonl；大語料用 corpus_large.jsonl）")
    ap.add_argument("--queries", default="queries.jsonl",
                    help="查詢檔名（預設 queries.jsonl；大語料用 queries_large.jsonl）")
    args = ap.parse_args()
    k = args.k

    # 用一個暫存記憶檔，避免污染你平常的記憶
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    os.environ["PI_MEMORY_PATH"] = tmp.name

    # 在設定好環境變數後才 import，確保記憶寫到暫存檔
    import sys
    sys.path.insert(0, str(ROOT))
    from memory.core import capture, retrieve, make_observation, set_memory_path
    set_memory_path(tmp.name)

    corpus = load_jsonl(HERE / args.corpus)
    queries = load_jsonl(HERE / args.queries)

    # corpus 用固定 id（方便對標準答案），不走 sha256
    for row in corpus:
        obs = make_observation(row["summary"], tags=row.get("tags", []))
        obs["id"] = row["id"]
        capture(obs)

    recalls, rrs, ndcgs = [], [], []
    rows = []
    for q in queries:
        relevant = set(q["relevant_ids"])
        hits = retrieve(q["query"], k)
        ranked_ids = [h["id"] for h in hits]

        hit_set = set(ranked_ids) & relevant
        recall = len(hit_set) / len(relevant) if relevant else 0.0

        rr = 0.0
        for rank, rid in enumerate(ranked_ids, start=1):
            if rid in relevant:
                rr = 1.0 / rank
                break

        nd = ndcg_at_k(ranked_ids, relevant, k)
        recalls.append(recall); rrs.append(rr); ndcgs.append(nd)
        rows.append((q["query"], ranked_ids[:k], sorted(relevant), recall, rr, nd))

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print(f"\n=== Memory Retrieval Benchmark (k={k}, {len(queries)} queries) ===\n")
    if args.per_query:
        for query, got, rel, rc, rr, nd in rows:
            mark = "✓" if rc > 0 else "✗"
            print(f"{mark} {query}")
            print(f"    got@{k}: {got}")
            print(f"    gold : {rel}   Recall={rc:.2f} RR={rr:.2f} nDCG={nd:.2f}\n")

    print(f"Recall@{k} : {avg(recalls):.3f}")
    print(f"MRR       : {avg(rrs):.3f}")
    print(f"nDCG@{k}  : {avg(ndcgs):.3f}")
    print()
    is_large = "large" in args.corpus
    table = REFERENCE_LARGE if is_large else REFERENCE
    ref = table.get(k, table[5])
    label = "大語料 corpus_large（自行探索用）" if is_large else "小語料 corpus（測試基準）"
    print(f"--- 參考基準（pure BM25, {label}）---")
    print(f"Recall@{k} ≈ {ref['recall']:.3f} | MRR ≈ {ref['mrr']:.3f} | nDCG@{k} ≈ {ref['ndcg']:.3f}")
    print("（純 BM25 接近上述參考值為正常。任務二的混合排序若有效，分數應「不低於」純 BM25。）\n")
    os.unlink(tmp.name)


# 由助教的參考解實測填入（見題目）。學生看到這組數字當作「該打到哪」的目標。
REFERENCE = {
    3: {"recall": 0.810, "mrr": 0.810, "ndcg": 0.802},
    5: {"recall": 0.810, "mrr": 0.810, "ndcg": 0.802},
}

# 大語料 corpus_large.jsonl / queries_large.jsonl（100 筆 / 40 題，自行探索用）的純 BM25 實測
REFERENCE_LARGE = {
    3: {"recall": 0.812, "mrr": 0.821, "ndcg": 0.782},
    5: {"recall": 0.838, "mrr": 0.826, "ndcg": 0.795},
}

if __name__ == "__main__":
    main()
