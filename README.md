# Pi Memory — Cross-Session Memory for a Local Coding Agent

> Adds persistent, cross-session memory to the [Pi coding agent](https://github.com/earendil-works/pi): captures observations from a session to disk, then retrieves and re-injects the most relevant ones into context at the start of the next session — entirely on local models, no cloud API calls.

`Course project` · Netdb Lab, NCKU · AIASE 2026 · Individual
**Stack:** Python (standard library core) · BM25 (from scratch) · sentence-transformers (optional hybrid) · TypeScript bridge (provided, unmodified)

## Overview

The system implements a four-stage loop — Capture, Store, Retrieve, Inject — hung off two Pi lifecycle hooks. During a session, the agent calls a `remember` tool, which the provided TypeScript bridge forwards as a subprocess call into `memory.cli`; `capture()` writes the observation to a JSON file, deduplicated by the SHA-256 hash of its summary. At the start of the next session, `before_agent_start` triggers `retrieve()`, which scores every stored observation against the current task with BM25 and returns the top-k. `build_injection()` then packs the highest-scoring observations into a token budget (default 2000) and hands the resulting text back to Pi before the model sees its first message. The disk-persisted store is what makes this survive across sessions, unlike Pi's built-in `/compact`, which only compresses one session's own conversation.

## Key Design Decisions

| Decision | Rationale |
| --- | --- |
| Core BM25 (`bm25_search`) uses only the standard library, no third-party dependencies | Public and hidden unit tests must pass without installing `sentence-transformers`; keeps the graded path dependency-free and deterministic |
| Hybrid retrieval gated behind `PI_RETRIEVAL=hybrid`, off by default | Core tests always exercise the deterministic BM25 path; the probabilistic embedding path is opt-in and isolated in `hybrid_search()` |
| `alpha=0.5` fusion: `alpha*BM25 + (1-alpha)*cosine`, both min-max normalized | Balances exact lexical hits (BM25) against semantic/cross-language matches (embedding) rather than picking one exclusively |
| Multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) for hybrid | Test queries mix English and Chinese; a multilingual model puts both in the same embedding space so a Chinese query can retrieve an English-written memory |
| Hybrid falls back to plain BM25 if `sentence-transformers` is missing or encoding fails | Keeps the CLI usable in environments without the optional dependency installed, without crashing |

## Challenges

**Problem.** On the 21-query benchmark (`queries.jsonl`, k=5), plain BM25 scored Recall@5 = 0.810 (17/21) — it missed 4 queries. BM25 is purely lexical (lowercase + alphanumeric/CJK-character tokenization, no stemming or synonyms), so it scores zero overlap whenever the query's wording doesn't literally match the stored memory's wording. The 4 misses broke down into: 2 vocabulary mismatches (`picture` vs. `image/file size`; `team sync` vs. `weekly meeting`), 1 cross-language + semantic case (a Chinese query about gradual feature rollout mapping to an English memory about `feature flag`), and 1 pure semantic gap (`code style` mapping to `linter/ESLint`).
**Approach.** Added `hybrid_search()`: BM25 score and embedding cosine similarity are each min-max normalized, then fused as `alpha*BM25 + (1-alpha)*cosine` with the multilingual model, so semantically close but lexically different text still scores highly.
**Result.** On the same 21-query benchmark, hybrid retrieval reached Recall@5 = 1.000 (21/21), MRR 0.917 (up from 0.810), and nDCG@5 0.938 (up from 0.802) — all 4 BM25 misses were recovered.

## Limitations

- BM25 has no stemming or synonym handling; any lexical mismatch between query and stored wording that embeddings aren't enabled for will still be missed.
- Token budget accounting uses `len(text) // 4` as an estimate, not a real tokenizer, so the actual token count injected can differ from the model's own count.
- `JsonStore` has no expiry, pruning, or max-size logic (`load` / `add` / `all` / `clear` only) — the memory file grows without bound as more observations are captured.
- Hybrid retrieval's first run downloads the embedding model from Hugging Face; without that one-time network access it silently falls back to BM25-only, which can mask why cross-language recall regressed.

## Running It

```bash
git clone https://github.com/Netdb-NCKU/hw4-pi-memory-shu0518.git
cd hw4-pi-memory-shu0518
pip install -r requirements.txt

pytest -q                                        # core unit tests (stdlib only)

python benchmark/run_benchmark.py --k 5 --per-query                       # BM25-only
PI_RETRIEVAL=hybrid python benchmark/run_benchmark.py --k 5 --per-query   # hybrid (macOS/Linux)

# CLI, without running the agent
PYTHONPATH=. python -m memory.cli capture --summary "this project uses pnpm test"
PYTHONPATH=. python -m memory.cli retrieve --query "pnpm test" --k 5
PYTHONPATH=. python -m memory.cli inject --query "run tests" --budget 2000
```

## Structure

    memory/bm25.py       bm25_search (core, stdlib-only) + hybrid_search (BM25 + embedding fusion)
    memory/store.py      JsonStore: JSON persistence, SHA-256 dedup on summary
    memory/core.py       capture / retrieve / build_injection wiring
    memory/cli.py        CLI entry point the Pi bridge subprocess calls
    pi-bridge/extension.ts   TypeScript bridge (provided, unmodified)
    benchmark/            corpus.jsonl / queries.jsonl (+ _large variants), run_benchmark.py
    tests/test_memory.py  Public unit tests
