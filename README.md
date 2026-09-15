# HW4：Pi Memory — 為本地 Coding Agent 打造持久記憶

為 [Pi coding agent](https://github.com/earendil-works/pi) 加上一層 context window 之外的**跨 session 持久記憶**：把 agent 互動中產生的觀察（observation）捕捉並寫入硬碟，下次 session 開始時依當前任務檢索最相關的幾筆、在 token 預算內注入回 context。核心、benchmark、評分皆為 **Python**，整個系統跑在本地模型上，**不呼叫任何雲端 API**。

- Repo：<https://github.com/Netdb-NCKU/hw4-pi-memory-shu0518>
- 繳交檔案：`memory/`、`tests/`、`benchmark/`、`pi-bridge/`、`README.md`、`REPORT.md`、`requirements.txt`、`demo/`

---

## 1. 設計概述

記憶系統的全流程是 **Capture → Store → Retrieve → Inject**，掛在 Pi 的兩個生命週期攔截點上：

```
  Pi（本地模型，gemma4:26b @ Ollama）
      │  before_agent_start（開工前注入）／ remember 工具（互動中捕捉）
      ▼
  pi-bridge/extension.ts        ← 已提供的 TS bridge，不需修改
      │  subprocess 呼叫 python -m memory.cli
      ▼
  memory/ （Python 核心）
      ├─ core.py    capture / retrieve / build_injection 串接
      ├─ store.py   JsonStore：JSON 持久層、SHA-256 去重
      └─ bm25.py    bm25_search（確定性 BM25）／ hybrid_search（任務二）
      ▼
  記憶檔（JSON，持久於硬碟）
```

四個階段各自的職責：

- **Capture**：agent 在互動中呼叫 `remember` 工具，bridge 透過 CLI 把該筆觀察送進 `capture()`。
- **Store**：`JsonStore` 以 `summary` 的 SHA-256 當 id 去重後寫入 JSON 檔，重啟後可完整讀回。
- **Retrieve**：給定查詢，用標準 BM25 對所有 observation 計分，回傳前 K 筆（預設確定性路徑）。
- **Inject**：`build_injection()` 依分數高到低，在 token 預算（預設 2000）內逐筆塞入，組成一段注入文字交還給 Pi，在模型讀取訊息前補進 context。

兩個攔截點互補：**捕捉**負責把知識寫進硬碟，**注入**負責在下次 session 依任務需求把知識取回。中間的硬碟存儲正是跨 session 的關鍵——這也是它與 Pi 內建 `/compact`（僅在單一 session 內壓縮對話、不跨 session 保存）的根本差異。

> 確定性與機率性的分界、各題錯誤分析、token 預算取捨等設計討論詳見 [`REPORT.md`](REPORT.md)。

---

## 2. 環境

### 開發與核心參數

| 項目 | 值 |
|---|---|
| Python 版本 | **3.14.3**（作業要求 >= 3.10） |
| 核心依賴 | 標準函式庫（capture / store / retrieve / BM25 皆無第三方相依） |
| 測試 | `pytest = 9.0.3` |
| BM25 參數 | `k1 = 1.5`、`b = 0.75` |
| Hybrid 融合權重 | `alpha = 0.5`（可由 `PI_HYBRID_ALPHA` 調整） |
| Embedding 模型（任務二） | `paraphrase-multilingual-MiniLM-L12-v2`（多語言，跨語言題友善） |
| Token 估算 | `len(text) // 4`（確定、不綁定特定 tokenizer） |
| 記憶檔位置 | 環境變數 `PI_MEMORY_PATH`，未設時預設 `~/.pi-memory.json` |

### 本地 LLM（僅 demo 用）

模型強弱不影響核心評分（核心由單元測試與 benchmark 客觀驗證），只在錄 demo 時用到。本專案的設定為：

| 項目 | 值 |
|---|---|
| 模型型號 | `gemma4:26b` |
| Context size | 模型上限 256K；本機實跑 num_ctx = 32768 |
| 推論後端 | Ollama（OpenAI-compatible 本地 endpoint） |
| VRAM | 12 GB |

### 可調環境變數

| 變數 | 預設 | 用途 |
|---|---|---|
| `PI_MEMORY_PATH` | `~/.pi-memory.json` | 記憶 JSON 檔位置 |
| `PI_RETRIEVAL` | `bm25` | 設為 `hybrid` 啟用任務二的 hybrid 檢索 |
| `PI_HYBRID_ALPHA` | `0.5` | hybrid 融合權重，夾限於 `[0, 1]` |
| `PI_HYBRID_MODEL` | 多語言 MiniLM | 指定 `sentence-transformers` 模型名 |
| `PYTHON` | `python3` | bridge 呼叫的 Python 執行檔（Windows 建議設為 `python`） |

---

## 3. 安裝與測試

在乾淨環境（建議 Python 3.14.3，或 >= 3.10 的相近版本）：

```bash
git clone https://github.com/Netdb-NCKU/hw4-pi-memory-shu0518.git
cd hw4-pi-memory-shu0518
pip install -r requirements.txt
```

執行核心單元測試，應全部通過：

```bash
pytest -q
```

> 核心（`bm25_search` / `JsonStore`）僅用標準函式庫，即使不裝 `sentence-transformers` 也能完整通過所有單元測試；`sentence-transformers` 只在啟用任務二 hybrid 時才需要。

---

## 4. 跑 Benchmark

benchmark 是迭代用的 study tool，**不直接計入自動評分**，但量測結果與錯誤分析需寫進 `REPORT.md`。兩組語料皆已提供。

```bash
# 基準語料（30 筆 / 21 題，預設）
python benchmark/run_benchmark.py --k 5 --per-query

# 大語料（100 筆 / 40 題，自行探索用）
python benchmark/run_benchmark.py --corpus corpus_large.jsonl --queries queries_large.jsonl --k 5 --per-query
```

啟用任務二的 hybrid 檢索時，需先設定 `PI_RETRIEVAL=hybrid`（設定方式依作業系統而異）：

```bash
# macOS / Linux（bash / zsh）：行內設定
PI_RETRIEVAL=hybrid python benchmark/run_benchmark.py --k 5 --per-query
```

```powershell
# Windows PowerShell：先設環境變數再執行
$env:PI_RETRIEVAL = "hybrid"; python benchmark/run_benchmark.py --k 5 --per-query
```

```bat
:: Windows 命令提示字元（cmd）
set PI_RETRIEVAL=hybrid && python benchmark/run_benchmark.py --k 5 --per-query
```

本專案在基準語料（`queries.jsonl`，21 題，k=5）量到的分數：

| 指標 | 純 BM25 | Hybrid（BM25 + 多語言 embedding） |
|---|---|---|
| Recall@5 | 0.810 | 1.000 |
| MRR | 0.810 | 0.917 |
| nDCG@5 | 0.802 | 0.938 |

純 BM25 漏接的 4 題與成因（詞彙不匹配、跨語言、語義落差）以及 hybrid 為何修得好，詳見 [`REPORT.md`](REPORT.md) 第 2 節。

---

## 5. 掛載到 Pi 並重現 Demo

bridge（`pi-bridge/extension.ts`）透過 `python -m memory.cli` 以 subprocess 呼叫本作業的 Python 記憶系統，因此**必須讓 Python 找得到 `memory/` 套件**。關鍵是：在 repo 根目錄執行，並設定 `PYTHONPATH=.`。

### macOS / Linux

```bash
cd hw4-pi-memory-shu0518
PYTHONPATH=. pi -e ./pi-bridge/extension.ts
```

### Windows（PowerShell）

倉庫內附 `demo.ps1`，已一併設好 UTF-8 輸出與所需環境變數，直接在 repo 根目錄執行即可：

```powershell
.\demo.ps1
```

`demo.ps1` 等同於：

```powershell
chcp 65001 | Out-Null            # 終端機切 UTF-8，避免中文亂碼
$env:PYTHON         = "python"   # bridge 預設呼叫 python3，Windows 改用 python
$env:PYTHONPATH     = "."        # 讓 python -m memory.cli 找得到 memory/ 套件
$env:PI_MEMORY_PATH = "$PWD\memory.json"  # 記憶檔放 repo 根目錄（而非家目錄）
$env:PYTHONUTF8     = "1"
pi -e ./pi-bridge/extension.ts
```

> bridge 內部會以 `extension.ts` 所在的 repo 根目錄為工作目錄（`cwd`），並一併帶上 `PYTHONPATH`，所以即使透過 `~/.pi/agent/extensions/` 自動載入，理論上也能定位到 `memory/`；但跨平台行為仍建議以上述「在 repo 根目錄啟動」的方式為準。

### 重現「跨 session 記憶生效」

demo 影片展示的流程：

1. **Session A（捕捉）**：對 agent 說出一條專案慣例，例如「這專案用 pnpm，不要用 npm；測試指令是 pnpm test」。模型呼叫 `remember` 工具，bridge 透過 `capture` 寫入記憶檔。
2. **關掉 agent、重開（新 session）**。
3. **Session B（注入 → 取回）**：對 agent 說「幫我跑測試」。`before_agent_start` 觸發 `inject`，BM25 檢索到 Session A 那筆慣例並注入 context，agent 不必再問就知道要跑 `pnpm test`。

也可不開模型、直接用 CLI 驗證記憶迴路：

```bash
PYTHONPATH=. python -m memory.cli capture --summary "this project uses pnpm test"
PYTHONPATH=. python -m memory.cli retrieve --query "pnpm test" --k 5
PYTHONPATH=. python -m memory.cli inject --query "run tests" --budget 2000
```

**Demo 影片連結**：https://youtu.be/ULUnd4TUvIo 

---

## 6. 任務二：Hybrid Retrieval（BM25 + 本地 embedding）

本專案實作的進階功能是 **hybrid retrieval**，用以修補純 BM25（lexical / 字面檢索）接不到的語義與跨語言題。

- 在純 BM25 之外加上 embedding 餘弦相似度，兩路分數各自 min-max 正規化後以 `alpha * BM25 + (1 - alpha) * cosine`（預設 `alpha = 0.5`）融合。
- 採多語言模型 `paraphrase-multilingual-MiniLM-L12-v2`，使「查英文 / 記憶寫中文」這類跨語言題也落在同一語義空間，因而能被取回。
- 首次執行會從 Hugging Face 下載模型並快取至 `~/.cache/huggingface`，之後可完全離線；**執行與 demo 期間不呼叫任何雲端推論 API**。

設計上刻意保留純 BM25 路徑不受污染：

- `memory/bm25.py::bm25_search()` 維持標準、確定性的 BM25，供公開與隱藏單元測試驗證；hybrid 邏輯獨立寫在 `hybrid_search()`。
- hybrid 由 `PI_RETRIEVAL=hybrid` **閘控**，預設不啟用，核心測試一律走確定性 BM25。
- 當本地沒有 `sentence-transformers`、或模型載入 / encode 失敗時，`hybrid_search()` 會**優雅退回純 BM25**，輸出仍走同一個確定性介面。

啟用方式與 benchmark 改進數據見上方第 4 節，完整錯誤分析見 [`REPORT.md`](REPORT.md)。

---

## 7. 專案結構

```
hw4-pi-memory-shu0518/
├── memory/
│   ├── __init__.py
│   ├── bm25.py             # bm25_search（核心）＋ hybrid_search（任務二）
│   ├── store.py            # JsonStore：JSON 持久層、SHA-256 去重
│   ├── core.py             # capture / retrieve / build_injection 串接（已提供）
│   └── cli.py              # 給 Pi bridge 的 CLI（已提供）
├── tests/
│   └── test_memory.py      # 公開單元測試
├── benchmark/
│   ├── corpus.jsonl        # 基準語料 30 筆
│   ├── queries.jsonl       # 基準查詢 21 題 + 標準答案
│   ├── corpus_large.jsonl  # 大語料 100 筆
│   ├── queries_large.jsonl # 大語料查詢 40 題
│   ├── run_benchmark.py    # Recall@k / MRR / nDCG
│   └── README.md
├── pi-bridge/
│   └── extension.ts        # 輕量化 TS bridge（已提供，不需修改）
├── fixtures/observations.json
├── demo/                   # demo 影片連結 / 截圖
│   └── README.md
├── demo.ps1                # Windows 一鍵啟動 Pi + bridge
├── requirements.txt
├── README.md
├── REPORT.md
└── models.json.example
```

---

## 8. 持續整合（CI）

每次 `push` 由 GitHub Actions 自動檢查基本可執行性：必要檔案存在、`.env` 未被 commit。繳交前請確認 CI 顯示綠色 ✅。本專案不含任何真實 API Key（跑本地模型，無需金鑰）。

> CI 僅檢查基本可執行性；BM25 排序正確性由評分時的隱藏測試集（格式同公開測試、資料不同）驗證。