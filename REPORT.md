# HW4 反思報告

## 1. 我如何判斷「記憶有效」？這個指標為何不可被作弊？
「記憶有沒有用」本身是主觀概念，無法直接評分。本系統把它拆解成一個可客觀檢驗的問題：**當一個查詢進來時，那筆真正相關的 observation 有沒有被 `retrieve()` 取回、而且排得夠前面？** 一旦這樣定義，主觀的「有用」就映射到三個標準檢索指標：
 
- **Recall@k**：正確記憶有沒有出現在前 k 筆裡（有沒有「想起來」）。
- **MRR**：第一個正確記憶排在第幾名的倒數（想起來的東西排得夠不夠前）。
- **nDCG@k**：考慮排序位置的折扣後增益（整體排序品質）。
這三者都對照 benchmark 提供的標準答案（gold label）計算，不是系統自我宣稱。換言之，「記憶有效」被重新定義為「在獨立的標準答案上，檢索排序達到可量化的門檻」。
 
### 為什麼這套指標作不了弊
 
關鍵在於**指標來自外部、固定的標準答案，而核心檢索是確定性的純函式**，這兩點同時成立時，沒有「灌水」的空間：
 
1. **隱藏測試集擋掉硬編碼。** 自動評分用的是格式相同、資料不同的隱藏測試集。`bm25_search()` 是 query 與 docs 的純函式，沒有任何 `if query == "..."` 之類的特例分支；就算想針對公開測試硬編碼答案，換一組隱藏資料就會失效。要在隱藏集上拿分，唯一的辦法是真的把通用的 BM25 排序做對。
2. **確定性讓「碰運氣」不可能。** `bm25_search()` 不呼叫任何模型、不含隨機性，`sorted(..., reverse=True)` 是穩定排序、同分維持輸入順序，因此同一組 `(query, docs)` 每次輸出完全相同。不能靠重跑取得較好的一次，也不能用隨機性矇混；分數要嘛重現得出來，要嘛重現不出來。
3. **指標衡量的是與真值的吻合度，不是自陳。** Recall / MRR / nDCG 全部以 benchmark 的標準答案為基準計算，且隱藏測試與人工抽查會回頭比對 REPORT 宣稱的分數是否屬實。要把指標做漂亮，只能讓檢索真的變好，沒有捷徑。
簡言之：**想作弊只有兩條路——硬編碼（被隱藏測試集擋掉）或引入非確定性（破壞可重現性、一樣被擋）**，兩條路都走不通，這正是作業強調的 Verifiability Mindset。

## 2. 我的 benchmark 分數

### 分數對比（基準語料 `queries.jsonl`，21 題，k=5）
 
| 指標 | 純 BM25 | Hybrid（BM25 + 多語言 embedding） | 改進 |
|---|---|---|---|
| Recall@5 | 0.810 | **1.000** | +0.190 |
| MRR | 0.810 | **0.917** | +0.107 |
| nDCG@5 | 0.802 | **0.938** | +0.136 |

重現方式：
 
```bash
# 純 BM25（各平台相同）
python benchmark/run_benchmark.py --k 5 --per-query
# Hybrid（macOS / Linux；Windows 寫法見第 7 點）
PI_RETRIEVAL=hybrid python benchmark/run_benchmark.py --k 5 --per-query
```

一個值得指出的對應：純 BM25 的 Recall@5 = 0.810 ≈ **17/21**，剛好就是 21 題中漏掉 4 題的結果；hybrid 把這 4 題全部接回，Recall@5 因此來到 21/21 = 1.000。下面的錯誤分析正是針對這漏掉的 4 題。
 
### 純 BM25 接不到的 4 題與成因
 
純 BM25 是 **lexical（字面）檢索**：`tokenize()` 只做小寫化與英數字／單一 CJK 字元切分，不做 stemming、不懂同義或語義，分數完全來自查詢詞與文件詞的**字面重疊**。只要查詢用字和記憶實際用字對不上，BM25 的 tf 就是 0、貢獻為 0。這 4 題正好踩在它的三種天花板上：
 
| # | 查詢 | 失敗類型 | 原因 |
|---|---|---|---|
| 1 | how big can an uploaded picture be? | 詞彙不匹配 | 查詢說 *picture*，記憶寫的是 *image / file size*，字面不重疊 |
| 2 | when is the team sync meeting each week? | 詞彙不匹配 | 查詢說 *team sync*，記憶寫的是 *weekly meeting*，同義但不同字 |
| 3 | 我要怎麼把新功能慢慢開放給部分使用者？ | 跨語言 + 語義 | 「慢慢開放給部分使用者」對應的記憶概念是 *feature flag / gradual rollout*，既跨語言又是語義轉換 |
| 4 | 怎麼確認我的程式碼風格符合規範？ | 語義落差 | 「程式碼風格符合規範」對應 *linter / ESLint*，是概念對應而非字面對應 |
 
### Hybrid 為什麼修得好
 
`hybrid_search()` 在純 BM25 之外加上 embedding 餘弦相似度，兩路分數各自 min-max 正規化後以 `alpha*BM25 + (1-alpha)*cosine`（預設 alpha=0.5）融合。embedding 把文字投影到語義向量空間，*picture* 與 *image*、*team sync* 與 *weekly meeting* 在向量空間中相近，即使字面不重疊也能拿到高相似度；而採用多語言模型 `paraphrase-multilingual-MiniLM-L12-v2` 讓中文查詢與英文記憶落在同一語義空間，因此跨語言的第 3 題也接得到。這就是業界記憶系統普遍從純 lexical 走向 hybrid 的原因——lexical 提供精準的字面命中，embedding 補上語義與跨語言的召回。
 
---

## 3. 系統哪些部分是確定性的、哪些是機率性的？為什麼這樣切？
本作業的設計哲學是「**確定性外殼包住機率性核心**」。在本系統中，這條分界線畫在一個明確的標準上：**凡是自動評分要驗證的，必須是確定性的；機率性的部分一律被包在確定性介面之內，且預設不啟用。**
 
**確定性外殼（單元測試驗證的部分）：**
 
- `bm25_search()`：純函式，無模型、無隨機，穩定排序，同分依輸入順序——同輸入必得同輸出。
- `JsonStore`：`id = sha256(summary)` 決定去重、`add()` 的存在性檢查、`_persist()` 的 JSON 寫入——皆為確定性。
- `build_injection()` 的預算截斷：給定檢索結果，截斷行為完全可預期。
- `retrieve()` 的模式開關：預設 `PI_RETRIEVAL=bm25`，公開與隱藏測試一律走確定性路徑。
**機率性核心（不供測試、被外殼包住）：**
 
- `hybrid_search()` 的 embedding 相似度：浮點運算、依模型權重、可能受裝置影響，本質上非位元級可重現。
- 更上游的捕捉決策：由本地 LLM 透過 `remember` 工具決定「該記什麼」，這是機率性的。
分界的具體實現方式有三層保護：(1) 機率性路徑用 `PI_RETRIEVAL` 環境變數**閘控**，預設不啟用，所以核心行為永遠確定；(2) `hybrid_search()` 在模型載不到或 encode 失敗時**優雅退回純 BM25**，輸出仍走同一個確定性介面；(3) hybrid 的邏輯**獨立成函式**，沒有污染 `bm25_search()`。也就是說，「相關性怎麼判斷」可以是機率性的（embedding），但它的輸出最終都流經同一個確定、可測的契約（前 k 筆、去重、預算截斷）——這就是外殼與核心的分工。
 
---

## 4. 注入的 token 預算我設多少？太多 / 太少各有什麼代價？

`build_injection()`（`core.py`）的策略是：**先把 header 計入用量，再依檢索分數由高到低逐筆塞入，碰到第一筆會超出預算的就停止。** token 用量以 `estimate_tokens(text) = max(1, len(text)//4)` 粗估。
 
幾個取捨的考量與誠實的權衡：
 
- **為何要有預算。** 本地小模型 context 是稀缺資源。注入過多記憶會把實際任務擠出 context、增加延遲，也可能用無關記憶干擾模型。預算把注入成本封頂。
- **為何按分數高到低塞。** `retrieve()` 已依相關度排序，優先注入最相關的記憶，確保有限的 budget 花在最可能有用的內容上。
- **碰到超預算就 `break`（而非跳過繼續找小的）。** 這是刻意的簡單取捨：保持嚴格的相關度順序、不讓低排名的小記憶插隊到被略過的高排名記憶之前。代價是可能浪費一點尾端預算（後面若有更短、能塞得下的記憶會被一起放棄）。這在小語料下影響很小，但若要更充分利用預算，可改成「跳過超量者、繼續嘗試後續較短記憶」的 best-fit 策略——這是一個明確可調的 trade-off。
- **`len//4` 的估算為何夠用。** 真實 token 數依 tokenizer 而異，但這個啟發式**確定、且不綁定任何特定 tokenizer**，跨模型可用。用估算精度換取確定性與零相依，符合本作業確定性外殼的原則。
- **丟誰。** 超預算時優先放棄相關度最低者（排序在後），因為低分通常意味低相關，是「最不可惜」的丟棄對象。
---

## 5. 我的記憶外掛和 `/compact` 是互補還是競爭？

兩者都處理「context 容納不下所有資訊」的問題，但作用範圍互補、不互相取代：
 
| | `/compact` | 本記憶系統 |
|---|---|---|
| 作用範圍 | **單一 session 內** | **跨 session** |
| 機制 | 壓縮／摘要當前對話以騰出 context | capture 寫入硬碟、下次 retrieve 取回 |
| 持久性 | 暫時，`/new` 或關掉 session 即消失 | 持久保存於 `memory.json` |
| 取用方式 | 把壓縮後的近期對話整段留在 context | 依當前查詢**選擇性檢索**相關片段 |
| 解決的問題 | session 內的 context 壓力 | session 間的知識遺忘 |
 
關鍵差異在於：`/compact` 是對「最近發生的一切」做有損壓縮，且結果仍綁在這次 session；它**無法**讓你在下週重開 agent 時問「這專案測試指令是什麼」還能想起來。本系統正好補上這個跨 session 的空缺——把觀察寫進硬碟，下次依任務需求**按相關度檢索**特定幾筆注入，而不是把所有東西都留著。兩者可並存：`/compact` 管 session 內的即時壓力，本系統管 session 間的長期知識。
 
---

## 6. 我的自動記憶系統 vs 人工手寫 `PROGRESS.md`，各有何優缺點？何時寧可手寫？

`PROGRESS.md` 是常見的土法記憶：開一個檔案，人工把專案狀態、慣例寫進去，下次手動貼回 context。兩者比較：
 
| 面向 | 手寫 PROGRESS.md | 本記憶系統 |
|---|---|---|
| 寫入 | 人工，靠紀律維護 | 自動，agent 呼叫 `remember` 即捕捉 |
| 去重 | 無，容易重複與矛盾累積 | SHA-256（依 summary）自動去重 |
| 取用 | 整檔讀進 context（全有或全無） | 依查詢相關度排序、只注入相關幾筆 |
| 預算 | 無，檔案無上限成長、終將塞爆 context | `build_injection` 在 token 預算內截斷 |
| 擴展性 | 差，內容一多就難用 | BM25／hybrid 在大量 observation 上仍能精準取回 |
| 時效 | 容易過期、無人更新 | 可加 decay／矛盾偵測（任務二）維持新鮮度 |
 
但要誠實看待 trade-off：`PROGRESS.md` 的優點是**透明、人可審閱、零基礎設施**，很適合承載高層次的專案敘事與里程碑。本系統的強項則是**規模化承載大量零碎事實，並做外科手術式的精準注入**——你不需要把整份筆記塞進 context，只取當下任務相關的那幾筆。兩者其實可以共存：`PROGRESS.md` 放人工策劃的高層狀態，記憶系統處理「用什麼套件管理器、commit 用什麼語言、測試怎麼跑」這類需要按需檢索的長尾慣例。
 
---

## 7. 環境記錄

### 系統與依賴（與程式碼一致，可直接採用）
 
| 項目 | 值 |
|---|---|
| Python 版本 | 3.14.3（要求 >= 3.10） |
| BM25 參數 | `k1 = 1.5`、`b = 0.75` |
| Hybrid 融合權重 | `alpha = 0.5`（`PI_HYBRID_ALPHA` 可調） |
| Embedding 模型 | `paraphrase-multilingual-MiniLM-L12-v2`（多語言，供跨語言題） |
| Token 估算 | `len // 4` |
| 記憶檔位置 | `PI_MEMORY_PATH` 或預設 `~/.pi-memory.json` |
 
### 本地 LLM（demo 用，請填入你的實際設定）
 
| 項目 | 值 |
|---|---|
| 模型型號 | Gemma4:26b |
| Context size | 模型上限 256K；本機實跑 num_ctx = 32768 |
| 推論後端 | Ollama   |
| VRAM | 12 GB |

### 重現指令
 
安裝與核心測試（各平台相同）：
 
```bash
pip install -r requirements.txt
pytest -q                                            # 核心單元測試
python benchmark/run_benchmark.py --k 5 --per-query  # 純 BM25
```
 
Hybrid 需設定環境變數 `PI_RETRIEVAL=hybrid`，設定方式依作業系統而異：
 
```bash
# macOS / Linux（bash / zsh）：行內設定環境變數
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
 
---