# SEC EDGAR Financial RAG

A fully self-hosted RAG pipeline over SEC 10-K/10-Q filings for 10 tickers (AAPL, MSFT, JPM, GS, JNJ, PFE, XOM, WMT, KO, TSLA), built to demonstrate production RAG engineering: custom HTML/table parsing, dense retrieval + reranking, grounded generation, and LLM observability — all running locally via vLLM, Qdrant, and LangSmith. No external API calls.

## Architecture

```
SEC EDGAR → build_dataset.py → parse_and_chunk.py → embed_chunks.py → Qdrant
                                                                          ↓
                                              retrieve.py → (rerank) → generate.py
                                                                          ↓
                                                                    Streamlit demo
```

Three models served locally via vLLM on one ~12GB GPU:

| Model | Role | Port |
|---|---|---|
| `BAAI/bge-large-en-v1.5` | Embeddings | 8000 |
| `BAAI/bge-reranker-base` | Reranking (cross-encoder) | 8001 |
| `microsoft/Phi-4-mini-instruct` | Generation | 8002 |

## Project structure

```
localChatbot/
  config.yaml / requirements.txt   <- shared across the whole pipeline
  data/               build_dataset.py, raw/, processed/, embedded/
  parsingChunking/     parse_and_chunk.py, dom_walker.py, table_cleaner.py
  embedding/           embed_chunks.py, load_qdrant.py, visualize_embeddings.py
  retrieval/           retrieve.py
  generation/          generate.py
  demo/                app.py (Streamlit)
  eval/                golden_questions.json, run_retrieval_eval.py, generate_batch.py
```

Every script locates `config.yaml` by walking upward from its own location, so it can be run from anywhere in the tree.

## Setup

```bash
pip install -r requirements.txt
```

Edit `config.yaml`: set a real contact email (SEC EDGAR requires it in the User-Agent) and, once vLLM/Qdrant are running, confirm the service URLs/ports match.

```bash
# 1. Build the raw corpus (SEC EDGAR, rate-limited)
python data/build_dataset.py

# 2. Parse + chunk (custom DOM walker + table cleaner)
python parsingChunking/parse_and_chunk.py
python parsingChunking/check_token_limits.py

# 3. Embed
vllm serve BAAI/bge-large-en-v1.5 --task embed          # port 8000
python embedding/embed_chunks.py

# 4. Load into Qdrant (run natively in WSL2, not Docker w/ Windows bind mount — see Troubleshooting)
cd ~/qdrant && ./qdrant                                  # port 6333
python embedding/load_qdrant.py

# 5. Retrieve / generate
vllm serve BAAI/bge-reranker-base --port 8001            # reranker (if using --rerank)
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve microsoft/Phi-4-mini-instruct --port 8002 \
  --gpu-memory-utilization 0.8 --max-model-len 8192 --enforce-eager

python retrieval/retrieve.py --query "What was Apple's gross margin in FY2022?"
python generation/generate.py --query "..." --rerank

# 6. Demo UI
streamlit run demo/app.py
```

Every stage is idempotent — safe to rerun; only new/changed inputs are reprocessed. **Exception:** any chunk-schema change (new metadata field, chunking logic change) requires a full reset (`rm -rf data/processed data/embedded` + wipe Qdrant) since it changes what's stored, not just what's new.

## Why SEC EDGAR

Free, public, no auth. Comes with structured metadata (CIK, accession number, filing date, form type) for free, which becomes chunk-level payload metadata for filtering at retrieval time. Same data source a real fintech company would use.

## Corpus scope

10 tickers across 6 sectors (Tech, Financials, Healthcare, Energy, Consumer staples, Automotive/Industrial), 10-K + 10-Q since 2022, ~190 filings, ~88K chunks. Sector diversity supports single-doc lookups, cross-period trend questions, and cross-company comparisons — the latter two are what actually stress-test retrieval, not just single-doc lookup.

## Key design decisions

**Custom parser, not a document library.** SEC filings from different preparers (Workiva, DFIN) use inconsistent, non-semantic HTML — no real `<h1>-<h4>` tags, headings implemented as bold spans, some preparers use `<p>` instead of `<div>` for ~85% of content, tables padded with empty spacer cells. `dom_walker.py` + `table_cleaner.py` were iteratively built and fixed against real filings from each preparer (Tesla, Microsoft, Goldman Sachs each surfaced a genuinely distinct heading/content-detection gap), verified against the specific filing that surfaced each one.

**Tables split, never truncated.** Oversized tables (routinely 1000+ tokens) are split at parse time — by row group and, when a table is also too wide, by column/period group — rather than truncated at embedding time, which would silently discard real financial data with zero way to retrieve it later. Split pieces carry `table_id`/`group_index`/`group_count` so `retrieve.py` reconstructs the full table from its siblings at query time.

**Embedding quality was measured, not assumed.** `visualize_embeddings.py` computes silhouette scores (full-space + LDA on held-out data) to confirm real separation exists where it matters (chunk type: text vs. table) and is deliberately absent where exact metadata filtering already covers it (ticker, filing section) — see `embedding/visualize_embeddings.py`.

**Hybrid search (dense + BM25) was tried and reverted.** Measured against the golden set, it regressed every metric (Hit Rate@10 filtered: 76.9% → 69.2%; MRR: 0.351 → 0.262) without even fixing the retrieval gaps it targeted. `search()` uses pure dense retrieval; documented here as a real negative result, not deleted.

**Reranking measurably helps.** Cross-encoder reranking (`BAAI/bge-reranker-base`, served via vLLM's Cohere-compatible `/rerank`) re-orders retrieval's own candidates — it can't fix a candidate that's missing entirely, only promote one that's buried. Confirmed: Hit Rate@10 84.6%, MRR 0.660 (filtered + reranked) vs. 76.9% / 0.351 without.

## Two production-grounding fixes (citation faithfulness)

Manual and batch testing (`eval/generate_batch.py`) against the golden question set surfaced two distinct, confirmed faithfulness failure modes — both required a code-level fix; prompt-only instructions did not reliably generalize.

**1. Wrong-period citation.** The model would sometimes cite a real, correctly-formatted source from the *wrong* fiscal year/quarter even when the correct-period source was also present in context. A dedicated prompt rule (checking period before citing) did not close the gap on retest. Fixed at the code level instead:
- `extract_target_years()` (`generate.py`) parses the year(s) named in the user's question.
- `filter_matching_period()` filters retrieved chunks to only those whose `fiscal_period_end` matches — falling back to the unfiltered set if that would empty the results. Deliberately does **not** match on `years_mentioned` (too permissive — MD&A sections routinely discuss several years in one chunk, defeating the filter).

**2. Table chunks were unretrievable for well-answered questions.** A verified-correct fact (e.g. Apple's gross margin %) existed in the corpus, correctly tagged, but never got retrieved — because a bare pipe-delimited numeric table has far less semantic surface for embedding similarity than prose. Fixed with `build_table_preamble()` (`parsingChunking/parse_and_chunk.py`): prepends a natural-language frame (ticker, form type, period, section) to each table chunk before embedding, so the vector captures what the table is *about*, not just its numbers. Token budget is reserved for this preamble before table-splitting decisions (`effective_max_tokens`).

Both fixes verified via a full pipeline rebuild (`rm -rf data/processed data/embedded` → reparse → re-embed → reload Qdrant) and confirmed via targeted live queries.

## Observability (LangSmith)

`generate.py`, `retrieve.py`, and `demo/app.py` are instrumented with LangSmith `@traceable` decorators, giving a nested trace per query:

```
rag_query / rag_demo_query (chain)
  ├─ embedding
  ├─ rerank
  └─ generation
```

Each LLM/embedding call's token usage (prompt/completion/total) and latency is captured automatically from vLLM's OpenAI-compatible `usage` field — no manual instrumentation needed beyond the decorator.

Setup:
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-key>
export LANGCHAIN_PROJECT=sec-edgar-rag
```
Must be exported in the same shell that runs the traced script — env vars don't carry across terminals.

## Evaluation

Retrieval-only metrics (Hit Rate@K, MRR — judge-free, don't require a generation LLM):

| | Unfiltered | Filtered | Filtered + reranked |
|---|---|---|---|
| Hit Rate@10 | 61.5% | 76.9% | 84.6% |
| MRR | 0.301 | 0.351 | 0.660 |

```bash
python eval/run_retrieval_eval.py --rerank
python eval/diagnose_misses.py --ids q3 q6 q12   # ranking issue vs. genuine absence
python eval/generate_batch.py --rerank --limit 6 # full generation pass over the golden set
```

13 golden questions, all 10 tickers covered, each traced back to a fact independently verified against raw filing HTML (see `golden_questions.json`'s `verification_method` field — 1 of 13 is `pipeline_retrieval`-verified rather than `raw_html`, weaker evidence, flagged explicitly).

## Known limitations

This is a portfolio project, not a production system — these are documented rather than chased further:

- **q3, q12 (retrieval misses):** genuinely absent from the top-50 candidates even before reranking — a retrieval-recall gap reranking can't fix. Root cause hypothesis: numbers-dense table content embeds weakly relative to surrounding narrative prose (consistent with the embedding-quality measurement above).
- **q6 (limit-sensitivity):** correct at `--limit 4`, incorrect at `--limit 5/6` — same underlying reconstructed table content produces a different answer depending on retrieval depth. In both cases the failure mode is a safe refusal or a citation mismatch, never a confident fabricated number.
- **No production hardening:** no auth, rate limiting, or concurrency handling on the demo — intentional scope decision for a portfolio project, not an oversight.
- **`run_retrieval_eval.py`/embedding pipeline network calls** were originally verified via mocks in a sandbox without GPU access; since then, verified for real against the live corpus (see Evaluation table above).

## Troubleshooting quick reference

vLLM + WSL2 issues hit while standing up the generation server, in the order they surfaced:

| Symptom | Cause | Fix |
|---|---|---|
| `RuntimeError: UVA is not available` | vLLM's V2 GPU model runner needs a CUDA feature WSL2 doesn't fully support | `VLLM_USE_V2_MODEL_RUNNER=0` |
| `Could not find nvcc` | FlashInfer sampler JIT-compiles kernels; WSL2 GPU passthrough ships the driver, not the compiler | add `VLLM_USE_FLASHINFER_SAMPLER=0` |
| `No available memory for the cache blocks` | Model loads at full bf16 (~7.6GB), not the smaller quantized footprint originally assumed | see final fix below |
| `Failed to find class TensorCoreTiledLayout` | Pre-quantized checkpoint depends on `torchao.prototype.awq` — unstable, version-incompatible | abandoned pre-quantized checkpoints |
| `Unknown quantization method: bitsandbytes` | vLLM 0.28.0 moved BitsAndBytes to an out-of-tree plugin | abandoned quantization entirely |
| `nvidia-smi` shows `N/A` per-process memory | Normal on WSL2 (WDDM driver model) — use the aggregate total instead | not a bug |

**Final working approach — no quantization, just memory-budget tuning:**
```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve microsoft/Phi-4-mini-instruct --port 8002 \
  --gpu-memory-utilization 0.8 --max-model-len 8192 --enforce-eager
```
`--enforce-eager` disables CUDA graph capture (a large, unpredictable memory consumer); `--max-model-len 8192` avoids reserving KV cache for the model's unused 128K context window. The `0.8` utilization value was derived from real load-log arithmetic, not guessed — see git history if the exact derivation is needed for a different GPU size.

**Qdrant on WSL2:** run the native Linux binary with storage on WSL2's own ext4 filesystem, not Docker with a Windows bind mount — Qdrant's own docs confirm a real data-corruption bug with that combination (vector data zeroed on restart).

## Next steps

- Full RAGAS metrics (context precision/recall, faithfulness, answer relevancy) now that a generation LLM exists
- Demo GIF for portfolio presentation
