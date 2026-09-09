# Financial RAG dataset builder

Builds the raw document corpus for the financial RAG portfolio project by
pulling 10-K / 10-Q filings directly from SEC EDGAR.

## Project structure

One task per directory, one shared config:

```
localChatbot/
  config.yaml           <- single shared config for the whole pipeline
  requirements.txt      <- single shared dependency list for the whole pipeline
  README.md             <- this file
  data/
    build_dataset.py
    raw/                <- created by build_dataset.py
    processed/          <- created by parse_and_chunk.py
    embedded/           <- created by embed_chunks.py
  parsingChunking/
    parse_and_chunk.py
    dom_walker.py
    table_cleaner.py
    check_token_limits.py
  embedding/
    embed_chunks.py
```

`requirements.txt` belongs at the root for the same reason `config.yaml`
does: it's a single shared dependency list covering every stage (BS4/lxml
for parsing, LangChain for chunking, requests/transformers for embedding),
not owned by any one subdirectory -- move it there alongside `config.yaml`
if you haven't already.

Every script finds `config.yaml` by searching upward from its own location
until it finds one (`find_config()` in each script) -- it doesn't matter
that `build_dataset.py` lives in `data/`, `parse_and_chunk.py` lives in
`parsingChunking/`, and `embed_chunks.py` lives in `embedding/`, since they all
walk up to the same project root either way. Data-directory paths inside
`config.yaml` (`output_dir`, `processed_dir`, `embedded_dir`) are then
resolved relative to wherever that config file actually lives, not
relative to any individual script or your terminal's current directory --
so every script agrees on where `data/raw`, `data/processed`, and
`data/embedded` are, regardless of which one of them you run or from
where. Verified by simulating this exact directory layout and running each
script from its actual location before this was written up.

Commands below assume running from the project root (`localChatbot/`),
matching where this README and `config.yaml` live -- but since path
resolution searches upward rather than depending on your current
directory, `cd parsingChunking && python parse_and_chunk.py` works exactly
as well as `python parsingChunking/parse_and_chunk.py` from the root.
Whichever's more convenient in the moment.

## Why SEC EDGAR

- Free, public, no auth, no scraping gray area — safe to publish alongside
  your project on GitHub.
- Structured metadata (CIK, accession number, filing date, form type) comes
  for free with every filing, which is exactly what you need for chunk-level
  metadata later (ticker / fiscal period / doc type filtering at retrieval
  time).
- It's the same data source a real fintech company would use, so it reads as
  a credible choice in an interview, not a toy dataset.

## Why this scope

10 tickers across 6 sectors, 10-K + 10-Q since 2022 (~60-80 documents):

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT |
| Financials | JPM, GS |
| Healthcare | JNJ, PFE |
| Energy | XOM |
| Consumer staples | WMT, KO |
| Automotive / industrial | TSLA |

Sector diversity matters because it lets your eval set include genuinely
useful RAG questions: single-document lookups ("what was AAPL's Q2 FY24
gross margin"), cross-period trend questions ("how has JPM's net interest
margin changed since 2022"), and cross-company comparisons ("compare
capex as % of revenue between XOM and TSLA") — the last two are what
actually stress-test retrieval quality, not just single-doc lookup.

Adjust `config.yaml` freely if you want to narrow scope later (fewer
tickers, shorter date range) for faster iteration on chunking/retrieval,
then widen again once that's solid. Note that widening scope increases
download time (SEC's rate limit is the binding constraint, not your
machine).

## Setup and run

This needs internet access, so run it locally, not in a sandboxed
environment:

```bash
pip install -r requirements.txt
```

Edit `config.yaml` and replace `your_email@example.com` with a real email —
SEC EDGAR requires a descriptive `User-Agent` with real contact info and
will block generic ones.

```bash
python data/build_dataset.py
```

Expect it to take a few minutes given the rate-limit delay (0.15s between
requests, well under SEC's 10 req/sec cap) across ~10 companies' filing
histories.

## Output structure

```
data/raw/
  AAPL/
    10-K_2023-09-30.htm
    10-K_2023-09-30.htm.meta.json
    10-Q_2024-03-30.htm
    10-Q_2024-03-30.htm.meta.json
    ...
  MSFT/
    ...
  manifest.json          <- flat index of every filing + its metadata
```

Each `.meta.json` sidecar looks like:

```json
{
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "cik": 320193,
  "form_type": "10-K",
  "fiscal_period_end": "2023-09-30",
  "filing_date": "2023-11-03",
  "accession_number": "0000320193-23-000106",
  "source_url": "https://www.sec.gov/Archives/edgar/data/...",
  "local_path": "data/raw/AAPL/10-K_2023-09-30.htm",
  "downloaded_at": "2026-08-31T12:00:00+00:00"
}
```

`company_name` comes from the `title` field in SEC's own ticker->CIK
mapping (`company_tickers.json`) -- free in the exact same API response
already being fetched for the CIK lookup, just not captured in an earlier
version of this script. It matters downstream for two reasons: a query
router/entity-extraction step can match on how people actually refer to a
company in conversation ("Apple") without first needing to resolve that to
a ticker symbol, and citations/retrieved-source display read better as
"Apple Inc." than "AAPL". `ticker` remains the field to use for exact
payload filtering in Qdrant (unambiguous, no casing/formatting variance);
`company_name` is for the natural-language-facing side of the pipeline.

This metadata schema is what gets carried through to chunk-level metadata
in the ingestion pipeline (ticker / doc_type / fiscal_period filters at
retrieval time).

## Parsing and chunking

Once `data/raw/` is populated, run:

```bash
python parsingChunking/parse_and_chunk.py
```

This is a hand-built parser (`dom_walker.py` + `table_cleaner.py`), not a
document-parsing library -- built and tuned against a real downloaded
filing rather than assumed to work out of the box. Two concrete problems
showed up when inspecting an actual Apple 10-Q and are explicitly handled:

- **Heavy layout padding in tables.** SEC filings generated by Workiva pad
  financial tables with `colspan`-based spacer cells for visual alignment.
  In the income statement table used to build this, **76% of cells were
  empty** (233 of 306). `table_cleaner.py` reconstructs the full grid
  respecting `colspan`, then drops columns that are empty across every row
  (24 columns collapsed to 5 in that example).
- **Currency symbols split into their own cell.** A row showing
  `Products | $ | 104,429` has the `$` and the number in separate cells.
  The cleaner merges a standalone `$` cell into its right-hand neighbor
  row-by-row (so it becomes `Products | $104,429`), then re-runs the
  empty-column drop since the source `$` column is now blank.
- No real `<h1>`-`<h4>` tags exist in these filings -- section headers
  (`Item 1. Financial Statements`, `Note 2 -- Revenue`, etc.) are plain
  bold `<span>` elements. `dom_walker.py` detects a heading as a `<div>`
  whose entire text content comes from bold spans (not just partially
  bold, to avoid misfiring on bold words inside normal paragraphs), and
  tracks a 2-level section path (PART/Item, then sub-heading) that gets
  attached to every chunk.
- Hidden inline-XBRL tagging blocks (`style="display:none"`) and pure
  layout tables that clean down to empty content (cover-page address
  blocks, spacer rows) are filtered out rather than becoming empty chunks.

Prose text is grouped by section and split with LangChain's
`RecursiveCharacterTextSplitter.from_huggingface_tokenizer`, token-budgeted
against whatever embedding model you configure (`embedding_model_id` /
`max_tokens` in `config.yaml`). Tables are always kept as one atomic chunk
-- never split mid-row.

**Defensive hard-cap on text chunk size.** Running this against the full
137-filing corpus surfaced real oversized chunks: a warning like `Token
indices sequence length is longer than the specified maximum sequence
length for this model (678 > 512)`, and manual inspection confirmed actual
output chunks up to ~700 tokens despite `max_tokens: 512`. Root cause
traces to documented, currently-open bugs in LangChain's HF-tokenizer-based
`RecursiveCharacterTextSplitter` around `chunk_overlap` handling
(langchain-ai/langchain#34804, #30184) -- rather than depend on pinning
down the exact internal mechanism, `enforce_max_tokens()` re-verifies the
actual token count with our own tokenizer after the splitter runs and
hard-splits at the token level (bypassing the splitter's separator/merge
logic entirely, so it's correct regardless of the upstream bug) if
anything still exceeds budget. Every chunk record now also carries
`token_count`, computed directly rather than estimated, so this is
directly checkable from the JSONL going forward instead of needing to
reverse-engineer it from character counts.

First run downloads the tokenizer for `embedding_model_id` from Hugging
Face, so it needs internet access the first time (cached locally after
that).

It's idempotent -- filings that already have a `.chunks.jsonl` output are
skipped on rerun. Note: if you ran this before the `enforce_max_tokens` fix
was added, delete and re-run the affected `.chunks.jsonl` files (idempotent
skip means it won't reprocess them on its own).

**What's been verified vs. not:** the DOM-walking, table-cleaning, and
defensive token-cap logic were tested directly -- the DOM/table logic
against a real downloaded 10-Q with manual line-by-line inspection, and
`enforce_max_tokens` with an integration test that reproduces the exact
678-token scenario observed on the real corpus run and confirms it's
correctly split with zero data loss. The LangChain splitter itself is used
as-is from its documented API; its output is verified rather than trusted,
which is what caught the sizing bug above in the first place.

**Percent-sign merge fix:** year-over-year "Change %" comparison tables
(common in Item 7 MD&A -- e.g. "Products and Services Performance", "Gross
Margin", "Effective Tax Rate") split the `%` symbol into its own cell,
mirroring the `$`-splitting problem but trailing instead of leading
(`7 | %` instead of `7%`). Affected roughly 13% of table chunks in a
sampled 10-K (7 of 55), clustered entirely in Item 7. Root cause,
confirmed against three separate tables: the percentage number always
sits in a colspan=2 cell (right-aligned per the numeric rule above)
immediately followed by a standalone colspan=1 `%` cell -- so
`merge_percent_columns()` merges a standalone `%` into its LEFT neighbor
(mirror image of `merge_dollar_columns()`, which merges a standalone `$`
into its RIGHT neighbor, since `$` is a leading prefix and `%` is a
trailing suffix). Verified: zero standalone `%` cells remain across all
55 tables in the sampled 10-K after the fix, down from 7.

**Row-group splitting for oversized tables:** large multi-year financial
statements routinely exceed the 512-token budget -- some tables in this
project's real corpus ran past 1400 tokens. The original design kept
tables atomic (never split, to avoid breaking a row) and left oversized
ones to be truncated at embedding time -- but for a 1400-token table
truncated to ~504, that meant silently discarding almost two-thirds of
its content, with zero way to retrieve the discarded part via search (the
embedding vector has no information about it at all, not even
approximately). Replaced with proper splitting: `detect_header_row_count()`
identifies leading rows with no numeric data (period-date headers,
section titles like "Net sales:" -- reusing the same `_NUMERIC_CELL`
classifier behind the right-alignment fix) and `split_grid_into_row_groups()`
packs data rows into multiple chunks, repeating those header rows at the
top of each group so column/period meaning survives the split, splitting
only *between* rows, never mid-row. Verified against a real 32-row, ~727-
token income statement (3 fiscal years) across several budget sizes: every
original row appears exactly once across the resulting groups (automated
equality check, not eyeballed), and every group stays within budget.

**The remaining edge case -- a single row too large even with its
header.** Initially checked how real this risk was on Apple's 10-K alone:
across 604 non-empty rows, the largest was ~183 words, from a narrative
audit discussion, not a numeric table -- suggesting the case might be
rare-to-nonexistent. That hypothesis didn't survive contact with the rest
of the corpus: running the full 10-ticker set surfaced it repeatedly on
JPM, KO, and PFE. Two distinct real patterns emerged (found by scanning
actual flagged rows, not guessed at):

1. **Genuinely wide tables** (e.g. JPM's "Markets revenue": Fixed Income
   Markets / Equity Markets / Total Markets reported for 2025, 2024, and
   2023 side by side -- 22 columns in one row) -- a structurally different
   problem than "too many rows"; row-splitting can't help here, since
   every row is oversized regardless of how few are packed together, the
   problem is column count. **Status: fixed.**
   `detect_period_column_groups()` finds repeating period groups anchored
   on bare year labels ("2025", "2024", "2023" -- deliberately distinct
   from `detect_header_row_count()`'s date handling, since a bare year
   matches `_NUMERIC_CELL` and wouldn't be caught as a header by that
   function). `split_grid_into_column_groups()` then splits the table into
   one complete, correctly-labeled sub-table per period -- column 0 (the
   row label) is repeated in every group, so each piece stands alone as a
   full "2025 Markets revenue breakdown" rather than a column slice
   missing its labels. This runs as a pre-step before row-splitting, since
   a table can be both wide AND tall: each resulting narrower group still
   goes through `split_grid_into_row_groups()` afterward if it alone is
   still oversized. Verified against two independent real tables -- JPM's
   original case, and (as a generalization check, not overfitting to one
   example) Apple's differently-shaped "Products and Services Performance"
   table -- both producing an exact multiset match between original and
   reconstructed data cells (excluding the intentionally-repeated label
   column): no loss, no duplication, not eyeballed. Full end-to-end
   integration verified too: JPM's real table run through the actual
   `process_filing()` function produced 12 correctly-linked chunks (one
   `table_id`, sequential `group_index`), none needing the
   `row_too_large` fallback.
2. **A narrative explanation crammed into one cell of an otherwise short
   row** (e.g. Pfizer's product-revenue table: five short numeric cells
   followed by one cell containing a full paragraph explaining the
   revenue driver; Coca-Cola's "Critical Audit Matters" hit this every
   year 2021-2025). **Status: fixed.** `split_oversized_row()` finds the
   dominant cell in an oversized row and splits ONLY that cell's content
   (`_split_text_into_pieces()`, preferring bullet-point boundaries, then
   sentence boundaries, then a hard word-count split as a last resort),
   producing multiple sub-rows that each repeat the row's other short
   cells alongside just one piece of the narrative -- so context (which
   product, which numbers) isn't lost. `split_grid_into_row_groups()` runs
   this as a pre-processing pass before packing rows into groups. Verified
   against the real Pfizer Paxlovid row: split into 4 sub-rows, and an
   automated check confirms the narrative content reconstructs word-for-
   word across those sub-rows with zero loss or duplication -- not just
   eyeballed. This is now a permanent regression test in
   `table_cleaner.py`.

   **A related bug this surfaced**: Pfizer's narrative cell happened to
   use bullet points, so the sentence-boundary fallback path in
   `_split_text_into_pieces()` went untested by that example. Testing a
   *different* real "row too large" warning -- Coca-Cola's "Critical
   Audit Matters" text, continuous prose with no bullets -- found the
   naive sentence-splitting regex incorrectly treated abbreviations like
   "U.S." as sentence ends, breaking a real sentence in half ("...the
   U.S." | "Tax Court issued an opinion..."). No data was lost (the
   word-for-word check still passed), but the split point was wrong,
   which would degrade retrieval/generation quality on any prose
   containing common abbreviations. Fixed with `_split_into_sentences()`:
   skips a candidate split point when the word immediately before the
   period is very short (<=2 letters -- catches "U.S.", "U.K.", "Mr",
   "Dr" without needing each one individually listed) or matches a small
   list of common longer abbreviations the length check alone wouldn't
   catch ("Inc.", "Corp.", "etc."). Verified against the real KO row:
   "the U.S. Tax Court...for tax years 2007 through 2009." now stays
   together as one sentence, confirmed by an explicit check that no
   sub-row ends mid-sentence at "U.S." -- also now a permanent regression
   test.

With both patterns fixed, `parse_and_chunk.py` still prints an unmissable
`[ROW TOO LARGE]` warning naming the ticker/section/token count for the
residual case where a row remains oversized even after both fixes are
attempted (a dominant cell with no usable split boundary, or some
not-yet-seen third pattern), every table chunk carries a queryable
`row_too_large` boolean, and `check_token_limits.py` summarizes these
across the whole corpus after a run -- so the true remaining extent
across all 10 tickers is known, not assumed, and any genuinely new
pattern still surfaces loudly rather than silently degrading.

*A note on process, in the interest of being straightforward about it:
the narrative-cell fix above was built in an earlier working session but
wasn't actually verified or delivered at the time -- it existed as
untested code that never made it into the files being handed over. Caught
and fixed by re-checking the actual file contents against what had been
delivered, rather than assuming prior described work was complete.*

**Sibling-linking metadata (`table_id` / `group_index` / `group_count`):**
splitting an oversized table into multiple chunks solves the embedding
blind-spot problem (each piece gets an accurate, untruncated vector) but
introduces a different loss if left unaddressed: at generation time, only
whichever single group matched the query gets retrieved -- the LLM never
sees the table's other groups, even though they're the same table, same
topic. The standard fix for this in RAG system design is "parent-document"
or "auto-merging" retrieval: search precisely at the individual-chunk
level, but at generation time reconstruct the full table by pulling in
sibling chunks. That merging logic belongs in the retrieval/query step
(not yet built), but the metadata it depends on is generated here, since
it comes directly from the splitting logic -- retrofitting it after the
whole corpus is already embedded and loaded would mean redoing this work.
Every table chunk (split or not) carries `table_id` (one per source
`<table>`, shared across all its pieces), `group_index` (0-based position
within that table), and `group_count` (total pieces, `1` for tables that
didn't need splitting) -- a uniform schema so retrieval code never needs
to special-case whether a given table was split. Verified with an
integration test: a small table correctly gets `group_count=1`; a large
one splits into multiple chunks all sharing one `table_id` with
sequential `group_index`; two different source tables get different
`table_id`s.

**Numeric right-alignment fix:** an earlier version of this cleaner left a
cosmetic misalignment where rows without a `$` prefix (e.g. "Services")
landed one column left of rows that had one (e.g. "Products"). Root cause:
Workiva represents a `$`-prefixed value as two separate colspan=1 cells
(`$`, then the number) but a plain value as a single colspan=2 cell of the
same total width -- and the original grid builder placed every cell's text
at the *start* of its span, which is correct for left-aligned labels but
wrong for right-aligned numbers. The fix: cells whose content is purely
numeric/currency-shaped (`$`, digits, `()`, `-`, `,`, `.`, `%`) are placed
at the *end* of their span instead of the start; label and header cells
keep left-alignment. Verified against both the income statement and the
balance sheet table in the sample filing -- both now produce consistently
aligned columns regardless of which rows have a `$` prefix.

### Output structure

```
data/processed/
  AAPL/
    10-K_2023-09-30.chunks.jsonl
    10-Q_2024-03-30.chunks.jsonl
    ...
  JPM/
    ...
```

Each line in a `.chunks.jsonl` file is one chunk record:

```json
{
  "chunk_id": "chunk_435f1053beb0da90",
  "ticker": "AAPL",
  "company_name": "Apple Inc.",
  "cik": 320193,
  "form_type": "10-Q",
  "fiscal_period_end": "2021-12-25",
  "filing_date": "2022-01-27",
  "source_url": "https://www.sec.gov/Archives/edgar/data/...",
  "section": "Item 1. Financial Statements > CONDENSED CONSOLIDATED STATEMENTS OF OPERATIONS (Unaudited)",
  "chunk_type": "table",
  "token_count": 187,
  "row_too_large": false,
  "table_id": "chunk_a1b2c3d4e5f6a7b8",
  "group_index": 0,
  "group_count": 1,
  "text": "..."
}
```

`row_too_large` is only meaningful for `chunk_type: "table"` -- `true`
only when a single row still exceeds `max_tokens` even after
`split_oversized_row()` has tried to fix it (currently: the wide-table
case, not yet handled, or the rare residual where a dominant cell has no
usable split boundary). `false` for every normal chunk, including ones
from a successfully-split oversized table -- that's the expected, working
path, not the edge case.

`table_id` / `group_index` / `group_count` (also table-only) identify
which source `<table>` a chunk came from and its position among that
table's pieces -- present and uniform (`group_count: 1` for tables that
didn't need splitting) so a future retrieval step can reconstruct a full
table from its parts regardless of whether splitting happened. See
"Sibling-linking metadata" above for why this exists now rather than
being deferred to the retrieval step.

## Embedding

Once `data/processed/` has chunks, start the embedding model with vLLM
(separate terminal, needs a GPU for reasonable speed though it'll run on
CPU too):

```bash
vllm serve BAAI/bge-large-en-v1.5 --task embed
```

Then run:

```bash
python embedding/embed_chunks.py
```

This calls vLLM's OpenAI-compatible `/v1/embeddings` endpoint directly over
HTTP, sending raw text strings -- deliberately **not** using
`langchain_openai.OpenAIEmbeddings`, which pre-tokenizes text client-side
with `tiktoken` (OpenAI's tokenizer) before sending integer token IDs to
the server. Since bge-large-en-v1.5 uses a BERT vocabulary, not tiktoken's,
that produces a real, documented `"Token id ... is out of vocabulary"`
error against this exact model. Sending raw strings lets vLLM tokenize
with the model's own correct tokenizer server-side, avoiding the mismatch
entirely.

**Oversized chunks:** bge-large-en-v1.5 has a hard 512-token limit, and
vLLM errors rather than silently truncating. Tables are now split into
row-groups at *parse* time when oversized (see the parsing section above)
rather than left atomic and truncated here -- so this truncation path is
now a rare fallback, not the primary mechanism, only hit by the edge case
of a single row too large to fit even with its header. When it does fire,
only the *embedding* is computed from the truncated version; the record's
`text` field keeps the full original, so retrieval still finds the chunk
via an imperfect-but-usable vector while generation sees the complete
row. Each output record has an explicit `embedding_truncated` boolean so
this stays visible rather than silent.

**Special-token buffer, found from a real production failure (not
anticipated in advance):** running against the full 137-filing corpus
produced `0` chunks embedded despite the run completing -- every request
failed with `400 Bad Request`, and it finished suspiciously fast (every
call rejected near-instantly, not actually running inference). Root
cause: the length check originally counted tokens without
`add_special_tokens=False`, inconsistent with `parse_and_chunk.py` (which
always passes it explicitly) -- for a BERT-family tokenizer, the default
silently adds `[CLS]`/`[SEP]` (2 extra tokens). But the deeper issue
wasn't just local miscounting: vLLM's *server* also adds its own
`[CLS]`/`[SEP]` when tokenizing raw text before checking it against the
512-token limit. Since `parse_and_chunk.py` targets exactly 512 content
tokens per chunk (the configured `chunk_size`), nearly every chunk sat
right at that boundary -- so once the server added its own 2 specials on
top, virtually everything exceeded the true limit and got rejected,
regardless of whether this script's own check had flagged it as
oversized. Fixed by reserving an 8-token safety buffer
(`SPECIAL_TOKEN_BUFFER`) below `max_tokens`, applied uniformly to every
chunk rather than only ones already far over budget. Also fixed:
`embed_batch()` previously discarded the HTTP response body on error
(`resp.raise_for_status()` only gives a bare status code) -- it now
surfaces vLLM's actual diagnostic message, which is what makes failures
like this actually debuggable instead of requiring inference from
symptoms alone.

Output mirrors `data/processed/`'s structure under `data/embedded/`, same
idempotent-skip behavior as the earlier stages.

**What's been verified vs. not:** the batching, response-index-sorting
(vLLM's embedding responses aren't guaranteed to preserve request order --
verified this matters by testing against a deliberately shuffled mock
response), truncation, and idempotency logic were all tested against a
mocked vLLM server, since no GPU/network is available in the dev sandbox
to run a real one. The actual HTTP contract with a live vLLM server has
not been exercised end-to-end. Run it against a small batch first (a
single ticker) and spot-check a few output records before processing the
full corpus.

### Output structure

```
data/embedded/
  AAPL/
    10-K_2023-09-30.chunks.jsonl
    ...
```

Each line is a chunk record plus two added fields:

```json
{
  "chunk_id": "chunk_435f1053beb0da90",
  "ticker": "AAPL",
  ...
  "chunk_type": "table",
  "text": "...",
  "embedding": [0.0123, -0.0456, ...],
  "embedding_truncated": false
}
```

## Next step

Load these embedded chunks into Qdrant, with the metadata fields (ticker /
form_type / fiscal_period_end at minimum) set up as filterable payload
fields -- that's what makes retrieval precise later.
