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
    visualize_embeddings.py
    load_qdrant.py
    .loaded_files.json  <- created by load_qdrant.py, gitignored
  retrieval/
    retrieve.py
  eval/
    golden_questions.json
    run_retrieval_eval.py
    diagnose_misses.py
    generate_batch.py
  generation/
    generate.py
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

### Pagination bug fix -- a real, confirmed bug that silently under-downloaded 4 of 10 tickers

A real production run of this project's corpus surfaced a significant,
confirmed bug: `manifest.json` showed AAPL/JNJ/KO/MSFT/PFE/TSLA each with
a complete ~19 filings spanning the full 2022-2026 range, while **JPM and
GS only had their most recent 4 filings each, XOM only had 1, and WMT was
missing its earliest ~4-5 filings from 2022** -- all four gaps at the OLD
end of the date range, no errors, the script completed "successfully"
every time.

Root cause: SEC's `submissions.json` endpoint only holds "at least one
year's...or 1,000...whichever is more" of a filer's most recent filings
of **all** form types (10-K, 10-Q, 8-K, proxy statements, everything) in
its primary `filings.recent` block -- not just the form types this
project cares about. A filer that submits heavily in other forms (major
financial institutions and large multinationals especially -- exactly
JPM, GS, and XOM) can have its older 10-K/10-Qs pushed entirely out of
that window, into supplementary paginated JSON files
(`filings.files`, each pointing to a
`CIK##########-submissions-NNN.json` page) that an earlier version of
this script never checked. `filter_filings()`'s own docstring literally
said "Flattens the 'recent' filings block" -- it never looked anywhere
else.

Fixed with `get_all_filing_pages()`: fetches the primary `recent` block
plus every page listed in `filings.files`, respecting the same rate
limit for each extra request, before `filter_filings()` filters across
all of them uniformly. Verified against two mocked scenarios: one
mirroring the exact confirmed bug pattern (a 4-filing "recent" block plus
a paginated page holding 5 older filings) -- correctly recovers all 9,
including the oldest filing that only exists on the paginated page; and
the normal case (no `files` key, or an empty one, matching AAPL/JNJ's
real behavior) -- confirmed zero extra requests made and identical
results to before, so this fix only activates for filers that actually
need it.

**This means a fresh full re-download is needed** for the corpus to
actually reflect complete history for JPM, GS, XOM, and WMT -- delete
`data/raw/` and re-run `build_dataset.py`. Given the parsing fixes
found on Tesla and Microsoft (see "Parsing and chunking" below) already
required a full pipeline re-run, combine both into one: fresh
`build_dataset.py` → `parse_and_chunk.py` → `embed_chunks.py` →
wipe-and-reload into Qdrant, rather than doing this twice.

### A second, genuinely different real finding: XOM spans two CIKs

Re-running with the pagination fix above completely resolved JPM, GS, and
WMT (all three now show the complete, expected 19 filings spanning
2022-2026, verified against a fresh `manifest.json`). **XOM was still
stuck at exactly 1 filing** -- and critically, the script didn't crash;
it kept going and correctly downloaded the other tickers, which ruled out
a repeat of the pagination bug and pointed at something specific to XOM.

Root cause, confirmed via current news, not guessed at: **ExxonMobil
completed a "redomiciliation merger" on 2026-07-01**, reorganizing from
"Exxon Mobil Corporation" (a New Jersey corporation, CIK `34088`) into a
brand-new legal entity, "ExxonMobil Holdings Corporation" (a Texas
corporation, CIK `2115436`), which now trades under the same "XOM"
ticker. `company_tickers.json`'s current ticker→CIK mapping necessarily
points to the *new* entity -- which, being genuinely new, has almost no
filing history of its own yet. The pagination fix was working correctly;
there was simply nothing to paginate into, because the CIK it was
fetching legally didn't exist before July 2026. All of XOM's real
2022-2026 history sits under the *old*, now-superseded CIK, which the
current ticker mapping no longer surfaces at all.

Confirmed directly against the actual manifest: the single XOM entry
recorded `cik=2115436`, matching the new post-merger entity exactly.

Fixed with a `cik_overrides` config entry (`config.yaml`): lets a ticker
map to a list of CIKs instead of one, merging their filing histories.
XOM is configured with both `34088` (pre-merger) and `2115436`
(post-merger, current). `filter_filings()` now tags each result with its
source CIK (since a single ticker can span more than one), and
`download_filing()` reads that per-filing CIK when building the archive
URL, rather than assuming one CIK per ticker. Verified with a scenario
mirroring the real case exactly: an old-CIK page holding 2022-2023
filings merged with a new-CIK page holding the single 2026 filing --
confirmed the merge produces all 3 filings with correct per-filing CIK
tracking, and that each one's downloaded archive URL correctly uses its
*own* CIK rather than a single shared one. Also verified the
non-overridden case (e.g. AAPL) still resolves to its single current CIK
exactly as before, and that a ticker list falls back correctly if
`cik_overrides` isn't present in config at all -- zero behavior change
for the 9 tickers that don't need this.

**Worth remembering for any future ticker additions**: a ticker symbol
can persist across a corporate restructuring (merger, redomiciliation,
holding-company reorganization) while its underlying CIK does not --
"which CIK does this ticker currently map to" and "does that CIK's
history cover the full date range I want" are two different questions,
and this project's very first assumption (one ticker = one stable CIK)
turned out to not always hold.

### A third finding on the re-run: a cross-listed filing needed deduplication

Re-running with both fixes above brought XOM to 20 filings (19 expected,
plus one) -- one more than the clean baseline, worth checking rather than
assuming success. Inspecting XOM's entries directly found the actual
cause: XOM's Q2 2026 10-Q (accession `0000034088-26-000093`, filed
2026-08-03 -- shortly after the July 1 redomiciliation merger) appeared
under **both** CIKs with otherwise identical data (same accession number,
same period, same filing date). This makes sense given the timing: EDGAR
cross-lists a filing like this under both the predecessor and successor
CIK around a restructuring, for continuity.

Without deduplication, this filing would get "downloaded" twice under the
same output filename -- confirmed via the manifest that both entries
pointed to the exact same `local_path`, meaning the second write silently
overwrote the first (no actual duplicate file on disk), but left two
manifest entries and an ambiguous, order-dependent CIK recorded in the
surviving `.meta.json` sidecar (whichever CIK happened to process last).

Fixed by deduplicating merged filings by `accession_number`, keeping the
first occurrence -- since `cik_overrides` lists the predecessor CIK
before the successor, this naturally attributes a cross-listed filing to
whichever CIK's accession-number prefix it actually originated from.
Verified against the exact real scenario from the manifest: 3 filings in
(one unique, one cross-listed pair) → 2 out, with the surviving entry
correctly keeping the predecessor CIK (`34088`, matching the accession
number's own prefix) rather than whichever CIK happened to be processed
last. Scoped to only run when a ticker actually has multiple CIKs
configured, so this adds zero overhead or behavior change for the other
9 tickers.

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

**`<p>`-vs-`<div>` content-loss fix, found on Tesla's filings -- a
different class of bug than everything above.** Every fix up to this
point addressed table *formatting* quirks. This one is structural, and
much bigger in impact: Tesla's filings are prepared by DFIN (Donnelly
Financial Solutions / ActiveDisclosure), not Workiva like every other
filing this pipeline had been built and verified against. `dom_walker.py`
only ever checked `<div>` elements for heading/text content -- but DFIN
puts the vast majority of actual prose in `<p>` tags instead. Measured on
a real Tesla 10-K: **440,774 characters of real content sat in `<p>` tags
versus only 62,946 in `<div>`** -- meaning the pipeline was silently
missing roughly 87% of the document's text, not just section headings.
Confirmed concretely before fixing: `build_records()` produced 128 blocks
with **zero** detected sections and only 21 text blocks on this file.
Root cause, found by inspecting the actual markup around a known heading
("PART I."): DFIN wraps section headings in `<p>` tags with the bold span
nested inside an `<a>` (anchor/bookmark) link, a structure `walk_blocks()`
never even looked at.

Fixed by generalizing `classify_div()` into `classify_block()`, called on
both `<div>` and `<p>` elements in `walk_blocks()` rather than `<div>`
only. Since a leaf classification short-circuits recursion into that
element's children, a `<p>` nested inside an already-classified `<div>`
(or vice versa) is captured once as part of the outer element's text, not
double-counted -- verified no duplication risk from the two tag types
nesting either way. After the fix, the same Tesla 10-K produced 357
blocks with **all 357** carrying a detected section and 366,507 characters
of captured text -- both now permanent regression assertions in
`dom_walker.py`'s test suite (loose bounds rather than exact figures, so
they're not brittle to minor future heuristic changes, but would catch a
full regression back to "near-zero headings, ~63K chars captured").
Verified zero regression on every previously-validated Workiva-based
filing (AAPL 10-Q, AAPL 10-K, JPM 10-K, KO 10-K) -- the AAPL 10-Q's block
count matched the pre-fix baseline exactly (107 blocks, 72 text, 35
table), confirming `<p>`-tag handling only activates where it's actually
needed. `table_cleaner.py`'s colspan-based grid/merge logic required no
changes at all -- verified against a real Tesla revenue table (multi-year
comparison with `$`/`%` columns) and it worked correctly unmodified,
confirming the earlier fixes generalize to a different filing platform's
table conventions, not just Workiva's specifically. This is exactly the
kind of gap that stays invisible until a filing from a different
preparer is actually tested -- worth remembering if this project scales
to more tickers, since other filers may use yet other preparers with
their own conventions.

**Bold-on-element-itself fix, found on Microsoft's filings -- a second,
distinct heading-detection gap from Tesla's, from the SAME filing
preparer.** Microsoft's filings are also DFIN-generated, so the `<p>`-tag
fix above was expected to be directly relevant here -- confirmed: both
files are ~85-90% `<p>`-based, same as Tesla. But even after that fix,
`classify_block()` still detected zero `Item`/`PART` headings on MSFT's
filing (heading text like `"Debt"` and `"Intelligent Cloud"` came through
as sub-headings, but nothing above them). Root cause, found the same way
as the Tesla bug -- tracing the real markup around a known heading
(`"ITEM 1. BUSINESS"`): Microsoft's DFIN template applies bold styling
**directly on the `<p>` element's own `style` attribute**
(`<p style="...font-weight:bold...">TEXT</p>`), not via a nested
`<span>` like Workiva and Tesla's DFIN template both use.
`classify_block()` only ever checked descendant spans for boldness, never
the container element itself, so this pattern was invisible to it.

This confirms something worth remembering even within a single filing
preparer: Tesla and Microsoft are both DFIN-generated, yet use two
different internal heading conventions (nested bold span inside an anchor
link vs. bold styling directly on the container) -- "check which
preparer generated this filing" isn't sufficient on its own; the specific
markup pattern still needs verifying per filing. Fixed by also calling
`is_bold_span()` (a generic style-string check, despite its name) on the
container element itself, not just its descendant spans -- only an
explicit non-bold override in a nested span disqualifies a
bold-at-the-container-level element from being treated as a heading.
Verified: MSFT's 10-Q went from 175/264 (66%) to 335/335 (100%) blocks
with a detected section, and the 10-K from 368/400 (92%) to 541/541
(100%) -- both now with proper `"ITEM 1. BUSINESS > subsection"`
hierarchical structure rather than bare, unparented subsection titles.
Zero regression confirmed across all 8 previously-validated files (AAPL,
JPM, KO, WMT, TSLA x2 form types each) -- every one produced block counts
identical to their pre-fix baselines. `table_cleaner.py` again required
no changes -- a real MSFT income statement table rendered correctly
unmodified.

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

### A real bug found downstream, in generation, traced back to a gap here: inline bold sub-headings

`classify_block()`'s heading detection (see its docstring above) only
fires when a block's entire text is <=150 characters AND entirely bold --
correctly designed for a standalone short heading block, like `<div><span
style="font-weight:700">Gross Margin</span></div>` followed by a
separate paragraph. But a different, real pattern showed up in Goldman
Sachs' 10-K MD&A: a bold lead-in phrase glued onto the front of a much
longer paragraph, in the SAME block --
`<div><span style="font-weight:700">Net Interest Income.</span>Net
interest income in the consolidated statements of earnings was...`
(well over 150 characters once the whole paragraph is counted).
`classify_block()` never even reached the bold-check for this block,
since it fails the length gate first -- so "Net Interest Income" was
never captured as a `section_l2` update, and every chunk in that
subsection inherited whatever heading had been detected *before* it
(the broader parent, "Net Revenues").

This surfaced as a real, concrete generation error, not a hypothetical:
a chunk's `section` metadata said "... > Net Revenues" when the chunk's
actual content was specifically the "Net Interest Income" subsection --
one component of net revenues, not the total. That mismatch materially
contributed to the generation model answering a "net revenues" question
using a "net interest income" figure (see "Generation" below for the
full incident). Confirmed non-rare via a corpus-wide count before
deciding whether to fix it at the source: **886 of 49,207 text chunks
(1.8%) match this pattern** -- roughly 4-5 per filing on average, not a
one-off specific to Goldman Sachs' formatting.

Fixed with `split_inline_subheading()`: detects the same
"Heading.heading in lowercase..." pattern (a short, capitalized phrase
ending in a period, immediately followed by a capital letter with no
space -- distinct from an ordinary sentence or an abbreviation like
"U.S.", both excluded by requiring 2-40 characters between the leading
capital and the period). `walk_blocks()` now applies this check to every
`('text', ...)` result: when it matches, a `('heading', ...)` is yielded
first, then the remaining text separately -- so a block that's
structurally two things (a subsection label + its content) is treated as
two blocks, exactly as if the original HTML had actually separated them,
feeding `section_l2` tracking correctly with no changes needed to
`parse_and_chunk.py` at all.

Verified thoroughly before trusting it: the exact real GS pattern
(confirmed splitting correctly into `('heading', 'Net Interest
Income')` + the remaining paragraph text), a regression check that the
*original*, already-working short-standalone-heading detection is
completely unaffected, an ordinary long paragraph with no bold lead-in
passing through unchanged, and false-positive avoidance (the "U.S."
abbreviation case, same as before). Then confirmed against the real
`GS_10-K_2025-12-31.htm` sample file directly (not just synthetic test
HTML): re-running `build_records()` on it now correctly labels the
Net Interest Income chunk's section as "... > Net Interest Income"
instead of the old "... > Net Revenues". All of this project's existing
`dom_walker.py` regression tests (Tesla's `<p>`-tag fix, Microsoft's
bold-on-element fix) still pass unchanged, confirming this is a purely
additive fix -- it only changes behavior for blocks matching the new
pattern, everything else is untouched.

**This needs a full pipeline re-run to take effect on the actual
corpus** -- section metadata for a real, non-trivial fraction of chunks
changes, the same "schema/content change = full re-run" pattern
established throughout this project:

```bash
rm -rf data/processed data/embedded
python parsingChunking/parse_and_chunk.py
python parsingChunking/check_token_limits.py
python embedding/embed_chunks.py
# then wipe + reload Qdrant, as done for every previous parsing fix
```

`generate.py`'s `extract_subheading()` patch (added as an immediate
generation-time fix before this deeper root cause was traced) stays in
place as a defensive backstop -- harmless and redundant once the corpus
is reprocessed (the section will already contain the subheading, and
`build_context_blocks()`'s duplicate-avoidance check will correctly
skip re-appending it), but still useful protection against any similar
not-yet-discovered pattern, or before the reprocessing is actually run.

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

## Embedding verification

Before trusting 41K+ embedded chunks going into Qdrant, `embedding/visualize_embeddings.py`
gives a real, evidence-based check of embedding quality rather than assuming
the embedding step "worked" just because it ran without errors.

```bash
pip install -r requirements.txt
python embedding/visualize_embeddings.py
```

Samples up to 300 chunks per ticker (full-corpus PCA/plotting is slow and
visually unreadable at ~41K points; a sample shows the same structure),
and produces two complementary kinds of evidence:

**A 2D visualization** (`embedding_space.png`) -- reduces the 1024
dimensions to 2 via PCA and plots three views: colored by ticker, by
chunk type (text/table), and by filing section (SEC Item number, e.g.
"Item 7" = MD&A, "Item 8" = Financial Statements -- extracted from the
`section` field via `extract_item_category()`, real deterministic
metadata already captured during parsing, not something requiring topic
modeling to obtain).

**A quantitative separation check**, because a 2D PCA plot alone is a
genuinely unreliable way to judge this: PCA's top 2 components can capture
as little as ~10% of total variance and can be dominated by one strong
structural signal, visually burying or exaggerating other structure.
`print_separation_scores()` instead computes **silhouette score** (the
standard metric for "how well-separated are these labeled groups")
directly on the full 1024-dimensional embeddings for each of ticker,
chunk_type, and filing section.

**A low full-space silhouette score is itself ambiguous** between two
different explanations: no real signal, or a real signal that's just
concentrated in a small subspace and diluted when averaged uniformly
across ~1000+ mostly-unrelated dimensions. To tell these apart, the script
also fits a **Linear Discriminant Analysis (LDA)** -- which explicitly
finds the direction(s) that *maximize* label separation, unlike PCA which
ignores labels entirely -- and compares. If LDA recovers meaningfully more
separation than the full-space score, that's real signal, just diluted.
If LDA is also low, the full-space result reflects genuine absence of
separation, not a measurement artifact.

### Bugs found while building this, fixed before trusting the output

Consistent with the rest of this project: verify before trusting, and when
a check itself might be wrong, test the check.

- **Wrong assumption about embedding normalization.** An earlier version
  of this script asserted "bge-large embeddings are NOT L2-normalized by
  default" without verifying it. Real data proved this false (every
  vector has norm exactly 1.0). Fixed to detect and report the actual
  normalization state from the data rather than asserting an assumption.
- **LDA overfitting gave a false positive.** An earlier version fit LDA
  and evaluated separation on the *same* data, with no train/test split.
  Tested against a synthetic case built to have zero real signal (pure
  noise, no label correlation anywhere) -- it produced an LDA score of
  +0.13, appearing to show real separation that didn't exist. Root cause:
  with ~1024 dimensions and often only a few hundred samples per group,
  LDA has enough freedom to find an apparently-separating direction
  purely by overfitting to noise. Fixed by fitting LDA on a train split
  and evaluating separation only on a held-out test split it never saw --
  re-verified against the same synthetic no-signal case: the false
  positive disappeared, while a genuine (but deliberately diluted) signal
  in a separate synthetic case was still correctly recovered.
- **Roman numeral capitalization bug** in `extract_item_category()`:
  Python's `.title()` mangled "PART IV" into "Part Iv". Fixed by using
  `.upper()` on the extracted identifier instead.

### Actual results on this corpus (2,854 sampled chunks, 10 tickers)

| Grouping | Full-space silhouette | LDA (held-out) | Verdict |
|---|---|---|---|
| Ticker | +0.0016 | +0.1299 | Ambiguous -- doesn't clear the recovery bar, but not confidently "no signal" either |
| Chunk type (text/table) | +0.0375 | +0.6674 | Real, substantial signal -- concentrated in a small subspace, diluted across the full space |
| Filing section (Item #) | -0.0518 | -0.0044 | Genuine absence of separation -- confirmed, not a measurement artifact |

**Chunk type**: confirmed real. The embedding meaningfully distinguishes
tabular from narrative content, just not along the dominant global
variance directions -- expected and fine, since most of the embedding's
capacity is spent on actual semantic content, not a structural surface
feature.

**Filing section**: confirmed absent, and that's fine -- separate the
empirical result from the design argument for why. SEC Item boundaries
are a legal/structural categorization, not a tight semantic one (Item 7
alone spans revenue trends, liquidity, market risk, segment performance
-- genuinely different sub-topics under one regulatory label). The
embedding doesn't need to redundantly re-learn this boundary: `section`
already exists as exact, structured metadata for precise Qdrant payload
filtering. An embedding space that rigidly clustered by Item number could
even hurt retrieval that legitimately needs to span sections (e.g. a
"supply chain risk" query relevant to both Item 1A and Item 7).

**Ticker**: genuinely ambiguous rather than forced into a clean verdict --
the LDA score doesn't clear the recovery threshold, so there may be a
mild recoverable signal (company/product vocabulary) or there may not be.
Doesn't matter practically either way: ticker-based clustering in the
embedding space was never something needed for retrieval, since Qdrant
filters on `ticker` exactly.

**Bottom line**: real, verifiable structure where it's wanted (content
type), no false structure imposed where it isn't needed (ticker), and a
confirmed absence exactly where an alternative mechanism (exact metadata
filtering) already compensates (filing section). Nothing here blocks
using this embedded corpus in Qdrant.

## Loading into Qdrant

**Run Qdrant natively inside WSL2 first -- not Docker with a Windows bind
mount.** Qdrant's own troubleshooting docs confirm a real, documented data
corruption risk with that specific combination: "When you mount Windows
folder into Qdrant docker container, the Windows hypervisor creates a
shared mount, which is not fully POSIX-compatible," with a described
failure mode of "vector data will be lost (set to all zeros) after
service restart." A v1.16.2 changelog entry ("Fix Docker/WSL on Windows
with bind mount corrupting storage") confirms this is a real, tracked bug,
not hypothetical caution. Running the native Linux binary inside WSL2,
with storage on WSL2's own ext4 filesystem (not `/mnt/c/...`), avoids the
whole failure class entirely -- consistent with how vLLM is already set
up (native process in WSL2, not containerized):

```bash
QDRANT_VERSION=$(curl -s https://api.github.com/repos/qdrant/qdrant/releases/latest | grep tag_name | cut -d '"' -f 4)
wget https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
mkdir -p ~/qdrant && mv qdrant ~/qdrant/qdrant
mkdir -p ~/qdrant/storage
cd ~/qdrant && ./qdrant
```

Verify no filesystem-compatibility WARN/ERROR appears in the startup log,
and that `curl http://localhost:6333/collections` responds. Your project
files can stay on the Windows drive as normal -- only Qdrant's own
storage needs to live on WSL2's native filesystem; `load_qdrant.py`
reaches it over the network client regardless of which side its data
lives on.

Then:

```bash
pip install -r requirements.txt
python embedding/load_qdrant.py
```

### Why point IDs are UUIDv5, not the chunk_id string directly

Qdrant only accepts 64-bit unsigned integers or proper UUIDs as point IDs
-- arbitrary strings (like `"chunk_a1b2c3d4..."`) are rejected outright.
Worse, a community report found that a hex string *coincidentally* the
right length gets silently reinterpreted by Qdrant as a UUID in a
different canonical form than intended -- not something to stumble into
by luck. `point_id_for()` uses UUIDv5 (RFC 4122, name-based, under a
fixed namespace) to deterministically derive a valid UUID from each
`chunk_id`: the same input always produces the same UUID, which is what
makes re-running this script idempotent at the *point* level -- it
upserts (updates) the same point rather than creating a duplicate with a
fresh random ID.

### Idempotent per source file, not just per point

A local manifest (`embedding/.loaded_files.json`, gitignored -- regeneratable)
tracks which `.chunks.jsonl` files have already been fully upserted, so
re-running after adding new tickers/filings only loads what's new rather
than re-upserting the full ~41K-point corpus every time. Saved
incrementally after each file completes, so a crash partway through
doesn't lose progress on files already done.

### Collection setup

Vector size is inferred from the actual data (the first not-yet-loaded
record's embedding length) rather than hardcoded, so it can't silently go
stale if the embedding model ever changes. Distance metric is Cosine,
matching bge-large-en-v1.5's confirmed L2-normalized output (see
"Embedding verification" above -- every vector has norm exactly 1.0).

Payload indexes are created for every field retrieval will need to filter
on precisely:

| Field | Type | Why |
|---|---|---|
| `ticker`, `company_name`, `form_type`, `chunk_type`, `section`, `table_id` | keyword | Exact-match filtering |
| `cik`, `group_index`, `group_count` | integer | Exact match + range |
| `fiscal_period_end`, `filing_date` | datetime | Proper date-range filtering (e.g. "filings since 2023"), without manually converting to unix timestamps -- Qdrant's native `datetime` index type (v1.8.0+) handles RFC 3339 directly |
| `row_too_large`, `embedding_truncated` | bool | Lets retrieval/eval code account for known imperfect chunks if needed |

`fiscal_period_end`/`filing_date` get converted from plain `"YYYY-MM-DD"`
strings to full RFC 3339 (`"YYYY-MM-DDT00:00:00Z"`) specifically for the
indexed payload, rather than relying on how strictly a bare date parses
under a given client/server version.

**What's been verified vs. not:** `point_id_for()`, `to_rfc3339()`, and
the manifest save/load logic were unit tested directly. The full
collection-creation, payload-index, upsert, idempotent-skip, and
incremental-load control flow was verified end-to-end against a realistic
mock of the Qdrant client (tracking actual calls made, not just "did it
run") -- confirmed: correct vector size/distance inferred from real data,
payload correctly excludes the redundant `embedding` field, dates convert
correctly, a rerun with nothing new makes zero upsert calls, and adding a
new ticker's file after an initial load correctly uploads only the new
file while skipping the already-loaded one. The actual network calls
against a live Qdrant server have not been exercised (no server reachable
from the dev sandbox) -- run this against a small batch first and
spot-check a point or two (e.g. via the Qdrant web UI at
`http://localhost:6333/dashboard`) before trusting it on the full corpus.

## Retrieval

```bash
python retrieval/retrieve.py --query "What was Apple's gross margin in 2022?"
python retrieval/retrieve.py --query "JPM risk factors" --ticker JPM --form-type 10-K
python retrieval/retrieve.py --query "revenue trends" --ticker AAPL --ticker MSFT --limit 5
```

Requires both vLLM (`vllm serve BAAI/bge-large-en-v1.5 --runner pooling`)
and Qdrant (`cd ~/qdrant && ./qdrant`) running.

### Why query embedding reuses embed_chunks.py's approach exactly

A query and a corpus chunk must go through the *identical* embedding
process, or the vectors aren't comparable -- this isn't code-reuse for its
own sake, it's a correctness requirement. `embed_query()` sends raw text
directly to vLLM's `/v1/embeddings` endpoint, same as the embedding stage,
same reason: `langchain_openai.OpenAIEmbeddings` pre-tokenizes with
tiktoken client-side, the wrong vocabulary for a BERT-based model -- a
real, documented bug against this exact model (see "Embedding" section
above).

### Split-table reconstruction

This is the retrieval-time half of a design decision made back at the
parsing stage: `table_id`/`group_index`/`group_count` exist specifically
so a table that got split into multiple chunks (to give each piece an
accurate, untruncated embedding -- see "Row-group splitting for oversized
tables") can be reassembled into the full table at query time. Search
finds fragments precisely; `reconstruct_results()` ensures generation
later sees the complete table, not just whichever piece happened to match
-- this is exactly the context-loss problem raised earlier in this
project ("if a particular table has multiple rows about the same topic
and we split it, isn't there contextual loss?") and the metadata
added specifically to solve it, now actually wired up.

Mechanics: for any search hit that's part of a split table
(`group_count > 1`), all sibling pieces are fetched via `scroll` (an exact
metadata lookup by `table_id`, not a similarity search) and concatenated
in `group_index` order. If multiple pieces of the *same* table both
appear in the raw search hits, they're de-duplicated into one consolidated
result using the higher score. Pieces are concatenated without stripping
repeated header rows -- a deliberate simplicity choice: still correct and
complete either way, and the redundancy is a minor verbosity cost against
a generation LLM's much larger context budget, not a correctness problem.

**What's been verified vs. not:** every function was unit tested directly
-- `embed_query()` confirmed to send raw text (not pre-tokenized) to the
correct endpoint; `build_filter()` confirmed correct across no-filter,
single-ticker, multi-ticker, and combined ticker+form_type+date-range
cases; `reconstruct_results()` verified against a realistic scenario (two
pieces of the same split table both matching, plus an unrelated chunk) --
confirmed it correctly finds *all* sibling pieces (including one that
wasn't even in the raw search hits), de-duplicates to one result, keeps
the higher score, and leaves unrelated chunks untouched; `search()` and
`format_results()` verified for correct argument passing and output
rendering. A full `main()` run was verified end-to-end against a complete
mock of both vLLM and Qdrant. The actual network calls against live
servers have not been exercised (no GPU/Qdrant instance reachable from the
dev sandbox) -- run a few real queries and spot-check the results,
especially a query you know should hit a split table, before relying on
this.

## Retrieval evaluation

```bash
python eval/run_retrieval_eval.py
python eval/run_retrieval_eval.py --limit 5
```

Requires vLLM and Qdrant both running, same as retrieval itself.

### Why this isn't "real" RAGAS yet, and what it is instead

RAGAS's core metrics -- context precision, context recall, faithfulness,
answer relevancy -- need an LLM to judge relevance, or an actual generated
answer to assess. Neither exists yet (no generation LLM has been picked or
served -- see "Next step"). Rather than block retrieval evaluation on
that, `run_retrieval_eval.py` computes **Hit Rate@K** and **Mean
Reciprocal Rank (MRR)**: judge-free, standard information-retrieval
metrics that answer "did retrieval actually find the right chunk" without
needing any LLM to grade it. This is complementary to RAGAS, not a
replacement for it -- the golden question set built here is exactly what
full RAGAS metrics will consume once a judge/generation LLM exists.

Both metrics matter together, not just one: Hit Rate@K alone can't tell
"found at rank 1" from "found at rank 10" (both just count as a hit) --
MRR (which averages `1/rank` per question, `0` for a miss) is sensitive to
that difference. Verified this mattered concretely while building the
harness: a mocked scenario where filtered search found the answer at rank
1 and unfiltered search found the *same* answer at rank 2 showed identical
100% Hit Rate for both, while MRR correctly showed `1.000` vs `0.500`.

### Why the golden set reuses facts already verified earlier in this project

Every entry in `golden_questions.json` traces back to a real number
confirmed during this project -- not fabricated plausible-sounding
figures. Apple's gross margin table (FY2022 and FY2023, testing whether
retrieval distinguishes between years), Pfizer's Paxlovid revenue (the
exact row that motivated `split_oversized_row()`), Coca-Cola's
audit-matters tax figures (the exact text that motivated the
abbreviation-splitting fix), JPMorgan's Markets revenue table (the exact
table that motivated column-group splitting).

**Not all entries are verified the same way, and the difference matters.**
Each question carries an explicit `verification_method` field:

- `raw_html` (5 of 6 questions -- AAPL x2, PFE, KO, JPM): confirmed
  directly against the filing's raw source HTML with BeautifulSoup,
  independent of whether the parsing/embedding/retrieval pipeline being
  evaluated has any bugs.
- `pipeline_retrieval` (1 of 6 -- JNJ): confirmed only via a live
  `retrieve.py` query result, i.e. the pipeline's own output. This is
  weaker and somewhat circular -- an undiscovered pipeline bug (two
  numbers glued together, a value misattributed to the wrong row) could
  get canonized as "correct" rather than caught, since the ground truth
  came from the same system being tested. `run_retrieval_eval.py` reports
  a separate, more rigorous hit rate restricted to `raw_html`-verified
  questions only, and flags non-`raw_html` entries with `*` in the
  per-question table, rather than silently blending the two.

**Coverage status:** all 10 of 10 tickers now have at least one golden
question -- full ticker coverage achieved. Of the process along the way:
GS and XOM (both Workiva-generated) needed no new fixes at all, unlike
Tesla and Microsoft's DFIN-generated filings, which each surfaced a
genuinely distinct real bug (see "Parsing and chunking" above). 3 of 13
questions are now 10-Q (WMT, TSLA x1, XOM), still weighted toward 10-K.
**Remaining known gaps, worth being upfront about:** no cross-ticker
comparison question yet, even though the corpus was specifically built
with sector diversity to support exactly that kind of query. At 13
questions, aggregate Hit Rate/MRR still carry real statistical noise -- a
single lucky or unlucky result swings the percentage by several
percentage points. One question (JNJ, q4) remains `pipeline_retrieval`-
verified rather than `raw_html` -- would need JNJ's raw source HTML to
upgrade.

### Filtered vs. unfiltered comparison

Every question runs through retrieval twice: once with a metadata filter
(the question's known-correct `ticker`/`form_type`) and once as pure
unfiltered semantic search. This directly measures how much the metadata
filtering built throughout this project actually contributes over vector
similarity alone -- rather than assuming it helps.

**What's been verified vs. not:** `check_hit()` was unit tested directly
across hit-at-various-ranks, miss, and empty-results cases.
`golden_questions.json` was validated for required fields. The full
`run_eval()` flow -- including the MRR-vs-Hit-Rate distinction and the
total-miss path -- was verified against realistic mocked scenarios,
including running all 6 real golden questions through a total-miss mock
to confirm no crashes and correct 0% reporting. The actual network calls
against live vLLM/Qdrant have not been exercised (no GPU/Qdrant instance
reachable from the dev sandbox) -- run it for real and check whether the
reported hit rate matches what a manual spot-check of each question would
suggest.

### Diagnosing misses: diagnose_misses.py

A real run against the corrected 190-filing corpus (see "Pagination bug
fix" and related sections above) gave Hit Rate@10 of 76.9% (filtered) and
61.5% (unfiltered) -- filtering rescued 2 complete misses into confirmed
hits (q5, q11), but 3 questions missed even with the correct
ticker/form_type filter applied, and MRR (0.351 filtered) revealed that
even among hits, the correct chunk often wasn't landing near the top
(average hit rank ~4.4).

Hit/miss alone doesn't distinguish two very different problems: a
**ranking** issue (the right chunk exists and is retrievable, just scored
too low for a small top-K) versus a more fundamental **absence** (not
found even in a much larger candidate pool). `diagnose_misses.py`
re-runs a specific question with a much larger limit (default 50, vs. the
standard eval's 10) and reports which case it is, plus the actual top 10
results -- so you can see what's outranking the correct answer, not just
that something did.

```bash
python eval/diagnose_misses.py                  # auto-detects current misses
python eval/diagnose_misses.py --ids q3 q6 q12   # target specific questions
python eval/diagnose_misses.py --ids q12 --limit 100
```

Verified against two mocked scenarios before trusting it: a "found at
rank 25 of 50" case (correctly identified as a ranking problem, not
absence) and a "not found anywhere in 50" case (correctly identified as
the more fundamental gap) -- plus the auto-detect path (correctly
excludes a question that actually hits, targets only genuine misses) and
the `--ids` path (proceeds with valid IDs, warns clearly on an unknown
one rather than failing silently).

**Worth noting which misses have a built-in hypothesis vs. which don't**:
q3 (Pfizer Paxlovid) and q6 (JPMorgan Fixed Income Markets) both involve
heavily-restructured table content -- exactly the kind of specific,
numbers-dense chunk that "Embedding verification" above already found
embeddings represent weakly (filing-section separation scored near zero
even after the LDA cross-check). q12 (Goldman Sachs net revenues) is a
plain fact with no such excuse, making it the more informative one to
actually run this diagnostic against first.

## Hybrid search (dense + BM25) -- tried, measured, reverted

Running `diagnose_misses.py` against the real corpus gave concrete,
specific evidence, not just a hunch: q12 (Goldman Sachs net revenues) and
q3 (Pfizer Paxlovid) were both genuinely absent from the top 50 results,
outranked by narrative prose that was semantically *in the neighborhood*
but didn't contain the actual figure -- exactly the failure mode BM25's
exact term/number matching is built to catch, and dense embeddings were
already known to be weak on (see "Embedding verification" -- filing
section separation scored near zero even after the LDA cross-check).

Qdrant supports **server-side BM25** (`model="Qdrant/bm25"` -- no
separate model to serve), fused with the existing dense vectors via
Reciprocal Rank Fusion. Built and tested thoroughly (correct request
shape, correct fusion config, filter propagation confirmed via Qdrant's
own docs rather than assumed) -- see git history / earlier project notes
for the full implementation detail, since this was fully reverted after
being measured.

**Measured against the golden set, this was a real regression, not an
improvement -- on every single metric:**

| | Pure dense (before) | Hybrid (measured) |
|---|---|---|
| Hit Rate@10 filtered | 76.9% | 69.2% |
| Hit Rate@10 unfiltered | 61.5% | 46.2% |
| MRR filtered | 0.351 | 0.262 |
| MRR unfiltered | 0.301 | 0.147 (nearly halved) |

Worse still: **q3 and q12 -- the two cases that specifically motivated
adding BM25 -- were still both MISS** even with hybrid search in place.
The hypothesis wasn't even confirmed on its own primary targets. Two
previously-working questions (q9, q10) actually regressed from HIT to
MISS. Leading hypothesis for why, not yet independently confirmed: BM25
tokenization likely splits numbers on punctuation (`"58,283"` -> `"58"`
+ `"283"`), losing exactly the distinctive-token weighting the whole
approach was built around.

**Reverted** `retrieve.py`'s `search()` back to pure dense retrieval.
The collection's schema still has both `dense` and `sparse` named
vectors (from the hybrid attempt) -- rather than reloading the
collection a third time, the sparse vectors are simply left populated
and unused; `search()` now queries only `using="dense"`. This is
recorded here deliberately, not deleted from the README, in keeping with
this project's whole discipline: **a documented, measured "this didn't
work" is exactly as valuable as a documented fix** -- it's real evidence,
and it's what stopped an actual regression from being mistaken for
progress.

## Re-ranking

Re-ranking targets a different problem than hybrid search did, and it's
worth being precise about the difference so this doesn't repeat the same
mistake: a re-ranker **only re-orders candidates retrieval already
found** -- it cannot retrieve something new. `diagnose_misses.py` showed
q6 was present at rank 11 of 50 under pure dense search (a genuine
ranking problem), while q3 and q12 were absent even from the top 50 (a
retrieval problem re-ranking cannot touch). So re-ranking is expected to
help the low-MRR "found but buried" pattern (q1, q2, q4, q5, q6, q7, q8,
q13), but **not** q3 or q12 -- that would need a change to retrieval
itself, still an open problem.

### How it works: cross-encoder vs. bi-encoder

Dense/BM25 retrieval both compare the query and each chunk as
**independently computed** representations -- the model never sees them
together, which is exactly what makes searching 85,965 chunks fast (every
chunk's representation is pre-computed once, ahead of time). A
cross-encoder reranker instead reads the query and *one specific
candidate's actual text together*, letting it directly attend between
them -- far more precise, but too expensive to run against the whole
corpus, since nothing can be pre-computed until the query is known. That's
why it's always a second stage on top of a fast first-stage retriever,
never a replacement for it.

### Serving: a second vLLM instance, no new framework needed

vLLM natively supports serving cross-encoder rerankers via a
Cohere-compatible `/rerank` endpoint. Since embeddings already use
`BAAI/bge-large-en-v1.5`, `BAAI/bge-reranker-base` is a natural,
consistent choice from the same model family:

```bash
vllm serve BAAI/bge-reranker-base --port 8001
```

A second, separate vLLM process from the embedding server (which stays
on its own port, e.g. 8000) -- reranker models are much smaller than the
embedding model, so this should fit comfortably alongside it on the same
GPU. Configured via `reranker_base_url` / `reranker_model_id` /
`reranker_api_key` in `config.yaml`.

No collection reload needed for this change, unlike hybrid search --
re-ranking is a query-time-only step; nothing about how points are
stored changes.

### Usage

```bash
python retrieval/retrieve.py --query "..." --rerank
python eval/run_retrieval_eval.py --rerank
```

`retrieve.py --rerank`: retrieves `--rerank-candidates` (default 50) via
plain dense search, reranks them, truncates to the final `--limit` *after*
table-sibling reconstruction (reconstruction can reduce the count via
de-duplication, so truncating first would risk cutting off results that
would have merged away). Without `--rerank`, behavior is unchanged from
plain dense search -- confirmed via a test asserting no `/rerank` call is
ever made and the exact `--limit` is passed straight through to `search()`.

`run_retrieval_eval.py --rerank`: adds a third comparison column
(filtered + reranked) alongside the existing unfiltered/filtered ones, so
reranking's actual effect can be measured directly against the same
baseline -- the same discipline that caught hybrid search being a
regression rather than an improvement.

### Implementation details worth knowing

**A real bug found on the very first live run, worth understanding, not
just patching around**: `BAAI/bge-reranker-base` rejected requests with
`"This model's maximum context length is 512 tokens... your prompt
contains at least 513 input tokens"`. Root cause: a cross-encoder takes
the query AND document **concatenated together** as one input
(`[CLS] query [SEP] document [SEP]`) -- unlike the embedding model, which
only ever sees one piece of text at a time. Chunks were deliberately
sized to fit near 512 tokens *on their own*, for the embedding model's
limit; adding any query text on top of an already-near-512-token chunk
easily exceeds the reranker's own, separate 512-token limit.

Fixed with `_truncate_document_for_reranker()`: loads the reranker's own
tokenizer (same real-tokenizer approach as `embed_chunks.py`'s embedding
truncation fix, not a rough character-count guess), computes the query's
token count, and truncates the **document** (never the query -- the
query is short and needs to stay intact; the document is what has room to
give) to fit within `512 - query_tokens - 8` (the same special-token
safety buffer convention used throughout this project). Verified against
a scenario mirroring the exact real error (a 600-word document that
would combine with the query to exceed the limit): correctly truncates
to precisely fit the budget, confirmed the combined length lands exactly
at 512, confirmed it truncates the tail (keeping the document's
beginning) rather than the head, and confirmed a longer query correctly
shrinks the document's available budget rather than ever truncating the
query itself. Also verified end-to-end through the full `rerank()`
function that a short document sent alongside a long one is left
completely unchanged -- truncation only touches documents that actually
need it.

`rerank()` sorts results by `relevance_score` **explicitly, client-side**
-- the API contract only guarantees each result carries a score, not that
the response list arrives pre-sorted, so this isn't assumed. Re-ranked
hits are wrapped in a small `_RerankedHit` stand-in (carrying just
`.score` and `.payload`, the only two attributes `reconstruct_results()`
and `check_hit()` actually read) rather than mutating the original
`ScoredPoint`'s `.score` in place -- `qdrant_client`'s return objects
aren't guaranteed mutable, so this avoids depending on that.

**What's been verified vs. not:** `rerank()` was tested against a
deliberately unsorted response, confirming it explicitly re-sorts rather
than trusting API order, and confirming it correctly promotes a
candidate from the *worst* dense rank to the top based on the reranker's
score. The empty-hits case (no API call made) was verified. A full
`retrieve.py --rerank` run was verified end-to-end against a complete
mock, confirming the correct answer gets promoted and the non-rerank path
is completely unaffected. `run_retrieval_eval.py --rerank`'s three-way
comparison was verified end-to-end too, including a case where Hit Rate
stays unchanged (both already hits) while MRR still improves --
confirming the two metrics are tracking genuinely different things, not
redundant. The actual network calls against a live reranker, and,
critically, **whether re-ranking actually improves the measured Hit
Rate/MRR against the golden set**, have not been verified against a real
server -- run `eval/run_retrieval_eval.py --rerank` once the reranker is
serving and compare directly against the pure-dense baseline (Hit
Rate@10: 76.9% filtered / 61.5% unfiltered; MRR: 0.351 filtered / 0.301
unfiltered) before concluding it helped, exactly as hybrid search's
regression was only caught by measuring rather than assuming.

## Generation

Reranking (measured: Hit Rate@10 84.6%, MRR 0.660 filtered+reranked --
see "Re-ranking" above) confirmed retrieval is solid enough to build a
real generation layer on top. `generation/generate.py` retrieves context
(same pipeline as `retrieve.py`), builds a grounded prompt, calls the
generation LLM, and prints the answer with numbered citations mapped
back to `source_url`.

```bash
python generation/generate.py --query "What was Apple's gross margin in fiscal 2022?"
python generation/generate.py --query "..." --ticker AAPL --form-type 10-K --rerank
```

Requires vLLM (embedding, port 8000), Qdrant (port 6333), and now a third
vLLM instance serving the generation model (port 8002); a fourth
(reranker, port 8001) if using `--rerank`.

### Model selection: Phi-4-mini-instruct (BitsAndBytes inflight quantization)

Chosen after actually checking current options against this project's
real hardware constraint (a 12GB GPU already running the embedding model
and reranker), not picked by name recognition. Two real things were
checked and ruled out along the way, worth recording since they're
genuine findings, not just process notes:

- **Qwen 3.5** (the newest, most-hyped small-model release at the time)
  has a confirmed vLLM issue for text-only serving: its new hybrid
  architecture (Gated DeltaNet + MoE) has no registered
  `Qwen3_5ForCausalLM` text-only class in vLLM, only the full multimodal
  class -- loading a text-only checkpoint through it causes a weight
  prefix mismatch. A `--language-model-only` workaround exists but has
  its own reported issues. Newest isn't automatically safest.
- **DeepSeek's small, locally-deployable models are not DeepSeek's own
  architecture.** Confirmed directly from DeepSeek's own model card:
  their distilled 1.5B-70B checkpoints are "based on Qwen2.5 and Llama3
  series" -- DeepSeek's actual native models (R1, V3, V4) are all
  671B-parameter, genuinely datacenter-class, needing 320GB+ VRAM even at
  4-bit. The Llama-based distill (the one that would actually avoid Qwen)
  scores worse than the Qwen-based one on DeepSeek's own benchmarks, and
  is tuned for chain-of-thought math/logic reasoning -- overhead this
  task (read context, answer faithfully) doesn't need.

**Phi-4-mini-instruct**: 3.8B parameters -- comfortably small compared to
every larger alternative considered (Llama 4 Scout/Maverick at 109B-400B
total params, Gemma 4 at 26B with an actively-reported vLLM performance
regression on top) -- 128K context window (confirmed via multiple
independent sources: Microsoft's own Azure AI Foundry catalog, the
Hugging Face model card, and artificialanalysis.ai -- not assumed), MIT
licensed. Note: `Phi-4-mini-reasoning` and `Phi-4-mini-flash-reasoning`
are separate, math/logic-specialized variants -- the plain `-instruct`
tag is the right one for general RAG QA.

**Serve at full bf16 precision, no quantization at all** -- using
`--enforce-eager` and a reduced `--max-model-len` instead to fit the
memory budget. This was the *third* approach tried, not the first, and
the full detour is worth recording honestly rather than editing out,
since each step is a real, confirmed finding, not a guess:

1. **The bare tag with no quantization and default settings**: loads at
   full bf16 (~7.6GB weights) with the full 128K context window and
   CUDA graph capture both enabled -- doesn't fit alongside the
   embedding+reranker servers already running (`ValueError: No available
   memory for the cache blocks`).
2. **A pre-quantized third-party checkpoint** (`pytorch/Phi-4-mini-instruct-AWQ-INT4`)
   hit a genuine version-compatibility failure (`Failed to find class
   TensorCoreTiledLayout`), traced to that checkpoint depending on
   `torchao.prototype.awq` -- TorchAO's own still-evolving AWQ
   implementation ("prototype" in the module path was the real signal,
   missed at first).
3. **vLLM's native BitsAndBytes quantization** (`--quantization bitsandbytes`)
   seemed like the fix -- until vLLM's own v0.28.0 release notes revealed
   it had been *deliberately migrated out of vLLM's core* to an
   out-of-tree plugin in that exact version, confirmed via the original
   RFC discussion (low usage relative to maintenance burden was the
   stated reason).

Rather than keep chasing quantization-method compatibility against a
fast-moving vLLM release, dropped quantization entirely. Two other
settings turned out to matter far more than which quantization scheme
was used: `--enforce-eager` disables CUDA graph capture, a real,
sizeable, hard-to-predict memory consumer (confirmed by the load log
reporting exactly `0.0 GiB for CUDAGraph memory` once disabled, versus a
`-3.88 GiB` *negative* available-memory reading with it enabled at the
same utilization); and `--max-model-len 8192` avoids reserving KV cache
space for the model's full 128K context, which this project's actual
usage pattern (~10 retrieved chunks x ~500 tokens each) never needs
anyway.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve microsoft/Phi-4-mini-instruct --port 8002 \
  --gpu-memory-utilization 0.8 --max-model-len 8192 --enforce-eager
```

The `0.8` came from real arithmetic on an actual failed attempt, not a
guess: at `--gpu-memory-utilization 0.7`, the load log reported exactly
`7.17 GiB` for weights and `0.39 GiB` available for KV cache (needing
`1.0 GiB`) -- from those real numbers, `0.8` was calculated to leave a
comfortable margin, and the next attempt's log confirmed it: `1.59 GiB`
available KV cache, reconciling almost exactly with the predicted
outcome.

Getting a working generation server up on WSL2 needed a total of five
real, distinct fixes along the way (UVA, a missing CUDA compiler, the
full-precision-vs-quantized VRAM confusion, the TorchAO version
incompatibility, and BitsAndBytes' core removal) before landing on this
final, no-quantization approach -- all documented in full in
"Troubleshooting" below, since several of these can resurface on any
vLLM server, not just this one.

### Prompt design

The system prompt requires a citation (`[1]`, `[2]`, ...) for every
specific claim, an explicit "I don't know" when the context doesn't
contain the answer, and exact (not rounded) figures. This isn't just
formatting preference -- fabricated or ungrounded financial figures are
a genuinely bad failure mode for a system over real SEC filings, and
requiring citations is what makes faithfulness checkable, both by a
human reading the sources and later by RAGAS's faithfulness metric.
`build_context_blocks()` labels each retrieved chunk with its citation
number plus ticker/form_type/period/section, so the model has enough
metadata to distinguish, for example, Apple's FY2022 and FY2023 gross
margin figures rather than conflating them -- verified directly: a test
with both years' chunks in context confirmed each context block is
correctly and distinctly labeled.

### Implementation notes

`generate_answer()` uses `temperature=0.0` -- this is a factual-QA task
over financial filings, not creative generation, so determinism is
preferred over sampling diversity. If there are zero retrieval results,
`main()` prints a direct message and returns without ever calling the
generation LLM -- verified via a test asserting no `/v1/chat/completions`
call is made in that case, avoiding a wasted call and, more importantly,
avoiding asking the model to "answer" with no grounding at all.

**What's been verified vs. not:** `build_context_blocks()`,
`build_messages()`, and `format_sources()` were tested directly,
confirming correct numbering, metadata labeling, and that two chunks from
different fiscal years get distinct, correctly-labeled blocks rather than
being conflated. `generate_answer()` was tested for correct request
construction (right endpoint, right payload, right auth header behavior
with and without an API key) and correct response parsing. A full
`main()` run was verified end-to-end against a complete mock of all three
services (embedding, generation, Qdrant), confirming a citation in the
generated answer correctly traces back to the real source URL in the
printed source list. The zero-results path was verified separately. The
actual network calls against a live generation server, and, critically,
**whether the model's answers are actually faithful to the retrieved
context** (not just well-formatted), have not been verified against a
real server -- run this for real and read the answers against their
cited sources before trusting them, and this is exactly what RAGAS's
faithfulness metric should measure formally next.

### Three real faithfulness bugs found via manual testing, and a tool built to cover the rest of the golden set

Running real queries against the live generation server surfaced three
distinct, confirmed faithfulness bugs, each fixed with a targeted
addition to `SYSTEM_PROMPT` in `generate.py`:

1. **Fabricated totals** -- asked for Goldman Sachs' net revenues (a
   figure confirmed genuinely absent from the retrieved context, not a
   ranking problem -- see "Re-ranking" above), the model summed three
   unrelated partial figures (net interest income + investment banking +
   investment management) into a plausible-looking but fabricated
   $31.90B total, violating the instruction to admit insufficient
   information rather than guess. Fixed with rule 5: combining figures
   is only allowed when the user's question explicitly asks for a sum
   *and* each figure is individually stated in context -- not as a
   substitute for a specific reported total that isn't itself present.
2. **Metric substitution** -- even after fixing (1), the model then
   answered the same question using a single *real* but *wrong* figure
   ("net interest income," a component, presented as "net revenues," the
   aggregate) -- arguably riskier than outright fabrication, since it's
   grounded in a real citation. Fixed with rule 6: a related-but-different
   metric is not a substitute for the one asked about, however close the
   citation looks.
3. **Unit-conversion drift** -- asked to combine Apple's Products and
   Services revenue (a legitimate combination request, correctly
   distinguished from case 1), the model got the correct value
   ($394,328) but mislabeled the unit as "billion" instead of "million" --
   off by 1000x. Fixed by strengthening rule 3 to explicitly require
   preserving the source's stated unit, not just its digits.

Root-caused (1) and (2) further, tracing them back to `dom_walker.py`'s
inline sub-heading gap (see "Parsing and chunking" above) -- fixed at the
source and confirmed the corrected metadata changes the model's actual
behavior on re-test, not just the citation label.

**Each of these three bugs was found by manually trying a handful of
queries, not systematically** -- and a bug turned up nearly every time a
new query was tried, which is itself a signal the untested majority of
the golden set isn't necessarily clean, just unexamined. `eval/generate_batch.py`
runs the real `generate.py` pipeline (same functions, same order of
operations) across every golden question in one pass, rather than relying
on picking queries by hand:

```bash
python eval/generate_batch.py
python eval/generate_batch.py --rerank
python eval/generate_batch.py --ids q3 q6 q12
```

Includes a rough automated check -- does the expected fact string appear
anywhere in the generated answer -- but this is explicitly a proxy, not
a substitute for reading each one: an answer can contain the right
number and still misattribute it (exactly bug #2 above), and a correct
refusal will always show as "does not contain," which is the right
outcome, not a failure. Verified with mocks: the full pipeline end-to-end,
and the zero-retrieval-results case (confirmed no generation call is
wastefully made when there's nothing to answer from).

### A 4th real bug found running the batch for real: period/fiscal-year confusion

Running `generate_batch.py --limit 5` (a context-budget constraint --
see "Implementation notes" below) for real across all 13 golden
questions surfaced a fourth distinct faithfulness bug, arguably the most
concerning yet: **three questions (q7, q9, q10) got confident, cited,
wrong answers by citing a source from the wrong fiscal year or quarter**
-- not fabrication, not metric substitution, just picking a plausible
but incorrect period. Two of the three (q7, q9) are worse than a simple
miss: the *correct*-period source was present in context as `[1]`, and
the model cited a different one anyway.

Added rule 7 to `SYSTEM_PROMPT`, requiring the model to check a source's
stated period against the question before citing it -- **and re-testing
found it did not reliably generalize.** Re-running with `--rerank
--limit 5` still found period confusion in new instances (q1, q7),
despite the correct-period source again being present in context, and
reranking's different candidate set introduced a new regression: q12
(Goldman Sachs), which previously gave a correct refusal, now cited a
confident wrong-year figure. This is a real, useful negative result --
unlike rules 5 and 6, which held up cleanly on re-test, rule 7 alone
did not close this gap. Worth recording honestly rather than quietly
dropping in favor of the next fix, same as the hybrid-search reversion
earlier in this project.

### Escalating from a prompt rule to a structural fix: `years_mentioned`

Given a purely prompt-wording fix didn't generalize, escalated to a
metadata-level fix instead: `extract_years_mentioned()` in
`parse_and_chunk.py` scans each chunk's own text for every 4-digit year
explicitly mentioned (e.g. "...$648,125 million in fiscal 2024, up from
$611,289 million in fiscal 2023..." -> `[2023, 2024]`), stored as a new
`years_mentioned` field on every chunk (text and table).

Deliberately a **list**, not a single "the period" value -- a
comparative MD&A sentence or a multi-year table genuinely discusses more
than one year at once (exactly the WMT/Tesla pattern behind q7 and q9),
and collapsing that to one value would lose real information rather than
add it. This is also why `fiscal_period_end` alone wasn't sufficient to
to prevent the bug: it only records the *filing's* own period, not which
year a specific number *within* a chunk is actually about, and filings
routinely report prior-year comparatives in the same chunk as their own
period's figures.

`build_context_blocks()` in `generate.py` now appends a `(discusses:
2023, 2024)` label directly to the citation header when this field is
present and non-empty -- making the distinction explicit and visible
rather than something the model has to infer from a date, which is
exactly the inference step it was getting wrong twice in a row. Verified
with mocks: the real q7-shaped scenario (two WMT sources, correctly and
distinctly labeled by the years they actually discuss), and two
backward-compatibility cases -- no crash and no spurious label when
`years_mentioned` is absent (older, not-yet-reprocessed chunks) or an
empty list (a chunk with no year mentioned in its text at all).

Indexed in `load_qdrant.py` (`years_mentioned`, integer -- Qdrant
indexes each element of an array field individually) so it's available
for retrieval-time filtering later, even though the immediate use here
is only in the generation prompt, not filtering search results.

**This needs the same full pipeline re-run as any other chunk-schema
change** to take effect on the real corpus -- new field, same
"schema/content change = full re-run" pattern used throughout this
project:

```bash
rm -rf data/processed data/embedded
python parsingChunking/parse_and_chunk.py
python parsingChunking/check_token_limits.py
python embedding/embed_chunks.py
# then wipe + reload Qdrant
```

**What's been verified vs. not**: `extract_years_mentioned()` was
tested thoroughly in isolation -- the real comparative-sentence case,
single and no-year cases, duplicate removal, and false-positive checks
specifically against dollar figures and accession-number fragments
(both plausible sources of spurious 4-digit matches). The dict-literal
integration in `parse_and_chunk.py` was verified against a reproduction
of the exact code pattern, since the file's own dependencies
(`langchain-text-splitters`) weren't installable in the sandbox used to
build this -- **not yet verified via a live end-to-end run of
`parse_and_chunk.py` itself.** And critically, whether this actually
fixes q1/q7's period confusion once the corpus is reprocessed and
`generate_batch.py` is re-run -- rather than just adding a
well-motivated label -- has not been confirmed. Given rule 7 alone
looked reasonable on paper and didn't hold up, treat this the same way:
re-run the batch for real after reprocessing before considering this
bug closed.

## Troubleshooting

Real errors hit while running this project, and their fixes -- kept
here rather than buried in whichever pipeline-stage section happened to
be current when they came up, since several of these (anything
vLLM/WSL2-related especially) can resurface at any serving step, not just
the one where they were first seen.

### `RuntimeError: UVA is not available` (vLLM + WSL2)

**Symptom**: a `vllm serve ...` command fails during GPU worker
initialization with `RuntimeError: UVA is not available`, regardless of
which model is being served.

**Cause**: a known, documented vLLM + WSL2 issue (confirmed via multiple
independent reports, including vLLM's own GitHub issue tracker -- not
specific to this project's setup). vLLM's newer "V2" GPU model runner
tries to allocate a UVA (Unified Virtual Addressing) buffer during
worker initialization -- a CUDA feature WSL2's GPU virtualization layer
doesn't fully support the way native Linux does.

**Fix**: force vLLM onto the older model runner path, which doesn't
require UVA:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve <model> --port <port>
```

If that alone doesn't resolve it, one more environment variable is
typically needed alongside it -- **confirmed on this project, not just
theoretical**: fixing the UVA error alone surfaced a second, separate
failure immediately after --
`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda'
doesn't exist`. Cause: some vLLM components (the FlashInfer sampler)
JIT-compile CUDA kernels at runtime using `nvcc`, the CUDA *compiler* --
a separate thing from the NVIDIA *driver*, which is all a typical WSL2
GPU passthrough setup installs (enough to *run* GPU code, not compile
new kernels from source). Fix, both flags together:

```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve <model> --port <port>
```

(A `[W...] destroy_process_group() was not called` warning alongside the
`nvcc` error is just a harmless side effect of the crash's cleanup path --
not a separate problem, ignore it once the real error above it is fixed.)

**Worth checking, not just working around**: if this appears on a new
server (e.g. the generation model) but didn't on earlier ones (e.g.
embedding, reranker) run in the same session, that's a real signal worth
following up on, not just accepting -- possibly a different code path
triggered by that specific model, or a `vllm` version change between
when the earlier servers were started and now (`pip show vllm` to check).
The environment variable fix above works regardless of which it turns
out to be, but understanding *why* it appeared inconsistently is worth a
look if it keeps recurring unpredictably.

### `ValueError: No available memory for the cache blocks`

**Symptom**: after fixing the two errors above, `vllm serve` gets further
(engine initialization starts) but still fails, with a log line just
before it noting something like *"The current
--gpu-memory-utilization=0.2000 is equivalent to
--gpu-memory-utilization=0.1578 without CUDA graph memory profiling"*,
followed by this `ValueError`.

**Cause, confirmed on this project -- a real gap in earlier guidance, not
just a config tweak**: this project's model-selection reasoning for
`Phi-4-mini-instruct` (see above) was based on a quantized, not
full-precision, footprint. But `vllm serve microsoft/Phi-4-mini-instruct`
with no quantization flag loads the model at full bf16 precision instead
-- ~7.6GB for weights alone, not the much smaller figure the model was
actually chosen for. At a modest `--gpu-memory-utilization` value (and
CUDA graph memory profiling, on by default since vLLM v0.21.0, further
reducing the *effective* fraction below the number you actually set),
there isn't enough room for the bf16 weights, let alone any KV cache --
alongside the embedding and reranker servers already running. Bumping
`--gpu-memory-utilization` up a little does not fix this on its own; the
real issue is the model's precision, not the percentage. (Skip to the
final fix at the bottom of this section -- the two entries below cover a
detour taken before arriving at it.)

### `Failed to find class TensorCoreTiledLayout` (torchao version mismatch)

**Symptom**: after switching to a pre-quantized checkpoint to fix the
memory error above (`pytorch/Phi-4-mini-instruct-AWQ-INT4`) and
installing its `torchao` dependency, `vllm serve` fails with a pydantic
`value_error`: `Failed to find class TensorCoreTiledLayout in any of the
allowed modules: torchao.prototype.mx_formats, torchao.quantization,
...`.

**Cause**: the checkpoint's serialized quantization config references a
class at an import path that the currently-installed `torchao` version
no longer has there (moved or restructured). The real signal, easy to
miss: `torchao.prototype.awq` in that module list -- "prototype" means
actively-evolving, not-yet-stable code, prone to exactly this kind of
breaking change between versions. This is a version-compatibility trap
in unstable code, not something to fix by guessing at version pins.

**Attempted fix that led to the next error**: don't keep chasing
pre-quantized third-party checkpoints built against a moving target --
try vLLM's own native BitsAndBytes inflight quantization instead, which
quantizes the original, canonical checkpoint at load time (no
third-party serialized config to become incompatible). This seemed
right, per vLLM's official docs -- but see the next entry.

### `Unknown quantization method: bitsandbytes`

**Symptom**: `vllm serve ... --quantization bitsandbytes` fails
immediately with a pydantic `value_error` listing the actually-supported
quantization methods -- `bitsandbytes` is not among them, despite
official vLLM documentation describing it as supported.

**Cause, confirmed directly from vLLM's own release notes, not a
guess**: `pip show vllm` showed version `0.28.0` -- an extremely recent
build. That version's release notes explicitly list as a breaking
change: *"bitsandbytes support migrated to an out-of-tree plugin."* The
underlying RFC (vLLM GitHub issue tracker) explains why: the vLLM team
measured bitsandbytes usage at roughly 0.5% of users relative to the
maintenance burden it imposed on core weight-loading code, and
deliberately moved it out. `pip install bitsandbytes` only installs the
underlying quantization library, not the separate plugin vLLM 0.28.0
now needs to route requests to it -- and docs.vllm.ai's "latest" pages
hadn't yet caught up to this very recent change at the time this was
hit.

**Decision point, after five real errors in a row specifically chasing
quantization methods on this one recent vLLM version**: rather than
install an unfamiliar out-of-tree plugin (itself a low-usage, "roughly
0.5% of users" feature per the same RFC -- a real risk of yet another
compatibility surprise), drop quantization entirely. See the final fix
below.

### A note on `nvidia-smi` while debugging the above: `N/A` per-process memory is normal on WSL2

While working through the memory errors above, `nvidia-smi`'s per-process
memory column showed `N/A` for the running vLLM processes. This is
expected, not a bug -- confirmed directly from NVIDIA's own developer
forum: under Windows' WDDM driver model (which is how WSL2 accesses the
GPU), the OS manages memory allocations itself, so `nvidia-smi` (which
queries the driver for this data) genuinely cannot see per-process
usage. The **aggregate** total at the top of `nvidia-smi`'s output (e.g.
`494MiB / 12282MiB`) is a separate, device-level query and remains
trustworthy -- use that number, not the per-process breakdown, when
estimating how much VRAM is actually free on WSL2.

### Final fix: no quantization -- `--enforce-eager` + reduced `--max-model-len` instead

Once quantization itself turned out to be the less reliable lever (two
distinct real failures chasing it), the more robust fix was to stop
depending on it at all. Two things account for nearly the entire memory
gap that quantization was being used to paper over:

- **CUDA graph capture** is a real, sizeable, hard-to-predict memory
  consumer -- confirmed by comparing two real load attempts at the same
  `--gpu-memory-utilization`: with CUDA graphs enabled, available memory
  came back *negative* (`-3.88 GiB`); with `--enforce-eager` (which
  disables CUDA graph capture entirely, trading some per-token generation
  speed for a predictable memory footprint), the load log reported
  exactly `0.0 GiB for CUDAGraph memory`.
- **The model's default 128K context window** reserves KV cache space
  this project's actual usage (~10 retrieved chunks x ~500 tokens each,
  nowhere near 128K) never needs. `--max-model-len 8192` avoids that
  entirely.

```bash
VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve microsoft/Phi-4-mini-instruct --port 8002 \
  --gpu-memory-utilization 0.8 --max-model-len 8192 --enforce-eager
```

The `0.8` figure came from real arithmetic on an actual failed attempt,
not another guess: at `--gpu-memory-utilization 0.7`, the load log
reported `7.17 GiB` for weights and only `0.39 GiB` available for KV
cache (needing `1.0 GiB` -- a `0.61 GiB` shortfall). Backing out the
implied non-weight overhead from that real number (`0.7 x 12.28 GiB -
7.56 GiB actual budget ~= 1.04 GiB` used by other processes + baseline
vLLM overhead) and solving for a `0.8`-range utilization that leaves a
comfortable margin above the `1.0 GiB` requirement gave `0.8` -- and the
next attempt's real load log confirmed it closely: `1.59 GiB` available
KV cache, `12,966` total KV cache tokens, `1.58x` concurrency at the
8192 max length. Two harmless log lines also appear at this point,
neither indicating a problem: a `deep_gemm` import failure (that kernel
library only accelerates Mixture-of-Experts models -- Phi-4-mini is
dense, so this is irrelevant, same as an identical warning seen earlier
in this project while setting up the *embedding* server) and a
`--kv-cache-memory=...` suggestion (an alternative, more precise
byte-exact flag vLLM offers for future tuning, not an error).

If this still runs out of memory on a different machine or after
changing other settings, check `nvidia-smi`'s aggregate total (not the
per-process column -- see above) to see how much VRAM is genuinely
free, and work the same arithmetic from real numbers in the actual load
log rather than guessing at a `--gpu-memory-utilization` value blind.

## Next step

Run `generate.py` for real against a few golden-set questions and
manually check: does the answer actually say what the cited source says,
not just cite something plausible-sounding? Then extend evaluation to
full RAGAS metrics (context precision/recall, faithfulness, answer
relevancy) now that a judge/generation LLM exists -- these specifically
needed a generation step to exist first, which is why only retrieval-only
metrics (Hit Rate, MRR) were measurable until now.
