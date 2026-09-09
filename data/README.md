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
  qdrant/
    load_qdrant.py
    .loaded_files.json  <- created by load_qdrant.py, gitignored
  retrieval/
    retrieve.py
  eval/
    golden_questions.json
    run_retrieval_eval.py
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
python qdrant/load_qdrant.py
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

A local manifest (`qdrant/.loaded_files.json`, gitignored -- regeneratable)
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

## Next step

Pick and serve a generation LLM (through vLLM or Ollama), build the
prompt that combines a user's question with retrieved context, and wire
up citations back to `source_url`. Then extend evaluation to full RAGAS
metrics (context precision/recall, faithfulness, answer relevancy) now
that a judge/generation LLM exists -- and, as discussed, benchmark hybrid
(dense + BM25) search against this same golden set and pure-dense
baseline before deciding whether to add it, rather than assuming it
helps.
