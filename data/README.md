# Financial RAG dataset builder

Builds the raw document corpus for the financial RAG portfolio project by
pulling 10-K / 10-Q filings directly from SEC EDGAR.

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

Adjust `config.yaml` freely — add tickers, add `8-K` for earnings
announcements, push `since_year` back further, etc. Note that widening scope
increases download time (SEC's rate limit is the binding constraint, not
your machine).

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
python build_dataset.py
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

This metadata schema is what gets carried through to chunk-level metadata
in the ingestion pipeline (ticker / doc_type / fiscal_period filters at
retrieval time).

## Next step

Feed `data/raw/` into the parsing + table-aware chunking stage (docling or
`unstructured`), which is the next piece of the ingestion pipeline.
