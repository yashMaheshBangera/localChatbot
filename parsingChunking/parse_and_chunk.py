"""
Parse + chunk downloaded SEC filings for the financial RAG pipeline.

Replaces the earlier Docling-based approach with a hand-built BeautifulSoup
parser + LangChain text splitter. This version was built by first measuring
the real structure of a sample filing (see table_cleaner.py and
dom_walker.py) rather than assuming a library would handle it -- notably,
SEC/Workiva filings render financial tables with heavy colspan-based layout
padding (in one sample table, 76% of cells were empty spacer cells) and
split currency symbols into their own cell, both of which this pipeline
explicitly cleans up.

Pipeline per filing:
  1. Walk the HTML DOM in document order (dom_walker.py), classifying each
     block as heading / text / table, and tracking a 2-level section path
     (e.g. "Item 1. Financial Statements > CONDENSED CONSOLIDATED
     STATEMENTS OF OPERATIONS").
  2. Tables are cleaned (table_cleaner.py): colspan-aware grid reconstruction,
     empty spacer columns dropped, "$"/"%" cells merged into their neighbor.
     A table that fits within max_tokens stays one atomic chunk. An
     oversized table (large multi-year financial statements routinely
     exceed 512 tokens -- some in this project's real corpus ran past 1400)
     is split into multiple row-group chunks instead of truncated: never
     splitting a data row itself, and repeating the detected header rows
     (period dates, section labels) at the top of every group so column
     meaning survives the split. This replaced an earlier design where
     oversized tables were truncated at embedding time, silently discarding
     everything past the token budget -- for large tables that meant
     losing over half the content with zero way to retrieve it via search.
  3. Consecutive text blocks under the same section are grouped, then split
     with LangChain's RecursiveCharacterTextSplitter (token-budgeted against
     your embedding model's tokenizer).

Usage:
    pip install -r requirements.txt
    python parse_and_chunk.py

Idempotent: filings that already have output chunks are skipped on rerun.

NOTE ON TESTING: the DOM-walking, table-cleaning, and table row-group-
splitting logic (dom_walker.py, table_cleaner.py) were verified against
real downloaded filings, including automated data-integrity checks (every
original row appears exactly once across split groups, no loss or
duplication) across multiple budget sizes. The LangChain text-splitting
step was written against LangChain's current documented API but has NOT
been run end-to-end in the dev sandbox (no network to install
langchain-text-splitters/transformers there). Run this on a small batch
first and spot-check a few "text" chunks before processing your whole
corpus.
"""

import hashlib
import json
import re
from pathlib import Path

import yaml
from tqdm import tqdm
from bs4 import BeautifulSoup
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from dom_walker import build_records
from table_cleaner import (
    detect_header_row_count,
    split_grid_into_row_groups,
    detect_period_column_groups,
    split_grid_into_column_groups,
    grid_to_text,
)

_YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def extract_years_mentioned(text: str) -> list:
    """Returns every distinct 4-digit year explicitly mentioned in a
    chunk's own text, sorted ascending -- e.g. "...$648,125 million in
    fiscal 2024, up from $611,289 million in fiscal 2023..." ->
    [2023, 2024].

    Built to fix a real, confirmed problem: a chunk's `fiscal_period_end`
    metadata records only the FILING's own period -- but a single filing
    routinely reports comparative prior-year figures in the same chunk
    (MD&A narrative, multi-year tables), and the filing's own period is a
    weak, indirect signal for which year a specific number in the chunk
    text actually discusses. Confirmed via generate_batch.py run across
    the full golden set: the model cited a wrong-period source for
    fiscal-year questions (e.g. citing a FY2026 filing for a FY2024
    question) even when the correct-period source was ALSO present in
    context -- deliberately not fixed by prompt wording alone (rule 7
    was tried and did not reliably generalize), fixed instead by making
    the actual years a chunk discusses an explicit, structured field the
    prompt can state directly, rather than something the model has to
    infer from a date.

    Deliberately returns every year found, not a single "the" period --
    a comparative chunk genuinely covers more than one year at once, and
    collapsing that to one value would lose real information rather than
    add it. Range restricted to 1950-2049 to avoid matching unrelated
    4-digit numbers (dollar figures, accession number fragments) that
    happen to fall outside any plausible filing year."""
    years = {int(m.group(0)) for m in _YEAR_PATTERN.finditer(text)}
    return sorted(years)

# This script's own directory -- the starting point for finding config.yaml.
SCRIPT_DIR = Path(__file__).resolve().parent


def find_config(start: Path, filename: str = "config.yaml", max_levels: int = 6) -> Path:
    """Searches upward from `start` through parent directories for the
    shared project config. Scripts live in different task-specific
    subdirectories (data/, parsingChunking/, embed/, ...) while config.yaml
    lives once at the shared project root."""
    current = start
    for _ in range(max_levels):
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        f"Could not find {filename} searching upward from {start} "
        f"(checked {max_levels} levels). Make sure config.yaml exists at "
        f"your project root."
    )


def load_config() -> tuple:
    """Returns (config_dict, project_root)."""
    config_path = find_config(SCRIPT_DIR)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config, config_path.parent


def resolve_dir(project_root: Path, config: dict, key: str, default: str) -> Path:
    """Resolves a data-directory config value relative to the project root
    (where config.yaml lives), not relative to any individual script."""
    return (project_root / config.get(key, default)).resolve()


def build_tokenizer_and_splitter(embedding_model_id: str, max_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=max_tokens,
        chunk_overlap=max(0, max_tokens // 10),  # ~10% overlap for context continuity
    )
    return tokenizer, splitter


def enforce_max_tokens(text: str, tokenizer, max_tokens: int) -> list:
    """Defensive safety net, not a trust exercise: LangChain's HF-tokenizer-based
    RecursiveCharacterTextSplitter has multiple documented, currently-open
    correctness issues around chunk sizing with chunk_overlap (e.g.
    langchain-ai/langchain#34804, #30184) -- observed in practice on this
    project as text chunks coming out larger than the configured max_tokens.
    Rather than depend on pinning down the exact internal cause, this
    re-verifies the ACTUAL token count with our own tokenizer after the
    splitter runs, and hard-splits at the token level (guaranteed correct,
    since it bypasses the splitter's separator/merge logic entirely) if
    the splitter's output still exceeds the budget."""
    token_ids = tokenizer.encode(text, truncation=False, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]
    pieces = []
    for i in range(0, len(token_ids), max_tokens):
        piece_ids = token_ids[i:i + max_tokens]
        pieces.append(tokenizer.decode(piece_ids, skip_special_tokens=True))
    return pieces


def chunk_id_for(local_path: str, index) -> str:
    """index can be an int (per-chunk position) or a string (e.g.
    "table-3", used for table_id -- one per SOURCE table, not per output
    chunk). Either way, produces a stable, deterministic ID."""
    digest = hashlib.sha1(f"{local_path}:{index}".encode()).hexdigest()[:16]
    return f"chunk_{digest}"


def process_filing(tokenizer, splitter, meta: dict, max_tokens: int) -> list:
    with open(meta["local_path"], "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    blocks = build_records(soup)

    records = []
    idx = 0
    table_block_idx = 0
    for block in blocks:
        if block["type"] == "table":
            # One table_id per SOURCE table (not per output chunk), shared
            # across all pieces it gets split into -- lets a future
            # retrieval step reconstruct the full table by fetching all
            # chunks with the same table_id, even though each piece is
            # embedded and searched independently. See README: splitting an
            # oversized table into separate chunks solves the embedding
            # blind-spot problem, but without this linking metadata it
            # trades that for a DIFFERENT loss -- generation only seeing
            # whichever single group happened to match the query, not the
            # whole table. Present (group_count=1) even for tables that
            # didn't need splitting, so the schema is uniform and retrieval
            # code never needs to special-case "was this table split".
            table_id = chunk_id_for(meta["local_path"], f"table-{table_block_idx}")
            table_block_idx += 1

            table_token_count = len(
                tokenizer.encode(block["content"], truncation=False, add_special_tokens=False)
            )

            if table_token_count <= max_tokens:
                # Fits within budget -- stays one atomic chunk, as before.
                table_pieces = [block["content"]]
            else:
                # Oversized: split rather than truncate. See module
                # docstring for why this replaced embedding-time truncation.
                grid = block["grid"]

                def count_tokens(text):
                    return len(tokenizer.encode(text, truncation=False, add_special_tokens=False))

                # Two genuinely different oversized-table shapes, found
                # from real examples, not anticipated in advance:
                #   1. Many ROWS (e.g. a 32-row income statement) -- fixed
                #      by split_grid_into_row_groups, splitting between
                #      rows and repeating header rows in every group.
                #   2. Many COLUMNS (e.g. JPMorgan's "Markets revenue"
                #      table: Fixed Income/Equity/Total Markets x 3 years
                #      side by side, 22 columns) -- row-splitting can't
                #      help here, since EVERY row is oversized regardless
                #      of how few are packed together; the problem is
                #      column count, not row count. Detected via bare year
                #      labels ("2025", "2024", "2023") as anchors between
                #      repeating column groups -- verified against two
                #      independent real tables (JPMorgan's and, as a
                #      generalization check, Apple's differently-shaped
                #      "Products and Services Performance" table), both
                #      producing correctly-labeled, data-complete
                #      sub-tables (automated multiset check, not eyeballed).
                # Column-splitting runs FIRST when detected, since a table
                # can be both wide AND tall -- each resulting narrower
                # piece still goes through row-splitting afterward if IT
                # alone is still oversized.
                column_groups = detect_period_column_groups(grid)
                if column_groups:
                    column_split_grids = split_grid_into_column_groups(grid, column_groups)
                else:
                    column_split_grids = [grid]

                table_pieces = []
                for sub_grid in column_split_grids:
                    sub_token_count = count_tokens(grid_to_text(sub_grid))
                    if sub_token_count <= max_tokens:
                        table_pieces.append(grid_to_text(sub_grid))
                    else:
                        header_row_count = detect_header_row_count(sub_grid)
                        row_groups = split_grid_into_row_groups(
                            sub_grid, header_row_count, count_tokens, max_tokens
                        )
                        table_pieces.extend(grid_to_text(g) for g in row_groups)

                table_pieces = [p for p in table_pieces if p.strip()]  # drop any empty group

            group_count = len(table_pieces)
            for group_index, piece in enumerate(table_pieces):
                token_count = len(
                    tokenizer.encode(piece, truncation=False, add_special_tokens=False)
                )
                # This should be rare-to-nonexistent for genuine financial
                # tables (verified: largest single row across a full 10-K,
                # 604 rows checked, was ~183 words -- from a narrative audit
                # discussion, not a numeric table -- comfortably under
                # budget). But it's only verified against ONE filer; other
                # tickers/industries (e.g. a bank's collateral schedules)
                # could plausibly differ. Flagged LOUDLY rather than left to
                # embed_chunks.py's silent truncation fallback, so if this
                # ever actually fires, it's immediately visible and can be
                # fixed properly (grounded in a real example) rather than
                # speculatively engineered for now.
                row_too_large = token_count > max_tokens
                if row_too_large:
                    tqdm.write(
                        f"  [ROW TOO LARGE] {meta['ticker']} {meta['form_type']} "
                        f"{meta['fiscal_period_end']}, section={block['section']!r}: "
                        f"a single table row (plus header) is {token_count} tokens, "
                        f"exceeding max_tokens={max_tokens} even before any embedding-"
                        f"time truncation. This is the rare edge case noted in the "
                        f"README -- if you see this, it's worth pasting the row content "
                        f"back for a proper fix rather than relying on truncation."
                    )
                records.append({
                    "chunk_id": chunk_id_for(meta["local_path"], idx),
                    "ticker": meta["ticker"],
                    "company_name": meta["company_name"],
                    "cik": meta["cik"],
                    "form_type": meta["form_type"],
                    "fiscal_period_end": meta["fiscal_period_end"],
                    "filing_date": meta["filing_date"],
                    "source_url": meta["source_url"],
                    "section": block["section"],
                    "chunk_type": "table",
                    "token_count": token_count,
                    "row_too_large": row_too_large,
                    "table_id": table_id,
                    "group_index": group_index,
                    "group_count": group_count,
                    "years_mentioned": extract_years_mentioned(piece),
                    "text": piece,
                })
                idx += 1
        else:  # text block -- may need splitting
            pieces = splitter.split_text(block["content"])
            # Defensive re-check: verify the splitter actually respected
            # max_tokens rather than trusting it (see enforce_max_tokens).
            final_pieces = []
            for p in pieces:
                final_pieces.extend(enforce_max_tokens(p, tokenizer, max_tokens))

            for piece in final_pieces:
                token_count = len(
                    tokenizer.encode(piece, truncation=False, add_special_tokens=False)
                )
                records.append({
                    "chunk_id": chunk_id_for(meta["local_path"], idx),
                    "ticker": meta["ticker"],
                    "company_name": meta["company_name"],
                    "cik": meta["cik"],
                    "form_type": meta["form_type"],
                    "fiscal_period_end": meta["fiscal_period_end"],
                    "filing_date": meta["filing_date"],
                    "source_url": meta["source_url"],
                    "section": block["section"],
                    "chunk_type": "text",
                    "token_count": token_count,
                    "years_mentioned": extract_years_mentioned(piece),
                    "text": piece,
                })
                idx += 1
    return records


def main():
    config, project_root = load_config()
    raw_dir = resolve_dir(project_root, config, "output_dir", "data/raw")
    processed_dir = resolve_dir(project_root, config, "processed_dir", "data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found -- run build_dataset.py first."
        )
    manifest = json.loads(manifest_path.read_text())

    print(f"Loading tokenizer + text splitter ({config['embedding_model_id']})...")
    tokenizer, splitter = build_tokenizer_and_splitter(
        config["embedding_model_id"], config["max_tokens"]
    )

    total_chunks = 0
    for meta in tqdm(manifest, desc="Parsing filings"):
        out_path = processed_dir / meta["ticker"] / (
            Path(meta["local_path"]).stem + ".chunks.jsonl"
        )
        if out_path.exists():
            continue  # idempotent: skip already-processed filings

        try:
            records = process_filing(tokenizer, splitter, meta, config["max_tokens"])
        except Exception as e:
            print(f"  [error] {meta['ticker']} {meta['form_type']} "
                  f"{meta['fiscal_period_end']}: {e}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        total_chunks += len(records)
        tables = sum(1 for r in records if r["chunk_type"] == "table")
        tqdm.write(
            f"  {meta['ticker']} {meta['form_type']} {meta['fiscal_period_end']}: "
            f"{len(records)} chunks ({tables} table chunks)"
        )

    print(f"\nDone. {total_chunks} chunks written under {processed_dir}/")


if __name__ == "__main__":
    main()
