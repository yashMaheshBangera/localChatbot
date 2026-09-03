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
     empty spacer columns dropped, "$" cells merged into their neighbor --
     then kept as one atomic chunk per table (never split).
  3. Consecutive text blocks under the same section are grouped, then split
     with LangChain's RecursiveCharacterTextSplitter (token-budgeted against
     your embedding model's tokenizer).

Usage:
    pip install -r requirements.txt
    python parse_and_chunk.py

Idempotent: filings that already have output chunks are skipped on rerun.

NOTE ON TESTING: the DOM-walking and table-cleaning logic (dom_walker.py,
table_cleaner.py) were verified against a real downloaded filing. The
LangChain text-splitting step was written against LangChain's current
documented API but has NOT been run end-to-end (no network in the dev
sandbox to install langchain-text-splitters/transformers). Run this on a
small batch first and spot-check a few "text" chunks before processing your
whole corpus.
"""

import hashlib
import json
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


def chunk_id_for(local_path: str, index: int) -> str:
    digest = hashlib.sha1(f"{local_path}:{index}".encode()).hexdigest()[:16]
    return f"chunk_{digest}"


def process_filing(tokenizer, splitter, meta: dict, max_tokens: int) -> list:
    with open(meta["local_path"], "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    blocks = build_records(soup)

    records = []
    idx = 0
    for block in blocks:
        if block["type"] == "table":
            # Tables are intentionally left uncapped/atomic here (never split
            # mid-row) -- embed_chunks.py handles any table that exceeds the
            # embedding model's own limit at embedding time, keeping the full
            # text in the record while flagging embedding_truncated=true.
            token_count = len(
                tokenizer.encode(block["content"], truncation=False, add_special_tokens=False)
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
                "text": block["content"],
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
