"""
Verify that no "text" chunk in data/processed/ exceeds max_tokens.

Run this after parse_and_chunk.py to get a definitive answer -- rather than
inferring it from whether the "Token indices sequence length..." warning
appeared during the run, which can fire harmlessly for reasons unrelated to
the final output (see README's "Defensive hard-cap on text chunk size").

Usage:
    python check_token_limits.py
"""

import json
from pathlib import Path

import yaml

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


def main():
    config_path = find_config(SCRIPT_DIR)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    project_root = config_path.parent

    processed_dir = (project_root / config.get("processed_dir", "data/processed")).resolve()
    max_tokens = config["max_tokens"]

    violations = []
    total_text_chunks = 0
    total_table_chunks = 0
    missing_token_count = 0

    for path in sorted(processed_dir.rglob("*.chunks.jsonl")):
        with path.open() as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)

                if "token_count" not in record:
                    missing_token_count += 1
                    continue

                if record["chunk_type"] == "text":
                    total_text_chunks += 1
                    if record["token_count"] > max_tokens:
                        violations.append((path, line_num, record))
                else:
                    total_table_chunks += 1  # tables are intentionally uncapped

    print(f"Checked {total_text_chunks} text chunks, {total_table_chunks} table chunks "
          f"(max_tokens={max_tokens})")

    if missing_token_count:
        print(f"\n[warn] {missing_token_count} chunks have no token_count field -- "
              f"these are from BEFORE the token-count fix was added. Delete and "
              f"re-run parse_and_chunk.py on the affected filings.")

    if total_text_chunks == 0:
        print(f"\n[SKIPPED] No text chunks with a token_count field were found -- "
              f"nothing was actually verified. This is expected if every chunk "
              f"predates the fix (see the [warn] above); re-run parse_and_chunk.py "
              f"first, then run this check again.")
    elif violations:
        print(f"\n[FAIL] {len(violations)} text chunks still exceed {max_tokens} tokens:")
        for path, line_num, record in violations[:10]:
            print(f"  {path} line {line_num}: chunk_id={record['chunk_id']} "
                  f"token_count={record['token_count']}")
        if len(violations) > 10:
            print(f"  ...and {len(violations) - 10} more")
    else:
        print(f"\n[PASS] Checked {total_text_chunks} text chunks, all within "
              f"{max_tokens} tokens. The defensive hard-cap is working correctly.")


if __name__ == "__main__":
    main()
