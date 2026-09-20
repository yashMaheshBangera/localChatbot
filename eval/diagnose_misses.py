"""
Diagnostic tool for investigating specific golden-question misses from
run_retrieval_eval.py. Unlike the eval harness (which only reports
hit/miss within the standard top-K), this shows the ACTUAL retrieved
results -- so a genuine miss can be told apart from a ranking-quality
problem (right answer present, just ranked low) versus a real absence
(right answer not found even in a much larger candidate pool).

For each targeted question, runs a FILTERED search (ticker + form_type,
the same "best case" the eval harness measures) with a much larger limit
than the standard eval, then reports:
  - whether the expected fact appears anywhere in that expanded set, and
    at what rank if so
  - the actual top 10 results returned, so you can see what's ranking
    ABOVE the correct answer (or in its place, if it's absent entirely)

Usage:
    python eval/diagnose_misses.py                  # auto-detects current misses
    python eval/diagnose_misses.py --ids q3 q6 q12   # specific questions
    python eval/diagnose_misses.py --ids q12 --limit 100
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def find_config(start: Path, filename: str = "config.yaml", max_levels: int = 6) -> Path:
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
        f"(checked {max_levels} levels)."
    )


def load_config_and_retrieve_module():
    config_path = find_config(SCRIPT_DIR)
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    project_root = config_path.parent

    retrieval_dir = project_root / "retrieval"
    sys.path.insert(0, str(retrieval_dir))
    import retrieve as retrieve_module

    return config, retrieve_module


def find_current_misses(config, retrieve_module, golden_questions, limit=10):
    """Re-runs the standard filtered eval (same as run_retrieval_eval.py)
    to find which questions currently miss, so this script can be run
    with no arguments and auto-target whatever's currently broken."""
    from qdrant_client import QdrantClient
    client = QdrantClient(url=config.get("qdrant_url", "http://localhost:6333"),
                           api_key=config.get("qdrant_api_key"))
    collection_name = config.get("collection_name", "sec_filings")

    misses = []
    for q in golden_questions:
        query_vector = retrieve_module.embed_query(
            config.get("vllm_base_url", "http://localhost:8000"),
            config["embedding_model_id"], q["question"], config.get("vllm_api_key"),
        )
        query_filter = retrieve_module.build_filter(ticker=q.get("ticker"), form_type=q.get("form_type"))
        hits = retrieve_module.search(client, collection_name, query_vector, query_filter, limit)
        results = retrieve_module.reconstruct_results(client, collection_name, hits)
        found = any(q["expected_context_contains"] in r["payload"].get("text", "") for r in results)
        if not found:
            misses.append(q["id"])
    return misses


def diagnose(config, retrieve_module, question, expanded_limit):
    from qdrant_client import QdrantClient
    client = QdrantClient(url=config.get("qdrant_url", "http://localhost:6333"),
                           api_key=config.get("qdrant_api_key"))
    collection_name = config.get("collection_name", "sec_filings")

    print(f"\n{'=' * 90}")
    print(f"[{question['id']}] {question['question']}")
    print(f"Expected: {question['expected_context_contains']!r} "
          f"(ticker={question.get('ticker')}, form_type={question.get('form_type')})")
    print('=' * 90)

    query_vector = retrieve_module.embed_query(
        config.get("vllm_base_url", "http://localhost:8000"),
        config["embedding_model_id"], question["question"], config.get("vllm_api_key"),
    )
    query_filter = retrieve_module.build_filter(ticker=question.get("ticker"), form_type=question.get("form_type"))
    hits = retrieve_module.search(client, collection_name, query_vector, query_filter, expanded_limit)
    results = retrieve_module.reconstruct_results(client, collection_name, hits)

    expected = question["expected_context_contains"]
    found_rank = None
    for i, r in enumerate(results, 1):
        if expected in r["payload"].get("text", ""):
            found_rank = i
            break

    if found_rank:
        print(f"\n-> Found at rank {found_rank} of {expanded_limit} "
              f"(a RANKING problem: the right chunk exists and is retrievable, "
              f"it's just scored too low to make it into a smaller top-K)")
    else:
        print(f"\n-> NOT FOUND anywhere in the top {expanded_limit} "
              f"(a more fundamental problem: either the fact isn't in the corpus "
              f"in a form this query can match, or the semantic gap is larger "
              f"than just a ranking issue)")

    print(f"\nActual top 10 results (what's ranking ABOVE or in place of the correct answer):")
    for i, r in enumerate(results[:10], 1):
        p = r["payload"]
        marker = " <-- CORRECT" if expected in p.get("text", "") else ""
        section = (p.get("section") or "")[:50]
        preview = p.get("text", "")[:100].replace("\n", " ")
        print(f"  {i:2d}. score={r['score']:.4f} [{p.get('chunk_type')}] {section}{marker}")
        print(f"      {preview}...")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", default=None,
                         help="Specific golden question IDs to diagnose (e.g. q3 q6 q12). "
                              "If omitted, auto-detects current misses.")
    parser.add_argument("--limit", type=int, default=50,
                         help="Expanded candidate window to search within (default 50, "
                              "much larger than the standard eval's top-10)")
    args = parser.parse_args()

    config, retrieve_module = load_config_and_retrieve_module()

    import json
    golden_path = SCRIPT_DIR / "golden_questions.json"
    with open(golden_path) as f:
        golden_questions = json.load(f)

    if args.ids:
        targets = [q for q in golden_questions if q["id"] in args.ids]
        missing_ids = set(args.ids) - set(q["id"] for q in targets)
        if missing_ids:
            print(f"Warning: question ID(s) not found in golden set: {missing_ids}")
    else:
        print("No --ids given, auto-detecting current misses (standard top-10 filtered eval)...")
        miss_ids = find_current_misses(config, retrieve_module, golden_questions)
        targets = [q for q in golden_questions if q["id"] in miss_ids]
        print(f"Found {len(targets)} current miss(es): {[q['id'] for q in targets]}")

    if not targets:
        print("Nothing to diagnose.")
        return

    for q in targets:
        diagnose(config, retrieve_module, q, args.limit)


if __name__ == "__main__":
    main()
