"""
Batch-runs generation (not just retrieval) across every golden question,
so faithfulness spot-checks don't rely on picking a handful of queries by
hand. Reuses generate.py's actual functions -- same retrieval, prompt
construction, and generation call as a real `generate.py` invocation, just
looped over the whole golden set with a report at the end.

Built after finding three real generation-faithfulness bugs (fabricated
sums, metric substitution, unit-conversion drift) through ad hoc manual
testing on just 2-3 questions -- given a bug turned up nearly every time a
new query was tried, the untested majority of the golden set is a real
gap worth closing, not an assumed-clean area.

Prerequisites: same three vLLM servers + Qdrant as generate.py itself
(see generate.py's own docstring for exact commands).

Usage:
    python eval/generate_batch.py
    python eval/generate_batch.py --rerank
    python eval/generate_batch.py --ids q3 q6 q12
"""

import argparse
import json
import sys
from pathlib import Path
from langsmith import traceable

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


def load_config_and_modules():
    config_path = find_config(SCRIPT_DIR)
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    project_root = config_path.parent

    retrieval_dir = project_root / "retrieval"
    generation_dir = project_root / "generation"
    sys.path.insert(0, str(retrieval_dir))
    sys.path.insert(0, str(generation_dir))
    import retrieve as retrieve_module
    import generate as generate_module

    return config, retrieve_module, generate_module


def load_golden_questions(path: Path) -> list:
    with open(path) as f:
        return json.load(f)

@traceable(name="rag_eval_question", run_type="chain")
def run_one(config, retrieve_module, generate_module, question: dict,
            limit: int, use_rerank: bool, rerank_candidates: int) -> dict:
    """Runs the real generate.py pipeline for a single question -- same
    functions, same order of operations, as an actual `generate.py`
    invocation. Returns the answer, sources, and a rough automated check
    of whether the expected fact appears anywhere in the answer text."""
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=config.get("qdrant_url", "http://localhost:6333"),
        api_key=config.get("qdrant_api_key"),
    )
    collection_name = config.get("collection_name", "sec_filings")

    query_vector = retrieve_module.embed_query(
        config.get("vllm_base_url", "http://localhost:8000"),
        config["embedding_model_id"], question["question"],
        config.get("vllm_api_key"),
    )
    query_filter = retrieve_module.build_filter(
        ticker=question.get("ticker"), form_type=question.get("form_type"),
    )

    if use_rerank:
        hits = retrieve_module.search(client, collection_name, query_vector,
                                       query_filter, rerank_candidates)
        hits = retrieve_module.rerank(
            config.get("reranker_base_url", "http://localhost:8001"),
            config.get("reranker_model_id", "BAAI/bge-reranker-base"),
            question["question"], hits, config.get("reranker_api_key"),
        )
    else:
        hits = retrieve_module.search(client, collection_name, query_vector,
                                       query_filter, limit)

    results = retrieve_module.reconstruct_results(client, collection_name, hits)
    results = results[:limit]

    target_years = generate_module.extract_target_years(question["question"])
    results = generate_module.filter_matching_period(results, target_years)

    if not results:
        return {
            "id": question["id"], "question": question["question"],
            "answer": None, "sources": [], "answer_contains_expected": False,
            "no_context": True,
        }

    context_blocks = generate_module.build_context_blocks(results)
    messages = generate_module.build_messages(question["question"], context_blocks)
    answer = generate_module.generate_answer(
        config.get("generation_base_url", "http://localhost:8002"),
        config.get("generation_model_id", "microsoft/Phi-4-mini-instruct"),
        messages, config.get("generation_api_key"),
    )

    expected = question["expected_context_contains"]
    return {
        "id": question["id"], "question": question["question"],
        "answer": answer,
        "sources": generate_module.format_sources(results),
        "answer_contains_expected": expected in answer,
        "no_context": False,
    }


def print_report(rows: list):
    for r in rows:
        print(f"\n{'=' * 90}")
        print(f"[{r['id']}] {r['question']}")
        print('=' * 90)
        if r["no_context"]:
            print("(no retrieval results at all -- generation was not called)")
            continue
        marker = "CONTAINS expected fact" if r["answer_contains_expected"] else \
                 "does NOT contain expected fact (may be a correct refusal, or a real miss -- read it)"
        print(f"[{marker}]")
        print(f"\nAnswer:\n{r['answer']}")
        print(f"\nSources:\n{r['sources']}")

    n = len(rows)
    n_contains = sum(1 for r in rows if r["answer_contains_expected"])
    n_no_context = sum(1 for r in rows if r["no_context"])
    print(f"\n{'=' * 90}")
    print(f"{n_contains}/{n} answers contain the expected fact "
          f"({n_no_context} had no retrieval context at all)")
    print("This is a rough proxy, not a substitute for reading each answer -- "
          "an answer can contain the right number and still misattribute it "
          "(as happened with the real 'net interest income' vs 'net revenues' "
          "bug), and a correct refusal ('not enough information') will always "
          "show as NOT containing the expected fact, which is the right "
          "outcome, not a failure.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="+", default=None,
                         help="Specific golden question IDs to run (default: all)")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--rerank-candidates", type=int, default=50)
    parser.add_argument("--golden-set", type=str, default=None)
    args = parser.parse_args()

    config, retrieve_module, generate_module = load_config_and_modules()

    golden_path = Path(args.golden_set) if args.golden_set else (SCRIPT_DIR / "golden_questions.json")
    golden_questions = load_golden_questions(golden_path)

    if args.ids:
        golden_questions = [q for q in golden_questions if q["id"] in args.ids]
        missing = set(args.ids) - set(q["id"] for q in golden_questions)
        if missing:
            print(f"Warning: question ID(s) not found in golden set: {missing}")

    print(f"Running generation for {len(golden_questions)} question(s)"
          f"{' with reranking' if args.rerank else ''}...")

    rows = []
    for q in golden_questions:
        rows.append(run_one(config, retrieve_module, generate_module, q,
                             args.limit, args.rerank, args.rerank_candidates))

    print_report(rows)


if __name__ == "__main__":
    main()
