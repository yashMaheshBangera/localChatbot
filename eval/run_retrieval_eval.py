"""
Judge-free retrieval evaluation: for each question in golden_questions.json,
run it through the real retrieval pipeline (embed_query + search +
reconstruct_results, same functions retrieve.py uses) and check whether the
known-correct fact actually appears in the retrieved results.

WHY THIS ISN'T "REAL" RAGAS YET:
RAGAS's core metrics -- context precision, context recall, faithfulness,
answer relevancy -- need an LLM to judge relevance or a generated answer to
assess, neither of which exist until a generation LLM is picked and served
(see README's "Next step"). Rather than block retrieval evaluation on that,
this computes Hit Rate@K: a simpler, judge-free, immediately-runnable
metric that answers "did retrieval actually find the right chunk" without
needing any LLM to grade it. This is complementary to RAGAS, not a
replacement -- the golden set built here is exactly what full RAGAS
metrics will consume once a judge/generation LLM exists.

WHY THE GOLDEN SET USES FACTS ALREADY VERIFIED EARLIER IN THIS PROJECT:
Every entry in golden_questions.json traces back to a real number this
project directly confirmed by inspecting actual filing HTML -- Apple's
gross margin table, Pfizer's Paxlovid revenue (the exact row that
motivated split_oversized_row()), JNJ's fair value text, Coca-Cola's
audit-matters tax figures (the exact text that motivated the abbreviation-
splitting fix), JPMorgan's Markets revenue table (the exact table that
motivated column-group splitting). Not fabricated plausible-sounding
figures -- reused ground truth this project already established.

Tests BOTH filtered (using the golden question's ticker/form_type) and
unfiltered (pure semantic search) retrieval for each question, so the
report shows how much metadata filtering actually helps over similarity
search alone -- a genuinely interesting number given how much of this
project's design centered on exact metadata filtering.

Usage:
    python eval/run_retrieval_eval.py
    python eval/run_retrieval_eval.py --limit 5   # top-K to check against
"""

import argparse
import json
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


def load_golden_questions(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def check_hit(results: list, expected_substring: str):
    """Returns (hit: bool, rank: int or None, matched_ticker: str or None).
    rank is 1-indexed position of the first result whose text contains the
    expected substring; None if no result matched."""
    for i, r in enumerate(results, 1):
        text = r["payload"].get("text", "")
        if expected_substring in text:
            return True, i, r["payload"].get("ticker")
    return False, None, None


def run_eval(retrieve_module, config, golden_questions: list, limit: int,
             use_rerank: bool = False, rerank_candidates: int = 50) -> dict:
    """Runs every golden question through the real retrieval pipeline in
    both filtered and unfiltered modes. Returns a results dict with
    per-question outcomes and aggregate hit rates.

    If use_rerank is True, ALSO computes a third mode -- filtered +
    reranked -- retrieving rerank_candidates via dense search, re-scoring
    with the cross-encoder reranker, then narrowing to `limit`. This is
    what actually answers "does reranking help", rather than assuming it
    does: same discipline as measuring hybrid search directly rather than
    guessing, which is exactly what caught hybrid search being a real
    regression instead of an improvement."""
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=config.get("qdrant_url", "http://localhost:6333"),
        api_key=config.get("qdrant_api_key"),
    )
    collection_name = config.get("collection_name", "sec_filings")

    per_question = []
    for q in golden_questions:
        query_vector = retrieve_module.embed_query(
            config.get("vllm_base_url", "http://localhost:8000"),
            config["embedding_model_id"],
            q["question"],
            config.get("vllm_api_key"),
        )

        # Unfiltered: pure semantic search, no metadata help
        hits_unfiltered = retrieve_module.search(client, collection_name, query_vector,
                                                  query_filter=None, limit=limit)
        results_unfiltered = retrieve_module.reconstruct_results(client, collection_name, hits_unfiltered)
        hit_u, rank_u, ticker_u = check_hit(results_unfiltered, q["expected_context_contains"])

        # Filtered: using the golden question's known correct ticker/form_type
        # (simulates a system that has correctly identified these, e.g. via
        # entity extraction on the user's question -- not built yet, this
        # tests the retrieval side assuming that part works)
        query_filter = retrieve_module.build_filter(
            ticker=q.get("ticker"), form_type=q.get("form_type"),
        )
        hits_filtered = retrieve_module.search(client, collection_name, query_vector,
                                                query_filter=query_filter, limit=limit)
        results_filtered = retrieve_module.reconstruct_results(client, collection_name, hits_filtered)
        hit_f, rank_f, ticker_f = check_hit(results_filtered, q["expected_context_contains"])

        result_row = {
            "id": q["id"], "question": q["question"],
            "unfiltered_hit": hit_u, "unfiltered_rank": rank_u,
            "filtered_hit": hit_f, "filtered_rank": rank_f,
            "verification_method": q.get("verification_method", "unknown"),
        }

        if use_rerank:
            hits_wide = retrieve_module.search(client, collection_name, query_vector,
                                                query_filter=query_filter, limit=rerank_candidates)
            hits_reranked = retrieve_module.rerank(
                config.get("reranker_base_url", "http://localhost:8001"),
                config.get("reranker_model_id", "BAAI/bge-reranker-base"),
                q["question"], hits_wide, config.get("reranker_api_key"),
            )
            results_reranked = retrieve_module.reconstruct_results(client, collection_name, hits_reranked)
            results_reranked = results_reranked[:limit]
            hit_r, rank_r, ticker_r = check_hit(results_reranked, q["expected_context_contains"])
            result_row["reranked_hit"] = hit_r
            result_row["reranked_rank"] = rank_r

        per_question.append(result_row)

    n = len(per_question)
    hit_rate_unfiltered = sum(1 for r in per_question if r["unfiltered_hit"]) / n if n else 0
    hit_rate_filtered = sum(1 for r in per_question if r["filtered_hit"]) / n if n else 0

    # Mean Reciprocal Rank: complementary to Hit Rate, since Hit Rate alone
    # treats "found at rank 1" and "found at rank 10" identically (both
    # just count as a hit) -- MRR specifically rewards higher ranks,
    # 1/rank per question, 0 for a miss.
    mrr_unfiltered = sum(1 / r["unfiltered_rank"] if r["unfiltered_hit"] else 0
                          for r in per_question) / n if n else 0
    mrr_filtered = sum(1 / r["filtered_rank"] if r["filtered_hit"] else 0
                        for r in per_question) / n if n else 0

    # Separate, more rigorous hit rate restricted to raw_html-verified
    # questions only -- excludes any question whose ground truth was
    # sourced from this project's own pipeline output rather than an
    # independent check against the primary source document. Reporting
    # both numbers rather than silently blending them, since a
    # pipeline-sourced "ground truth" can't catch a bug in that same
    # pipeline -- it would just canonize it.
    raw_html_questions = [r for r in per_question if r["verification_method"] == "raw_html"]
    n_raw = len(raw_html_questions)
    hit_rate_filtered_raw_html = (
        sum(1 for r in raw_html_questions if r["filtered_hit"]) / n_raw if n_raw else None
    )

    if use_rerank:
        hit_rate_reranked = sum(1 for r in per_question if r["reranked_hit"]) / n if n else 0
        mrr_reranked = sum(1 / r["reranked_rank"] if r["reranked_hit"] else 0
                            for r in per_question) / n if n else 0
        report_extra = {"hit_rate_reranked": hit_rate_reranked, "mrr_reranked": mrr_reranked}
    else:
        report_extra = {}

    return {
        "per_question": per_question,
        "hit_rate_unfiltered": hit_rate_unfiltered,
        "hit_rate_filtered": hit_rate_filtered,
        "hit_rate_filtered_raw_html_only": hit_rate_filtered_raw_html,
        "n_raw_html_questions": n_raw,
        "mrr_unfiltered": mrr_unfiltered,
        "mrr_filtered": mrr_filtered,
        "n_questions": n,
        "limit": limit,
        "use_rerank": use_rerank,
        **report_extra,
    }


def print_report(report: dict):
    use_rerank = report.get("use_rerank", False)
    header = f"\n{'Q':4s} {'Unfiltered':^18s} {'Filtered':^18s}"
    if use_rerank:
        header += f" {'Reranked':^18s}"
    header += f" {'Verified':^10s}  Question"
    print(header)
    print("-" * (100 + (19 if use_rerank else 0)))

    for r in report["per_question"]:
        u = f"HIT @{r['unfiltered_rank']}" if r["unfiltered_hit"] else "MISS"
        f = f"HIT @{r['filtered_rank']}" if r["filtered_hit"] else "MISS"
        v = r["verification_method"]
        v_display = v if v == "raw_html" else f"*{v}"  # flag non-raw_html entries
        row = f"{r['id']:4s} {u:^18s} {f:^18s}"
        if use_rerank:
            reranked = f"HIT @{r['reranked_rank']}" if r["reranked_hit"] else "MISS"
            row += f" {reranked:^18s}"
        row += f" {v_display:^10s}  {r['question'][:45]}"
        print(row)

    print("-" * (100 + (19 if use_rerank else 0)))
    print(f"Hit Rate@{report['limit']} (unfiltered, pure semantic search): "
          f"{report['hit_rate_unfiltered']:.1%} ({report['n_questions']} questions)")
    print(f"Hit Rate@{report['limit']} (filtered by ticker+form_type):     "
          f"{report['hit_rate_filtered']:.1%} ({report['n_questions']} questions)")
    if use_rerank:
        print(f"Hit Rate@{report['limit']} (filtered + reranked):              "
              f"{report['hit_rate_reranked']:.1%} ({report['n_questions']} questions)")
    if report["hit_rate_filtered_raw_html_only"] is not None:
        print(f"Hit Rate@{report['limit']} (filtered, raw_html-verified questions ONLY): "
              f"{report['hit_rate_filtered_raw_html_only']:.1%} "
              f"({report['n_raw_html_questions']} questions)")
        print("(This is the more rigorous number -- excludes any question whose ground "
              "truth came from this pipeline's own output rather than an independent "
              "check against the primary source document. * marks those in the table above.)")
    print(f"MRR       (unfiltered, pure semantic search): {report['mrr_unfiltered']:.3f}")
    print(f"MRR       (filtered by ticker+form_type):     {report['mrr_filtered']:.3f}")
    if use_rerank:
        print(f"MRR       (filtered + reranked):              {report['mrr_reranked']:.3f}")
    print("(MRR is sensitive to rank, not just hit/miss -- 1.0 means every "
          "question's answer was found at rank 1; Hit Rate alone can't "
          "distinguish that from finding it at rank 10.)")

    if report["hit_rate_filtered"] > report["hit_rate_unfiltered"]:
        diff = report["hit_rate_filtered"] - report["hit_rate_unfiltered"]
        print(f"\nMetadata filtering improved hit rate by {diff:.1%} over pure semantic search.")
    elif report["hit_rate_filtered"] < report["hit_rate_unfiltered"]:
        print(f"\n[unexpected] Filtered hit rate is LOWER than unfiltered -- worth investigating, "
              f"since a correct ticker/form_type filter should never make the true match harder to find.")

    if use_rerank:
        if report["hit_rate_reranked"] > report["hit_rate_filtered"]:
            diff = report["hit_rate_reranked"] - report["hit_rate_filtered"]
            print(f"Reranking improved hit rate by {diff:.1%} over filtered dense search alone.")
        elif report["hit_rate_reranked"] < report["hit_rate_filtered"]:
            diff = report["hit_rate_filtered"] - report["hit_rate_reranked"]
            print(f"Reranking made hit rate WORSE by {diff:.1%} vs. filtered dense search alone "
                  f"-- same thing that happened with hybrid search; don't assume this helped, "
                  f"trust these numbers.")
        else:
            print("Reranking made no difference to hit rate (may still affect MRR -- check that too).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10, help="Top-K results to check against")
    parser.add_argument("--golden-set", type=str, default=None,
                         help="Path to golden questions JSON (default: golden_questions.json next to this script's project root)")
    parser.add_argument("--rerank", action="store_true",
                         help="Also compute a filtered+reranked mode alongside unfiltered/filtered, "
                              "for direct comparison (needs a second vLLM instance serving a reranker)")
    parser.add_argument("--rerank-candidates", type=int, default=50,
                         help="How many candidates to retrieve for reranking to choose from")
    args = parser.parse_args()

    config_path = find_config(SCRIPT_DIR)
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    project_root = config_path.parent

    # Import retrieve.py's functions -- assumes retrieval/ is a sibling
    # directory of this script, matching the project's established layout.
    retrieval_dir = project_root / "retrieval"
    sys.path.insert(0, str(retrieval_dir))
    import retrieve as retrieve_module

    golden_path = Path(args.golden_set) if args.golden_set else (SCRIPT_DIR / "golden_questions.json")
    if not golden_path.exists():
        raise FileNotFoundError(f"{golden_path} not found")
    golden_questions = load_golden_questions(golden_path)

    print(f"Loaded {len(golden_questions)} golden questions from {golden_path}")
    report = run_eval(retrieve_module, config, golden_questions, args.limit,
                       use_rerank=args.rerank, rerank_candidates=args.rerank_candidates)
    print_report(report)


if __name__ == "__main__":
    main()
