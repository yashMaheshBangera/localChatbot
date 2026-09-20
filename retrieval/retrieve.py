"""
Retrieval layer: embed a user's question through the same vLLM endpoint
used for the corpus, search Qdrant with vector similarity + optional
metadata filters, and reconstruct any split tables from their sibling
pieces before returning results.

Prerequisites (both running):
    vllm serve BAAI/bge-large-en-v1.5 --runner pooling
    cd ~/qdrant && ./qdrant

Usage:
    python retrieval/retrieve.py --query "What was Apple's gross margin in 2022?"
    python retrieval/retrieve.py --query "JPM risk factors" --ticker JPM --form-type 10-K
    python retrieval/retrieve.py --query "revenue trends" --ticker AAPL --ticker MSFT --limit 5

WHY QUERY EMBEDDING REUSES embed_chunks.py's HTTP APPROACH, NOT A NEW ONE:
Same reasoning as the embedding stage -- raw text sent directly to vLLM's
/v1/embeddings endpoint, not langchain_openai.OpenAIEmbeddings (which
pre-tokenizes with tiktoken client-side, wrong vocabulary for a BERT-based
model, a real documented bug against this exact model). A query and a
corpus chunk MUST go through the identical embedding process, or the
resulting vectors aren't comparable -- this isn't just code reuse for its
own sake, it's a correctness requirement.

WHY SPLIT TABLES GET RECONSTRUCTED HERE, NOT LEFT AS FRAGMENTS:
This is the retrieval-time half of a design decision made back at the
parsing stage: table_id/group_index/group_count exist specifically so a
table that got split into multiple chunks (to give each piece an accurate,
untruncated embedding) can be reassembled into the full table at query
time. Search finds fragments precisely (each with its own accurate
vector); reconstruction here ensures the LLM that eventually generates an
answer sees the complete table, not just whichever piece happened to match
the query -- see README's "Sibling-linking metadata" section for the full
reasoning and the context-loss problem this solves.
"""

import argparse
import json
from pathlib import Path

import requests

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


def load_config() -> tuple:
    config_path = find_config(SCRIPT_DIR)
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config, config_path.parent


def embed_query(base_url: str, model: str, query: str, api_key=None) -> list:
    """Embeds a single query string via vLLM's /v1/embeddings endpoint.
    Same raw-HTTP approach as embed_chunks.py's embed_batch() -- see
    module docstring for why this matters, not just style consistency."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/embeddings",
        headers=headers,
        json={"model": model, "input": [query]},
        timeout=60,
    )
    if not resp.ok:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} for url {resp.url}\n"
            f"Response body: {resp.text[:2000]}"
        )
    data = resp.json()["data"]
    return data[0]["embedding"]


def build_filter(ticker=None, tickers=None, form_type=None, chunk_type=None,
                  fiscal_period_start=None, fiscal_period_end=None):
    """Builds a Qdrant Filter from optional constraints. Returns None if
    nothing was specified (an unfiltered similarity search). ticker (one)
    and tickers (a list) are both accepted for CLI convenience; only one
    should be used per call."""
    from qdrant_client import models

    conditions = []

    if ticker:
        conditions.append(models.FieldCondition(
            key="ticker", match=models.MatchValue(value=ticker)
        ))
    elif tickers:
        conditions.append(models.FieldCondition(
            key="ticker", match=models.MatchAny(any=list(tickers))
        ))

    if form_type:
        conditions.append(models.FieldCondition(
            key="form_type", match=models.MatchValue(value=form_type)
        ))

    if chunk_type:
        conditions.append(models.FieldCondition(
            key="chunk_type", match=models.MatchValue(value=chunk_type)
        ))

    if fiscal_period_start or fiscal_period_end:
        range_kwargs = {}
        if fiscal_period_start:
            range_kwargs["gte"] = f"{fiscal_period_start}T00:00:00Z"
        if fiscal_period_end:
            range_kwargs["lte"] = f"{fiscal_period_end}T00:00:00Z"
        conditions.append(models.FieldCondition(
            key="fiscal_period_end", range=models.DatetimeRange(**range_kwargs)
        ))

    if not conditions:
        return None
    return models.Filter(must=conditions)


def search(client, collection_name: str, query_vector: list, query_filter=None, limit: int = 10):
    """Pure dense search. Reverted from hybrid (dense+BM25 RRF fusion)
    after measuring it against the golden set: hybrid made every metric
    WORSE, not better (Hit Rate@10 filtered 76.9% -> 69.2%, MRR filtered
    0.351 -> 0.262) -- a real, measured regression, not a hunch. See
    README's "Hybrid search" section for the full comparison and the
    leading hypothesis (BM25 tokenization likely splitting numbers like
    "58,283" on the comma, losing the exact-match benefit the whole
    approach was built around).

    The collection still has both 'dense' and 'sparse' named vectors
    (from the hybrid attempt) -- reverting the QUERY logic to
    dense-only, rather than reloading the collection back to a
    single-vector schema too, since the sparse vectors are harmless to
    leave populated and unused, and another full reload isn't worth
    doing twice. `using="dense"` is required now (unlike the original
    single-anonymous-vector collection) since Qdrant needs to know which
    named vector to search when a collection has more than one."""
    result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using="dense",
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return result.points


_RERANKER_MAX_CONTEXT = 512  # BAAI/bge-reranker-base's own limit, confirmed via the real 400 error
_RERANKER_SPECIAL_TOKEN_BUFFER = 8  # [CLS]/[SEP] around query+document, same safety margin as embed_chunks.py
_reranker_tokenizer_cache = {}


def _get_reranker_tokenizer(model: str):
    """Loads (and caches) the reranker's own tokenizer, same approach as
    embed_chunks.py's real tokenizer use for the embedding model -- not a
    rough character-count guess."""
    if model not in _reranker_tokenizer_cache:
        from transformers import AutoTokenizer
        _reranker_tokenizer_cache[model] = AutoTokenizer.from_pretrained(model)
    return _reranker_tokenizer_cache[model]


def _truncate_document_for_reranker(tokenizer, query_text: str, document_text: str) -> str:
    """Truncates document_text (not query_text) so that query + document
    together fit within the reranker's own context limit.

    Real, confirmed bug this fixes: a cross-encoder reranker takes the
    query AND document CONCATENATED as one input
    ("[CLS] query [SEP] document [SEP]") -- unlike the embedding model,
    which only ever sees one piece of text at a time. Chunks were sized
    to fit near 512 tokens ON THEIR OWN for embedding; adding any query
    text on top of an already-near-512-token chunk easily exceeds the
    reranker's separate 512-token limit. Confirmed via a real error:
    "This model's maximum context length is 512 tokens... your prompt
    contains at least 513 input tokens." Truncates the DOCUMENT, not the
    query -- the query is short and needs to stay intact; the document is
    what has room to give."""
    query_tokens = len(tokenizer.encode(query_text, add_special_tokens=False))
    budget_for_document = _RERANKER_MAX_CONTEXT - query_tokens - _RERANKER_SPECIAL_TOKEN_BUFFER

    doc_token_ids = tokenizer.encode(document_text, add_special_tokens=False)
    if len(doc_token_ids) <= budget_for_document:
        return document_text

    truncated_ids = doc_token_ids[:max(budget_for_document, 0)]
    return tokenizer.decode(truncated_ids, skip_special_tokens=True)


def rerank(base_url: str, model: str, query_text: str, hits: list, api_key=None):
    """Re-scores a list of dense-search hits using a cross-encoder
    reranker (e.g. BAAI/bge-reranker-base) served via vLLM's
    Cohere-compatible /rerank endpoint, and returns them re-sorted by the
    reranker's relevance score, highest first.

    Unlike dense/BM25 retrieval -- where the query and each chunk are
    each converted into a representation independently, then compared --
    a cross-encoder reads the query and one candidate's actual text
    TOGETHER, letting it directly attend between them. This is far more
    precise, but too expensive to run against the whole corpus, which is
    why it's always a SECOND stage: it can only re-order candidates
    retrieval already found, never retrieve something new. If the
    correct chunk wasn't in `hits` to begin with (see diagnose_misses.py
    -- confirmed true for q3 and q12), reranking cannot fix that; it
    specifically targets the "found but ranked too low" pattern (q6, and
    the broader low-MRR pattern across q1/q2/q4/q5/q7/q8/q13).

    Documents are truncated to fit the reranker's own combined
    query+document context limit before sending -- see
    _truncate_document_for_reranker for why this is necessary (a real,
    confirmed error, not a precaution). Ranking itself is unaffected by
    truncating the tail of a long document; the reranker still sees the
    vast majority of the content, just not literally every token.

    Explicitly sorts by relevance_score client-side rather than trusting
    the response is already ordered -- the API contract only guarantees
    each result carries a score, not that the list arrives pre-sorted."""
    if not hits:
        return hits

    tokenizer = _get_reranker_tokenizer(model)
    documents = [
        _truncate_document_for_reranker(tokenizer, query_text, h.payload.get("text", ""))
        for h in hits
    ]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        f"{base_url.rstrip('/')}/rerank",
        headers=headers,
        json={"model": model, "query": query_text, "documents": documents},
        timeout=60,
    )
    if not resp.ok:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} for url {resp.url}\n"
            f"Response body: {resp.text[:2000]}"
        )

    results = resp.json()["results"]
    results.sort(key=lambda r: r["relevance_score"], reverse=True)

    reranked = []
    for r in results:
        original_hit = hits[r["index"]]
        # Wrap rather than mutate original_hit.score in place -- qdrant_client's
        # ScoredPoint objects aren't guaranteed mutable, and this avoids
        # depending on that implementation detail.
        reranked.append(_RerankedHit(score=r["relevance_score"], payload=original_hit.payload))
    return reranked


class _RerankedHit:
    """Minimal stand-in for a qdrant ScoredPoint, carrying the reranker's
    score instead of the original dense-search score. Only needs .score
    and .payload -- the only two attributes reconstruct_results() and
    check_hit() actually read."""
    def __init__(self, score, payload):
        self.score = score
        self.payload = payload


def fetch_table_siblings(client, collection_name: str, table_id: str):
    """Fetches every chunk sharing the given table_id, sorted by
    group_index, via scroll (not search -- we want an exact metadata
    lookup, not a similarity ranking)."""
    from qdrant_client import models

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(
                key="table_id", match=models.MatchValue(value=table_id)
            )]
        ),
        limit=100,  # generous; no real table should ever split into more pieces than this
        with_payload=True,
    )
    return sorted(points, key=lambda p: p.payload.get("group_index", 0))


def reconstruct_results(client, collection_name: str, hits: list) -> list:
    """Post-processes raw search hits: any hit that's part of a split
    table (group_count > 1) gets its sibling pieces fetched and merged
    into one combined text, so generation sees the complete table rather
    than whichever single fragment matched the query. De-duplicates: if
    multiple pieces of the SAME table both appear in the raw hits, only
    one consolidated result is returned for that table, using the best
    (highest) score among its matching pieces.

    Pieces are concatenated in group_index order without removing
    repeated header rows -- a deliberate simplicity choice (see README):
    correct and complete either way, and the redundancy is a minor
    verbosity cost against a generation LLM's much larger context budget,
    not a correctness problem. Column-split tables (multiple complete
    per-period sub-tables) read naturally in this same concatenated
    order too, not just row-split ones."""
    seen_table_ids = {}
    results = []

    for hit in hits:
        table_id = hit.payload.get("table_id")
        group_count = hit.payload.get("group_count", 1)

        if table_id and group_count and group_count > 1:
            if table_id in seen_table_ids:
                existing = seen_table_ids[table_id]
                if hit.score > existing["score"]:
                    existing["score"] = hit.score
                continue

            siblings = fetch_table_siblings(client, collection_name, table_id)
            combined_text = "\n\n".join(s.payload.get("text", "") for s in siblings)

            merged_payload = dict(hit.payload)
            merged_payload["text"] = combined_text
            merged_payload["reconstructed_from_pieces"] = len(siblings)

            result = {"score": hit.score, "payload": merged_payload}
            seen_table_ids[table_id] = result
            results.append(result)
        else:
            results.append({"score": hit.score, "payload": dict(hit.payload)})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def format_results(results: list) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        p = r["payload"]
        lines.append(f"\n--- Result {i} (score={r['score']:.4f}) ---")
        lines.append(f"{p.get('ticker')} {p.get('form_type')} "
                      f"{p.get('fiscal_period_end', '')[:10]} | {p.get('section')}")
        if p.get("reconstructed_from_pieces"):
            lines.append(f"[Reconstructed from {p['reconstructed_from_pieces']} table pieces]")
        text = p.get("text", "")
        preview = text[:400] + ("..." if len(text) > 400 else "")
        lines.append(preview)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Natural language question")
    parser.add_argument("--ticker", action="append", default=None,
                         help="Filter to one or more tickers (repeatable)")
    parser.add_argument("--form-type", default=None, help="e.g. 10-K or 10-Q")
    parser.add_argument("--chunk-type", default=None, choices=["text", "table"])
    parser.add_argument("--fiscal-period-start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--fiscal-period-end", default=None, help="YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rerank", action="store_true",
                         help="Retrieve a wider candidate set via dense search, then "
                              "re-score with a cross-encoder reranker (needs a second "
                              "vLLM instance serving a reranker model -- see README). "
                              "Only re-orders candidates dense search already found; "
                              "cannot surface a chunk dense search missed entirely.")
    parser.add_argument("--rerank-candidates", type=int, default=50,
                         help="How many candidates to retrieve for the reranker to "
                              "choose from before narrowing to --limit (only used "
                              "with --rerank)")
    args = parser.parse_args()

    config, project_root = load_config()

    query_vector = embed_query(
        config.get("vllm_base_url", "http://localhost:8000"),
        config["embedding_model_id"],
        args.query,
        config.get("vllm_api_key"),
    )

    from qdrant_client import QdrantClient
    client = QdrantClient(
        url=config.get("qdrant_url", "http://localhost:6333"),
        api_key=config.get("qdrant_api_key"),
    )

    tickers = args.ticker if args.ticker and len(args.ticker) > 1 else None
    ticker = args.ticker[0] if args.ticker and len(args.ticker) == 1 else None

    query_filter = build_filter(
        ticker=ticker,
        tickers=tickers,
        form_type=args.form_type,
        chunk_type=args.chunk_type,
        fiscal_period_start=args.fiscal_period_start,
        fiscal_period_end=args.fiscal_period_end,
    )

    collection_name = config.get("collection_name", "sec_filings")

    if args.rerank:
        hits = search(client, collection_name, query_vector, query_filter, args.rerank_candidates)
        hits = rerank(
            config.get("reranker_base_url", "http://localhost:8001"),
            config.get("reranker_model_id", "BAAI/bge-reranker-base"),
            args.query, hits, config.get("reranker_api_key"),
        )
    else:
        hits = search(client, collection_name, query_vector, query_filter, args.limit)

    results = reconstruct_results(client, collection_name, hits)
    results = results[:args.limit]  # only actually truncates anything when --rerank widened the pool

    print(f"Query: {args.query!r}")
    print(f"Filter: {query_filter}")
    if args.rerank:
        print(f"Reranked {len(hits)} candidates (retrieved {args.rerank_candidates}) "
              f"-> {len(results)} final results after table reconstruction")
    else:
        print(f"{len(hits)} raw hits -> {len(results)} results after table reconstruction")
    print(format_results(results))


if __name__ == "__main__":
    main()
