"""
Generation layer: takes a user's question, retrieves context via the same
pipeline as retrieve.py (embed -> search -> optional rerank -> table
reconstruction), builds a grounded prompt instructing the model to answer
ONLY from the provided context and cite sources, calls the generation LLM,
and prints the answer with a numbered source list mapping back to
source_url.

Prerequisites (all three running):
    vllm serve BAAI/bge-large-en-v1.5 --runner pooling          # embeddings, port 8000
    vllm serve BAAI/bge-reranker-base --port 8001                # reranker (only if --rerank), port 8001
    VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
      vllm serve microsoft/Phi-4-mini-instruct --port 8002 \
      --gpu-memory-utilization 0.8 --max-model-len 8192 --enforce-eager
                                                                   # generation, port 8002 -- see
                                                                   # README's "Troubleshooting" for
                                                                   # the full five-incident story
                                                                   # behind this exact command on WSL2
    cd ~/qdrant && ./qdrant                                       # Qdrant, port 6333

Usage:
    python generation/generate.py --query "What was Apple's gross margin in fiscal 2022?"
    python generation/generate.py --query "..." --ticker AAPL --form-type 10-K --rerank

WHY microsoft/Phi-4-mini-instruct (served at full precision, no quantization):
Chosen after checking current model options against this project's actual
hardware constraint (a 12GB GPU already running the embedding model and
reranker) -- see README's "Generation" section for the full reasoning,
including two real things checked and ruled out along the way: Qwen 3.5
(newest release, but has a confirmed vLLM text-only-serving bug for its
new hybrid architecture) and DeepSeek's small local models (not actually
DeepSeek's own architecture -- they're distillations onto Qwen or Llama
base models). Phi-4-mini-instruct: ~3.8B params, ~7.17GB bf16 weights
(confirmed exact from vLLM's own load log) -- fits without any
quantization once CUDA graph capture (via --enforce-eager) and the
unnecessary 128K context default (via --max-model-len 8192) are both
removed from the memory budget. Quantization was tried first and
abandoned after two distinct real failures (see README's
"Troubleshooting" for the full story: a version-incompatible
pre-quantized checkpoint, then bitsandbytes turning out to have been
deliberately removed from this exact vLLM version's core) -- dropping
quantization entirely turned out to be more robust than continuing to
chase compatibility with a fast-moving vLLM release. Mature vLLM
support, 128K context
window (confirmed via multiple independent sources, not assumed), MIT
licensed.

WHY THE PROMPT REQUIRES CITATIONS AND EXPLICIT "I DON'T KNOW":
This is a RAG system over real SEC filings -- ungrounded or fabricated
financial figures are a genuinely bad failure mode, not just an
inconvenience. Requiring a citation for every claim, and an explicit
admission when the context doesn't contain the answer, is the whole point
of retrieval-augmented generation over a base model's parametric memory:
it makes faithfulness checkable (both by a human reading the sources, and
later by RAGAS's faithfulness metric) rather than just plausible-sounding.
"""

import argparse
import re
import sys
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = """You are a financial research assistant answering questions about SEC filings (10-K and 10-Q reports).

Answer the user's question using ONLY the information in the numbered context sources below. Follow these rules strictly:
1. Cite the source number(s) in brackets (e.g. [1], [2]) immediately after every specific fact, figure, or claim you state.
2. If the provided context does not contain enough information to answer the question, say so explicitly. Do not guess, and do not use any knowledge beyond what's given in the context.
3. Be precise with numbers -- reproduce figures exactly as given in the context, do not round or approximate. Preserve the exact unit of measure stated in the source (e.g. if a source states a figure "in millions", state your answer in millions too, using the same number -- do not convert it to billions, thousands, or any other unit, since this risks a scale error).
4. If multiple sources give conflicting information, note the discrepancy rather than picking one silently.
5. If the user's question explicitly asks for a sum, difference, or other combination of two or more figures, and each figure being combined is individually and unambiguously stated in the context, you may perform that calculation -- show which figures you combined and cite each one's source, and express the result in the same unit as the source figures (per rule 3 -- do not convert units as part of combining them). However, do not reconstruct a specific company-reported total or headline figure (e.g. "total net revenues", "total revenue") by adding together OTHER, different line items when the filing's own reported total is not itself stated in the context. You cannot verify such an assembled total is complete or matches what the company actually reported -- treat this case as insufficient information under rule 2 instead.
6. A figure for a different, related metric is not a substitute for the specific metric asked about, even if it is real, correctly cited, and numerically close. For example, "net interest income" is a component of a company's revenue, not the same thing as its "net revenues" or "total revenue" -- do not answer a question about one metric using a figure explicitly labeled as a different one. If the context contains only related-but-different metrics, and not the specific one asked about, treat this as insufficient information under rule 2.
7. PERIOD MATCHING IS MANDATORY: Before using any number, check that source's period/fiscal year (shown in its header or "discusses:" years) against what the question asks for. If multiple sources contain similar or related figures, you MUST select only the source whose period matches the question's stated year or quarter — even if another source's number looks more prominent or appears first. If NO provided source matches the requested period, say so explicitly rather than substituting a different period's figure."""

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


_SUBHEADING_PATTERN = re.compile(r"^([A-Z][A-Za-z ]{2,40}?)\.(?=[A-Z])")


def extract_subheading(text: str) -> str | None:
    """Detects a chunk's own bolded lead-in label, when present, e.g.
    "Net Interest Income.Net interest income in the consolidated..." ->
    "Net Interest Income". SEC filing MD&A sections commonly bold a
    short subsection label directly followed (no space, a parsing
    artifact from the original HTML) by a sentence that restates it in
    lowercase -- exactly the "Heading.heading in lowercase..." pattern
    this regex targets, distinct from an ordinary sentence or an
    abbreviation like "U.S." (both excluded by requiring a capital
    letter with no space immediately after the period, and a 2-40
    character label).

    Real, confirmed problem this addresses: a chunk's `section` payload
    field often only captures the broader PARENT section (e.g. "Net
    Revenues"), not the specific subsection a given chunk actually
    contains (e.g. "Net Interest Income", one component of net
    revenues, not the total). Feeding the model a citation header
    labeled with only the parent section materially contributed to it
    answering a "net revenues" question using a "net interest income"
    figure -- confirmed by inspecting the chunk's raw text directly."""
    match = _SUBHEADING_PATTERN.match(text)
    if match:
        return match.group(1).strip()
    return None

_TARGET_YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def extract_target_years(query: str) -> set[int]:
    """Extracts every fiscal year explicitly named in the user's question
    (e.g. "fiscal year 2022" -> {2022}; "fiscal years 2022 and 2023" ->
    {2022, 2023}). Used to filter retrieved context to matching-period
    sources before generation, rather than relying on prompt instructions
    alone -- confirmed via generate_batch.py that Phi-4-mini-instruct does
    not reliably enforce period-matching from instruction text alone, even
    with an explicit rule and the correct-period source present in
    context."""
    return {int(y) for y in _TARGET_YEAR_PATTERN.findall(query)}


def filter_matching_period(results: list, target_years: set[int]) -> list:
    """Keeps only chunks whose OWN filing period (fiscal_period_end's year)
    matches the question's target year(s). Deliberately does NOT also match
    on years_mentioned: confirmed that MD&A sections routinely discuss a
    rolling multi-year comparison in the same paragraph (e.g. a FY2024
    filing's Gross Margin section mentions 2022, 2023, and 2024 together),
    so years_mentioned alone is too permissive to discriminate between
    filings -- it matched almost every candidate regardless of which
    fiscal year the filing itself actually covers, silently defeating this
    filter entirely on the first attempt. fiscal_period_end is the
    authoritative source: it says definitively which period THIS filing
    covers, not just which years its text happens to reference."""
    if not target_years:
        return results

    def matches(r):
        p = r["payload"]
        period = p.get("fiscal_period_end") or ""
        period_year = int(period[:4]) if period[:4].isdigit() else None
        return period_year in target_years

    filtered = [r for r in results if matches(r)]
    return filtered if filtered else results

def build_context_blocks(results: list) -> list:
    """Formats each retrieved chunk with a citation number and enough
    metadata (ticker, form_type, period, section, years discussed) for
    the model -- and a human reading the sources afterward -- to know
    exactly which filing and which period a claim came from. Appends the
    chunk's own detected sub-heading (see extract_subheading), when
    present, since the section metadata alone can be too broad to
    distinguish a subsection figure from the total it's part of --
    confirmed as a real, contributing cause of a generation error, not a
    hypothetical.

    Also appends `years_mentioned` (see extract_years_mentioned in
    parse_and_chunk.py), when present, as an explicit "discusses: ..."
    label. Built to fix a second, separate confirmed error: prompt rule 7
    alone (requiring the model to check a source's period against the
    question) did not reliably generalize -- generate_batch.py run
    across the full golden set still found the model citing a
    wrong-fiscal-year source when the correct one was ALSO present in
    context, even with rule 7 in place. Stating which years a chunk's
    text actually discusses directly, rather than leaving the model to
    infer this from `fiscal_period_end` (a single date, which doesn't
    capture that a filing often reports prior-year comparatives in the
    same chunk), removes an inference step the model was getting wrong,
    rather than just re-wording the instruction to get it right."""
    blocks = []
    for i, r in enumerate(results, 1):
        p = r["payload"]
        period = (p.get("fiscal_period_end") or "")[:10]
        section = p.get("section")
        subheading = extract_subheading(p.get("text", ""))
        if subheading and subheading.lower() not in (section or "").lower():
            section = f"{section} > {subheading}" if section else subheading
        header = f"[{i}] {p.get('ticker')} {p.get('form_type')} (period ending {period}) - {section}"
        years = p.get("years_mentioned")
        if years:
            header += f" (discusses: {', '.join(str(y) for y in years)})"
        blocks.append(f"{header}\n{p.get('text', '')}")
    return blocks


def build_messages(query: str, context_blocks: list) -> list:
    context_text = "\n\n".join(context_blocks)
    user_message = f"Context sources:\n\n{context_text}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def generate_answer(base_url: str, model: str, messages: list, api_key=None,
                     temperature: float = 0.0, max_tokens: int = 400) -> str:
    """Calls the generation LLM's OpenAI-compatible /v1/chat/completions
    endpoint. temperature=0.0 by default -- this is a factual-QA task
    over financial filings, not creative generation; determinism is
    preferred over sampling diversity.

    max_tokens=400 (down from an earlier, arbitrary 1000): these are
    short, cited factual answers -- typically one or two sentences, on
    the order of 20-50 tokens -- not long-form writing, so 1000 was more
    headroom than the task needs. This matters beyond just efficiency:
    a real overflow was hit running generate_batch.py across the golden
    set (a --max-model-len 7168 server, chosen to fit alongside the
    embedding+reranker servers -- see README's "Troubleshooting" -- had a
    prompt at 6169 input tokens, which together with the OLD 1000-token
    request exceeded the server's total budget by 1 token). Lowering
    max_tokens is the correct fix here, not raising --max-model-len
    (already sized against a real, measured memory constraint) or
    reducing retrieved context (which would throw away real evidence the
    model should have)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=120,
    )
    if not resp.ok:
        raise requests.HTTPError(
            f"{resp.status_code} {resp.reason} for url {resp.url}\n"
            f"Response body: {resp.text[:2000]}"
        )
    return resp.json()["choices"][0]["message"]["content"]


def format_sources(results: list) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        p = r["payload"]
        period = (p.get("fiscal_period_end") or "")[:10]
        lines.append(
            f"[{i}] {p.get('ticker')} {p.get('form_type')} (period ending {period}) - "
            f"{p.get('section')}\n    {p.get('source_url')}"
        )
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
    parser.add_argument("--limit", type=int, default=10,
                         help="Max number of retrieved chunks to include as context")
    parser.add_argument("--rerank", action="store_true",
                         help="Retrieve a wider candidate set and re-score with the "
                              "cross-encoder reranker before building context")
    parser.add_argument("--rerank-candidates", type=int, default=50)
    args = parser.parse_args()

    config, retrieve_module = load_config_and_retrieve_module()

    query_vector = retrieve_module.embed_query(
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
    collection_name = config.get("collection_name", "sec_filings")

    tickers = args.ticker if args.ticker and len(args.ticker) > 1 else None
    ticker = args.ticker[0] if args.ticker and len(args.ticker) == 1 else None
    query_filter = retrieve_module.build_filter(
        ticker=ticker, tickers=tickers, form_type=args.form_type, chunk_type=args.chunk_type,
        fiscal_period_start=args.fiscal_period_start, fiscal_period_end=args.fiscal_period_end,
    )

    if args.rerank:
        hits = retrieve_module.search(client, collection_name, query_vector,
                                       query_filter, args.rerank_candidates)
        hits = retrieve_module.rerank(
            config.get("reranker_base_url", "http://localhost:8001"),
            config.get("reranker_model_id", "BAAI/bge-reranker-base"),
            args.query, hits, config.get("reranker_api_key"),
        )
    else:
        hits = retrieve_module.search(client, collection_name, query_vector,
                                       query_filter, args.limit)

    results = retrieve_module.reconstruct_results(client, collection_name, hits)
    results = results[:args.limit]

    target_years = extract_target_years(args.query)
    results = filter_matching_period(results, target_years)

    if not results:
        print("No relevant context found for this query -- nothing to answer from.")
        return

    context_blocks = build_context_blocks(results)
    messages = build_messages(args.query, context_blocks)
    answer = generate_answer(
        config.get("generation_base_url", "http://localhost:8002"),
        config.get("generation_model_id", "microsoft/Phi-4-mini-instruct"),
        messages,
        config.get("generation_api_key"),
    )

    print(f"Question: {args.query}\n")
    print(f"Answer:\n{answer}\n")
    print(f"Sources:\n{format_sources(results)}")


if __name__ == "__main__":
    main()
