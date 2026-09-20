"""
Streamlit demo UI for the local SEC EDGAR RAG pipeline.

Wraps the same functions generate.py and generate_batch.py already use
(embed -> search -> optional rerank -> period-filter -> generate) behind a
minimal chat-style interface, for portfolio demo purposes only -- this is
not meant to be a production frontend (no auth, no rate limiting, no
concurrency handling -- see README's "Scope" section for why that's an
explicit, considered decision for a portfolio project rather than an
oversight).

Prerequisites: the same three vLLM servers + Qdrant as generate.py itself
(see generate.py's own docstring for exact commands).

Usage:
    streamlit run demo/app.py
"""

import sys
from pathlib import Path

import streamlit as st

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


@st.cache_resource
def load_modules():
    """Cached so the config/module import only happens once per Streamlit
    process, not on every question."""
    config_path = find_config(SCRIPT_DIR)
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    project_root = config_path.parent

    sys.path.insert(0, str(project_root / "retrieval"))
    sys.path.insert(0, str(project_root / "generation"))
    import retrieve as retrieve_module
    import generate as generate_module

    return config, retrieve_module, generate_module


def answer_question(query: str, ticker, form_type, limit: int,
                     use_rerank: bool, rerank_candidates: int = 50):
    """Runs the real generate.py pipeline for one question -- same
    functions, same order of operations, as an actual generate.py /
    generate_batch.py invocation, including the fiscal-period filter
    (filter_matching_period) established this session to fix the
    citation-period bug."""
    config, retrieve_module, generate_module = load_modules()

    from qdrant_client import QdrantClient
    client = QdrantClient(
        url=config.get("qdrant_url", "http://localhost:6333"),
        api_key=config.get("qdrant_api_key"),
    )
    collection_name = config.get("collection_name", "sec_filings")

    query_vector = retrieve_module.embed_query(
        config.get("vllm_base_url", "http://localhost:8000"),
        config["embedding_model_id"], query,
        config.get("vllm_api_key"),
    )
    query_filter = retrieve_module.build_filter(ticker=ticker, form_type=form_type)

    if use_rerank:
        hits = retrieve_module.search(client, collection_name, query_vector,
                                       query_filter, rerank_candidates)
        hits = retrieve_module.rerank(
            config.get("reranker_base_url", "http://localhost:8001"),
            config.get("reranker_model_id", "BAAI/bge-reranker-base"),
            query, hits, config.get("reranker_api_key"),
        )
    else:
        hits = retrieve_module.search(client, collection_name, query_vector,
                                       query_filter, limit)

    results = retrieve_module.reconstruct_results(client, collection_name, hits)
    results = results[:limit]

    if not results:
        return None, []

    target_years = generate_module.extract_target_years(query)
    results = generate_module.filter_matching_period(results, target_years)

    context_blocks = generate_module.build_context_blocks(results)
    messages = generate_module.build_messages(query, context_blocks)
    answer = generate_module.generate_answer(
        config.get("generation_base_url", "http://localhost:8002"),
        config.get("generation_model_id", "microsoft/Phi-4-mini-instruct"),
        messages, config.get("generation_api_key"),
    )
    return answer, results


st.set_page_config(page_title="SEC Filings RAG Demo", page_icon="\U0001F4CA", layout="centered")
st.title("\U0001F4CA SEC Filings RAG — Local Demo")
st.caption(
    "Fully self-hosted RAG over 10-K/10-Q filings for AAPL, MSFT, JPM, GS, JNJ, "
    "PFE, XOM, WMT, KO, TSLA. Embedding, reranking, and generation all run "
    "locally via vLLM — no external API calls."
)

with st.sidebar:
    st.header("Retrieval settings")
    ticker = st.text_input("Ticker filter (optional)", "").strip().upper() or None
    form_type = st.selectbox("Form type", [None, "10-K", "10-Q"],
                              format_func=lambda x: x or "Any")
    limit = st.slider("Context chunks (--limit)", 2, 10, 6)
    use_rerank = st.checkbox("Use reranker", value=True)
    st.divider()
    st.caption(
        "This demo calls the same local vLLM servers used throughout "
        "development — embedding, reranking, and generation each on "
        "their own port, plus a local Qdrant instance. Nothing leaves "
        "this machine."
    )

query = st.text_input(
    "Ask a question about one of the ten companies' filings:",
    placeholder="What was Apple's total gross margin in fiscal year 2022?",
)

if st.button("Ask", type="primary") and query:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            answer, results = answer_question(query, ticker, form_type, limit, use_rerank)
        except Exception as e:
            st.error(
                "Something went wrong — is the full pipeline running? "
                "(embedding / reranker / generation vLLM servers + Qdrant)\n\n"
                f"{e}"
            )
            st.stop()

    if answer is None:
        st.warning("No relevant context found for this query.")
    else:
        st.subheader("Answer")
        st.text(answer)

        st.subheader("Sources")
        for i, r in enumerate(results, 1):
            p = r["payload"]
            period = (p.get("fiscal_period_end") or "")[:10]
            st.markdown(
                f"**[{i}]** {p.get('ticker')} {p.get('form_type')} "
                f"(period ending {period}) — {p.get('section')}  \n"
                f"[{p.get('source_url')}]({p.get('source_url')})"
            )
