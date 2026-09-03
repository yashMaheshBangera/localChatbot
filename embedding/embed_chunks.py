"""
Embed chunked SEC filings by calling a vLLM-served embedding model.

Serve the model first (separate terminal):

    vllm serve BAAI/bge-large-en-v1.5 --task embed

Then run this script:

    python embed_chunks.py

WHY THIS DOESN'T USE langchain_openai.OpenAIEmbeddings:
That class pre-tokenizes text client-side with tiktoken (OpenAI's tokenizer)
before sending integer token IDs to the server. bge-large-en-v1.5 uses a
BERT vocabulary, not tiktoken's -- so a tiktoken token ID sent to vLLM often
doesn't correspond to anything valid in BERT's ~30K-token vocabulary,
producing a "Token id out of vocabulary" 400 error (a real, documented
issue against this exact model). This script instead sends raw text
strings directly to vLLM's /v1/embeddings endpoint over plain HTTP -- vLLM
tokenizes with the model's OWN correct tokenizer server-side, which avoids
the mismatch entirely.

WHY CHUNKS ARE LENGTH-CHECKED BEFORE SENDING:
bge-large-en-v1.5 has a hard 512-token limit; vLLM errors (rather than
silently truncating) on anything longer. Table chunks from parse_and_chunk.py
are deliberately kept atomic/unsplit, so a large table can plausibly exceed
512 tokens. This script checks token length with the same tokenizer used
for chunking, and truncates with a logged warning rather than letting one
oversized table crash the whole batch.

NOTE ON TESTING: this script is written against vLLM's documented OpenAI-
compatible /v1/embeddings API but has NOT been run against a live vLLM
server (no network/GPU in the dev sandbox). Run it against a small batch
first and inspect the output before trusting it on the full corpus --
same discipline applied throughout this project.
"""

import json
from pathlib import Path

import requests
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

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


def embed_batch(base_url: str, model: str, texts: list, api_key: str | None = None) -> list:
    """Sends raw text strings (not pre-tokenized IDs) to a vLLM /v1/embeddings
    endpoint. Returns a list of embedding vectors in the same order as texts."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = requests.post(
        f"{base_url.rstrip('/')}/v1/embeddings",
        headers=headers,
        json={"model": model, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # /v1/embeddings responses aren't guaranteed to preserve input order --
    # each item carries its own "index", so sort defensively rather than
    # trust response order matching request order.
    data.sort(key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def load_chunks(processed_dir: Path):
    """Yields (source_path, record) for every chunk across all .chunks.jsonl
    files under processed_dir."""
    for path in sorted(processed_dir.rglob("*.chunks.jsonl")):
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield path, json.loads(line)


def main():
    config, project_root = load_config()
    processed_dir = resolve_dir(project_root, config, "processed_dir", "data/processed")
    embedded_dir = resolve_dir(project_root, config, "embedded_dir", "data/embedded")
    embedded_dir.mkdir(parents=True, exist_ok=True)

    embedding_model_id = config["embedding_model_id"]
    max_tokens = config["max_tokens"]
    vllm_base_url = config.get("vllm_base_url", "http://localhost:8000")
    vllm_api_key = config.get("vllm_api_key")  # None if server has no auth
    batch_size = config.get("embedding_batch_size", 32)

    print(f"Loading tokenizer for length-checking ({embedding_model_id})...")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_id)

    all_chunks = list(load_chunks(processed_dir))
    if not all_chunks:
        raise FileNotFoundError(
            f"No .chunks.jsonl files found under {processed_dir} -- "
            f"run parse_and_chunk.py first."
        )
    print(f"Found {len(all_chunks)} chunks across all filings.")

    # Group by source file so output mirrors the processed_dir structure
    by_source = {}
    for source_path, record in all_chunks:
        by_source.setdefault(source_path, []).append(record)

    total_embedded = 0
    total_truncated = 0
    for source_path, records in tqdm(by_source.items(), desc="Embedding filings"):
        rel = source_path.relative_to(processed_dir)
        out_path = embedded_dir / rel
        if out_path.exists():
            continue  # idempotent: skip already-embedded filings

        # Length-check against the model's max tokens. If a chunk is too
        # long (tables in particular, since they're kept atomic/unsplit),
        # the EMBEDDING is computed from a truncated version, but the
        # record's "text" field keeps the FULL original -- so retrieval
        # still finds the chunk (via an imperfect but usable vector) while
        # generation still sees the complete table, not a truncated one.
        # embedding_truncated makes this explicit rather than silent, so
        # retrieval/eval code can account for it if needed (e.g. weighting
        # such matches differently, or flagging them for a future rerun
        # once a longer-context embedding model is used).
        texts = []
        truncated_flags = []
        for r in records:
            token_ids = tokenizer.encode(r["text"], truncation=False)
            if len(token_ids) > max_tokens:
                tqdm.write(
                    f"  [warn] {rel} chunk {r['chunk_id']} ({r['chunk_type']}) "
                    f"is {len(token_ids)} tokens, truncating to {max_tokens} "
                    f"for embedding (full text kept in the record)"
                )
                token_ids = token_ids[:max_tokens]
                texts.append(tokenizer.decode(token_ids, skip_special_tokens=True))
                truncated_flags.append(True)
                total_truncated += 1
            else:
                texts.append(r["text"])
                truncated_flags.append(False)

        # Batch requests rather than one-per-chunk, for throughput
        embeddings = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                embeddings.extend(
                    embed_batch(vllm_base_url, embedding_model_id, batch, vllm_api_key)
                )
        except requests.RequestException as e:
            tqdm.write(f"  [error] {rel}: {e}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for record, embedding, was_truncated in zip(records, embeddings, truncated_flags):
                record_with_embedding = dict(record)
                record_with_embedding["embedding"] = embedding
                record_with_embedding["embedding_truncated"] = was_truncated
                f.write(json.dumps(record_with_embedding) + "\n")

        total_embedded += len(records)

    print(f"\nDone. {total_embedded} chunks embedded "
          f"({total_truncated} truncated for exceeding {max_tokens} tokens) "
          f"under {embedded_dir}/")


if __name__ == "__main__":
    main()
