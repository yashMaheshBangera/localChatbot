"""
Load embedded chunks (data/embedded/**/*.chunks.jsonl) into a Qdrant
collection, with the rich metadata (ticker, form_type, fiscal_period_end,
company_name, table_id/group_index/group_count for split-table
reconstruction, etc.) set up as filterable payload fields.

Prerequisite: Qdrant running natively in WSL2 (NOT Docker with a Windows
bind mount -- confirmed, documented data corruption risk on that specific
combination, see README):
    cd ~/qdrant && ./qdrant

Usage:
    pip install -r requirements.txt
    python qdrant/load_qdrant.py

WHY POINT IDS ARE UUIDv5, NOT THE chunk_id STRING DIRECTLY:
Qdrant only accepts 64-bit unsigned integers or proper UUIDs as point IDs
-- arbitrary strings (like our "chunk_a1b2c3d4...") are rejected. UUIDv5
(RFC 4122, name-based) deterministically derives a valid UUID from the
chunk_id string under a fixed namespace: the same chunk_id always produces
the same UUID, which is what makes re-running this script idempotent --
it upserts (updates) the same point rather than creating a duplicate with
a new random ID.

WHY fiscal_period_end/filing_date GET CONVERTED TO FULL RFC 3339 STRINGS:
Qdrant's "datetime" payload index type (supports proper range filtering,
e.g. "filings since 2023-01-01", without manually converting to unix
timestamps) expects RFC 3339 format. Our stored values are plain
"YYYY-MM-DD" dates; converted to "YYYY-MM-DDT00:00:00Z" here to avoid
relying on how strictly a bare date parses under a given client/version.

Idempotent per SOURCE FILE (not just per point): a local manifest
(qdrant/.loaded_files.json) tracks which .chunks.jsonl files have already
been fully upserted, so re-running after adding new tickers/filings only
loads what's new rather than re-upserting the full ~41K+ point corpus
every time. Saved incrementally after each file, so a crash partway
through doesn't lose progress on files already completed.
"""

import json
import uuid
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent

# Fixed namespace for deriving point UUIDs from chunk_id strings -- must
# stay constant across runs, or the same chunk_id would map to a
# different UUID each time, breaking idempotency. Generated once
# (uuid.uuid4()) and hardcoded, not regenerated per run.
POINT_ID_NAMESPACE = uuid.UUID("7f3c9e1a-4b6d-4a2f-9c8e-1d5b3a7f2e6c")


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


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_ID_NAMESPACE, chunk_id))


def to_rfc3339(date_str) -> str:
    """Converts a plain 'YYYY-MM-DD' date string to a full RFC 3339
    datetime string ('YYYY-MM-DDT00:00:00Z'), which Qdrant's datetime
    payload index expects. Passes through unchanged if already
    datetime-shaped or falsy."""
    if not date_str:
        return date_str
    if "T" in str(date_str):
        return date_str
    return f"{date_str}T00:00:00Z"


def load_manifest(manifest_path: Path) -> set:
    if not manifest_path.exists():
        return set()
    with open(manifest_path) as f:
        return set(json.load(f))


def save_manifest(manifest_path: Path, loaded_files: set):
    with open(manifest_path, "w") as f:
        json.dump(sorted(loaded_files), f, indent=2)


def record_to_point(record: dict):
    """Converts one chunk record into a Qdrant PointStruct with TWO named
    vectors: 'dense' (the already-computed bge-large embedding, unchanged
    from before) and 'sparse' (BM25, computed server-side by Qdrant itself
    from the chunk's raw text via model="Qdrant/bm25" -- no separate
    model to serve, no extra embedding step on this side).

    Built to address a real, measured weakness found via
    diagnose_misses.py: specific numeric/table content (e.g. "Goldman
    Sachs net revenues... $58,283") was losing out to generic narrative
    prose in dense-only retrieval, even when the prose was less relevant
    -- exactly the failure mode BM25's exact term/number matching is
    supposed to help with. Fusion happens at query time (see retrieve.py);
    this function just needs to supply both vectors per point.

    Imports qdrant_client lazily (inside the function, not at module
    load) so this module's pure logic -- point_id_for, to_rfc3339,
    manifest handling -- can be unit tested without qdrant_client
    installed."""
    from qdrant_client import models

    payload = {k: v for k, v in record.items() if k != "embedding"}
    if "fiscal_period_end" in payload:
        payload["fiscal_period_end"] = to_rfc3339(payload["fiscal_period_end"])
    if "filing_date" in payload:
        payload["filing_date"] = to_rfc3339(payload["filing_date"])

    return models.PointStruct(
        id=point_id_for(record["chunk_id"]),
        vector={
            "dense": record["embedding"],
            "sparse": models.Document(text=record["text"], model="Qdrant/bm25"),
        },
        payload=payload,
    )


PAYLOAD_INDEXES = [
    # (field_name, schema_type)
    ("ticker", "keyword"),
    ("company_name", "keyword"),
    ("cik", "integer"),
    ("form_type", "keyword"),
    ("fiscal_period_end", "datetime"),
    ("filing_date", "datetime"),
    ("chunk_type", "keyword"),
    ("section", "keyword"),
    ("table_id", "keyword"),
    ("group_index", "integer"),
    ("group_count", "integer"),
    ("row_too_large", "bool"),
    ("embedding_truncated", "bool"),
    # Every year explicitly mentioned in the chunk's own text (see
    # extract_years_mentioned in parse_and_chunk.py) -- an array field;
    # Qdrant indexes each element individually, so a filter like
    # years_mentioned=2024 matches any chunk whose text mentions 2024,
    # regardless of how many other years it also discusses. Indexed now
    # so it's available for retrieval-time filtering later, even though
    # the immediate use (README's "Generation" section) is surfacing it
    # in the generation prompt, not filtering search results with it yet.
    ("years_mentioned", "integer"),
]


def ensure_collection(client, collection_name: str, vector_size: int):
    """Creates the collection if it doesn't already exist, then ensures
    every payload index in PAYLOAD_INDEXES is present. Both parts are
    safe to call repeatedly -- collection creation is skipped if it
    already exists, and creating an already-existing payload index is a
    no-op in Qdrant rather than an error.

    Two NAMED vectors, not one anonymous vector like before: 'dense'
    (cosine distance, matching bge-large's confirmed L2-normalized
    output) and 'sparse' (BM25, no distance metric -- sparse vectors use
    their own comparison, not a distance config). This is a genuine
    schema change from the single-vector collection used before hybrid
    search, so it requires a fresh collection -- an existing single-vector
    collection can't just have a sparse vector bolted on without also
    renaming its one anonymous vector to 'dense' first."""
    from qdrant_client import models

    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )
        print(f"Created collection '{collection_name}' "
              f"(dense: size={vector_size}, distance=Cosine; sparse: BM25)")
    else:
        print(f"Collection '{collection_name}' already exists, reusing it")

    schema_map = {
        "keyword": models.PayloadSchemaType.KEYWORD,
        "integer": models.PayloadSchemaType.INTEGER,
        "bool": models.PayloadSchemaType.BOOL,
        "datetime": models.PayloadSchemaType.DATETIME,
    }
    for field_name, schema_type in PAYLOAD_INDEXES:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=schema_map[schema_type],
        )
    print(f"Ensured {len(PAYLOAD_INDEXES)} payload indexes")


def main():
    config, project_root = load_config()
    embedded_dir = (project_root / config.get("embedded_dir", "data/embedded")).resolve()

    if not embedded_dir.exists():
        raise FileNotFoundError(
            f"{embedded_dir} not found -- run embed_chunks.py first."
        )

    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=config.get("qdrant_url", "http://localhost:6333"),
        api_key=config.get("qdrant_api_key"),
    )
    collection_name = config.get("collection_name", "sec_filings")
    batch_size = config.get("qdrant_batch_size", 256)

    all_files = sorted(embedded_dir.rglob("*.chunks.jsonl"))
    if not all_files:
        raise FileNotFoundError(f"No .chunks.jsonl files found under {embedded_dir}")

    manifest_path = SCRIPT_DIR / ".loaded_files.json"
    loaded_files = load_manifest(manifest_path)

    # Peek at the first not-yet-loaded file's first record to determine
    # vector size from the actual data, rather than hardcoding a number
    # that would silently go stale if the embedding model ever changes.
    vector_size = None
    for path in all_files:
        if str(path.relative_to(embedded_dir)) in loaded_files:
            continue
        with path.open() as f:
            first_line = f.readline()
            if first_line.strip():
                vector_size = len(json.loads(first_line)["embedding"])
                break

    if vector_size is not None:
        ensure_collection(client, collection_name, vector_size)
    else:
        print("Nothing new to load -- all files already in the manifest.")
        return

    total_upserted = 0
    total_skipped_files = 0

    for path in tqdm(all_files, desc="Loading files into Qdrant"):
        rel = str(path.relative_to(embedded_dir))
        if rel in loaded_files:
            total_skipped_files += 1
            continue

        records = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        points = [record_to_point(r) for r in records]

        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(collection_name=collection_name, points=batch)

        total_upserted += len(points)
        loaded_files.add(rel)
        save_manifest(manifest_path, loaded_files)  # incremental, survives a crash

    print(f"\nDone. {total_upserted} points upserted, "
          f"{total_skipped_files} already-loaded files skipped.")
    count = client.count(collection_name=collection_name).count
    print(f"Collection '{collection_name}' now has {count} total points.")


if __name__ == "__main__":
    main()
