"""
Diagnostic: inspect how Docling actually serializes one specific table.

Run this locally (needs `docling` + `transformers` installed) against the
10-Q you sent Claude, to see the real chunked output for the income
statement table before deciding whether any cleanup logic is needed.

Usage:
    python inspect_table_output.py path/to/10-Q_2021-12-25.htm
"""

import sys

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.labels import DocItemLabel
from transformers import AutoTokenizer


def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_table_output.py path/to/file.htm")
        sys.exit(1)

    path = sys.argv[1]

    print("Converting document...")
    doc = DocumentConverter().convert(source=path).document

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5"),
        max_tokens=512,
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    chunks = list(chunker.chunk(dl_doc=doc))
    print(f"Total chunks: {len(chunks)}")

    # Find chunks that contain a table and mention "net sales" -- our
    # income statement candidate from the manual inspection.
    matches = []
    for i, chunk in enumerate(chunks):
        labels = {item.label for item in chunk.meta.doc_items}
        if DocItemLabel.TABLE in labels and "net sales" in chunk.text.lower():
            matches.append((i, chunk))

    print(f"Matching table chunks: {len(matches)}\n")

    for i, chunk in matches:
        enriched = chunker.contextualize(chunk=chunk)
        token_count = tokenizer.count_tokens(enriched)
        print(f"--- Chunk {i} ({token_count} tokens) ---")
        print(enriched)
        print()

    if not matches:
        print("No matching chunk found -- try adjusting the search string, "
              "or print chunk.text for every table-labeled chunk to locate "
              "the income statement manually.")


if __name__ == "__main__":
    main()
