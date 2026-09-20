"""
Visualize the embedded chunk corpus by reducing 1024-dim embeddings down to
2D (PCA) and plotting them, colored by ticker.

This is a genuine sanity check, not just a pretty picture: if chunks from
the same company cluster together, and/or table chunks separate visually
from text chunks, that's real evidence the embeddings captured meaningful
structure rather than being noise. If everything looks like a uniform
blob with no structure at all, that's worth investigating before trusting
this data going into Qdrant.

Usage:
    pip install scikit-learn matplotlib
    python embedding/visualize_embeddings.py

Samples up to --per-ticker chunks from each ticker (default 300) rather
than loading and plotting the full ~41K corpus, since PCA and the plot
itself get slow and visually unreadable at that scale, and a
representative sample shows the same clustering structure.
"""

import argparse
import json
import random
import re
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

_ITEM_PATTERN = re.compile(r'^(PART|Item)\s+([IVXLC0-9]+[A-Za-z]?)', re.IGNORECASE)


def extract_item_category(section):
    """Extracts a normalized top-level content category (e.g. "Item 7",
    "Item 1A", "Part IV") from a chunk's section path, for grouping chunks
    by which standardized part of the filing they came from. Item numbers
    are standardized across every 10-K/10-Q filer (Item 1 = Business, Item
    1A = Risk Factors, Item 7 = MD&A, Item 8 = Financial Statements, etc.),
    so this is real content-type metadata already captured during parsing
    -- no topic modeling needed to get this level of categorization."""
    if not section:
        return "(no section)"
    top_level = section.split(" > ")[0].strip()
    m = _ITEM_PATTERN.match(top_level)
    if m:
        prefix, identifier = m.groups()
        # .upper() rather than .title() for the identifier -- title()
        # mangles roman numerals ("IV" -> "Iv"), .upper() handles both
        # roman numerals and letter suffixes ("7A") correctly.
        return f"{prefix.capitalize()} {identifier.upper()}"
    return "(other)"


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


def load_sample(embedded_dir: Path, per_ticker: int, seed: int = 0):
    """Loads up to per_ticker chunks from each ticker subdirectory,
    sampled randomly (not just the first N) so the sample isn't biased
    toward whichever filing happens to sort first."""
    rng = random.Random(seed)
    records_by_ticker = {}

    for ticker_dir in sorted(embedded_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        all_records = []
        for path in ticker_dir.glob("*.chunks.jsonl"):
            with path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        all_records.append(json.loads(line))
        if not all_records:
            continue
        sampled = rng.sample(all_records, min(per_ticker, len(all_records)))
        records_by_ticker[ticker] = sampled

    return records_by_ticker


def print_separation_scores(embeddings, tickers, chunk_types, categories, min_group_size=10):
    """Computes silhouette scores -- the standard metric for 'how well
    separated are these labeled groups' -- directly on the FULL embedding
    space, not a lossy 2D PCA projection. This is a more rigorous check
    than eyeballing a scatter plot: PCA's top 2 components only capture a
    small fraction of total variance and can be dominated by one strong
    structural signal (e.g. text vs table), visually burying a real but
    subtler topical signal that a plot alone wouldn't reveal.

    Score ranges from -1 to 1: near 0 means no real separation (groups
    overlap as much as random chance would suggest), positive and
    increasingly large means the groups are genuinely distinguishable in
    the embedding space, negative means points are actually closer to
    other groups than their own. Uses cosine distance, the standard
    choice for comparing text embeddings.

    Groups smaller than min_group_size are dropped before scoring --
    silhouette score is unreliable and can be misleadingly noisy on very
    small groups (e.g. an Item that only had 2-3 sampled chunks).

    ALSO runs a Linear Discriminant Analysis (LDA) check alongside the raw
    full-space score. A low full-space score is genuinely ambiguous
    between two different explanations: (a) the labels carry no real
    signal in this embedding space, or (b) there IS a real signal, just
    concentrated in a small subspace and diluted when averaged uniformly
    across all ~1000+ dimensions (most of which encode unrelated content).
    Unlike PCA, LDA explicitly finds the direction(s) that MAXIMIZE label
    separation, so silhouette computed on the LDA projection distinguishes
    these two cases empirically instead of guessing.

    LDA is fit on a TRAIN split and evaluated only on a held-out TEST
    split, not the same data it was fit on. This matters: an earlier
    version without the split gave a FALSE POSITIVE of +0.13 (above the
    naive "+0.1 is meaningful" bar) on a synthetic case constructed to
    have zero real signal -- with ~1024 dimensions and often only a few
    hundred samples per group, LDA has enough freedom to find an
    apparently-separating direction purely by overfitting to noise.
    Evaluating on held-out data is what actually distinguishes a real,
    generalizing signal from that kind of illusion. Verified against two
    controlled synthetic cases before trusting this: a real-but-diluted
    signal (5 of 1024 dims) correctly recovers a high held-out LDA score,
    while a genuinely-signal-free case correctly stays low on held-out
    data (unlike the non-cross-validated version, which falsely showed
    separation on that same noise)."""
    from sklearn.metrics import silhouette_score
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from collections import Counter

    print("\n--- Quantitative separation check (full 1024-dim space, not PCA) ---")

    for label_name, labels in [("ticker", tickers), ("chunk_type", chunk_types),
                                 ("filing section (Item #)", categories)]:
        counts = Counter(labels)
        keep_groups = {g for g, c in counts.items() if c >= min_group_size}
        mask = [l in keep_groups for l in labels]
        n_kept = sum(mask)
        n_groups = len(keep_groups)

        if n_groups < 2 or n_kept < min_group_size * 2:
            print(f"  {label_name}: not enough groups/samples with >= "
                  f"{min_group_size} chunks each to score, skipping")
            continue

        filtered_embeddings = embeddings[mask]
        filtered_labels = [l for l, m in zip(labels, mask) if m]

        full_score = silhouette_score(filtered_embeddings, filtered_labels, metric='cosine')

        # Cross-validated LDA: fit on a TRAIN split only, then check
        # whether the separation holds up on a HELD-OUT test split. This
        # matters because with ~1024 dimensions and often only a few
        # hundred samples per group, LDA can find an apparently-separating
        # direction purely by overfitting to noise -- confirmed with a
        # synthetic all-noise test case that produced a false-positive
        # LDA score of +0.13 (above the naive +0.1 "meaningful" bar) when
        # fit and evaluated on the SAME data with no train/test split.
        # Evaluating only on held-out data is what actually distinguishes
        # a real, generalizing signal from an overfit illusion.
        from sklearn.model_selection import train_test_split
        try:
            idx = np.arange(len(filtered_labels))
            train_idx, test_idx = train_test_split(
                idx, test_size=0.3, random_state=0,
                stratify=filtered_labels,
            )
            lda = LinearDiscriminantAnalysis()
            lda.fit(filtered_embeddings[train_idx], [filtered_labels[i] for i in train_idx])
            test_projected = lda.transform(filtered_embeddings[test_idx])
            test_labels = [filtered_labels[i] for i in test_idx]
            lda_score = silhouette_score(test_projected, test_labels, metric='euclidean')
        except Exception:
            lda_score = None

        dropped = len(labels) - n_kept
        line = (f"  {label_name}: full-space = {full_score:+.4f}, "
                f"LDA-projected = {lda_score:+.4f}" if lda_score is not None
                else f"  {label_name}: full-space = {full_score:+.4f}, LDA failed")
        line += (f" ({n_groups} groups, {n_kept} chunks scored"
                 + (f", {dropped} dropped for undersized groups" if dropped else "")
                 + ")")
        print(line)

        if lda_score is not None:
            if lda_score > full_score + 0.2:
                print(f"    -> LDA recovers meaningfully more separation: real signal, "
                      f"diluted across the full space (most dims encode other content)")
            elif lda_score < 0.1:
                print(f"    -> LDA does NOT recover strong separation either: the low "
                      f"full-space score reflects a genuine lack of separation for this "
                      f"grouping, not just a measurement artifact")
            else:
                print(f"    -> LDA and full-space scores are in the same rough range: "
                      f"no strong evidence of a diluted-but-real signal")

    print(
        "\n  Full-space score alone is genuinely ambiguous when low -- could mean "
        "either 'no real signal' or 'real signal, diluted across many "
        "dimensions'. The LDA comparison above distinguishes those two cases "
        "empirically rather than assuming one or the other."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-ticker", type=int, default=300,
                         help="Max chunks to sample per ticker (default 300)")
    parser.add_argument("--out", type=str, default="embedding_space.png",
                         help="Output image filename")
    args = parser.parse_args()

    import yaml
    config_path = find_config(SCRIPT_DIR)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    project_root = config_path.parent
    embedded_dir = (project_root / config.get("embedded_dir", "data/embedded")).resolve()

    if not embedded_dir.exists():
        raise FileNotFoundError(
            f"{embedded_dir} not found -- run embed_chunks.py first."
        )

    print(f"Sampling up to {args.per_ticker} chunks per ticker from {embedded_dir}...")
    records_by_ticker = load_sample(embedded_dir, args.per_ticker)

    if not records_by_ticker:
        raise FileNotFoundError(f"No embedded chunks found under {embedded_dir}")

    all_records = [r for recs in records_by_ticker.values() for r in recs]
    print(f"Loaded {len(all_records)} chunks across {len(records_by_ticker)} tickers: "
          f"{', '.join(sorted(records_by_ticker.keys()))}")

    embeddings = np.array([r["embedding"] for r in all_records])
    tickers = [r["ticker"] for r in all_records]
    chunk_types = [r["chunk_type"] for r in all_records]
    categories = [extract_item_category(r.get("section")) for r in all_records]

    print(f"Embedding matrix shape: {embeddings.shape}")
    norms = np.linalg.norm(embeddings, axis=1)
    is_normalized = bool(np.allclose(norms, 1.0, atol=1e-3))
    print(f"Vector norms: min={norms.min():.3f}, max={norms.max():.3f}, mean={norms.mean():.3f} "
          + ("(L2-normalized -- every vector has norm ~1.0, typical for "
             "embedding models optimized for cosine similarity)"
             if is_normalized else
             "(NOT L2-normalized -- a range rather than all exactly 1.0)"))

    unique_categories = sorted(set(categories))
    print(f"Content categories found: {len(unique_categories)} -- "
          f"{', '.join(unique_categories)}")

    print_separation_scores(embeddings, tickers, chunk_types, categories)

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    print("Running PCA (1024 dims -> 2)...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_
    print(f"Explained variance: PC1={explained[0]:.1%}, PC2={explained[1]:.1%} "
          f"(low numbers are normal for PCA on high-dim embeddings -- this is a "
          f"rough 2D shadow of a 1024-dim space, not the full picture)")

    unique_tickers = sorted(set(tickers))
    colors = mpl.colormaps.get_cmap('tab10').resampled(len(unique_tickers))
    ticker_to_color = {t: colors(i) for i, t in enumerate(unique_tickers)}

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Left plot: colored by ticker
    ax = axes[0]
    for t in unique_tickers:
        mask = [tk == t for tk in tickers]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=t, alpha=0.6, s=15,
                   color=ticker_to_color[t])
    ax.set_title("Embedding space colored by ticker")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.legend(markerscale=2, fontsize=8, loc='best')

    # Middle plot: colored by chunk_type (text vs table)
    ax = axes[1]
    type_colors = {"text": "tab:blue", "table": "tab:orange"}
    for ct, color in type_colors.items():
        mask = [c == ct for c in chunk_types]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=ct, alpha=0.5, s=15, color=color)
    ax.set_title("Embedding space colored by chunk type")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.legend(markerscale=2, fontsize=8, loc='best')

    # Right plot: colored by SEC Item category (e.g. Item 7 = MD&A) --
    # real, deterministic content-type metadata already captured during
    # parsing, not something requiring topic modeling to obtain.
    ax = axes[2]
    cat_colors = mpl.colormaps.get_cmap('tab20').resampled(max(len(unique_categories), 1))
    for i, cat in enumerate(unique_categories):
        mask = [c == cat for c in categories]
        pts = coords[mask]
        ax.scatter(pts[:, 0], pts[:, 1], label=cat, alpha=0.6, s=15,
                   color=cat_colors(i))
    ax.set_title("Embedding space colored by filing section (Item #)")
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1%} var)")
    ax.legend(markerscale=2, fontsize=7, loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    out_path = SCRIPT_DIR / args.out
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {out_path}")
    print(
        "\nWhat to look for: if same-ticker points cluster together in the "
        "left plot, the embeddings are capturing company-specific vocabulary "
        "and content -- a good sign. If text/table points separate in the "
        "middle plot, that's expected (very different content shapes). In "
        "the right plot, chunks from the same Item (e.g. all the Item 8 "
        "financial-statement chunks, or all the Item 1A risk-factor chunks) "
        "clustering together would be strong evidence the embeddings "
        "captured genuine topical structure, not just surface features. A "
        "uniform, structureless blob in any plot would be worth "
        "investigating before trusting this data in Qdrant."
    )


if __name__ == "__main__":
    main()
