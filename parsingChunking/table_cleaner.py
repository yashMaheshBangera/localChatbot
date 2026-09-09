from bs4 import BeautifulSoup
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


import re

_NUMERIC_CELL = re.compile(r'^[\$\(\)\-\d,\.\%\s]*$')  # digits, $, (), -, comma, period, %, whitespace


def build_grid(table_tag):
    rows = table_tag.find_all('tr')
    parsed_rows = []
    max_cols = 0
    for tr in rows:
        cells = tr.find_all(['td', 'th'])
        row_cells = []
        col_cursor = 0
        for c in cells:
            colspan = int(c.get('colspan', 1) or 1)
            text = c.get_text(" ", strip=True)
            row_cells.append((col_cursor, colspan, text))
            col_cursor += colspan
        max_cols = max(max_cols, col_cursor)
        parsed_rows.append(row_cells)

    grid = []
    for row_cells in parsed_rows:
        row = [""] * max_cols
        for start, colspan, text in row_cells:
            if not text:
                continue  # blank cell -- position doesn't matter, leave grid default ""
            # Numeric/currency values are right-aligned within a merged cell
            # (matches the accounting convention used in these filings, e.g.
            # a colspan=2 "19,516" cell occupies the same visual position as
            # a separate "$" + "104,429" pair in other rows). Labels and
            # headers stay left-aligned at the start of their span.
            if _NUMERIC_CELL.match(text) and colspan > 1:
                col = start + colspan - 1
            else:
                col = start
            if col < max_cols:
                row[col] = text
        grid.append(row)
    return grid


def drop_empty_columns(grid):
    if not grid or not grid[0]:
        return grid
    n_cols = len(grid[0])
    keep = [c for c in range(n_cols) if any(row[c].strip() for row in grid)]
    return [[row[c] for c in keep] for row in grid]


def merge_dollar_columns(grid):
    """Merges a standalone '$' cell into its right-hand neighbor, row by row,
    leaving the source cell blank (same row length preserved so column
    alignment across rows isn't disturbed). The now fully-blank '$' column
    gets removed by a subsequent drop_empty_columns pass."""
    new_grid = []
    for row in grid:
        new_row = list(row)
        for c in range(len(new_row) - 1):
            if new_row[c].strip() == "$":
                nxt = new_row[c + 1].strip()
                new_row[c + 1] = f"${nxt}"
                new_row[c] = ""
        new_grid.append(new_row)
    return new_grid


def merge_percent_columns(grid):
    """Merges a standalone '%' cell into its LEFT-hand neighbor, row by row
    -- the mirror image of merge_dollar_columns. '$' is a leading prefix
    (Products | $ | 104,429), so it merges rightward; '%' is a trailing
    suffix (7 | %), so it merges leftward. Verified against three separate
    tables (Products and Services Performance, Gross Margin, Effective Tax
    Rate) in a real 10-K: the percentage number always sits in a colspan=2
    cell (right-aligned per the numeric rule above) immediately followed by
    a standalone colspan=1 '%' cell -- so merging left always lands on the
    correct number, never on a label. Leaves the source '%' cell blank; the
    now fully-blank '%' column gets removed by a subsequent
    drop_empty_columns pass."""
    new_grid = []
    for row in grid:
        new_row = list(row)
        for c in range(1, len(new_row)):
            if new_row[c].strip() == "%":
                prev = new_row[c - 1].strip()
                new_row[c - 1] = f"{prev}%"
                new_row[c] = ""
        new_grid.append(new_row)
    return new_grid


def grid_to_text(grid):
    lines = []
    for row in grid:
        if not any(cell.strip() for cell in row):
            continue
        lines.append(" | ".join(cell.strip() for cell in row))
    return "\n".join(lines)


def detect_header_row_count(grid, max_header_rows=4, max_header_row_chars=200):
    """Returns how many consecutive LEADING rows are header-like: labels,
    section titles, or period dates like "September 24, 2022" (which
    contain digits but also letters, so they correctly fail _NUMERIC_CELL).
    These rows get repeated at the top of every split group when a table
    is too large for one chunk, so column/period meaning survives.

    Requires BOTH no numeric content AND a short total row length.
    Numeric-only was found insufficient on its own from a real bug: a
    stock-compensation footnote table's data rows contain prose
    descriptions (award terms, vesting conditions) with no numeric cells
    at all -- "no numeric content" alone misclassified the first REAL data
    row (1567 characters) as a header, which then got repeated into every
    split group, making every one of them oversized. True headers/section
    dividers in real filings are consistently short (under ~60 chars in
    the cases seen); 200 is a safety margin above that, not a tight fit.

    Capped at max_header_rows as a further safety bound against degenerate
    tables with many consecutive short label-only rows."""
    count = 0
    for row in grid:
        has_numeric_data = any(
            cell.strip() and _NUMERIC_CELL.match(cell.strip())
            for cell in row
        )
        row_length = sum(len(cell) for cell in row)
        if has_numeric_data or row_length > max_header_row_chars:
            break
        count += 1
        if count >= max_header_rows:
            break
    return count


_YEAR_LABEL = re.compile(r'^(19|20)\d{2}$')  # bare 4-digit year, e.g. "2025"


def detect_period_column_groups(grid):
    """Detects a genuinely WIDE table made of repeating period groups side
    by side -- e.g. JPMorgan's "Markets revenue" table, which reports
    Fixed Income/Equity/Total Markets figures for 2025, 2024, and 2023 all
    in one 22-column row. Splitting such a table by ROWS doesn't help --
    every row is oversized regardless of how few rows are packed together,
    since the problem is column count, not row count.

    Detection anchors on bare year labels ("2025", not "September 24,
    2022" -- which contains letters and is handled by
    detect_header_row_count instead). A bare year matches _NUMERIC_CELL
    (it's just digits), so it wouldn't be caught as a header by that
    function -- this is deliberately a separate, narrower check.

    Returns a list of (start_col, end_col) ranges, one per detected
    period group (column 0, the row label, is NOT included in any range --
    callers should always add it back when extracting a group), or None
    if fewer than 2 year-anchors are found (not a clear repeating-group
    table -- most tables have none of this and should skip straight to
    the normal row-splitting path)."""
    anchor_cols = []
    for row in grid:
        for col_idx, cell in enumerate(row):
            if _YEAR_LABEL.match(cell.strip()):
                anchor_cols.append(col_idx)
        if anchor_cols:
            break  # use the first row that has year anchors

    if len(anchor_cols) < 2:
        return None

    anchor_cols = sorted(set(anchor_cols))
    groups = []
    start = 0
    for end in anchor_cols:
        groups.append((start, end))
        start = end + 1
    return groups


def split_grid_into_column_groups(grid, column_groups):
    """Splits a wide table into multiple narrower grids along
    column_groups boundaries (see detect_period_column_groups). Column 0
    (the row label) is included in every resulting grid, for every row --
    not just the header -- so each group is a complete, independently
    meaningful sub-table (e.g. a full "2025 Markets revenue" breakdown
    with every row correctly labeled), not a column slice missing its row
    labels. Verified against a real 22-column JPMorgan table: an automated
    check confirmed every original data cell (excluding the intentionally-
    repeated label column) appears in exactly one resulting group -- exact
    multiset match, no loss, no duplication."""
    sub_grids = []
    for start, end in column_groups:
        cols = sorted(set([0] + [c for c in range(start, end + 1)]))
        sub_grid = [[row[c] for c in cols if c < len(row)] for row in grid]
        sub_grids.append(sub_grid)
    return sub_grids


_COMMON_ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'dr', 'jr', 'sr', 'vs', 'etc', 'inc', 'co', 'corp',
    'ltd', 'no', 'st', 'ave', 'approx', 'gov', 'assn', 'dept', 'est',
    'vol', 'fig', 'rev', 'gen', 'rep', 'sen', 'prof', 'hon',
}


def _split_into_sentences(text):
    """Splits text into sentences, filtering out false-positive splits at
    abbreviations (e.g. "U.S.", "Mr.", "Inc.") that a naive period-based
    regex would incorrectly treat as sentence ends.

    Found from a real bug: naive splitting on Coca-Cola's audit-matters
    text broke mid-sentence at "U.S." ("...the U.S." | "Tax Court issued
    an opinion...") -- the words weren't lost (an automated word-for-word
    check confirmed that), but the split point was wrong, which would
    degrade retrieval/generation quality on any prose containing common
    abbreviations, not just this one case.

    Not a full NLP sentence tokenizer -- a practical heuristic: skips a
    candidate split point when the token immediately before the period is
    either very short (<=2 letters, catches initials and two-letter
    abbreviations like "U.S.", "U.K.", "Mr", "Dr", "vs" without needing
    them individually listed) or matches a small list of common
    abbreviations 2+ letters long that the length check alone wouldn't
    catch ("Inc.", "Corp.", "etc.")."""
    candidates = list(re.finditer(r'(?<=[.!?])\s+(?=[A-Z])', text))

    sentences = []
    last_end = 0
    for m in candidates:
        split_pos = m.start()
        preceding = text[:split_pos]
        prev_word = re.search(r'([A-Za-z]+)\.$', preceding)
        is_abbreviation = bool(prev_word) and (
            len(prev_word.group(1)) <= 2
            or prev_word.group(1).lower() in _COMMON_ABBREVIATIONS
        )
        if is_abbreviation:
            continue
        sentences.append(text[last_end:split_pos].strip())
        last_end = m.end()
    sentences.append(text[last_end:].strip())
    return [s for s in sentences if s]


def _split_text_into_pieces(text, count_tokens_fn, max_tokens):
    """Splits a large block of text into pieces each within max_tokens,
    preferring natural boundaries. Used only for the rare case of a single
    oversized table cell -- found from a real example: a stock-compensation
    footnote table where a "Terms" column contained multiple bullet-point
    clauses (award vesting conditions, payout ranges, etc) totaling 1600+
    characters in one cell.

    Boundary preference, in order: bullet-point markers (•, ●, ▪) if
    present (common in exactly this kind of footnote), then sentence
    boundaries (abbreviation-aware, see _split_into_sentences), then a
    hard word-count split as a last resort so this always terminates with
    compliant pieces rather than giving up."""
    if count_tokens_fn(text) <= max_tokens:
        return [text]

    bullet_markers = ['\u2022', '\u25cf', '\u25aa']  # bullet, black circle, black square
    found_marker = next((m for m in bullet_markers if m in text), None)

    if found_marker:
        raw_units = text.split(found_marker)
        units = []
        for i, u in enumerate(raw_units):
            u = u.strip()
            if not u:
                continue
            # Re-attach the marker to every unit except a possible leading
            # fragment before the first bullet (rare, but don't invent a
            # bullet that wasn't there in the source).
            units.append(f"{found_marker} {u}" if i > 0 or text.strip().startswith(found_marker) else u)
    else:
        units = _split_into_sentences(text)

    if not units:
        units = [text]

    pieces = []
    current = []
    current_tokens = 0
    for unit in units:
        unit_tokens = count_tokens_fn(unit)

        if unit_tokens > max_tokens:
            # A single bullet/sentence is itself too large (rare) -- hard
            # word-chunk it so this always terminates with compliant output.
            if current:
                pieces.append(' '.join(current))
                current, current_tokens = [], 0
            words = unit.split()
            for i in range(0, len(words), max_tokens):
                pieces.append(' '.join(words[i:i + max_tokens]))
            continue

        if current and current_tokens + unit_tokens > max_tokens:
            pieces.append(' '.join(current))
            current, current_tokens = [], 0

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        pieces.append(' '.join(current))

    return pieces if pieces else [text]


def split_oversized_row(row, count_tokens_fn, budget_for_row):
    """Handles the rare case of a single row too large even with its
    header repeated. Finds whichever cell holds the bulk of the row's
    content and splits ONLY that cell's text (via _split_text_into_pieces),
    producing multiple sub-rows that each repeat the row's OTHER cells --
    e.g. for the real footnote-table case this was built against, each
    sub-row still shows "Senior and other key management" (who the award
    type applies to) alongside just one bullet point of the full
    description, instead of losing that context.

    Falls back to returning the row unchanged (still oversized) if the
    largest cell isn't actually the bottleneck, or splitting it doesn't
    reduce the piece count -- the caller's row_too_large flag still
    catches this rare residual case."""
    if not row:
        return [row]

    cell_tokens = [count_tokens_fn(c) for c in row]
    largest_idx = max(range(len(row)), key=lambda i: cell_tokens[i])
    other_tokens = sum(t for i, t in enumerate(cell_tokens) if i != largest_idx)
    budget_for_cell = budget_for_row - other_tokens - 5  # small safety margin

    if budget_for_cell < 20 or cell_tokens[largest_idx] <= budget_for_cell:
        return [row]  # other cells alone dominate, or it already fits

    pieces = _split_text_into_pieces(row[largest_idx], count_tokens_fn, budget_for_cell)
    if len(pieces) <= 1:
        return [row]  # splitting didn't actually help

    sub_rows = []
    for piece in pieces:
        new_row = list(row)
        new_row[largest_idx] = piece
        sub_rows.append(new_row)
    return sub_rows


def split_grid_into_row_groups(grid, header_row_count, count_tokens_fn, max_tokens):
    """Splits a table's grid into row-groups that each fit within
    max_tokens once rendered to text, repeating the header rows at the top
    of every group so column/period meaning survives the split. Never
    splits a data row's OTHER cells -- only ever expands a row that's
    individually oversized (via split_oversized_row) or splits BETWEEN
    rows.

    If a row is still too large after attempting a cell split (rare -- the
    row's other cells alone dominate the budget, or the oversized cell has
    no usable boundaries to split on), it's emitted as its own still-
    oversized group; embed_chunks.py's truncation-with-safety-buffer
    remains the final fallback for that residual case."""
    header_rows = grid[:header_row_count]
    data_rows = grid[header_row_count:]

    if not data_rows:
        return [grid]  # nothing to split -- header-only or empty table

    header_text = grid_to_text(header_rows) if header_rows else ""
    header_tokens = count_tokens_fn(header_text) if header_text else 0

    # Pre-process: expand any row that (with header) would still exceed
    # max_tokens by splitting its largest cell's content, so the packing
    # pass below mostly just needs to pack rows that already fit.
    expanded_data_rows = []
    for row in data_rows:
        row_tokens = count_tokens_fn(grid_to_text([row]))
        if header_tokens + row_tokens > max_tokens:
            sub_rows = split_oversized_row(row, count_tokens_fn, max_tokens - header_tokens)
            expanded_data_rows.extend(sub_rows)
        else:
            expanded_data_rows.append(row)

    groups = []
    current_rows = []
    current_tokens = header_tokens

    for row in expanded_data_rows:
        row_tokens = count_tokens_fn(grid_to_text([row]))

        if current_rows and current_tokens + row_tokens > max_tokens:
            groups.append(header_rows + current_rows)
            current_rows = []
            current_tokens = header_tokens

        current_rows.append(row)
        current_tokens += row_tokens

    if current_rows:
        groups.append(header_rows + current_rows)

    return groups


def table_to_cleaned_text(table_tag):
    grid = build_grid(table_tag)
    grid = drop_empty_columns(grid)
    grid = merge_dollar_columns(grid)
    grid = merge_percent_columns(grid)
    grid = drop_empty_columns(grid)
    return grid_to_text(grid), grid


def _test_against(soup, search_text, label):
    tables = soup.find_all('table')
    target = None
    for t in tables:
        if search_text in t.get_text(" ", strip=True):
            target = t
            break
    if target is None:
        print(f"[{label}] not found, skipping")
        return

    raw_grid = build_grid(target)
    cleaned_text, cleaned_grid = table_to_cleaned_text(target)
    print(f"--- {label} ---")
    print(f"raw: {len(raw_grid)}x{len(raw_grid[0])}  cleaned: {len(cleaned_grid)}x{len(cleaned_grid[0])}")
    print(cleaned_text)
    print()


def _test_split_oversized_row(soup, table_search_text, row_search_text, label,
                               abbreviation_check=None):
    """Verifies split_oversized_row against a real narrative-cell case:
    one dominant cell (e.g. a "why revenue changed" explanation) alongside
    otherwise-short numeric cells. Checks BOTH that a split actually
    happens under a forced small budget, and -- more importantly -- that
    reconstructing the narrative cell's content across all sub-rows
    matches the original word-for-word, so this isn't just "looks
    reasonable" but verified lossless.

    table_search_text and row_search_text are separate on purpose: a
    product name like "Paxlovid" alone matches many unrelated tables in a
    pharma 10-K (legal proceedings, summaries, etc), not just the specific
    product-revenue table with the narrative cell -- table_search_text
    should be distinctive text unique to the target table.

    abbreviation_check, if given, is a string (e.g. "U.S.") that must NOT
    appear as the last word of any sub-row except the final one --
    catches the class of bug found on Coca-Cola's audit-matters text,
    where naive sentence-splitting broke mid-sentence at an abbreviation
    ("...the U.S." | "Tax Court issued...")."""
    tables = soup.find_all('table')
    target = None
    for t in tables:
        if table_search_text in t.get_text(" ", strip=True):
            target = t
            break
    if target is None:
        print(f"[{label}] table not found, skipping")
        return

    grid = build_grid(target)
    grid = drop_empty_columns(grid)
    grid = merge_dollar_columns(grid)
    grid = merge_percent_columns(grid)
    grid = drop_empty_columns(grid)

    row = next((r for r in grid if any(row_search_text in c for c in r)), None)
    if row is None:
        print(f"[{label}] row containing {row_search_text!r} not found after cleaning, skipping")
        return

    dominant_cell = max(row, key=len)
    if len(dominant_cell) < 100:
        print(f"[{label}] no dominant narrative cell found (largest cell only "
              f"{len(dominant_cell)} chars), skipping")
        return

    def count_tokens(text):
        return len(text.split())

    sub_rows = split_oversized_row(row, count_tokens, budget_for_row=60)
    dominant_col = row.index(dominant_cell)
    reconstructed = " ".join(sr[dominant_col] for sr in sub_rows)

    match = dominant_cell.split() == reconstructed.split()
    print(f"[{label}] split into {len(sub_rows)} sub-rows, "
          f"word-for-word reconstruction match: {match}")
    if not match:
        print(f"  MISMATCH -- original {len(dominant_cell.split())} words, "
              f"reconstructed {len(reconstructed.split())} words")

    if abbreviation_check:
        bad_split = any(
            sr[dominant_col].rstrip().endswith(abbreviation_check)
            for sr in sub_rows[:-1]
        )
        print(f"  abbreviation-aware split check ({abbreviation_check!r} "
              f"never ends a non-final sub-row): {not bad_split}")


def _test_column_group_split(soup, table_search_text, label):
    """Verifies detect_period_column_groups + split_grid_into_column_groups
    against a real genuinely-wide table (many repeating period groups side
    by side, e.g. JPMorgan's 22-column "Markets revenue" table). Checks
    that groups are actually detected, and -- the real correctness check,
    not just "did it run" -- that every original data cell (excluding the
    intentionally-repeated label column) appears in exactly one resulting
    group: an exact multiset match, no loss, no duplication."""
    tables = soup.find_all('table')
    target = None
    for t in tables:
        if table_search_text in t.get_text(" ", strip=True):
            target = t
            break
    if target is None:
        print(f"[{label}] table not found, skipping")
        return

    grid = build_grid(target)
    grid = drop_empty_columns(grid)
    grid = merge_dollar_columns(grid)
    grid = merge_percent_columns(grid)
    grid = drop_empty_columns(grid)

    groups = detect_period_column_groups(grid)
    if not groups:
        print(f"[{label}] no column groups detected, skipping")
        return

    sub_grids = split_grid_into_column_groups(grid, groups)

    original_data_cells = sorted(
        row[c].strip() for row in grid for c in range(1, len(row)) if row[c].strip()
    )
    reconstructed_data_cells = sorted(
        row[c].strip()
        for sg in sub_grids for row in sg for c in range(1, len(row))
        if row[c].strip()
    )
    match = original_data_cells == reconstructed_data_cells

    print(f"[{label}] detected {len(groups)} column groups, "
          f"data integrity (exact multiset match): {match}")
    if not match:
        print(f"  MISMATCH -- {len(original_data_cells)} original data cells "
              f"vs {len(reconstructed_data_cells)} reconstructed")


if __name__ == "__main__":
    import os

    def _load(path):
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return BeautifulSoup(f.read(), 'lxml')

    # Sample files saved under stable, ticker-prefixed names in samples/,
    # since the generic filenames these get uploaded as (matching the real
    # build_dataset.py output pattern, e.g. "10-K_2025-12-31.htm") collide
    # across different tickers that share a fiscal year-end -- a JPM
    # upload overwriting a same-named PFE upload at that path is exactly
    # what broke the Paxlovid test below once, hence saving stable copies.
    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')

    aapl_10q_path = os.path.join(samples_dir, 'AAPL_10-Q_2021-12-25.htm')
    if os.path.exists(aapl_10q_path):
        soup = _load(aapl_10q_path)
        # Two different table shapes, to confirm the numeric right-alignment
        # fix (see README) generalizes rather than being overfit to one table.
        _test_against(soup, "Net sales", "Income statement (10-Q)")
        _test_against(soup, "Total assets", "Balance sheet (10-Q)")
    else:
        print("[Income statement / Balance sheet] AAPL 10-Q sample not present, skipping")

    aapl_10k_path = os.path.join(samples_dir, 'AAPL_10-K_2022-09-24.htm')
    if os.path.exists(aapl_10k_path):
        soup_10k = _load(aapl_10k_path)
        # Percent-merge fix, verified against three different tables from a
        # real 10-K -- see README for the colspan pattern that motivated this.
        _test_against(soup_10k, "Net sales by category", "Products and Services Performance (10-K)")
        _test_against(soup_10k, "Gross margin percentage", "Gross Margin (10-K)")
        _test_against(soup_10k, "Effective tax rate", "Effective Tax Rate (10-K)")
        # Column-group splitting, generalization check: a different real
        # wide table than the JPM one this fix was originally built against.
        _test_column_group_split(soup_10k, "Net sales by category",
                                  "Products and Services column-group split (AAPL 10-K)")
    else:
        print("[Products/Services/Gross Margin/Tax Rate] AAPL 10-K sample not present, skipping")

    pfe_10k_path = os.path.join(samples_dir, 'PFE_10-K_2025-12-31.htm')
    if os.path.exists(pfe_10k_path):
        soup_pfe = _load(pfe_10k_path)
        # split_oversized_row / narrative-cell-in-row fix, verified against
        # a real Pfizer 10-K product-revenue table (one dominant explanation
        # cell among short numeric cells -- see README).
        _test_split_oversized_row(
            soup_pfe,
            table_search_text="Declines primarily driven by",
            row_search_text="Paxlovid",
            label="Paxlovid narrative-cell split (PFE 10-K)",
        )
    else:
        print("[Paxlovid narrative-cell split] PFE 10-K sample not present, skipping "
              "(re-upload PFE's 10-K to samples/PFE_10-K_2025-12-31.htm to restore this test)")

    jpm_10k_path = os.path.join(samples_dir, 'JPM_10-K_2025-12-31.htm')
    if os.path.exists(jpm_10k_path):
        soup_jpm = _load(jpm_10k_path)
        # Column-group splitting, the case this fix was originally built
        # against: JPMorgan's 22-column "Markets revenue" table.
        _test_column_group_split(soup_jpm, "Year ended December 31, (in millions, except where otherwise noted)",
                                  "Markets revenue column-group split (JPM 10-K)")
    else:
        print("[Markets revenue column-group split] JPM 10-K sample not present, skipping")

    ko_10k_path = os.path.join(samples_dir, 'KO_10-K_2022-12-31.htm')
    if os.path.exists(ko_10k_path):
        soup_ko = _load(ko_10k_path)
        # Abbreviation-aware sentence splitting, the case this fix was
        # built against: Coca-Cola's "Critical Audit Matters" text is
        # continuous prose (no bullet points, unlike Pfizer's case), and
        # naive splitting broke mid-sentence at "U.S."
        _test_split_oversized_row(
            soup_ko,
            table_search_text="Description of the Matter",
            row_search_text="Description of the Matter",
            label="Critical Audit Matters sentence split (KO 10-K)",
            abbreviation_check="U.S.",
        )
    else:
        print("[Critical Audit Matters sentence split] KO 10-K sample not present, skipping")

    tsla_10k_path = os.path.join(samples_dir, 'TSLA_10-K_2021-12-31.htm')
    if os.path.exists(tsla_10k_path):
        soup_tsla = _load(tsla_10k_path)
        # Tesla uses DFIN (Donnelly Financial Solutions) as its filing
        # preparer, not Workiva like every other filing this pipeline was
        # built against -- confirms table_cleaner.py's colspan-based
        # grid/merge logic generalizes to a different filing platform's
        # HTML conventions, not just Workiva's specifically. (The dom_walker
        # <p>-vs-<div> gap this same filing surfaced is a separate issue,
        # tested in dom_walker.py, not here.)
        _test_against(soup_tsla, "Total revenues", "Total revenues breakdown (TSLA 10-K, DFIN-generated)")
    else:
        print("[Total revenues breakdown] TSLA 10-K sample not present, skipping")
