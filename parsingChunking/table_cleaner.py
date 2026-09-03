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


if __name__ == "__main__":
    with open('/mnt/user-data/uploads/10-Q_2021-12-25.htm', 'r', encoding='utf-8', errors='replace') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'lxml')

    # Two different table shapes, to confirm the numeric right-alignment fix
    # (see README) generalizes rather than being overfit to one table.
    _test_against(soup, "Net sales", "Income statement (10-Q)")
    _test_against(soup, "Total assets", "Balance sheet (10-Q)")

    # Percent-merge fix, verified against three different tables from a
    # real 10-K -- see README for the colspan pattern that motivated this.
    with open('/mnt/user-data/uploads/10-K_2022-09-24.htm', 'r', encoding='utf-8', errors='replace') as f:
        html_10k = f.read()
    soup_10k = BeautifulSoup(html_10k, 'lxml')
    _test_against(soup_10k, "Net sales by category", "Products and Services Performance (10-K)")
    _test_against(soup_10k, "Gross margin percentage", "Gross Margin (10-K)")
    _test_against(soup_10k, "Effective tax rate", "Effective Tax Rate (10-K)")
