"""DOM walker: turns a filing's HTML into an ordered sequence of blocks
(heading / text / table), tracking current section context.

Tested standalone against real SEC filing structure before being wired into
the full pipeline -- see the heading heuristic and table cleaner validation
that motivated this design.
"""

from bs4 import BeautifulSoup, NavigableString
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from table_cleaner import table_to_cleaned_text


def is_bold_span(span):
    style = span.get('style', '') or ''
    return 'font-weight:700' in style or 'font-weight:bold' in style


def classify_block(element):
    """Returns ('heading', text) or ('text', text) or None (skip/container).

    Generalized from an earlier div-only version after finding a real,
    confirmed gap: not every SEC filing preparer uses <div> as the primary
    content container. Tesla's DFIN-generated filings put the vast
    majority of actual prose in <p> tags instead -- 440,774 characters in
    <p> vs. only 62,946 in <div> in a real 10-K, meaning the div-only
    version was silently missing ~87% of the document's text, not just
    section headings. This function works identically for either tag,
    called on both in walk_blocks() below."""
    if element.find('table'):
        return None  # container that wraps a table -- don't treat as text
    full_text = element.get_text(strip=True)
    if not full_text:
        return None

    spans = element.find_all('span')
    if spans and len(full_text) <= 150:
        bold_text = ''.join(s.get_text() for s in spans if is_bold_span(s))
        non_bold_text = ''.join(s.get_text() for s in spans if not is_bold_span(s))
        if bold_text.strip() and not non_bold_text.strip():
            return ('heading', full_text)

    return ('text', full_text)


def walk_blocks(soup):
    """Yields ('heading'|'text'|'table', content) in document order.

    Recursion rule: once a <div> or <p> is classified as a leaf (heading
    or text), its children are not visited separately (we already
    captured its full text). Container divs/paragraphs with no direct
    text/table of their own are descended into. <table> elements are
    emitted whole and not recursed into.

    Both <div> and <p> are checked (see classify_block's docstring for
    why) -- since a leaf classification short-circuits recursion into that
    element's children, a <p> nested inside an already-classified <div>
    (or vice versa) is captured once, as part of the outer element's full
    text, not double-counted. Verified no regression on Workiva-generated
    filings (which use <div> almost exclusively) after adding <p> handling.
    """
    body = soup.body or soup

    def is_hidden(tag):
        style = tag.get('style', '') or ''
        return 'display:none' in style.replace(' ', '')

    def _walk(node):
        for child in node.children:
            if isinstance(child, NavigableString):
                continue
            if is_hidden(child):
                continue  # skip inline-XBRL tagging blocks and other hidden content
            if child.name == 'table':
                yield ('table', child)
                continue
            if child.name in ('div', 'p'):
                result = classify_block(child)
                if result is not None:
                    yield result
                    continue
                # container (no direct text, or wraps a table) -- recurse
                yield from _walk(child)
                continue
            # any other tag (span at top level, etc.) -- recurse defensively
            yield from _walk(child)

    yield from _walk(body)


HEADING_LEVEL_1 = __import__('re').compile(r'^(PART\s|Item\s+\d)', __import__('re').IGNORECASE)


def build_records(soup):
    """Walks blocks, tracks a 2-level section path, groups consecutive text
    blocks, and returns a flat list of {'type', 'section', 'content'}."""
    section_l1 = None  # PART / Item level
    section_l2 = None  # sub-heading level
    records = []
    text_buffer = []

    def flush_text():
        nonlocal text_buffer
        if text_buffer:
            joined = "\n\n".join(text_buffer)
            section = " > ".join(s for s in [section_l1, section_l2] if s)
            records.append({'type': 'text', 'section': section or None, 'content': joined})
            text_buffer = []

    for kind, content in walk_blocks(soup):
        if kind == 'heading':
            flush_text()
            if HEADING_LEVEL_1.match(content):
                section_l1 = content
                section_l2 = None
            else:
                section_l2 = content
        elif kind == 'text':
            text_buffer.append(content)
        elif kind == 'table':
            cleaned, grid = table_to_cleaned_text(content)
            if not cleaned.strip():
                continue  # pure layout/spacer table (e.g. cover-page address block) -- skip
            flush_text()
            section = " > ".join(s for s in [section_l1, section_l2] if s)
            records.append({'type': 'table', 'section': section or None, 'content': cleaned, 'grid': grid})

    flush_text()
    return records


if __name__ == "__main__":
    import os

    samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')

    aapl_10q_path = os.path.join(samples_dir, 'AAPL_10-Q_2021-12-25.htm')
    if not os.path.exists(aapl_10q_path):
        print(f"[Income statement / Balance sheet] AAPL 10-Q sample not present at {aapl_10q_path}, skipping")
        soup = None
    else:
        with open(aapl_10q_path, 'r', encoding='utf-8', errors='replace') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'lxml')

    if soup is not None:
        records = build_records(soup)
        print(f"Total blocks: {len(records)}")
        kinds = {}
        for r in records:
            kinds[r['type']] = kinds.get(r['type'], 0) + 1
        print("By type:", kinds)

        print("\n--- First 15 blocks (type, section, content preview) ---")
        for r in records[:15]:
            preview = r['content'][:70].replace("\n", " ")
            print(f"[{r['type']:5s}] section={r['section']!r:60s} content={preview!r}")

    # Regression guard for the real <p>-vs-<div> bug found on Tesla's
    # filings: Tesla's filing preparer (DFIN/ActiveDisclosure) puts ~87%
    # of actual prose content in <p> tags rather than <div> (440,774 vs.
    # 62,946 characters in a real 10-K) -- every other filing tested in
    # this project uses Workiva, which is primarily <div>-based, so this
    # gap was invisible until a filing from a different preparer was
    # actually tested. Fixed by generalizing classify_div() into
    # classify_block(), checked against both tag types in walk_blocks().
    tsla_path = os.path.join(samples_dir, 'TSLA_10-K_2021-12-31.htm')
    if os.path.exists(tsla_path):
        with open(tsla_path, 'r', encoding='utf-8', errors='replace') as f:
            html_tsla = f.read()
        soup_tsla = BeautifulSoup(html_tsla, 'lxml')
        records_tsla = build_records(soup_tsla)
        sections_found = sum(1 for r in records_tsla if r['section'])
        text_chars = sum(len(r['content']) for r in records_tsla if r['type'] == 'text')
        print(f"\n[TSLA 10-K, DFIN-generated] {len(records_tsla)} blocks, "
              f"{sections_found} with a detected section, {text_chars} text chars captured")
        # Loose bounds, not exact figures -- guards against a full
        # regression back to "zero headings detected, ~63K chars captured"
        # (the actual pre-fix numbers) without being brittle to minor
        # future changes in heading-detection heuristics.
        assert sections_found > 100, (
            f"REGRESSION: only {sections_found} sections detected on TSLA's DFIN-generated "
            f"filing (expected 100+) -- the <p>-tag heading detection may be broken again."
        )
        assert text_chars > 300000, (
            f"REGRESSION: only {text_chars} text chars captured on TSLA's DFIN-generated "
            f"filing (expected 300,000+) -- may be back to only reading <div> tags."
        )
        print("PASS: <p>-tag handling correctly captures Tesla's DFIN-generated content")
    else:
        print("\n[TSLA <p>-tag regression check] TSLA 10-K sample not present, skipping")
