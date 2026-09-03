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


def classify_div(div):
    """Returns ('heading', text) or ('text', text) or None (skip/container)."""
    if div.find('table'):
        return None  # container that wraps a table -- don't treat as text
    full_text = div.get_text(strip=True)
    if not full_text:
        return None

    spans = div.find_all('span')
    if spans and len(full_text) <= 150:
        bold_text = ''.join(s.get_text() for s in spans if is_bold_span(s))
        non_bold_text = ''.join(s.get_text() for s in spans if not is_bold_span(s))
        if bold_text.strip() and not non_bold_text.strip():
            return ('heading', full_text)

    return ('text', full_text)


def walk_blocks(soup):
    """Yields ('heading'|'text'|'table', content) in document order.

    Recursion rule: once a <div> is classified as a leaf (heading or text),
    its children are not visited separately (we already captured its full
    text). Container divs with no direct text/table of their own are
    descended into. <table> elements are emitted whole and not recursed
    into.
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
            if child.name == 'div':
                result = classify_div(child)
                if result is not None:
                    yield result
                    continue
                # container div (no direct text, or wraps a table) -- recurse
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
            records.append({'type': 'table', 'section': section or None, 'content': cleaned})

    flush_text()
    return records


if __name__ == "__main__":
    with open('/mnt/user-data/uploads/10-Q_2021-12-25.htm', 'r', encoding='utf-8', errors='replace') as f:
        html = f.read()
    soup = BeautifulSoup(html, 'lxml')

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
