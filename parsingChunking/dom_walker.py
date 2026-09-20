"""DOM walker: turns a filing's HTML into an ordered sequence of blocks
(heading / text / table), tracking current section context.

Tested standalone against real SEC filing structure before being wired into
the full pipeline -- see the heading heuristic and table cleaner validation
that motivated this design.
"""

from bs4 import BeautifulSoup, NavigableString
import re
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
    called on both in walk_blocks() below.

    Checks for bold styling in TWO places, not just one -- a second real
    gap, found on Microsoft's DFIN-generated filings. Workiva (and Tesla's
    DFIN template) wrap heading text in a nested <span style="...bold...">
    inside the container; Microsoft's DFIN template instead puts the bold
    style directly on the <p> element itself
    (<p style="...font-weight:bold...">TEXT</p>, no span at all). A
    version that only checked descendant spans found zero headings on
    MSFT's filing despite it clearly having them -- is_bold_span() is
    actually a generic style-string check (despite its name), so it's
    called here on the element itself as well as on any nested spans."""
    if element.find('table'):
        return None  # container that wraps a table -- don't treat as text
    full_text = element.get_text(strip=True)
    if not full_text:
        return None

    if len(full_text) <= 150:
        spans = element.find_all('span')

        if is_bold_span(element):
            # Bold applied at the container level. Only an explicit
            # non-bold override in a nested span should disqualify this
            # from being a heading (e.g. a mostly-bold line with one
            # incidental non-bold word/footnote marker inside it).
            non_bold_override = ''.join(s.get_text() for s in spans if not is_bold_span(s))
            if not non_bold_override.strip():
                return ('heading', full_text)
        elif spans:
            # Bold applied via nested spans instead (Workiva's convention,
            # and Tesla's <p><a><span style="...bold...">).
            bold_text = ''.join(s.get_text() for s in spans if is_bold_span(s))
            non_bold_text = ''.join(s.get_text() for s in spans if not is_bold_span(s))
            if bold_text.strip() and not non_bold_text.strip():
                return ('heading', full_text)

    return ('text', full_text)


_INLINE_SUBHEADING_PATTERN = re.compile(r"^([A-Z][A-Za-z ]{2,40}?)\.(?=[A-Z])")


def split_inline_subheading(text):
    """Detects a bold lead-in glued onto the front of a longer paragraph
    in the SAME block, e.g. "Net Interest Income.Net interest income in
    the consolidated statements..." -> ("Net Interest Income", "Net
    interest income in the consolidated statements...").

    Real, confirmed gap this fixes: classify_block()'s heading detection
    only fires when len(full_text) <= 150 AND the entire block is bold --
    correctly designed for a standalone short heading block, but it never
    even reaches the bold-check for a block like this one, which is a
    single long paragraph (usually well over 150 chars) where only the
    first few words are bold and the rest continues as ordinary prose in
    the SAME block. Confirmed as a real, non-hypothetical problem via a
    genuine generation error (a chunk's section metadata said "Net
    Revenues" -- the parent section -- when the chunk's actual content
    was the "Net Interest Income" subsection, contributing to the model
    answering a net-revenues question with a net-interest-income figure)
    and confirmed non-rare via a corpus-wide count: 886 of 49,207 text
    chunks (1.8%) match this pattern -- roughly 4-5 per filing on
    average, not a one-off.

    Uses the same regex as generate.py's extract_subheading() -- same
    validated pattern (tested against the real failing case plus false
    positives like "U.S.Government..." and ordinary prose), applied here
    at parse time instead of generation time so every downstream
    consumer of chunk metadata (retrieve.py, diagnose_misses.py,
    generate.py, eval reports) sees the correct, complete section path,
    not just the one place a patch happened to be added first.

    Returns (heading, remaining_text) if the pattern matches, or
    (None, text) unchanged otherwise."""
    match = _INLINE_SUBHEADING_PATTERN.match(text)
    if match:
        heading = match.group(1).strip()
        remaining = text[match.end(1) + 1:]  # +1 skips the period itself
        return heading, remaining
    return None, text


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

    A ('text', ...) result is additionally checked for an inline bold
    lead-in (see split_inline_subheading) -- when found, a ('heading', ...)
    is yielded first, then the remaining text, so a bold-lead-in-glued-onto-
    a-paragraph block behaves like the two separate blocks it structurally
    represents, feeding section_l2 tracking in parse_and_chunk.py correctly
    without that module needing any changes of its own.
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
                    kind, content = result
                    if kind == 'text':
                        heading, remaining = split_inline_subheading(content)
                        if heading is not None:
                            yield ('heading', heading)
                            if remaining.strip():
                                yield ('text', remaining)
                            continue
                    yield result
                    continue
                # container (no direct text, or wraps a table) -- recurse
                yield from _walk(child)
                continue
            # any other tag (span at top level, etc.) -- recurse defensively
            yield from _walk(child)

    yield from _walk(body)


HEADING_LEVEL_1 = re.compile(r'^(PART\s|Item\s+\d)', re.IGNORECASE)


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

    msft_path = os.path.join(samples_dir, 'MSFT_10-K_2022-06-30.htm')
    if os.path.exists(msft_path):
        with open(msft_path, 'r', encoding='utf-8', errors='replace') as f:
            html_msft = f.read()
        soup_msft = BeautifulSoup(html_msft, 'lxml')
        records_msft = build_records(soup_msft)
        sections_found = sum(1 for r in records_msft if r['section'])
        # Regression guard for a second real, distinct heading-detection
        # gap found on this same file: Microsoft's DFIN template applies
        # bold styling directly on the <p> element itself
        # (<p style="...font-weight:bold...">TEXT</p>), not via a nested
        # <span> like Workiva and Tesla's DFIN template both use. A
        # version that only checked descendant spans found ZERO "Item
        # N"/"PART" headings on this file despite them clearly being
        # present -- confirmed before fixing by tracing the real markup
        # around "ITEM 1. BUSINESS". Fixed by also checking is_bold_span()
        # on the container element itself, not just its descendant spans.
        print(f"[MSFT 10-K, DFIN-generated, bold-on-element] {len(records_msft)} blocks, "
              f"{sections_found}/{len(records_msft)} with a detected section")
        assert sections_found == len(records_msft), (
            f"REGRESSION: only {sections_found}/{len(records_msft)} blocks have a detected "
            f"section on MSFT's DFIN-generated filing -- the bold-on-element-itself heading "
            f"detection may be broken again."
        )
        print("PASS: bold-on-element heading detection correctly captures MSFT's Item/PART headings")
    else:
        print("\n[MSFT bold-on-element regression check] MSFT 10-K sample not present, skipping")
