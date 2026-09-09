"""
SEC EDGAR dataset builder for a financial RAG portfolio project.

Downloads 10-K / 10-Q primary documents for a configured list of tickers
directly from SEC EDGAR's free, public data APIs, and writes each filing
alongside a JSON metadata sidecar (ticker, fiscal period, form type, source
URL, etc). That metadata is what your ingestion pipeline should attach to
every chunk later, so retrieval can be filtered/scoped precisely.

SEC EDGAR fair access rules (https://www.sec.gov/os/webmaster-faq#developers):
  - Max ~10 requests/second
  - Must send a descriptive User-Agent with a real name + email
Both are respected here. Set your contact info in config.yaml before running.

Usage:
    pip install -r requirements.txt
    python build_dataset.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"

REQUEST_DELAY_SECONDS = 0.15  # stays comfortably under SEC's 10 req/sec limit

# This script's own directory -- the starting point for finding config.yaml.
SCRIPT_DIR = Path(__file__).resolve().parent


def find_config(start: Path, filename: str = "config.yaml", max_levels: int = 6) -> Path:
    """Searches upward from `start` through parent directories for the
    shared project config. Scripts live in different task-specific
    subdirectories (data/, parsingChunking/, embed/, ...) while config.yaml
    lives once at the shared project root -- this lets every script find it
    without needing to know how deep it's nested."""
    current = start
    for _ in range(max_levels):
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current.parent == current:  # reached filesystem root
            break
        current = current.parent
    raise FileNotFoundError(
        f"Could not find {filename} searching upward from {start} "
        f"(checked {max_levels} levels). Make sure config.yaml exists at "
        f"your project root."
    )


def load_config() -> tuple:
    """Returns (config_dict, project_root) -- project_root is the directory
    config.yaml was actually found in, used to resolve data-directory paths
    relative to it rather than relative to any individual script."""
    config_path = find_config(SCRIPT_DIR)
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config, config_path.parent


def resolve_dir(project_root: Path, config: dict, key: str, default: str) -> Path:
    """Resolves a data-directory config value relative to the project root
    (where config.yaml lives), not relative to any individual script."""
    return (project_root / config.get(key, default)).resolve()


def build_session(user_agent: str) -> requests.Session:
    if "your_email@example.com" in user_agent:
        raise ValueError(
            "Set a real contact email in config.yaml's user_agent field before "
            "running -- SEC EDGAR requires this and will block generic/missing "
            "User-Agents."
        )
    session = requests.Session()
    session.headers.update({
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
    })
    return session


def get_ticker_to_cik(session: requests.Session) -> dict:
    """Returns {ticker: {"cik": int, "name": str}} using SEC's official
    ticker -> CIK mapping. The mapping also includes a "title" field (the
    company's registered name) that's free in the same response -- captured
    here rather than discarded, so downstream chunks can carry a
    human-readable company name alongside the ticker."""
    resp = session.get(TICKERS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {
        v["ticker"].upper(): {"cik": v["cik_str"], "name": v["title"]}
        for v in data.values()
    }


def get_filings_for_cik(session: requests.Session, cik: int) -> dict:
    resp = session.get(SUBMISSIONS_URL.format(cik=cik), timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_all_filing_pages(session: requests.Session, submissions: dict) -> list:
    """Returns every columnar filing block for this filer: the primary
    'recent' block plus any additional paginated blocks listed in
    submissions['filings']['files'].

    Real, confirmed bug this fixes: SEC's submissions.json only holds "at
    least one year's...or 1,000...whichever is more" of a filer's most
    RECENT filings of ALL form types (10-K, 10-Q, 8-K, proxy statements,
    everything) in the primary 'recent' block -- not just the form types
    this project cares about. A filer that submits heavily in OTHER forms
    (8-Ks, proxy statements -- exactly what large financial institutions
    and multinationals tend to do) can have its older 10-K/10-Qs pushed
    out of that window entirely, silently, with no error. An earlier
    version of this function only ever read 'recent', which meant it
    quietly under-downloaded exactly this class of filer.

    Confirmed as the real cause on a real run of this project's corpus:
    AAPL/JNJ/KO/MSFT/PFE/TSLA all downloaded a complete, expected ~19
    filings each (full 2022-2026 range), while JPM and GS -- both
    high-filing-volume financial institutions -- only got their most
    recent 4 filings each (roughly the last year), XOM only got 1 (its
    single most recent filing), and WMT was missing its earliest ~4-5
    filings from 2022. All four gaps were at the OLD end of the date
    range, consistent with exactly this failure mode, and none of them
    surfaced as an error -- the script completed "successfully" every
    time, just with an incomplete result for these specific tickers."""
    pages = [submissions["filings"]["recent"]]
    for extra in submissions["filings"].get("files", []):
        resp = session.get(SUBMISSIONS_PAGE_URL.format(name=extra["name"]), timeout=30)
        resp.raise_for_status()
        pages.append(resp.json())
        time.sleep(REQUEST_DELAY_SECONDS)  # this is a real request too -- respect the rate limit
    return pages


def filter_filings(pages: list, form_types: list, since_year: int, cik: int) -> list:
    """Filters filings by form type + year across every columnar filing
    block for this filer (see get_all_filing_pages for why there can be
    more than one). Tags each result with its source `cik`, since a
    single ticker can now span multiple CIKs (see cik_overrides in
    config.yaml -- needed when a ticker persisted across a corporate
    restructuring that changed the underlying legal entity/CIK)."""
    results = []
    for page in pages:
        n = len(page["accessionNumber"])
        for i in range(n):
            form = page["form"][i]
            filing_date = page["filingDate"][i]
            year = int(filing_date[:4])
            if form in form_types and year >= since_year:
                results.append({
                    "cik": cik,
                    "form": form,
                    "accession_number": page["accessionNumber"][i],
                    "filing_date": filing_date,
                    "primary_document": page["primaryDocument"][i],
                    "report_date": page["reportDate"][i],
                })
    return results


def download_filing(session: requests.Session, filing: dict, dest_dir: Path) -> Path:
    accession_nodash = filing["accession_number"].replace("-", "")
    url = ARCHIVE_URL.format(
        cik=filing["cik"], accession_nodash=accession_nodash, filename=filing["primary_document"]
    )
    resp = session.get(url, timeout=30)
    resp.raise_for_status()

    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filing["primary_document"]).suffix or ".htm"
    out_path = dest_dir / f"{filing['form']}_{filing['report_date']}{ext}"
    out_path.write_bytes(resp.content)
    return out_path


def main():
    config, project_root = load_config()
    session = build_session(config["user_agent"])

    print("Fetching SEC ticker -> CIK mapping...")
    ticker_to_cik = get_ticker_to_cik(session)
    time.sleep(REQUEST_DELAY_SECONDS)

    raw_dir = resolve_dir(project_root, config, "output_dir", "data/raw")
    manifest = []

    for ticker in config["tickers"]:
        ticker = ticker.upper()
        entry = ticker_to_cik.get(ticker)
        if entry is None:
            print(f"  [skip] {ticker}: not found in SEC ticker list")
            continue
        company_name = entry["name"]

        # Normally just the current CIK from company_tickers.json. Some
        # tickers need more than one -- see cik_overrides in config.yaml
        # (confirmed real case: XOM persisted across a 2026-07-01
        # corporate restructuring that changed its underlying CIK,
        # silently losing pre-restructuring history if only the current
        # CIK is used).
        cik_list = config.get("cik_overrides", {}).get(ticker, [entry["cik"]])

        all_filings = []
        for cik in cik_list:
            print(f"Fetching filings for {ticker} ({company_name}, CIK {cik})...")
            submissions = get_filings_for_cik(session, cik)
            time.sleep(REQUEST_DELAY_SECONDS)

            pages = get_all_filing_pages(session, submissions)
            if len(pages) > 1:
                print(f"  CIK {cik} has {len(pages)} filing pages (high filing volume "
                      f"in other form types pushed older filings into paginated files)")

            filings = filter_filings(pages, config["form_types"], config["since_year"], cik)
            print(f"  found {len(filings)} matching filings under CIK {cik}")
            all_filings.extend(filings)

        if len(cik_list) > 1:
            # A single filing can legitimately be cross-listed under BOTH
            # a predecessor and successor CIK around a corporate
            # restructuring -- confirmed real case: XOM's Q2 2026 10-Q
            # (accession 0000034088-26-000093, filed shortly after the
            # 2026-07-01 redomiciliation merger) appeared under both the
            # old CIK (34088) and the new one (2115436) with otherwise
            # identical data. Without deduplication, that filing would get
            # "downloaded" twice under the same output filename (silently
            # overwriting itself -- no duplicate file on disk, but two
            # manifest entries and an ambiguous, order-dependent CIK
            # recorded in its .meta.json sidecar). Deduplicated here by
            # accession_number, keeping the FIRST occurrence -- given
            # cik_overrides lists the predecessor CIK before the successor
            # one, this naturally attributes a cross-listed filing to
            # whichever CIK's accession-number prefix it actually
            # originated from, not just whichever happened to process last.
            seen_accessions = set()
            deduped_filings = []
            n_dupes = 0
            for f in all_filings:
                if f["accession_number"] in seen_accessions:
                    n_dupes += 1
                    continue
                seen_accessions.add(f["accession_number"])
                deduped_filings.append(f)
            if n_dupes:
                print(f"  {ticker}: removed {n_dupes} filing(s) cross-listed under "
                      f"multiple CIKs (same accession number)")
            all_filings = deduped_filings

        if len(cik_list) > 1:
            print(f"  {ticker} total across {len(cik_list)} CIKs: {len(all_filings)} filings")

        for filing in all_filings:
            try:
                dest_dir = raw_dir / ticker
                out_path = download_filing(session, filing, dest_dir)
                time.sleep(REQUEST_DELAY_SECONDS)

                meta = {
                    "ticker": ticker,
                    "company_name": company_name,
                    "cik": filing["cik"],
                    "form_type": filing["form"],
                    "fiscal_period_end": filing["report_date"],
                    "filing_date": filing["filing_date"],
                    "accession_number": filing["accession_number"],
                    "source_url": ARCHIVE_URL.format(
                        cik=filing["cik"],
                        accession_nodash=filing["accession_number"].replace("-", ""),
                        filename=filing["primary_document"],
                    ),
                    "local_path": str(out_path),
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
                meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
                meta_path.write_text(json.dumps(meta, indent=2))
                manifest.append(meta)
                print(f"  saved {out_path.name}")
            except requests.HTTPError as e:
                print(f"  [error] {ticker} {filing['form']} {filing['report_date']}: {e}")

    manifest_path = raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {len(manifest)} filings saved. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
