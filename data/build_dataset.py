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
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{filename}"

REQUEST_DELAY_SECONDS = 0.15  # stays comfortably under SEC's 10 req/sec limit


def load_config(path: str = "data/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


def filter_filings(submissions: dict, form_types: list, since_year: int) -> list:
    """Flattens the 'recent' filings block and filters by form type + year."""
    recent = submissions["filings"]["recent"]
    results = []
    n = len(recent["accessionNumber"])
    for i in range(n):
        form = recent["form"][i]
        filing_date = recent["filingDate"][i]
        year = int(filing_date[:4])
        if form in form_types and year >= since_year:
            results.append({
                "form": form,
                "accession_number": recent["accessionNumber"][i],
                "filing_date": filing_date,
                "primary_document": recent["primaryDocument"][i],
                "report_date": recent["reportDate"][i],
            })
    return results


def download_filing(session: requests.Session, cik: int, filing: dict, dest_dir: Path) -> Path:
    accession_nodash = filing["accession_number"].replace("-", "")
    url = ARCHIVE_URL.format(
        cik=cik, accession_nodash=accession_nodash, filename=filing["primary_document"]
    )
    resp = session.get(url, timeout=30)
    resp.raise_for_status()

    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filing["primary_document"]).suffix or ".htm"
    out_path = dest_dir / f"{filing['form']}_{filing['report_date']}{ext}"
    out_path.write_bytes(resp.content)
    return out_path


def main():
    config = load_config()
    session = build_session(config["user_agent"])

    print("Fetching SEC ticker -> CIK mapping...")
    ticker_to_cik = get_ticker_to_cik(session)
    time.sleep(REQUEST_DELAY_SECONDS)

    raw_dir = Path(config.get("output_dir", "data/raw"))
    manifest = []

    for ticker in config["tickers"]:
        ticker = ticker.upper()
        entry = ticker_to_cik.get(ticker)
        if entry is None:
            print(f"  [skip] {ticker}: not found in SEC ticker list")
            continue
        cik = entry["cik"]
        company_name = entry["name"]

        print(f"Fetching filings for {ticker} ({company_name}, CIK {cik})...")
        submissions = get_filings_for_cik(session, cik)
        time.sleep(REQUEST_DELAY_SECONDS)

        filings = filter_filings(submissions, config["form_types"], config["since_year"])
        print(f"  found {len(filings)} matching filings")

        for filing in filings:
            try:
                dest_dir = raw_dir / ticker
                out_path = download_filing(session, cik, filing, dest_dir)
                time.sleep(REQUEST_DELAY_SECONDS)

                meta = {
                    "ticker": ticker,
                    "company_name": company_name,
                    "cik": cik,
                    "form_type": filing["form"],
                    "fiscal_period_end": filing["report_date"],
                    "filing_date": filing["filing_date"],
                    "accession_number": filing["accession_number"],
                    "source_url": ARCHIVE_URL.format(
                        cik=cik,
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
