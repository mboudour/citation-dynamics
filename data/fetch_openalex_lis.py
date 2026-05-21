"""
fetch_openalex_lis.py
---------------------
Fetches all Library and Information Science (LIS) journal articles from
OpenAlex (1975–2024) and saves the result as a pandas DataFrame pickle,
in the same format as the Dimensions.ai dataset used in the companion analysis.

Usage:
    python fetch_openalex_lis.py [--email your@email.com] [--output OpenAlex_LIS_1975_2024.pkl]

OpenAlex concept ID for Library and Information Science: C136764020
API documentation: https://docs.openalex.org/

Output columns (matching Dimensions format):
    id              : OpenAlex work ID (e.g. W2741809807)
    doi             : DOI string (without https://doi.org/ prefix)
    title           : Paper title
    abstract        : Reconstructed plain-text abstract
    year            : Publication year (int)
    journal         : Source journal name
    authors         : List of author display names
    author_ids      : List of OpenAlex author IDs
    times_cited     : Cited-by count
    reference_ids   : List of referenced OpenAlex work IDs
    is_oa           : Boolean — is the paper Open Access?
"""

import argparse
import time
import pickle
import requests
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONCEPT_ID   = "C136764020"          # OpenAlex concept: Library and Information Science
YEAR_MIN     = 1975
YEAR_MAX     = 2024
PER_PAGE     = 200                   # Max allowed by OpenAlex
SLEEP_S      = 0.1                   # Polite delay between requests (seconds)
MAX_RETRIES  = 5                     # Retries on transient HTTP errors


def reconstruct_abstract(inverted_index: dict | None) -> str:
    """Convert OpenAlex inverted-index abstract to plain text."""
    if not inverted_index:
        return ""
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions))


def fetch_page(url: str, params: dict, headers: dict | None = None, retries: int = MAX_RETRIES) -> dict:
    """GET a single page from the OpenAlex API with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers or {}, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s …")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code} — retrying ({attempt+1}/{retries}) …")
                time.sleep(5 * (attempt + 1))
        except requests.RequestException as e:
            print(f"  Request error: {e} — retrying ({attempt+1}/{retries}) …")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries.")


def fetch_all_works(email: str | None = None, api_key: str | None = None) -> list[dict]:
    """
    Retrieve all LIS journal articles from OpenAlex using cursor-based pagination.
    Returns a list of raw work dictionaries.
    """
    base_url = "https://api.openalex.org/works"
    params = {
        "filter": (
            f"concepts.id:{CONCEPT_ID},"
            f"type:article,"
            f"publication_year:{YEAR_MIN}-{YEAR_MAX}"
        ),
        "select": (
            "id,doi,title,abstract_inverted_index,authorships,"
            "publication_year,primary_location,open_access,"
            "referenced_works,cited_by_count"
        ),
        "per-page": PER_PAGE,
        "cursor": "*",
    }
    if email:
        params["mailto"] = email   # Polite pool — higher rate limit

    # API key passed as Authorization header (never in URL/params)
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    works = []
    page_num = 0

    # First request to get total count
    data = fetch_page(base_url, params, headers=headers)
    total = data["meta"]["count"]
    print(f"Total works to fetch: {total:,}")

    while True:
        page_num += 1
        results = data.get("results", [])
        if not results:
            break
        works.extend(results)
        fetched = len(works)
        if page_num % 50 == 0 or fetched >= total:
            pct = 100 * fetched / total
            print(f"  Page {page_num:4d} — {fetched:,}/{total:,} ({pct:.1f}%)")

        next_cursor = data.get("meta", {}).get("next_cursor")
        if not next_cursor:
            break

        params["cursor"] = next_cursor
        time.sleep(SLEEP_S)
        data = fetch_page(base_url, params, headers=headers)

    print(f"Fetched {len(works):,} works in {page_num} pages.")
    return works


def parse_works(raw_works: list[dict]) -> pd.DataFrame:
    """Parse raw OpenAlex work dicts into a clean DataFrame."""
    records = []
    for w in raw_works:
        # Authors
        authorships = w.get("authorships") or []
        authors    = [a["author"]["display_name"] for a in authorships
                      if a.get("author") and a["author"].get("display_name")]
        author_ids = [a["author"]["id"] for a in authorships
                      if a.get("author") and a["author"].get("id")]

        # Journal
        primary = w.get("primary_location") or {}
        source  = primary.get("source") or {}
        journal = source.get("display_name", "")

        # DOI — strip prefix
        doi_raw = w.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "").strip()

        # Abstract
        abstract = reconstruct_abstract(w.get("abstract_inverted_index"))

        # References (list of OpenAlex IDs)
        refs = w.get("referenced_works") or []

        # Open Access
        oa_info = w.get("open_access") or {}
        is_oa   = bool(oa_info.get("is_oa", False))

        records.append({
            "id":           w.get("id", ""),
            "doi":          doi,
            "title":        w.get("title") or "",
            "abstract":     abstract,
            "year":         w.get("publication_year"),
            "journal":      journal,
            "authors":      authors,
            "author_ids":   author_ids,
            "times_cited":  w.get("cited_by_count", 0),
            "reference_ids": refs,
            "is_oa":        is_oa,
        })

    df = pd.DataFrame(records)
    df = df[df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values("year").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch LIS data from OpenAlex")
    parser.add_argument(
        "--email", default=None,
        help="Your email for OpenAlex polite pool (recommended for higher rate limits)"
    )
    parser.add_argument(
        "--output", default="OpenAlex_LIS_1975_2024.pkl",
        help="Output pickle filename"
    )
    parser.add_argument(
        "--api-key", default=None, dest="api_key",
        help="OpenAlex API key (passed as Authorization header; never stored in output)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OpenAlex LIS Fetcher")
    print(f"  Concept : Library and Information Science ({CONCEPT_ID})")
    print(f"  Years   : {YEAR_MIN}–{YEAR_MAX}")
    print(f"  Email   : {args.email or '(none — anonymous pool)'}")
    print(f"  API key : {'provided (Authorization header)' if args.api_key else '(none)'}")
    print(f"  Output  : {args.output}")
    print("=" * 60)

    raw = fetch_all_works(email=args.email, api_key=args.api_key)
    print("Parsing records …")
    df = parse_works(raw)

    print(f"\nDataset summary:")
    print(f"  Articles          : {len(df):,}")
    print(f"  Year range        : {df['year'].min()}–{df['year'].max()}")
    print(f"  With abstract     : {(df['abstract'].str.len() > 10).sum():,} "
          f"({100*(df['abstract'].str.len() > 10).mean():.1f}%)")
    print(f"  With references   : {(df['reference_ids'].apply(len) > 0).sum():,} "
          f"({100*(df['reference_ids'].apply(len) > 0).mean():.1f}%)")
    print(f"  Mean times cited  : {df['times_cited'].mean():.2f}")

    out_path = Path(args.output)
    df.to_pickle(out_path)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
