"""
test_query_density.py
---------------------
Tests a list of Dimensions keyword queries and reports, for each:
  - total articles fetched (corpus size)
  - number of internal citation edges (positive pairs available)
  - citation graph density
  - number of nodes with at least one internal citation
  - estimated balanced dataset size (2 × internal edges)

Usage:
    python test_query_density.py

Edit the QUERIES list below to add or remove candidate queries.
Results are printed to stdout and saved to query_density_report.csv
in the same directory as this script.

Requirements: dimcli, pandas, pyarrow
Place Dim_key.txt in the same directory as this script.
"""

import dimcli
import time
import datetime
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

HERE     = Path(__file__).parent
KEY_FILE = HERE / "Dim_key.txt"

YEAR_MIN = 1975
YEAR_MAX = 2024
FIELDS   = "id+year+reference_ids"   # minimal fields — no abstract needed for density test
MAX_ARTICLES = 60_000                 # stop fetching once corpus exceeds this (saves API quota)

# ── Candidate queries to test ──────────────────────────────────────────────────
# Format: (label, keyword_string)
# Add as many as you like. Run overnight if needed.

QUERIES = [
    # ── Replacements for environmental_engineering ──
    ("constructed_wetlands",          "constructed wetlands"),
    ("membrane_bioreactor",           "membrane bioreactor"),
    ("anaerobic_digestion",           "anaerobic digestion"),
    ("soil_bioremediation",           "soil bioremediation"),
    ("wastewater_treatment",          "wastewater treatment"),
    ("phytoremediation",              "phytoremediation"),
    ("landfill_leachate",             "landfill leachate"),
    ("air_pollution_control",         "air pollution control"),

    # ── Replacements for art_history ──
    ("byzantine_art",                 "Byzantine art"),
    ("renaissance_painting",          "Renaissance painting"),
    ("dutch_golden_age",              "Dutch Golden Age painting"),
    ("iconography_medieval",          "iconography medieval"),
    ("museum_studies",                "museum studies"),
    ("visual_culture_theory",         "visual culture theory"),
    ("baroque_art",                   "Baroque art"),
    ("impressionism_painting",        "Impressionism painting"),
]

# ── Dimensions login ───────────────────────────────────────────────────────────

with open(KEY_FILE) as f:
    api_key = f.read().strip()

dimcli.login(key=api_key, endpoint="https://app.dimensions.ai/api/dsl.json")
dsl = dimcli.Dsl()

# ── Helper ─────────────────────────────────────────────────────────────────────

def fetch_corpus(keyword: str) -> pd.DataFrame:
    """Fetch id, year, reference_ids for a keyword query, year by year."""
    all_records = {}
    for year in range(YEAR_MIN, YEAR_MAX + 1):
        if len(all_records) >= MAX_ARTICLES:
            print(f"    [cap reached at {MAX_ARTICLES:,}]", flush=True)
            break
        try:
            result = dsl.query_iterative(f"""
                search publications
                in title_abstract_only
                for "{keyword}"
                where year = {year}
                and type = "article"
                return publications[{FIELDS}]
            """, verbose=False)
            df = result.as_dataframe()
            print(f"    {year}: {len(df):,} records", flush=True)
            for rec in df.to_dict(orient="records"):
                pid = rec.get("id")
                if pid and pid not in all_records:
                    all_records[pid] = rec
        except Exception as e:
            print(f"    ERROR year {year}: {e}", flush=True)
        time.sleep(0.8)

    if not all_records:
        return pd.DataFrame(columns=["id", "year", "reference_ids"])

    rows = list(all_records.values())
    df = pd.DataFrame({
        "id":            [r.get("id", "") for r in rows],
        "year":          [int(r.get("year") or 0) for r in rows],
        "reference_ids": [
            [str(x) for x in (r.get("reference_ids") or []) if isinstance(x, str) and str(x).startswith("pub.")]
            for r in rows
        ],
    })
    df = df[df["year"] > 0].reset_index(drop=True)
    return df


def compute_stats(df: pd.DataFrame) -> dict:
    """Compute citation graph statistics from a corpus dataframe."""
    corpus_ids = set(df["id"])
    n_articles = len(df)

    # internal edges: citations where both citing and cited paper are in corpus
    internal_edges = 0
    citing_nodes   = set()
    cited_nodes    = set()

    for _, row in df.iterrows():
        refs = row["reference_ids"]
        if not isinstance(refs, list):
            continue
        internal = [r for r in refs if r in corpus_ids]
        if internal:
            citing_nodes.add(row["id"])
            cited_nodes.update(internal)
            internal_edges += len(internal)

    n_nodes_with_edges = len(citing_nodes | cited_nodes)
    # density = edges / (n * (n-1))  [directed, no self-loops]
    n = n_articles
    density = internal_edges / (n * (n - 1)) if n > 1 else 0.0
    # density among only nodes that participate in at least one internal edge
    n2 = n_nodes_with_edges
    density_active = internal_edges / (n2 * (n2 - 1)) if n2 > 1 else 0.0

    return {
        "n_articles":          n_articles,
        "n_internal_edges":    internal_edges,
        "n_nodes_active":      n_nodes_with_edges,
        "density_full":        density,
        "density_active":      density_active,
        "balanced_pairs":      2 * internal_edges,
    }


# ── Main loop ──────────────────────────────────────────────────────────────────

results = []

for label, keyword in QUERIES:
    print(f"\n{'='*60}")
    print(f"[{datetime.datetime.now():%H:%M:%S}]  Testing: {label!r}  (keyword: {keyword!r})")
    print(f"{'='*60}")

    df = fetch_corpus(keyword)
    stats = compute_stats(df)

    row = {"label": label, "keyword": keyword, **stats}
    results.append(row)

    print(f"  Articles          : {stats['n_articles']:>8,}")
    print(f"  Internal edges    : {stats['n_internal_edges']:>8,}")
    print(f"  Active nodes      : {stats['n_nodes_active']:>8,}")
    print(f"  Density (full)    : {stats['density_full']:.2e}")
    print(f"  Density (active)  : {stats['density_active']:.2e}")
    print(f"  Balanced pairs    : {stats['balanced_pairs']:>8,}  ← target ≥ 1,000")

    # save incrementally so partial results are not lost on interruption
    pd.DataFrame(results).to_csv(HERE / "query_density_report.csv", index=False)
    print(f"  [saved to query_density_report.csv]")

# ── Final report ───────────────────────────────────────────────────────────────

print(f"\n\n{'='*60}")
print("FINAL REPORT")
print(f"{'='*60}")
report = pd.DataFrame(results).sort_values("n_internal_edges", ascending=False)
print(report.to_string(index=False))
print(f"\nSaved to: {HERE / 'query_density_report.csv'}")
