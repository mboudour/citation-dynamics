"""
fetch_dimensions_archaeology.py
---------------------------------
Fetches archaeology journal articles from Dimensions.ai (1975–2024)
using a single keyword search, year by year, and saves two outputs:

  1. Dimensions_archaeology_1975_2024_raw.json.gz   — raw API response
  2. Dimensions_archaeology_1975_2024.parquet       — processed: id, year, title,
                                                         abstract, reference_ids

Keyword: "archaeology" (searched in title and abstract)

Folder structure (run from data_collection_scripts/):
    Citation Dynamics/
    └── computations/
        └── data_collection/
            ├── data/                              ← outputs saved here
            └── data_collection_scripts/
                ├── Dim_key.txt
                └── fetch_dimensions_archaeology.py

Requirements:
    pip install dimcli pandas pyarrow
"""

import dimcli
import json
import gzip
import time
import datetime
import pandas as pd
from pathlib import Path

HERE     = Path(__file__).parent
DATA_DIR = HERE.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

with open(HERE / "Dim_key.txt") as f:
    api_key = f.read().strip()

dimcli.login(key=api_key, endpoint="https://app.dimensions.ai/api/dsl.json")
dsl = dimcli.Dsl()

KEYWORD  = "archaeology"
YEAR_MIN = 1975
YEAR_MAX = 2024
FIELDS   = "id+year+title+abstract+reference_ids"

all_records = {}

for year in range(YEAR_MIN, YEAR_MAX + 1):
    print(f"[{datetime.datetime.now():%H:%M:%S}]  Year {year} ...", end=" ", flush=True)
    try:
        result = dsl.query_iterative(f"""
            search publications
            in title_abstract_only
            for "{KEYWORD}"
            where year = {year}
            and type = "article"
            return publications[{FIELDS}]
        """)
        df = result.as_dataframe()
        added = 0
        for rec in df.to_dict(orient="records"):
            pid = rec.get("id")
            if pid and pid not in all_records:
                try:
                    rec["abstract"] = str(rec.get("abstract") or "")
                except Exception:
                    rec["abstract"] = ""
                all_records[pid] = rec
                added += 1
        print(f"{len(df):,} fetched, {added:,} new (total: {len(all_records):,})", flush=True)
    except Exception as e:
        print(f"ERROR: {e} — skipping year {year}", flush=True)
    time.sleep(1)

print(f"\n{'='*50}")
print(f"TOTAL UNIQUE RECORDS : {len(all_records):,}")

raw_list = list(all_records.values())
raw_path = DATA_DIR / "Dimensions_archaeology_1975_2024_raw.json.gz"
with gzip.open(raw_path, "wt", encoding="utf-8") as f:
    json.dump(raw_list, f, ensure_ascii=False, default=str)
print(f"Saved raw     → {raw_path}  ({raw_path.stat().st_size / 1e6:.1f} MB)")

def extract_ref_ids(raw_refs):
    if not isinstance(raw_refs, list):
        return []
    return [str(item) for item in raw_refs if isinstance(item, str) and str(item).startswith("pub.")]

processed = pd.DataFrame({
    "id":            [r.get("id", "") for r in raw_list],
    "year":          [int(r.get("year", 0)) for r in raw_list],
    "title":         [r.get("title") or "" for r in raw_list],
    "abstract":      [r.get("abstract") or "" for r in raw_list],
    "reference_ids": [extract_ref_ids(r.get("reference_ids")) for r in raw_list],
})
processed = processed[processed["year"] > 0].sort_values("year").reset_index(drop=True)

parquet_path = DATA_DIR / "Dimensions_archaeology_1975_2024.parquet"
processed.to_parquet(parquet_path, index=False, engine="pyarrow")
print(f"Saved parquet → {parquet_path}  ({parquet_path.stat().st_size / 1e6:.1f} MB)")

refs_nonempty = (processed["reference_ids"].apply(len) > 0).sum()
corpus_ids    = set(processed["id"])
internal      = processed["reference_ids"].apply(lambda refs: sum(1 for r in refs if r in corpus_ids)).sum()

print(f"\nProcessed summary:")
print(f"  Records          : {len(processed):,}")
print(f"  Year range       : {processed['year'].min()}–{processed['year'].max()}")
print(f"  With references  : {refs_nonempty:,} ({100*refs_nonempty/len(processed):.1f}%)")
print(f"  With abstract    : {(processed['abstract'].str.len() > 10).sum():,}")
print(f"  Internal edges   : {internal:,}")
