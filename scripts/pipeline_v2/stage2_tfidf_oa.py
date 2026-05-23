"""
Stage 2 — TF-IDF Semantic Similarity (OpenAlex v2)
====================================================
Computes cosine similarity between title+abstract pairs using TF-IDF vectors.
This is a computationally efficient alternative to SBERT that runs in minutes
rather than hours on CPU-only hardware.

Outputs (saved to results/oa/):
  oa_pairs_v2_sbert.parquet  — pairs with semantic_similarity column added
  oa_sbert_stats_v2.json     — encoding statistics
"""

import json, os, time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

PAIRS_PATH = "/home/ubuntu/pipeline_v2/results/oa/oa_pairs_v2.parquet"
DATA_PATH  = "/home/ubuntu/data/OpenAlex_LIS_1975_2024.parquet"
OUT_DIR    = "/home/ubuntu/pipeline_v2/results/oa"
LOG_FILE   = f"{OUT_DIR}/stage2_oa.log"
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("Loading pairs and article data...")
pairs_df = pd.read_parquet(PAIRS_PATH)
art_df   = pd.read_parquet(DATA_PATH, columns=["id","title","abstract"])
log(f"Pairs: {len(pairs_df):,} | Articles: {len(art_df):,}")

# Collect unique article IDs needed
needed_ids = set(pairs_df["citing_id"].tolist()) | set(pairs_df["cited_id"].tolist())
art_df = art_df[art_df["id"].isin(needed_ids)].copy().reset_index(drop=True)
log(f"Unique articles to encode: {len(art_df):,}")

# Build text
def make_text(row):
    t = str(row["title"])   if pd.notna(row["title"])    else ""
    a = str(row["abstract"]) if pd.notna(row["abstract"]) else ""
    return (t + " " + a).strip()

art_df["text"] = art_df.apply(make_text, axis=1)
id_to_idx = {pid: i for i, pid in enumerate(art_df["id"])}

# TF-IDF vectorization
log("Fitting TF-IDF vectorizer (max_features=50000, sublinear_tf=True)...")
t0 = time.time()
vectorizer = TfidfVectorizer(
    max_features=50000,
    sublinear_tf=True,
    min_df=2,
    ngram_range=(1, 2),
    strip_accents="unicode",
    analyzer="word"
)
tfidf_matrix = vectorizer.fit_transform(art_df["text"].tolist())
log(f"TF-IDF matrix shape: {tfidf_matrix.shape} in {time.time()-t0:.1f}s")

# Compute pairwise similarities in batches
log("Computing pairwise cosine similarities in batches...")
t0 = time.time()

BATCH_SIZE = 10000
sims = []
missing = 0

citing_ids = pairs_df["citing_id"].tolist()
cited_ids  = pairs_df["cited_id"].tolist()

n_batches = (len(pairs_df) + BATCH_SIZE - 1) // BATCH_SIZE
for b in range(n_batches):
    start = b * BATCH_SIZE
    end   = min(start + BATCH_SIZE, len(pairs_df))
    
    batch_citing = citing_ids[start:end]
    batch_cited  = cited_ids[start:end]
    
    # Get indices
    ci_idxs = [id_to_idx.get(pid) for pid in batch_citing]
    cd_idxs = [id_to_idx.get(pid) for pid in batch_cited]
    
    for ci, cd in zip(ci_idxs, cd_idxs):
        if ci is None or cd is None:
            sims.append(0.0)
            missing += 1
        else:
            # Dot product of L2-normalized TF-IDF vectors = cosine similarity
            a = tfidf_matrix[ci]
            b_vec = tfidf_matrix[cd]
            sim = float((a * b_vec.T).toarray()[0, 0])
            # Normalize by norms
            norm_a = float(np.sqrt(a.multiply(a).sum()))
            norm_b = float(np.sqrt(b_vec.multiply(b_vec).sum()))
            if norm_a > 0 and norm_b > 0:
                sim = sim / (norm_a * norm_b)
            else:
                sim = 0.0
            sims.append(sim)
    
    if (b + 1) % 10 == 0:
        log(f"Batch {b+1}/{n_batches} done ({end:,}/{len(pairs_df):,} pairs)")

log(f"Similarities computed in {time.time()-t0:.1f}s. Missing: {missing:,}")

pairs_df["semantic_similarity"] = sims
out_path = f"{OUT_DIR}/oa_pairs_v2_sbert.parquet"
pairs_df.to_parquet(out_path, index=False)
log(f"Saved to {out_path}")

# Verify
log(f"Similarity stats: min={min(sims):.4f}, max={max(sims):.4f}, mean={sum(sims)/len(sims):.4f}")

stats = {
    "method": "TF-IDF cosine similarity",
    "n_articles_encoded": int(len(art_df)),
    "n_pairs": int(len(pairs_df)),
    "tfidf_max_features": 50000,
    "tfidf_ngram_range": [1, 2],
    "missing_embeddings": int(missing),
    "sim_mean": float(sum(sims)/len(sims)),
    "sim_std": float(np.std(sims))
}
with open(f"{OUT_DIR}/oa_sbert_stats_v2.json", "w") as f:
    json.dump(stats, f, indent=2)

log("Stage 2 TF-IDF complete.")
