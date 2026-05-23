"""
Stage 4 — Figures (v2)
=======================
Generates all figures for both datasets and comparative analysis.
Uses only stored CV results (no model retraining).

Outputs saved to:
  results/figures/dim/   — Dimensions figures
  results/figures/oa/    — OpenAlex figures
  results/figures/comp/  — Comparative figures
"""

import json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns

warnings.filterwarnings("ignore")

DIM_PAIRS  = "/home/ubuntu/pipeline_v2/results/dim/dim_pairs_v2.parquet"
OA_PAIRS   = "/home/ubuntu/pipeline_v2/results/oa/oa_pairs_v2_sbert.parquet"
DIM_CV     = "/home/ubuntu/pipeline_v2/results/dim/dim_cv_results_v2.json"
OA_CV      = "/home/ubuntu/pipeline_v2/results/oa/oa_cv_results_v2.json"
DIM_SHAP   = "/home/ubuntu/pipeline_v2/results/dim/dim_shap_v2.json"
OA_SHAP    = "/home/ubuntu/pipeline_v2/results/oa/oa_shap_v2.json"
DIM_ABL    = "/home/ubuntu/pipeline_v2/results/dim/dim_ablation_v2.json"
OA_ABL     = "/home/ubuntu/pipeline_v2/results/oa/oa_ablation_v2.json"
DIM_TEMP   = "/home/ubuntu/pipeline_v2/results/dim/dim_temporal_holdout_v2.json"
OA_TEMP    = "/home/ubuntu/pipeline_v2/results/oa/oa_temporal_holdout_v2.json"
DIM_DATA   = "/home/ubuntu/data/Dimensions_LIS_1975_2024.parquet"
OA_DATA    = "/home/ubuntu/data/OpenAlex_LIS_1975_2024.parquet"

FIG_DIM  = "/home/ubuntu/pipeline_v2/results/figures/dim"
FIG_OA   = "/home/ubuntu/pipeline_v2/results/figures/oa"
FIG_COMP = "/home/ubuntu/pipeline_v2/results/figures/comp"
for d in [FIG_DIM, FIG_OA, FIG_COMP]:
    os.makedirs(d, exist_ok=True)

LOG_FILE = "/home/ubuntu/pipeline_v2/results/stage4.log"

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

PALETTE = {"Dimensions": "#2196F3", "OpenAlex": "#FF5722"}
MODEL_COLORS = ["#4CAF50","#2196F3","#FF9800","#9C27B0"]
STYLE = {"figure.dpi": 150, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
plt.rcParams.update(STYLE)

# ── Load data ─────────────────────────────────────────────────────────────────
log("Loading data and results...")
dim_cv   = json.load(open(DIM_CV))
oa_cv    = json.load(open(OA_CV))
dim_shap = json.load(open(DIM_SHAP))
oa_shap  = json.load(open(OA_SHAP))
dim_abl  = json.load(open(DIM_ABL))
oa_abl   = json.load(open(OA_ABL))
dim_temp = json.load(open(DIM_TEMP))
oa_temp  = json.load(open(OA_TEMP))

dim_pairs = pd.read_parquet(DIM_PAIRS)
oa_pairs  = pd.read_parquet(OA_PAIRS)

dim_art = pd.read_parquet(DIM_DATA, columns=["year","times_cited"])
oa_art  = pd.read_parquet(OA_DATA,  columns=["year","cited_by_count"] if "cited_by_count" in pd.read_parquet(OA_DATA, columns=["year"]).columns else ["year","times_cited"])

# ── Fig DIM-1: Publications per year ─────────────────────────────────────────
log("Fig DIM-1: Publications per year (Dimensions)...")
dim_art["year"] = pd.to_numeric(dim_art["year"], errors="coerce")
oa_art["year"]  = pd.to_numeric(oa_art["year"],  errors="coerce")

fig, ax = plt.subplots(figsize=(9, 4))
dim_yr = dim_art.groupby("year").size()
ax.bar(dim_yr.index, dim_yr.values, color=PALETTE["Dimensions"], alpha=0.8, width=0.8)
ax.set_xlabel("Year"); ax.set_ylabel("Number of Articles")
ax.set_title("Dimensions.ai — Publications per Year (LIS, 1975–2024)")
ax.set_xlim(1974, 2025)
plt.tight_layout()
plt.savefig(f"{FIG_DIM}/dim_publications_per_year.png", bbox_inches="tight")
plt.close()
log("  Saved dim_publications_per_year.png")

# ── Fig OA-1: Publications per year ──────────────────────────────────────────
log("Fig OA-1: Publications per year (OpenAlex)...")
fig, ax = plt.subplots(figsize=(9, 4))
oa_yr = oa_art.groupby("year").size()
ax.bar(oa_yr.index, oa_yr.values, color=PALETTE["OpenAlex"], alpha=0.8, width=0.8)
ax.set_xlabel("Year"); ax.set_ylabel("Number of Articles")
ax.set_title("OpenAlex — Publications per Year (LIS, 1975–2024)")
ax.set_xlim(1974, 2025)
plt.tight_layout()
plt.savefig(f"{FIG_OA}/oa_publications_per_year.png", bbox_inches="tight")
plt.close()
log("  Saved oa_publications_per_year.png")

# ── Fig DIM-2: Feature distributions ─────────────────────────────────────────
log("Fig DIM-2: Feature distributions (Dimensions)...")
dim_feats = ["prestige_cited","temporal_gap","common_refs","jaccard_refs","common_citers"]
fig, axes = plt.subplots(1, len(dim_feats), figsize=(16, 4))
for ax, feat in zip(axes, dim_feats):
    pos = dim_pairs[dim_pairs["label"]==1][feat]
    neg = dim_pairs[dim_pairs["label"]==0][feat]
    ax.hist(pos, bins=40, alpha=0.6, color="#2196F3", label="Citing", density=True)
    ax.hist(neg, bins=40, alpha=0.6, color="#FF5722", label="Non-citing", density=True)
    ax.set_title(feat.replace("_"," ").title(), fontsize=9)
    ax.set_xlabel("Value"); ax.set_ylabel("Density")
axes[0].legend(fontsize=8)
plt.suptitle("Dimensions — Feature Distributions by Pair Type", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_DIM}/dim_feature_distributions.png", bbox_inches="tight")
plt.close()
log("  Saved dim_feature_distributions.png")

# ── Fig OA-2: Feature distributions ──────────────────────────────────────────
log("Fig OA-2: Feature distributions (OpenAlex)...")
oa_feats = ["prestige_cited","activity_citing","temporal_gap","common_refs","jaccard_refs","common_citers","semantic_similarity"]
fig, axes = plt.subplots(1, len(oa_feats), figsize=(20, 4))
for ax, feat in zip(axes, oa_feats):
    pos = oa_pairs[oa_pairs["label"]==1][feat]
    neg = oa_pairs[oa_pairs["label"]==0][feat]
    ax.hist(pos, bins=40, alpha=0.6, color="#2196F3", label="Citing", density=True)
    ax.hist(neg, bins=40, alpha=0.6, color="#FF5722", label="Non-citing", density=True)
    ax.set_title(feat.replace("_"," ").title(), fontsize=8)
    ax.set_xlabel("Value"); ax.set_ylabel("Density")
axes[0].legend(fontsize=7)
plt.suptitle("OpenAlex — Feature Distributions by Pair Type", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG_OA}/oa_feature_distributions.png", bbox_inches="tight")
plt.close()
log("  Saved oa_feature_distributions.png")

# ── Fig DIM-3: Model comparison bar chart ────────────────────────────────────
log("Fig DIM-3: Model comparison (Dimensions)...")
models = list(dim_cv.keys())
metrics = ["auc","pr_auc","f1","accuracy","precision","recall","mcc"]
metric_labels = ["AUC-ROC","PR-AUC","F1","Accuracy","Precision","Recall","MCC"]
x = np.arange(len(metrics))
width = 0.2
fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, color) in enumerate(zip(models, MODEL_COLORS)):
    means = [dim_cv[name]["mean"][m] for m in metrics]
    stds  = [dim_cv[name]["std"][m]  for m in metrics]
    ax.bar(x + i*width, means, width, yerr=stds, label=name, color=color, alpha=0.85, capsize=3)
ax.set_xticks(x + width*1.5); ax.set_xticklabels(metric_labels, rotation=20, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0.85, 1.02)
ax.set_title("Dimensions — Model Performance Comparison (5-fold CV)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIM}/dim_model_comparison.png", bbox_inches="tight")
plt.close()
log("  Saved dim_model_comparison.png")

# ── Fig OA-3: Model comparison bar chart ─────────────────────────────────────
log("Fig OA-3: Model comparison (OpenAlex)...")
models_oa = list(oa_cv.keys())
fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, color) in enumerate(zip(models_oa, MODEL_COLORS)):
    means = [oa_cv[name]["mean"][m] for m in metrics]
    stds  = [oa_cv[name]["std"][m]  for m in metrics]
    ax.bar(x + i*width, means, width, yerr=stds, label=name, color=color, alpha=0.85, capsize=3)
ax.set_xticks(x + width*1.5); ax.set_xticklabels(metric_labels, rotation=20, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0.85, 1.02)
ax.set_title("OpenAlex — Model Performance Comparison (5-fold CV)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_OA}/oa_model_comparison.png", bbox_inches="tight")
plt.close()
log("  Saved oa_model_comparison.png")

# ── Fig DIM-4: SHAP feature importance ───────────────────────────────────────
log("Fig DIM-4: SHAP (Dimensions)...")
feats_d = dim_shap["features"]
vals_d  = dim_shap["mean_abs_shap"]
order_d = np.argsort(vals_d)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([feats_d[i].replace("_"," ").title() for i in order_d],
        [vals_d[i] for i in order_d], color=PALETTE["Dimensions"], alpha=0.85)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Dimensions — SHAP Feature Importance (Gradient Boosting)")
plt.tight_layout()
plt.savefig(f"{FIG_DIM}/dim_shap.png", bbox_inches="tight")
plt.close()
log("  Saved dim_shap.png")

# ── Fig OA-4: SHAP feature importance ────────────────────────────────────────
log("Fig OA-4: SHAP (OpenAlex)...")
feats_o = oa_shap["features"]
vals_o  = oa_shap["mean_abs_shap"]
order_o = np.argsort(vals_o)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([feats_o[i].replace("_"," ").title() for i in order_o],
        [vals_o[i] for i in order_o], color=PALETTE["OpenAlex"], alpha=0.85)
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("OpenAlex — SHAP Feature Importance (Gradient Boosting)")
plt.tight_layout()
plt.savefig(f"{FIG_OA}/oa_shap.png", bbox_inches="tight")
plt.close()
log("  Saved oa_shap.png")

# ── Fig DIM-5: Ablation ───────────────────────────────────────────────────────
log("Fig DIM-5: Ablation (Dimensions)...")
baseline_d = dim_abl["all_features"]
abl_feats_d = [k.replace("without_","") for k in dim_abl if k != "all_features"]
abl_drops_d = [baseline_d - dim_abl[f"without_{f}"] for f in abl_feats_d]
order_ad = np.argsort(abl_drops_d)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([abl_feats_d[i].replace("_"," ").title() for i in order_ad],
        [abl_drops_d[i] for i in order_ad], color=PALETTE["Dimensions"], alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("AUC Drop (baseline − without feature)")
ax.set_title(f"Dimensions — Feature Ablation (RF, baseline AUC={baseline_d:.4f})")
plt.tight_layout()
plt.savefig(f"{FIG_DIM}/dim_ablation.png", bbox_inches="tight")
plt.close()
log("  Saved dim_ablation.png")

# ── Fig OA-5: Ablation ────────────────────────────────────────────────────────
log("Fig OA-5: Ablation (OpenAlex)...")
baseline_o = oa_abl["all_features"]
abl_feats_o = [k.replace("without_","") for k in oa_abl if k != "all_features"]
abl_drops_o = [baseline_o - oa_abl[f"without_{f}"] for f in abl_feats_o]
order_ao = np.argsort(abl_drops_o)[::-1]
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh([abl_feats_o[i].replace("_"," ").title() for i in order_ao],
        [abl_drops_o[i] for i in order_ao], color=PALETTE["OpenAlex"], alpha=0.85)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xlabel("AUC Drop (baseline − without feature)")
ax.set_title(f"OpenAlex — Feature Ablation (RF, baseline AUC={baseline_o:.4f})")
plt.tight_layout()
plt.savefig(f"{FIG_OA}/oa_ablation.png", bbox_inches="tight")
plt.close()
log("  Saved oa_ablation.png")

# ── Fig OA-6: Semantic similarity distribution ────────────────────────────────
log("Fig OA-6: Semantic similarity distribution (OpenAlex)...")
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(oa_pairs[oa_pairs["label"]==1]["semantic_similarity"], bins=50,
        alpha=0.6, color="#2196F3", label="Citing pairs", density=True)
ax.hist(oa_pairs[oa_pairs["label"]==0]["semantic_similarity"], bins=50,
        alpha=0.6, color="#FF5722", label="Non-citing pairs", density=True)
ax.set_xlabel("Cosine Similarity (SPECTER)")
ax.set_ylabel("Density")
ax.set_title("OpenAlex — Semantic Similarity by Pair Type")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_OA}/oa_semantic_similarity.png", bbox_inches="tight")
plt.close()
log("  Saved oa_semantic_similarity.png")

# ── Fig COMP-1: AUC-ROC comparison ───────────────────────────────────────────
log("Fig COMP-1: AUC-ROC comparison (Dimensions vs OpenAlex)...")
common_models = [m for m in dim_cv if m in oa_cv]
dim_aucs = [dim_cv[m]["mean"]["auc"] for m in common_models]
oa_aucs  = [oa_cv[m]["mean"]["auc"]  for m in common_models]
dim_stds = [dim_cv[m]["std"]["auc"]  for m in common_models]
oa_stds  = [oa_cv[m]["std"]["auc"]   for m in common_models]
x = np.arange(len(common_models))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, dim_aucs, 0.35, yerr=dim_stds, label="Dimensions", color=PALETTE["Dimensions"], alpha=0.85, capsize=4)
ax.bar(x + 0.2, oa_aucs,  0.35, yerr=oa_stds,  label="OpenAlex",   color=PALETTE["OpenAlex"],   alpha=0.85, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(common_models, rotation=15, ha="right")
ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.93, 1.01)
ax.set_title("Dimensions vs OpenAlex — AUC-ROC Comparison (5-fold CV)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_COMP}/comp_auc_comparison.png", bbox_inches="tight")
plt.close()
log("  Saved comp_auc_comparison.png")

# ── Fig COMP-2: PR-AUC comparison ────────────────────────────────────────────
log("Fig COMP-2: PR-AUC comparison...")
dim_prauc = [dim_cv[m]["mean"]["pr_auc"] for m in common_models]
oa_prauc  = [oa_cv[m]["mean"]["pr_auc"]  for m in common_models]
dim_prauc_std = [dim_cv[m]["std"]["pr_auc"] for m in common_models]
oa_prauc_std  = [oa_cv[m]["std"]["pr_auc"]  for m in common_models]
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, dim_prauc, 0.35, yerr=dim_prauc_std, label="Dimensions", color=PALETTE["Dimensions"], alpha=0.85, capsize=4)
ax.bar(x + 0.2, oa_prauc,  0.35, yerr=oa_prauc_std,  label="OpenAlex",   color=PALETTE["OpenAlex"],   alpha=0.85, capsize=4)
ax.set_xticks(x); ax.set_xticklabels(common_models, rotation=15, ha="right")
ax.set_ylabel("PR-AUC"); ax.set_ylim(0.90, 1.01)
ax.set_title("Dimensions vs OpenAlex — PR-AUC Comparison (5-fold CV)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_COMP}/comp_prauc_comparison.png", bbox_inches="tight")
plt.close()
log("  Saved comp_prauc_comparison.png")

# ── Fig COMP-3: Full metrics heatmap ─────────────────────────────────────────
log("Fig COMP-3: Full metrics heatmap...")
all_metrics = ["auc","pr_auc","f1","accuracy","precision","recall","mcc"]
metric_labels_short = ["AUC","PR-AUC","F1","Acc","Prec","Rec","MCC"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, cv_data, title in zip(axes, [dim_cv, oa_cv], ["Dimensions", "OpenAlex"]):
    mat = np.array([[cv_data[m]["mean"][met] for met in all_metrics] for m in common_models])
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0.88, vmax=1.0)
    ax.set_xticks(range(len(all_metrics))); ax.set_xticklabels(metric_labels_short, rotation=30, ha="right")
    ax.set_yticks(range(len(common_models))); ax.set_yticklabels(common_models, fontsize=9)
    for i in range(len(common_models)):
        for j in range(len(all_metrics)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if mat[i,j] < 0.96 else "white")
    ax.set_title(f"{title} — Model Metrics")
    plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig(f"{FIG_COMP}/comp_metrics_heatmap.png", bbox_inches="tight")
plt.close()
log("  Saved comp_metrics_heatmap.png")

# ── Fig COMP-4: SHAP comparison ───────────────────────────────────────────────
log("Fig COMP-4: SHAP comparison (common features)...")
common_feats = ["prestige_cited","temporal_gap","common_refs","jaccard_refs","common_citers"]
dim_shap_vals = {f: v for f, v in zip(dim_shap["features"], dim_shap["mean_abs_shap"])}
oa_shap_vals  = {f: v for f, v in zip(oa_shap["features"],  oa_shap["mean_abs_shap"])}
# Normalise to sum=1
dim_total = sum(dim_shap_vals.get(f, 0) for f in common_feats)
oa_total  = sum(oa_shap_vals.get(f, 0)  for f in common_feats)
dim_norm  = [dim_shap_vals.get(f, 0) / dim_total if dim_total > 0 else 0 for f in common_feats]
oa_norm   = [oa_shap_vals.get(f, 0)  / oa_total  if oa_total  > 0 else 0 for f in common_feats]
x = np.arange(len(common_feats))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, dim_norm, 0.35, label="Dimensions", color=PALETTE["Dimensions"], alpha=0.85)
ax.bar(x + 0.2, oa_norm,  0.35, label="OpenAlex",   color=PALETTE["OpenAlex"],   alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f.replace("_"," ").title() for f in common_feats], rotation=20, ha="right")
ax.set_ylabel("Normalised Mean |SHAP value|")
ax.set_title("Dimensions vs OpenAlex — SHAP Feature Importance (Normalised)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_COMP}/comp_shap_comparison.png", bbox_inches="tight")
plt.close()
log("  Saved comp_shap_comparison.png")

# ── Fig COMP-5: Temporal hold-out comparison ──────────────────────────────────
log("Fig COMP-5: Temporal hold-out comparison...")
common_temp = [m for m in dim_temp if m in oa_temp]
dim_t_aucs = [dim_temp[m]["auc"] for m in common_temp]
oa_t_aucs  = [oa_temp[m]["auc"]  for m in common_temp]
x = np.arange(len(common_temp))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - 0.2, dim_t_aucs, 0.35, label="Dimensions", color=PALETTE["Dimensions"], alpha=0.85)
ax.bar(x + 0.2, oa_t_aucs,  0.35, label="OpenAlex",   color=PALETTE["OpenAlex"],   alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(common_temp, rotation=15, ha="right")
ax.set_ylabel("AUC-ROC"); ax.set_ylim(0.85, 1.01)
ax.set_title("Temporal Hold-out AUC (train ≤2015, test 2016–2020)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_COMP}/comp_temporal_holdout.png", bbox_inches="tight")
plt.close()
log("  Saved comp_temporal_holdout.png")

log("Stage 4 Figures complete. All figures saved.")
