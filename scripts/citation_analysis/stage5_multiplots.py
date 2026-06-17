"""
Stage 5: Per-Dataset Multiplots (2×5 grids)
============================================
Reads the Stage 2 pair files and Stage 3 JSON results and produces
four publication-quality multiplot figures, each a 2×5 grid of subplots
(one panel per dataset):

  fig5_roc_multiplot.png      – CV ROC curves (all 4 classifiers)
  fig6_rolling_multiplot.png  – Rolling temporal AUC (Gradient Boosting)
  fig7_ablation_multiplot.png – Leave-one-out ablation AUC drop per feature
  fig8_shap_multiplot.png     – Normalized SHAP importance per feature

All outputs go to:
  <script_dir>/figures/

Usage:
  python stage5_multiplots.py
"""

import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
FEAT_DIR    = SCRIPT_DIR / "features"
RESULTS_DIR = SCRIPT_DIR / "results"
FIG_DIR     = SCRIPT_DIR / "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Dataset registry ───────────────────────────────────────────────────────
DATASETS = [
    "darkmatter", "LIS", "fatigue_crack", "environmental_engineering",
    "neuroblastoma", "osteosarcoma", "political_participation",
    "welfare_state", "archaeology", "art_history",
]

LABELS = {
    "darkmatter":               "Dark Matter",
    "LIS":                      "Info Literacy (LIS)",
    "fatigue_crack":            "Fatigue Crack",
    "environmental_engineering":"Env. Engineering",
    "neuroblastoma":            "Neuroblastoma",
    "osteosarcoma":             "Osteosarcoma",
    "political_participation":  "Political Partic.",
    "welfare_state":            "Welfare State",
    "archaeology":              "Archaeology",
    "art_history":              "Art History",
}

DISCIPLINE_COLORS = {
    "darkmatter":               "#1f77b4",   # Science – blue
    "LIS":                      "#1f77b4",
    "fatigue_crack":            "#ff7f0e",   # Engineering – orange
    "environmental_engineering":"#ff7f0e",
    "neuroblastoma":            "#2ca02c",   # BioMed – green
    "osteosarcoma":             "#2ca02c",
    "political_participation":  "#d62728",   # Social Science – red
    "welfare_state":            "#d62728",
    "archaeology":              "#9467bd",   # Humanities – purple
    "art_history":              "#9467bd",
}

FEATURES = ["prestige_cited", "temporal_gap", "common_refs",
            "jaccard_refs", "common_citers", "semantic_similarity"]

FEATURE_LABELS = {
    "prestige_cited":    "Prestige",
    "temporal_gap":      "Temp. Gap",
    "common_refs":       "Common Refs",
    "jaccard_refs":      "Jaccard",
    "common_citers":     "Co-citers",
    "semantic_similarity": "Semantic",
}

MODEL_COLORS = {
    "Logistic Regression": "#4878cf",
    "Linear SVM":          "#6acc65",
    "Random Forest":       "#d65f5f",
    "Gradient Boosting":   "#b47cc7",
}

# ── Rolling windows (same as manuscript) ──────────────────────────────────
ROLLING_WINDOWS = [
    ("2005-2010", 2004, 2005, 2010),
    ("2010-2015", 2009, 2010, 2015),
    ("2015-2020", 2014, 2015, 2020),
    ("2018-2025", 2017, 2018, 2025),
]

# ══════════════════════════════════════════════════════════════════════════
# Helper: load pairs
# ══════════════════════════════════════════════════════════════════════════
def load_pairs(dataset):
    path = FEAT_DIR / f"{dataset}_pairs_stage2.parquet"
    if not path.exists():
        path = FEAT_DIR / f"{dataset}_pairs_stage1.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df

# ══════════════════════════════════════════════════════════════════════════
# Fig 5 – CV ROC multiplot
# ══════════════════════════════════════════════════════════════════════════
def make_roc_multiplot():
    print("Generating fig5_roc_multiplot.png ...")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        df = load_pairs(ds)
        if df is None:
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        X = df[FEATURES].values.astype(float)
        y = df["label"].values
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
            "Linear SVM":          CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED)),
            "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=SEED),
        }

        for mname, model in models.items():
            tprs, aucs = [], []
            mean_fpr = np.linspace(0, 1, 200)
            for tr, te in skf.split(X, y):
                sc = StandardScaler()
                model.fit(sc.fit_transform(X[tr]), y[tr])
                yp = model.predict_proba(sc.transform(X[te]))[:, 1]
                fpr, tpr, _ = roc_curve(y[te], yp)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                aucs.append(roc_auc_score(y[te], yp))
            mean_tpr = np.mean(tprs, axis=0)
            mean_auc = np.mean(aucs)
            ax.plot(mean_fpr, mean_tpr,
                    color=MODEL_COLORS[mname], lw=1.5,
                    label=f"{mname} ({mean_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_title(LABELS[ds], fontsize=10,
                     color=DISCIPLINE_COLORS[ds], fontweight="bold")
        ax.set_xlabel("FPR", fontsize=8)
        ax.set_ylabel("TPR", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="lower right")

    fig.suptitle("CV ROC Curves Across 10 Datasets", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fig5_roc_multiplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
# Fig 6 – Rolling temporal AUC multiplot
# ══════════════════════════════════════════════════════════════════════════
def make_rolling_multiplot():
    print("Generating fig6_rolling_multiplot.png ...")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        df = load_pairs(ds)
        if df is None:
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        X = df[FEATURES].values.astype(float)
        y = df["label"].values

        # citing_year column
        if "citing_year" not in df.columns:
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No year col", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        years = df["citing_year"].values
        gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)

        window_labels, window_aucs = [], []
        for wlabel, train_end, test_start, test_end in ROLLING_WINDOWS:
            tr_mask = years <= train_end
            te_mask = (years >= test_start) & (years <= test_end)
            if tr_mask.sum() < 10 or te_mask.sum() < 4:
                continue
            # ensure both classes present
            if len(np.unique(y[te_mask])) < 2:
                continue
            sc = StandardScaler()
            gb.fit(sc.fit_transform(X[tr_mask]), y[tr_mask])
            yp = gb.predict_proba(sc.transform(X[te_mask]))[:, 1]
            auc = roc_auc_score(y[te_mask], yp)
            window_labels.append(wlabel)
            window_aucs.append(auc)

        if window_aucs:
            color = DISCIPLINE_COLORS[ds]
            ax.plot(range(len(window_labels)), window_aucs,
                    marker="o", color=color, lw=2, ms=6)
            ax.set_xticks(range(len(window_labels)))
            ax.set_xticklabels(window_labels, rotation=30, ha="right", fontsize=7)
            ax.set_ylim([max(0.4, min(window_aucs) - 0.05), 1.02])
            for i, v in enumerate(window_aucs):
                ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7)
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)

        ax.set_title(LABELS[ds], fontsize=10,
                     color=DISCIPLINE_COLORS[ds], fontweight="bold")
        ax.set_ylabel("AUC", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Rolling Temporal AUC (Gradient Boosting) Across 10 Datasets",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fig6_rolling_multiplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
# Fig 7 – Ablation multiplot
# ══════════════════════════════════════════════════════════════════════════
def make_ablation_multiplot():
    print("Generating fig7_ablation_multiplot.png ...")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        path = RESULTS_DIR / f"{ds}_ablation.json"
        if not path.exists():
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No results", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        with open(path) as f:
            ablation = json.load(f)

        baseline = ablation.get("all_features", 0)
        drops, feat_names = [], []
        for feat in FEATURES:
            key = f"without_{feat}"
            if key in ablation:
                drop = max(0.0, baseline - ablation[key])
                drops.append(drop)
                feat_names.append(FEATURE_LABELS[feat])

        if not drops:
            ax.text(0.5, 0.5, "No ablation data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(LABELS[ds])
            continue

        color = DISCIPLINE_COLORS[ds]
        bars = ax.barh(feat_names, drops, color=color, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, drops):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)

        ax.set_title(LABELS[ds], fontsize=10,
                     color=DISCIPLINE_COLORS[ds], fontweight="bold")
        ax.set_xlabel("AUC Drop", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature Ablation: AUC Drop per Feature Across 10 Datasets",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fig7_ablation_multiplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
# Fig 8 – SHAP multiplot
# ══════════════════════════════════════════════════════════════════════════
def make_shap_multiplot():
    print("Generating fig8_shap_multiplot.png ...")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for idx, ds in enumerate(DATASETS):
        ax = axes[idx]
        path = RESULTS_DIR / f"{ds}_shap.json"
        if not path.exists():
            ax.set_title(LABELS[ds])
            ax.text(0.5, 0.5, "No results", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        with open(path) as f:
            shap_data = json.load(f)

        feats = shap_data.get("features", FEATURES)
        vals  = np.array(shap_data.get("mean_abs_shap", []))
        if vals.sum() > 0:
            vals = vals / vals.sum()   # normalize to sum=1

        feat_names = [FEATURE_LABELS.get(f, f) for f in feats]

        color = DISCIPLINE_COLORS[ds]
        bars = ax.barh(feat_names, vals, color=color, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", fontsize=7)

        ax.set_xlim([0, 1.05])
        ax.set_title(LABELS[ds], fontsize=10,
                     color=DISCIPLINE_COLORS[ds], fontweight="bold")
        ax.set_xlabel("Norm. SHAP", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Normalized SHAP Feature Importance Across 10 Datasets",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = FIG_DIR / "fig8_shap_multiplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    make_roc_multiplot()
    make_rolling_multiplot()
    make_ablation_multiplot()
    make_shap_multiplot()
    print("\nStage 5 complete. Four multiplots saved to figures/")
