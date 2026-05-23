"""
Stage 3 — ML Training (Dimensions v2)
======================================
5-fold stratified CV for LR, SVM, RF, GB.
Metrics: AUC-ROC, PR-AUC, F1, Accuracy, Precision, Recall, MCC.
Also: SHAP feature importance, leave-one-out ablation, temporal hold-out.

Outputs (saved to results/dim/):
  dim_cv_results_v2.json
  dim_shap_v2.json
  dim_ablation_v2.json
  dim_temporal_holdout_v2.json
"""

import json, os, time, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, accuracy_score, precision_score,
                              recall_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler
import shap

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

PAIRS_PATH = "/home/ubuntu/pipeline_v2/results/dim/dim_pairs_v2.parquet"
OUT_DIR    = "/home/ubuntu/pipeline_v2/results/dim"
LOG_FILE   = f"{OUT_DIR}/stage3_dim.log"
os.makedirs(OUT_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("[%H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("Loading Dimensions pairs v2...")
df = pd.read_parquet(PAIRS_PATH)
log(f"Loaded {len(df):,} pairs")

FEATURES = ["prestige_cited","temporal_gap","common_refs","jaccard_refs","common_citers"]
X = df[FEATURES].values.astype(float)
y = df["label"].values

# ── 5-fold CV ─────────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=SEED)),
    "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=SEED),
}

cv_results = {}
for name, model in models.items():
    log(f"Training {name}...")
    fold_metrics = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)
        model.fit(Xtr, ytr)
        yprob = model.predict_proba(Xte)[:, 1]
        ypred = model.predict(Xte)
        fold_metrics.append({
            "auc":       roc_auc_score(yte, yprob),
            "pr_auc":    average_precision_score(yte, yprob),
            "f1":        f1_score(yte, ypred),
            "accuracy":  accuracy_score(yte, ypred),
            "precision": precision_score(yte, ypred),
            "recall":    recall_score(yte, ypred),
            "mcc":       matthews_corrcoef(yte, ypred),
        })
        log(f"  Fold {fold+1}/5 AUC={fold_metrics[-1]['auc']:.4f} PR-AUC={fold_metrics[-1]['pr_auc']:.4f}")
    means = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    stds  = {k: float(np.std( [m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    cv_results[name] = {"mean": means, "std": stds, "folds": fold_metrics}
    log(f"  {name} mean AUC={means['auc']:.4f} ± {stds['auc']:.4f}")

with open(f"{OUT_DIR}/dim_cv_results_v2.json", "w") as f:
    json.dump(cv_results, f, indent=2)
log("CV results saved.")

# ── SHAP (best model = GB) ─────────────────────────────────────────────────────
log("Computing SHAP values (Gradient Boosting)...")
scaler_full = StandardScaler()
Xsc = scaler_full.fit_transform(X)
gb = GradientBoostingClassifier(n_estimators=200, random_state=SEED)
gb.fit(Xsc, y)
explainer = shap.TreeExplainer(gb)
shap_vals  = explainer.shap_values(Xsc[:5000])
mean_abs   = np.abs(shap_vals).mean(axis=0).tolist()
shap_out   = {"features": FEATURES, "mean_abs_shap": mean_abs}
with open(f"{OUT_DIR}/dim_shap_v2.json", "w") as f:
    json.dump(shap_out, f, indent=2)
log("SHAP saved.")

# ── Ablation ──────────────────────────────────────────────────────────────────
log("Running leave-one-out ablation (Random Forest)...")
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)
ablation = {}
# Baseline
base_aucs = []
for tr, te in skf.split(X, y):
    sc = StandardScaler()
    rf.fit(sc.fit_transform(X[tr]), y[tr])
    base_aucs.append(roc_auc_score(y[te], rf.predict_proba(sc.transform(X[te]))[:,1]))
ablation["all_features"] = float(np.mean(base_aucs))
log(f"  Baseline AUC={ablation['all_features']:.4f}")
# Leave one out
for i, feat in enumerate(FEATURES):
    mask = [j for j in range(len(FEATURES)) if j != i]
    Xm = X[:, mask]
    aucs = []
    for tr, te in skf.split(Xm, y):
        sc = StandardScaler()
        rf.fit(sc.fit_transform(Xm[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], rf.predict_proba(sc.transform(Xm[te]))[:,1]))
    ablation[f"without_{feat}"] = float(np.mean(aucs))
    log(f"  without {feat}: AUC={ablation[f'without_{feat}']:.4f} (drop={ablation['all_features']-ablation[f'without_{feat}']:.4f})")

with open(f"{OUT_DIR}/dim_ablation_v2.json", "w") as f:
    json.dump(ablation, f, indent=2)
log("Ablation saved.")

# ── Temporal hold-out ─────────────────────────────────────────────────────────
log("Running temporal hold-out evaluation (train ≤2015, test 2016–2020)...")
train_mask = df["citing_year"] <= 2015
test_mask  = (df["citing_year"] >= 2016) & (df["citing_year"] <= 2020)
Xtr_t, ytr_t = X[train_mask], y[train_mask]
Xte_t, yte_t = X[test_mask],  y[test_mask]
log(f"  Train: {train_mask.sum():,} pairs | Test: {test_mask.sum():,} pairs")

temporal_results = {}
for name, model in models.items():
    sc = StandardScaler()
    model.fit(sc.fit_transform(Xtr_t), ytr_t)
    yprob = model.predict_proba(sc.transform(Xte_t))[:, 1]
    ypred = model.predict(sc.transform(Xte_t))
    temporal_results[name] = {
        "auc":    float(roc_auc_score(yte_t, yprob)),
        "pr_auc": float(average_precision_score(yte_t, yprob)),
        "f1":     float(f1_score(yte_t, ypred)),
        "mcc":    float(matthews_corrcoef(yte_t, ypred)),
    }
    log(f"  {name}: AUC={temporal_results[name]['auc']:.4f}")

with open(f"{OUT_DIR}/dim_temporal_holdout_v2.json", "w") as f:
    json.dump(temporal_results, f, indent=2)
log("Temporal hold-out saved.")

log("Stage 3 Dimensions complete.")
