# Semantic Alignment, Chronological Distance, and Citation-Network Prominence in Citation Formation: Evidence from Ten Citation Corpora
By Moses Boudourides

---

## Overview

This repository contains all scripts, data outputs, and results for a comparative study of citation formation across ten topical bibliographic corpora spanning five broad scholarly domains (Science, Engineering, BioMed, Social Science, Humanities). A temporally-aware machine learning framework is applied to predict whether one scientific paper will cite another, using keyword-defined corpora of approximately 26,000–393,000 peer-reviewed articles each, retrieved from [Dimensions.ai](https://www.dimensions.ai) (1975–2024).

A central design principle is **strict temporal causality**: all features are computed using only information available prior to the citing paper's publication year, eliminating the future-topology data leakage that affects many prior citation prediction studies.

The study decomposes citation prediction into three explanatory components — **Semantic alignment (S)**, **Chronological distance (T)**, and **Citation-network prominence (N_prom)** — and reports their marginal contributions across all ten corpora.

The paper is available in `paper/manuscript4.pdf`.

---

## Datasets

Ten thematic bibliographic datasets are retrieved from Dimensions.ai using keyword searches in title and abstract, restricted to journal articles published 1975–2024.
*See `outputs/tables/table1_corpus_overview.csv` for corpus construction queries and abstract availability.*

**Table 1. Corpus construction overview**

| Domain | Dataset | Keywords | Year range | Papers | Total unique citations | Mean ref length | Papers with abstracts |
|---|---|---|---|---|---|---|---|
| Science | Protein Folding | "protein folding" | 1975–2024 | 82,428 | 1,161,663 | 47.1 | 79,455 (96.4%) |
| Science | CRISPR | "CRISPR" | 2002–2024 | 64,484 | 1,138,453 | 45.5 | 60,316 (93.5%) |
| Engineering | Additive Manufacturing | "additive manufacturing" | 1975–2024 | 58,541 | 816,148 | 40.0 | 56,901 (97.2%) |
| Engineering | Corrosion Protection | "corrosion protection" | 1975–2024 | 26,486 | 323,246 | 29.3 | 25,432 (96.0%) |
| BioMed | Neuroblastoma | "neuroblastoma" | 1975–2024 | 57,319 | 775,233 | 28.7 | 51,616 (90.1%) |
| BioMed | Osteosarcoma | "osteosarcoma" OR "bone sarcoma" | 1975–2024 | 61,187 | 649,724 | 24.3 | 54,528 (89.1%) |
| Social Science | Income Inequality | "income inequality" | 1975–2024 | 40,279 | 487,905 | 26.1 | 38,666 (96.0%) |
| Social Science | Organizational Behavior | "organizational behavior" | 1975–2024 | 75,622 | 856,527 | 29.0 | 68,985 (91.2%) |
| Humanities | Film Studies | "film studies" | 1975–2024 | 393,342 | 3,866,640 | 26.0 | 387,458 (98.5%) |
| Humanities | Memory Studies | "memory studies" | 1975–2024 | 347,195 | 4,047,258 | 35.9 | 343,757 (99.0%) |

**Table 2. Citation graph statistics**

| Domain | Dataset | Papers | Papers in internal citations | Internal citations | Density | GCC size |
|---|---|---|---|---|---|---|
| Science | Protein Folding | 82,428 | 73,433 (89.1%) | 889,384 | 2.62e-04 | 72,970 |
| Science | CRISPR | 64,484 | 49,224 (76.3%) | 753,711 | 3.63e-04 | 48,937 |
| Engineering | Additive Manufacturing | 58,541 | 48,937 (83.6%) | 575,853 | 3.36e-04 | 48,190 |
| Engineering | Corrosion Protection | 26,486 | 18,861 (71.2%) | 125,884 | 3.59e-04 | 18,112 |
| BioMed | Neuroblastoma | 57,319 | 44,290 (77.3%) | 326,670 | 1.99e-04 | 43,662 |
| BioMed | Osteosarcoma | 61,187 | 46,131 (75.4%) | 389,874 | 2.08e-04 | 45,365 |
| Social Science | Income Inequality | 40,279 | 26,974 (67.0%) | 136,433 | 1.68e-04 | 26,189 |
| Social Science | Organizational Behavior | 75,622 | 42,269 (55.9%) | 242,381 | 8.48e-05 | 39,904 |
| Humanities | Film Studies | 393,342 | 280,525 (71.3%) | 903,486 | 1.17e-05 | 264,167 |
| Humanities | Memory Studies | 347,195 | 247,507 (71.3%) | 1,820,834 | 3.02e-05 | 237,708 |

---

## Data Collection

All datasets are fetched using the scripts in `scripts/data_collection/` (e.g., `fetch_dimensions_crispr.py`, `fetch_6_new_datasets.py`), which iterate year by year (1975–2024) for each keyword and deduplicate across keywords within the same corpus.

**Requirements:** A valid [Dimensions.ai](https://www.dimensions.ai) API key.

---

## Pipeline and Scripts

The machine learning pipeline is executed sequentially through five stages located in `scripts/citation_analysis/`:

1. **`stage1_feature_engineering.py`**: Constructs temporally-capped directed acyclic citation graphs (DAGs) and samples positive and hard-negative pairs within strict ±3-year windows, computing structural features (N_prom) strictly prior to the citing year.
2. **`stage2_sbert_semantic.py` & `stage2b_directional_semantic.py`**: Computes semantic text embeddings using Sentence-BERT (`all-MiniLM-L6-v2`) on titles and abstracts, yielding the directional semantic alignment (S) feature.
3. **`stage3_ml_training.py`**: Trains classifiers (Logistic Regression, Linear SVM, Random Forest, Gradient Boosting) using 5-fold cross-validation, temporal holdout, and S-T-N_prom feature ablation.
4. **`stage4_comparative_figures.py`**: Generates comparative performance visualizations (AUC bar charts, temporal holdout stability, SHAP heatmaps).
5. **`stage5_multiplots.py`**: Generates multi-panel figures across all 10 datasets (rolling temporal AUC, ablation multiplots, SHAP multiplots).

### Outputs

All generated figures and tables are available in the `outputs/` directory:

- `outputs/figures/`: Multi-dataset visualizations (fig1–fig11)
- `outputs/tables/`: Dataset overviews and citation graph statistics
- `outputs/results/`: S-T-N_prom decomposition results per corpus (JSON) and summary (CSV)
- `outputs/reports/`: Interpretation analysis

---

## Requirements

```
Python 3.11
dimcli>=1.0
pandas>=2.0
pyarrow>=14.0
networkx>=3.0
numpy>=1.26
scikit-learn>=1.3
sentence-transformers>=2.7
torch>=2.1
matplotlib>=3.7
seaborn>=0.12
tabulate>=0.9
```

---

## Key Findings

The central contribution of this study is the S-T-N_prom decomposition of citation formation, applied consistently across ten keyword-defined scholarly corpora. The decomposition isolates the marginal predictive contribution of each component — directional semantic alignment (S), chronological distance (T), and citation-network prominence (N_prom) — under temporally rigorous evaluation conditions that eliminate future-topology data leakage.

**Chronological distance dominates across all corpora.** The T-only model achieves AUC scores ranging from 0.577 (Memory Studies) to 0.824 (Protein Folding), substantially outperforming both S-only and N_prom-only models in every corpus. The marginal contribution of T beyond S (ΔT) is positive in all ten corpora, confirming that chronological distance carries discriminative information that semantic alignment cannot provide. This finding is consistent with the Matthew effect and preferential attachment dynamics in citation networks: papers that have been in the literature longer have accumulated more citations and are therefore more likely to be cited again, independently of their semantic content.

**Semantic alignment adds little beyond chronological distance — and in most corpora actively reduces predictive power.** ΔS < 0 in all ten corpora, meaning that adding directional semantic alignment to a model that already includes T reduces rather than improves AUC. This result does not imply that semantic content is irrelevant to citation formation in general; rather, within keyword-defined corpora where the candidate-generation protocol already restricts both topical and chronological distance variation, semantic alignment contributes little additional predictive information beyond what chronological distance already captures. Notably, in four of ten corpora, cited papers are marginally *less* semantically aligned with the citing paper than non-cited candidates — a result unusual in the citation prediction literature, and most plausibly arising because keyword-defined corpora compress semantic variation among candidate papers, reducing the discriminative value of semantic alignment below chance.

**Citation-network prominence contributes negligibly once chronological distance is known.** ΔN|ST ≈ 0 in eight of ten corpora, indicating that temporal indegree and PageRank provide no independent predictive signal beyond T. This is itself a network-scientific finding: within temporally and topically constrained candidate pools, citation-network prominence is largely a proxy for chronological position rather than an independent structural signal. The two exceptions — Income Inequality and Organizational Behavior — exhibit non-trivial Cohen's d for N_prom (0.366 and 0.342 respectively), yet ΔN|ST remains near zero, revealing a collinearity paradox in which prominence separates cited from non-cited candidates in isolation but adds nothing once T is already in the model.

**When citations occur despite weak semantic alignment, they are temporally proximate.** In nine of ten corpora, the lowest-similarity decile of citation pairs is more temporally recent than the average positive pair (ΔT < 0), suggesting that chronological recency compensates for semantic distance. No consistent prominence pattern emerges among semantically surprising citations across corpora.

**The Income Inequality corpus is the notable exception.** It is the one corpus where the T curve is completely flat (d_T = 0.087) and N_prom is the primary separator (d_Nprom = 0.366), demonstrating that chronological distance dominance is not a universal law but a corpus-specific profile that varies with the citation culture of each scholarly community.

All conclusions are conditional on the keyword-defined corpus design and the ±3-year hard-negative candidate space employed in this study. The full S-T-N_prom decomposition results are in `outputs/results/stn_decomposition_summary.csv`.

---

## Copyright

© 2026 Moses Boudourides. All rights reserved.
This repository and its contents are made available for academic peer review purposes only. No part of this work may be reproduced, distributed, or used in any form without the express written permission of the authors.
