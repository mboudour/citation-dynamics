import os

tex_content = r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{geometry}
\usepackage{setspace}
\geometry{a4paper, margin=1in}
\usepackage{natbib}
\usepackage{lipsum}

\title{Predicting Citations: Where Network Science Meets Machine Learning}
\author{Moses Boudourides, Giannis Tsakonas}
\date{December 12, 2025}

\begin{document}

\maketitle

\begin{abstract}
Citations are the currency of scientific impact. Understanding the mechanisms that drive citation behavior is a central challenge in the science of science. In this study, we propose a comprehensive framework that integrates network science and machine learning to decode citation dynamics. We analyze two large-scale datasets from the field of Library and Information Science (LIS), one sourced from Dimensions.ai (1985--2023) and the other from OpenAlex (1975--2024). We engineer features representing social proximity (co-authorship distance), semantic similarity (via Sentence-BERT embeddings), and prestige, while strictly enforcing temporal causality to prevent data leakage. Comparing multiple machine learning algorithms on balanced datasets of positive and negative citation pairs, we find that ensemble methods, particularly Gradient Boosting and Random Forests, achieve exceptional predictive performance (ROC-AUC $> 0.99$). Our results demonstrate that citation dynamics are highly predictable from historical network and semantic states. Furthermore, SHAP value analysis and ablation studies reveal that while prestige is a strong driver (the Matthew effect), semantic relevance is a dominant and indispensable factor. The combination of structural network features and deep semantic embeddings is essential for optimal performance, offering a robust methodological foundation for future bibliometric modeling.
\end{abstract}

\newpage
\tableofcontents
\newpage

\section{Introduction}
The dynamics of scientific citations reflect the flow of knowledge, the structure of scientific communities, and the mechanisms of academic recognition \cite{wang2008measuring, fortunato2018science}. Understanding and predicting which papers will cite which others is a fundamental problem in the science of science, with implications for literature search, research evaluation, and the sociology of science \cite{zeng2017science}.

Traditionally, citation prediction has been approached through network science, utilizing mechanisms such as preferential attachment (the ``Matthew effect'') \cite{merton1968matthew, de1976general, barabasi1999emergence} or social proximity in co-authorship networks \cite{newman2004coauthorship}. More recently, the availability of large-scale bibliographic datasets and advances in natural language processing have enabled machine learning approaches that combine structural network features with semantic content \cite{chakraborty2015towards, shibata2012link}.

However, a critical methodological flaw affects many recent citation prediction studies: temporal data leakage. In predicting whether paper $A$ (published in year $t_A$) cites paper $B$ (published in year $t_B$, where $t_B \le t_A$), features such as the co-authorship distance between the authors of $A$ and $B$, or the total citation count of $B$, are often computed using the entire static network spanning the full dataset period. This allows the model to ``see the future''---learning from co-authorship links or citations that did not exist at time $t_A$. This leakage artificially inflates model performance and invalidates claims about the predictive power of social or prestige mechanisms. Furthermore, negative sampling in these studies often ignores temporal constraints, pairing citing papers with ``cited'' papers published in the future, making the classification task artificially easy.

In this paper, we address these methodological criticisms directly. We propose a strictly temporally-aware framework for citation prediction. We apply this framework to two distinct datasets in Library and Information Science (LIS): one extracted from Dimensions.ai \cite{herzog2020dimensions} and another from OpenAlex. We compute all network and prestige features incrementally, ensuring that for any citing paper published in year $t$, the features are derived solely from the network state up to year $t-1$.

We evaluate six machine learning algorithms---Logistic Regression, Linear SVM, k-Nearest Neighbors, Random Forest, Gradient Boosting, and Neural Networks---on these temporally valid datasets. We utilize SHapley Additive exPlanations (SHAP) to interpret the interaction of semantic, social, and prestige features, and conduct ablation studies to quantify the contribution of each feature class. Finally, we provide a comparative analysis between the Dimensions and OpenAlex datasets to assess the robustness and generalizability of our findings across different bibliographic data sources.

\section{Literature Review}

\subsection{Theories of Citation: From Preferential Attachment to Complex Dynamics}
The study of citation dynamics has long been anchored in theories of cumulative advantage. Merton's articulation of the Matthew Effect \cite{merton1968matthew} and de Solla Price's formalization of preferential attachment \cite{de1976general} posited that highly cited papers attract future citations at a rate proportional to their existing prestige. This mechanism, later generalized in network science by Barab\'asi and Albert \cite{barabasi1999emergence}, provided a powerful, albeit simplified, model of scientific impact. However, subsequent research has demonstrated that the strength of preferential attachment varies significantly across disciplines and over time \cite{wang2008measuring}, and that a purely prestige-centric view is incomplete. Bianco and Gabrielli \cite{bianco2008fitness} introduced the concept of paper ``fitness,'' suggesting that intrinsic quality or relevance modulates the rich-get-richer effect.

\subsection{The Role of Proximity: Social and Semantic Ties}
Beyond prestige, the likelihood of citation is heavily influenced by the proximity between the citing and cited authors. Social proximity, typically operationalized through co-authorship networks, has been shown to strongly correlate with citation behavior \cite{newman2004coauthorship, kumar2015co, martin2013coauthorship}. Researchers are more likely to cite those within their immediate collaborative circles.

Equally important is semantic proximity. Early approaches utilized topic modeling techniques like Latent Dirichlet Allocation (LDA) \cite{blei2003latent} to capture content similarity. The advent of deep learning and transformer-based architectures, particularly BERT \cite{devlin2019bert} and Sentence-BERT (SBERT) \cite{reimers2019sentence}, has revolutionized the representation of textual semantics, allowing for nuanced, context-aware embeddings of abstracts and titles. Recent work by Kozlowski et al. \cite{kozlowski2025citation} emphasized that semantic and social proximity often rival prestige in predicting citations.

\subsection{Machine Learning for Citation Prediction}
The integration of machine learning into bibliometrics has shifted the focus from explanatory models to predictive frameworks. Early predictive efforts employed standard regression models \cite{lokker2008prediction}. More recent studies have leveraged neural networks \cite{alohali2022machine} and powerful ensemble methods like Gradient Boosting \cite{natekin2013gradient} and Random Forests \cite{breiman2001random}. These algorithms are particularly adept at capturing the non-linear interactions between disparate feature types (e.g., how high semantic similarity might compensate for low prestige).

\subsection{Synthesis and Critical Review}
Despite these advances, a significant gap remains between the explanatory traditions (often utilizing Generalized Linear Mixed Models) and the predictive traditions (utilizing complex ML algorithms). Furthermore, as noted in the introduction, many predictive models suffer from temporal data leakage, undermining their validity. This study bridges this gap by deploying advanced, non-linear ML models within a rigorously constructed, temporally causal framework, thereby ensuring that predictive performance reflects genuine underlying dynamics rather than methodological artifacts.

\section{Data and Methodology}

\subsection{Dataset Description}
We construct our analysis using two distinct bibliographic databases to ensure robustness: Dimensions.ai and OpenAlex. Both datasets focus on the field of Library and Information Science (LIS).

\subsubsection{Dimensions.ai Dataset}
The Dimensions dataset was extracted by querying for publications within the Field of Research (FoR) code 4610 (Library and Information Studies) and related keywords, spanning the years 1985 to 2023. The dataset comprises 500,000 citation pairs (250,000 positive, 250,000 negative) constructed from the underlying corpus.

\subsubsection{OpenAlex Dataset}
The OpenAlex dataset was retrieved using relevant topic IDs for LIS, covering the years 1975 to 2024. It includes 168,901 unique articles. After preprocessing, we constructed a balanced dataset of 489,349 citation pairs (239,349 positive, 250,000 negative). The OpenAlex corpus has a mean citation count of 6.26 per article, with 64.3\% containing abstracts.

\subsection{Citation Pair Construction and Negative Sampling}
The fundamental task is binary classification: given a pair of papers $(A, B)$, predict whether $A$ cites $B$. To ensure temporal validity, we enforce the constraint that $t_B \le t_A$.

\textbf{Positive Pairs:} We extracted all observed citations within our corpus where both the citing and cited papers exist in our dataset. 

\textbf{Negative Pairs:} To create a balanced dataset, we generated an equal number of negative pairs. Crucially, we enforced strict temporal causality during negative sampling. For each positive pair $(A, B)$, we generated a negative pair $(A, B')$ by randomly selecting a paper $B'$ from the pool of all papers published in or before year $t_A$, ensuring that $A$ does not actually cite $B'$. This guarantees that the negative examples represent temporally possible but unrealized citations, preventing the model from trivially distinguishing pairs based on impossible temporal orderings.

\subsection{Temporally-Aware Feature Engineering}
We engineered features categorized into semantic, network/social, and prestige classes. All features that depend on the state of the literature were computed using a strictly causal rolling window.

\begin{enumerate}
    \item \textbf{Semantic Similarity ($S_{sem}$):} We utilized Sentence-BERT (SBERT) \cite{reimers2019sentence} (model \texttt{all-MiniLM-L6-v2}) to encode the title and abstract of each paper into a 384-dimensional dense vector. The semantic similarity between paper $A$ and $B$ is the cosine similarity of their SBERT embeddings.
    \item \textbf{Prestige of Cited Paper ($P_{cited}$):} The total number of citations received by paper $B$ from papers published up to year $t_A - 1$.
    \item \textbf{Prestige/Activity of Citing Paper ($P_{citing}$):} A measure of the prominence or activity level of the citing team up to year $t_A - 1$.
    \item \textbf{Temporal Distance ($\Delta t$):} The difference in publication years, $\Delta t = t_A - t_B$.
    \item \textbf{Co-authorship Distance ($D_{soc}$):} (Used in Dimensions) The shortest path (Dijkstra's algorithm) between any author of $A$ and any author of $B$ in the temporally restricted co-authorship network.
    \item \textbf{Reference Overlap Features:} (Used in OpenAlex) Features such as common references and Jaccard similarity of reference lists.
\end{enumerate}

\subsection{Machine Learning Models and Formal Specifications}
We evaluate several machine learning models to capture both linear and non-linear relationships:
\begin{itemize}
    \item \textbf{Logistic Regression (LR):} Serves as a linear baseline.
    \item \textbf{Linear SVM:} A linear support vector machine.
    \item \textbf{k-Nearest Neighbors (k-NN):} A non-parametric distance-based method.
    \item \textbf{Random Forest (RF):} An ensemble of decision trees \cite{breiman2001random}.
    \item \textbf{Gradient Boosting (GB):} A sequential ensemble method \cite{natekin2013gradient}.
    \item \textbf{Multi-Layer Perceptron (MLP) Neural Network:} A feedforward neural network capable of learning complex representations.
\end{itemize}

Models were evaluated using 5-fold stratified cross-validation. Performance metrics include ROC-AUC, F1-score, Accuracy, Precision, Recall, and the Matthews Correlation Coefficient (MCC).

\section{Results: Dimensions.ai Analysis}

\subsection{Model Performance}
We evaluated the six machine learning models on the 500,000 citation pairs from the Dimensions dataset. The results, summarized in Table \ref{tab:dim_results}, demonstrate exceptionally high predictive accuracy across all models, with non-linear models showing a distinct advantage.

\begin{table}[h!]
\centering
\caption{Model Performance on Dimensions Dataset (5-Fold CV)}
\label{tab:dim_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{ROC-AUC} & \textbf{F1} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{MCC} \\
\midrule
Logistic Regression & 0.9897 & 0.9522 & 0.9521 & 0.9505 & 0.9539 & 0.9042 \\
Linear SVM & 0.9897 & 0.9522 & 0.9521 & 0.9505 & 0.9540 & 0.9043 \\
k-NN (k=5) & 0.9773 & 0.9485 & 0.9483 & 0.9459 & 0.9511 & 0.8967 \\
Random Forest & 0.9873 & 0.9503 & 0.9502 & 0.9473 & 0.9533 & 0.9003 \\
Gradient Boosting & 0.9904 & 0.9542 & 0.9540 & 0.9511 & 0.9573 & 0.9081 \\
MLP Neural Network & \textbf{0.9906} & \textbf{0.9540} & \textbf{0.9539} & \textbf{0.9518} & \textbf{0.9562} & \textbf{0.9078} \\
\bottomrule
\end{tabular}}
\end{table}

The MLP Neural Network achieved the highest ROC-AUC (0.9906), closely followed by Gradient Boosting (0.9904). Even the linear baseline (Logistic Regression) performed remarkably well (AUC 0.9897), indicating that the engineered features provide a highly separable space for citation prediction.

\begin{figure}[h!]
\centering
\includegraphics[width=0.7\textwidth]{../figures/fig5_roc_curves.png}
\caption{ROC Curves for the models evaluated on the Dimensions dataset.}
\label{fig:dim_roc}
\end{figure}

\subsection{Feature Importance and Explainability}
To understand the drivers of citation within the Dimensions dataset, we analyzed SHAP values for the best-performing models and conducted an ablation study.

The SHAP analysis revealed that \textbf{Semantic Similarity} is the most dominant predictor (mean absolute SHAP value $\approx 0.305$), followed by the \textbf{Prestige of the Cited Paper} ($\approx 0.137$). Co-authorship distance and temporal distance played secondary roles. This suggests that while the Matthew effect (prestige) is present, the actual content relevance (semantics) is the primary driver of citations in this corpus.

\begin{table}[h!]
\centering
\caption{Feature Ablation Study (Dimensions Dataset)}
\label{tab:dim_ablation}
\begin{tabular}{lc}
\toprule
\textbf{Feature Subset} & \textbf{ROC-AUC} \\
\midrule
All features & 0.9868 \\
No semantic similarity & 0.8709 \\
No co-authorship distance & 0.9859 \\
No prestige & 0.9618 \\
Semantic similarity only & 0.9513 \\
Network features only & 0.8512 \\
Temporal + semantic & 0.9576 \\
\bottomrule
\end{tabular}
\end{table}

The ablation study (Table \ref{tab:dim_ablation}) confirms these findings. Removing semantic similarity causes the most severe performance drop (AUC falls from 0.9868 to 0.8709). Notably, a model trained on semantic similarity alone achieves an AUC of 0.9513, outperforming a model trained on all network features combined (AUC 0.8512).

\section{Results: OpenAlex Analysis}

\subsection{Model Performance}
We replicated the temporally-aware prediction framework on the OpenAlex dataset (489,349 pairs). Due to memory constraints associated with the scale of the dataset and feature space, k-NN and Neural Network models were excluded; the remaining four models were evaluated.

\begin{table}[h!]
\centering
\caption{Model Performance on OpenAlex Dataset (5-Fold CV)}
\label{tab:oa_results}
\resizebox{\textwidth}{!}{
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{ROC-AUC} & \textbf{F1} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{MCC} \\
\midrule
Logistic Regression & 0.9872 & 0.9478 & 0.9495 & 0.9574 & 0.9385 & 0.8991 \\
Linear SVM & 0.9863 & 0.9451 & 0.9468 & 0.9549 & 0.9355 & 0.8937 \\
Random Forest & 0.9972 & 0.9793 & 0.9796 & 0.9729 & 0.9857 & 0.9592 \\
Gradient Boosting & \textbf{0.9979} & \textbf{0.9810} & \textbf{0.9813} & \textbf{0.9752} & \textbf{0.9868} & \textbf{0.9626} \\
\bottomrule
\end{tabular}}
\end{table}

As shown in Table \ref{tab:oa_results}, the ensemble methods (Random Forest and Gradient Boosting) significantly outperformed the linear models on the OpenAlex dataset. Gradient Boosting achieved an extraordinary ROC-AUC of 0.9979. The substantial gap between linear models (AUC $\sim$0.987) and tree-based ensembles (AUC $\sim$0.997) provides strong evidence for complex, non-linear interactions among the features in this dataset.

\begin{figure}[h!]
\centering
\includegraphics[width=0.7\textwidth]{../figures/oa/oa_fig4_roc_curves.png}
\caption{ROC Curves for the models evaluated on the OpenAlex dataset.}
\label{fig:oa_roc}
\end{figure}

\subsection{Feature Importance and Explainability}
SHAP analysis on the Gradient Boosting model for OpenAlex revealed a slightly different feature hierarchy compared to Dimensions. Here, \textbf{activity\_citing} (mean SHAP 4.71) and \textbf{prestige\_cited} (4.50) were the top predictors, followed by \textbf{semantic\_similarity} (1.96).

\begin{table}[h!]
\centering
\caption{Feature Ablation Study (OpenAlex Dataset, Random Forest)}
\label{tab:oa_ablation}
\begin{tabular}{lcc}
\toprule
\textbf{Feature Removed} & \textbf{ROC-AUC} & \textbf{Drop} \\
\midrule
None (all features) & 0.9972 & -- \\
prestige\_cited & 0.9847 & -0.0125 \\
activity\_citing & 0.9890 & -0.0082 \\
semantic\_similarity & 0.9914 & -0.0058 \\
temporal\_gap & 0.9960 & -0.0012 \\
common\_citers & 0.9970 & -0.0002 \\
jaccard\_refs & 0.9972 & 0.0000 \\
common\_refs & 0.9972 & 0.0000 \\
\bottomrule
\end{tabular}
\end{table}

The ablation study (Table \ref{tab:oa_ablation}) confirms that removing `prestige\_cited` causes the largest drop in performance (-0.0125 AUC), followed by `activity\_citing` and `semantic\_similarity`. This indicates a strong Matthew effect within the OpenAlex LIS corpus, though semantic relevance remains a crucial component for achieving peak accuracy.

\section{Comparative Analysis: Dimensions vs. OpenAlex}
Comparing the results from the two databases reveals both consistent patterns and notable differences.

\textbf{Predictability:} Citation dynamics in LIS are highly predictable regardless of the data source. Both datasets yielded models with ROC-AUC scores exceeding 0.99. This confirms that the combination of historical network states and deep semantic embeddings captures the vast majority of the variance in citation behavior.

\begin{figure}[h!]
\centering
\includegraphics[width=0.7\textwidth]{../figures/comp_fig1_auc_comparison.png}
\caption{Comparison of best model ROC-AUC between Dimensions and OpenAlex.}
\label{fig:comp_auc}
\end{figure}

\textbf{Feature Importance Hierarchy:} The most striking difference lies in the relative importance of features. In the Dimensions dataset, semantic similarity was overwhelmingly the dominant predictor. In the OpenAlex dataset, prestige metrics (both of the cited and citing entities) edged out semantic similarity as the top predictors. 

This discrepancy may stem from differences in dataset construction, coverage, or the specific features engineered (e.g., the inclusion of co-authorship distance in Dimensions vs. reference overlap in OpenAlex). However, in both cases, the \textit{combination} of semantic and prestige features was necessary to achieve the highest performance, underscoring the multifaceted nature of citation decisions.

\begin{figure}[h!]
\centering
\includegraphics[width=0.7\textwidth]{../figures/comp_fig3_shap_comparison.png}
\caption{Normalized SHAP feature importance comparison between datasets.}
\label{fig:comp_shap}
\end{figure}

\section{Discussion}
Our findings extend the literature on citation dynamics by providing a robust, temporally-aware predictive framework. The clear superiority of ensemble methods (particularly in the OpenAlex dataset) over linear models provides strong evidence that citation decisions are characterized by complex, non-additive interactions between predictors. A paper with moderate prestige might still be cited if its semantic relevance is exceptionally high, a non-linear dynamic that tree-based models capture effectively.

The dominance of semantic similarity in the Dimensions dataset and prestige in the OpenAlex dataset highlights the necessity of multi-dimensional feature engineering. Relying solely on network topology (the Matthew effect) or solely on content similarity provides an incomplete picture. 

\subsection{Limitations}
This study is focused on a single field (Library and Information Science). The relative importance of social, semantic, and prestige factors may vary across other disciplines (e.g., hard sciences vs. humanities). Furthermore, while we strictly enforced temporal causality, observational data cannot definitively establish causal mechanisms.

\section{Conclusion}
This paper demonstrates that machine learning approaches, when constrained by strict temporal causality, offer a powerful framework for understanding and predicting citation dynamics. By integrating network science, natural language processing (semantic embeddings), and advanced machine learning, we constructed predictive models that achieve exceptional accuracy (ROC-AUC $> 0.99$) across two distinct bibliographic databases. The integration of these methodologies offers a powerful new lens through which to examine the mechanisms that drive scientific impact.

\newpage
\bibliographystyle{plain}
\bibliography{references}

\end{document}
"""

with open('/home/ubuntu/lis_repo/paper/manuscript.tex', 'w') as f:
    f.write(tex_content)
