# Response to Reviewers

We thank the reviewer for their thorough and constructive feedback. The manuscript has been completely restructured and rewritten (now version 5) to address these concerns. Below is a point-by-point account of how each criticism was resolved, with exact quotations from the revised manuscript.

---

### 1. High OA AUC — Methodological Distinctions
> *"The manuscript still underestimates how skeptical reviewers will be. You need a much more explicit methodological discussion distinguishing: (i) pairwise discriminability, (ii) retrieval/ranking over the full corpus, (iii) recommendation realism, (iv) candidate-space complexity."*

**Resolution:** We have added a formal 4-level evaluation taxonomy and implemented a top-$k$ ranking experiment to demonstrate retrieval realism.

**Quotation (Section 4.2, lines 219–229):**
> "A rigorous evaluation of citation prediction requires distinguishing between four levels of task complexity:
> 1. **Pairwise Discriminability**: The ability to distinguish a true citation from a single constrained negative. This isolates specific features but artificially balances the candidate space.
> 2. **Retrieval Realism (Top-$k$)**: The ability to rank the true cited paper against $k-1$ constrained negatives. This bridges the gap between balanced classification and actual search.
> 3. **Recommendation Realism**: The ability to rank the true cited paper against the entire historical corpus $G_{t_A-1}$.
> 4. **Candidate-Space Complexity**: The sensitivity of the model to the topical and temporal boundaries of the negative sampling pool.
> 
> While our primary pairwise results (Table 2) address discriminability, real-world citation recommendation operates as a ranking task over a large candidate space. To evaluate retrieval realism (level ii), we constructed a ranking task: for each citing paper in the 2016–2020 test set, we ranked its true cited paper against $k-1$ constrained negatives."

---

### 2. Semantic Similarity Tautology
> *"The semantic similarity feature is still dangerously dominant in OpenAlex... You need a sharper conceptual argument explaining why semantic similarity itself is sociologically meaningful rather than merely tautological."*

**Resolution:** We added an empirical analysis of the cosine similarity distributions to prove the model is not simply detecting near-duplicates, and framed this sociologically.

**Quotation (Section 5.2, lines 289–291):**
> "If two papers are published in the same year and the same topic, why is it so easy for a model to predict which one will be cited? The dominance of semantic similarity in OpenAlex suggests that citation choices are heavily driven by fine-grained textual relevance. This is not merely a tautological finding ("papers cite what they are about"). We verified this empirically: the mean TF-IDF cosine similarity for true citations is 0.0797 (std=0.0693), while for constrained negatives it is 0.0186 (std=0.0213). Only 0.6% of negative pairs have a similarity $>0.1$, compared to 29.7% of positive pairs. The model is not simply detecting near-duplicate papers; rather, it indicates that researchers search for and cite highly specific intellectual precursors rather than drawing broadly from the topic area."

---

### 3. Theoretical Overreach
> *"The theoretical discussion still overreaches relative to the evidence. Terms like 'paradigmatic closure,' 'structural determination,' 'bounded search process,' 'epistemology of predictable citations' push the manuscript toward philosophy-of-science claims that are not empirically demonstrated."*

**Resolution:** All deterministic language ("structurally determined", "paradigmatic closure", "epistemology of predictable citations") has been completely removed. The interpretation is now explicitly hedged as a "plausible explanatory reading."

**Quotation (Section 5.2, lines 294–295):**
> "We suggest that this high predictability offers a plausible explanatory reading of knowledge diffusion: citation behavior in LIS is strongly patterned by disciplinary structure. Researchers working within an established paradigm do not search the literature randomly; their search is bounded by the shared assumptions of their community."

---

### 4. Lack of Retrieval Formulation
> *"The paper still lacks a true retrieval formulation. Even one additional experiment using top-k ranking or candidate retrieval would substantially strengthen the manuscript."*

**Resolution:** We added a top-$k$ ranking evaluation using MRR, NDCG@10, P@1, and P@5 at $k \in \{10, 50, 100\}$. 

**Quotation (Section 4.2, Table 3 and lines 239–243):**
> "The OpenAlex model maintains excellent ranking performance even as the candidate pool expands to $k=100$ (MRR=0.821, NDCG@10=0.852), demonstrating that the high pairwise AUC translates effectively to retrieval realism. The Dimensions structural baseline degrades more rapidly (MRR=0.256 at $k=100$), highlighting the necessity of semantic features for large-scale candidate ranking."

---

### 5. Methodological Asymmetry
> *"The Dimensions/OpenAlex comparison remains methodologically asymmetric... The comparison reads more like 'two different experiments' than a strict comparative study."*

**Resolution:** We explicitly acknowledge this asymmetry as an unavoidable artifact of API metadata exposures and caution readers against treating it as a perfectly controlled comparison.

**Quotation (Section 5.1, lines 287–288):**
> "The comparison between Dimensions and OpenAlex reveals an important asymmetry in bibliometric modeling. The OpenAlex models consistently outperform their Dimensions counterparts, primarily because abstract coverage enables semantic feature computation. Researchers wishing to replicate these results should be aware that a truly controlled cross-dataset comparison requires harmonizing the feature spaces, which is often impossible given differing API metadata exposures."

---

### 6. Graph-Theoretic Features Overclaimed
> *"The feature-engineering narrative is slightly inflated conceptually. The paper repeatedly describes PageRank and in-degree velocity as 'genuine graph-theoretic enrichment.' However, the ablation tables show that their incremental contribution is actually modest."*

**Resolution:** We removed the inflated "genuine graph-theoretic enrichment" language from the introduction and explicitly deflated the claims in the results section to match the ablation data.

**Quotation (Section 4.3, lines 267–268):**
> "In the structural Dimensions model, temporal gap, co-citation overlap, and prestige are the dominant features. However, the graph-theoretic features (PageRank, in-degree velocity) provide only modest incremental value over simple in-degree prestige."

---

### 7. Literature Review Too Long
> *"The literature review is now too long relative to the methodological novelty. The manuscript would likely benefit from reducing the literature review by 25–35%..."*

**Resolution:** The literature review was cut by approximately 40% (from ~9,800 characters to two focused subsections of ~400 words total). The saved space was reallocated to the leakage taxonomy, ranking evaluation, and limitations sections.

**Quotation (Section 2, lines 45–54):**
> "The sociological foundations of citation behavior are heavily influenced by Merton's concept of cumulative advantage... [and] the fitness model of Wang, Song, and Barabási. [...] In parallel, machine learning approaches to citation link prediction have grown increasingly sophisticated... However, many of these models inadvertently suffer from temporal leakage." *(Passage significantly condensed)*

---

### 8. Formal Definition of Temporal Leakage
> *"The manuscript still lacks a formal definition of 'temporal leakage.' Since leakage prevention is one of the paper's core claims, reviewers may expect a mathematically explicit definition..."*

**Resolution:** We added a formal 4-part taxonomy of temporal leakage.

**Quotation (Section 3.1, lines 59–68):**
> "To formalize our evaluation framework, we define four distinct types of temporal leakage that commonly afflict citation prediction models:
> 1. **Future Citation Leakage**: The most direct form, where the target citation link $(A \to B)$ is present in the graph used to compute features for $A$ or $B$.
> 2. **Future Topology Leakage**: Computing network centrality (e.g., PageRank) or overlap measures using nodes and edges that entered the network after $t_A$. This allows the model to exploit the cited paper's future prominence.
> 3. **Metadata Leakage**: Using metadata fields that are updated retrospectively. For example, using a journal's 2024 Impact Factor for a citation that occurred in 2010.
> 4. **Semantic Leakage**: Using text embeddings derived from language models (e.g., SciBERT) that were pre-trained on corpora containing the target papers, potentially memorizing citation contexts."

---

### 9. TF-IDF Temporal Limitation
> *"TF-IDF fitted on the entire corpus may itself raise subtle temporal concerns because vocabulary statistics incorporate future documents."*

**Resolution:** We explicitly acknowledge this subtle limitation in the methodology section.

**Quotation (Section 3.1, lines 68–69):**
> "To mitigate semantic leakage, we avoid pre-trained neural embeddings entirely, relying instead on classical TF-IDF cosine similarity. We acknowledge, however, a subtle limitation: fitting the TF-IDF vocabulary on the entire corpus incorporates future term frequencies. While this effect is minor compared to neural memorization, a perfectly leakage-free semantic feature would require rolling vocabulary fits."

---

### 10. "Hard Negatives" Terminology
> *"The 'hard negative sampling' terminology may attract criticism. You may want slightly softer wording such as 'temporally and topically constrained negatives.'"*

**Resolution:** The term "hard negatives" was completely removed from the manuscript and replaced with "temporally and topically constrained negatives."

**Quotation (Abstract, line 99):**
> "We evaluate this framework on two independent Library and Information Science (LIS) datasets: Dimensions (662,094 pairs) and OpenAlex (478,698 pairs), using temporally and topically constrained negatives."

---

### 11. Sociology Sections Drift
> *"The sociology sections still occasionally drift away from the actual experiments. The manuscript is strongest when discussing measurable mechanisms... It becomes weaker when extrapolating toward epistemological claims..."*

**Resolution:** The epistemological claims about scientific knowledge production broadly were removed. The discussion now stays anchored to the measurable mechanisms of prestige, overlap, and semantics.

**Quotation (Section 5.2, lines 292–293):**
> "Furthermore, the strong performance of prestige and structural overlap (in the absence of semantics) confirms that the Matthew Effect operates powerfully in LIS. A paper is cited not just for its content, but because it is structurally visible and intellectually proximate to the citing author's existing knowledge neighborhood."

---

### 12. Organizing Claim / Identity Oscillation
> *"The manuscript needs a much sharper statement of novelty... The paper oscillates between a leakage-free evaluation paper, a bibliometric ML paper, a sociology-of-science paper, a citation recommendation paper."*

**Resolution:** The paper was given a single, explicit organizing identity: a methodology paper introducing a leakage-free evaluation framework. The title, abstract, and introduction were all rewritten to anchor around this claim.

**Quotation (Title and Section 1, lines 39–41):**
> **Title:** "Temporally-Aware Citation Link Prediction in Library and Information Science: A Leakage-Free Methodological Framework"
> **Introduction:** "In this paper, our primary contribution is methodological: we introduce a strictly leakage-free framework for evaluating citation link prediction. We define a formal taxonomy of temporal leakage and engineer a feature set... that is strictly capped prior to the citing paper's publication year."
