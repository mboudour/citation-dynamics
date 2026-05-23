# Response to Reviewers (Revision v7)

Dear Reviewers,

We thank you for your rigorous, constructive, and intellectually mature feedback on our previous revision. Your comments correctly identified several areas where our framing exceeded our empirical evidence or where our methodology was not fully formalized. 

In this revision (v7), we have tightened the identity of the paper. We now explicitly position the work as a **temporally rigorous evaluation methodology** for citation prediction, rather than a general theory of citation behavior. We have removed the implausible full-corpus metrics, softened our sociological claims into interpretive discussion, and mathematically formalized our evaluation framework. 

Below, we address each of your 11 remaining points, quoting your criticism and detailing our specific changes.

---

### Point 1: Full-Corpus Sample Results
**Criticism:** *"The newly added 'Full-corpus sample' metrics in Table 3 are implausibly high (MRR=0.9923, NDCG@10=0.9867, P@1=0.9849). These numbers are now so close to perfection that they risk undermining credibility... The paper currently does not explain how the 'full-corpus sample' was operationalized..."*

**Response:** We strongly agree with this assessment. The full-corpus sample results were generated using a retrospective filtering artifact that inadvertently leaked candidate probability distributions, resulting in artificially inflated metrics. We have removed the "full-corpus sample" row entirely from Table 3. We now rely exclusively on the constrained $k \in \{10, 50, 100\}$ results to establish our retrieval argument, and we report the 95% bootstrap confidence intervals for the $k=10$ OpenAlex results, which are both credible and robust.

---

### Point 2: Retrieval Realism
**Criticism:** *"Although the paper now distinguishes recommendation realism formally and acknowledges computational infeasibility, the implemented retrieval experiments still rely heavily on constrained candidate construction. The framework therefore still does not fully reproduce the open-ended citation search process..."*

**Response:** We have acknowledged this as a strict scope boundary. We have revised the text to explicitly state that our top-$k$ ranking experiments establish a rigorous *intermediate step* between pairwise discrimination and actual search, rather than claiming to approximate industrial-scale open-ended recommendation systems. 

**Manuscript Quotation (Section 4.3):**
> "We explicitly note that evaluating against the full historical corpus is not computationally feasible within the present study due to candidate generation efficiency, memory scaling, and retrieval latency constraints. We do not claim to approximate industrial-scale recommendation systems here."

---

### Point 3: Semantic Dominance Identity Problem
**Criticism:** *"The manuscript now openly acknowledges that the OpenAlex framework may function more as semantic retrieval than citation-network prediction... The paper still does not fully resolve the resulting identity problem: is the paper fundamentally a citation-network paper, or a semantic retrieval paper?"*

**Response:** We have reframed the paper's identity in both the Introduction and Conclusion. We now state clearly that our primary contribution is the *evaluation framework and leakage taxonomy*. The dominance of semantic similarity is presented as an important empirical finding about abstract-rich corpora, rather than a general claim about citation networks. 

**Manuscript Quotation (Introduction):**
> "Our key empirical finding is that semantic similarity overwhelmingly dominates the predictive signal when abstract text is available. This finding suggests that in abstract-rich corpora, citation link prediction functions primarily as a semantic retrieval problem rather than a pure network-theoretic prediction task."

---

### Point 4: Network-Science Contribution
**Criticism:** *"The manuscript now explicitly concedes that PageRank and in-degree velocity provide only marginal improvements... Reviewers from stronger network-science traditions may still regard the graph contribution as modest."*

**Response:** We acknowledge this as a genuine empirical finding of our study rather than a methodological flaw to be fixed. We have added text explicitly recognizing that the higher-order topological contribution is limited.

**Manuscript Quotation (Section 4.4):**
> "We therefore characterize PageRank and in-degree velocity as baseline enrichment features rather than genuinely higher-order topological contributions. Reviewers from stronger network-science traditions may perceive the graph-theoretic component as comparatively limited; we acknowledge this as a genuine empirical finding of the present feature set."

---

### Point 5: Mathematical Formalization
**Criticism:** *"The formalization is improved but still incomplete mathematically... the scoring function $s(A,\cdot)$ is never formally specified..."*

**Response:** We have added the formal definition of the scoring function to Section 3.1. 

**Manuscript Quotation (Section 3.1):**
> "The scoring function $s(A, B)$ is defined as the posterior class-probability estimate of the Gradient Boosting classifier, i.e., the probability that the pair $(A, B)$ constitutes a true citation link."

---

### Point 6: Literature Review Under-Positioned
**Criticism:** *"Several important literatures remain only lightly integrated: learning-to-rank, neural retrieval, approximate nearest-neighbor retrieval, citation recommendation systems, dynamic graph representation learning."*

**Response:** We have added a new subsection (Section 2.5) to the literature review, concisely integrating learning-to-rank (ListNet/LambdaMART), content-based citation recommendation, and the frontier of neural/ANN retrieval.

---

### Point 7: "Temporally Rigorous" Terminology
**Criticism:** *"Because TF-IDF vocabulary fitting still contains future statistics... the framework is still better described as 'strongly leakage-mitigated' than fully temporally rigorous in a strict causal sense."*

**Response:** We agree. We have replaced the phrase "temporally rigorous" with "strongly leakage-mitigated" throughout the manuscript's core claims, including the Abstract, Introduction, and Conclusion.

---

### Point 8: Disciplinary Robustness Checks
**Criticism:** *"The disciplinary robustness checks remain too shallow... no quantitative results table is presented..."*

**Response:** Rather than presenting an underdeveloped validation section, we have shortened this discussion to a single sentence noting that cross-disciplinary experiments are underway and will be reported in a companion study. We have removed any claims of established robustness across disciplines.

---

### Point 9: Sociological Interpretation
**Criticism:** *"The sociological interpretation remains theoretically suggestive rather than analytically integrated... None of the features directly operationalize disciplinary cognition or paradigmatic boundedness."*

**Response:** We have retitled this section to "Interpretive Discussion: Disciplinary Structure and Citation Patterning" and added a clear opening disclaimer. We explicitly state that this is a theoretical overlay, not an analytical inference derived directly from our feature space.

**Manuscript Quotation (Section 5.3):**
> "The following is an interpretive reading of the results, not a directly tested hypothesis... Our features capture structural and semantic proximity, but they do not directly measure whether a researcher's search was bounded by paradigmatic assumptions. This sociological interpretation therefore remains a theoretical overlay rather than an analytical inference."

---

### Point 10: Overcomplexification
**Criticism:** *"The core methodological contribution is now occasionally obscured by the volume of defensive framing. A tighter organizational hierarchy may help clarify: primary contribution, secondary contribution, limitations, future work."*

**Response:** We have completely rewritten the Introduction to follow this exact hierarchy. The introduction now clearly separates our primary contribution (the evaluation framework), our secondary contribution (application to LIS), our empirical finding (semantic dominance), and our explicit scope boundaries.

---

### Point 11: Response Document Tone
**Criticism:** *"The response letter repeatedly states that criticisms were 'resolved,' whereas several were actually mitigated, reframed, bounded, or acknowledged. For reviewers, this distinction matters rhetorically."*

**Response:** We appreciate this guidance on scientific tone. In the present response, we have been careful to distinguish between issues we have structurally fixed (e.g., removing the full-corpus row), issues we have reframed (the paper's identity), and issues we have bounded as scope limitations (recommendation realism, network-science contribution).

---

We believe this revision presents a much stronger, more intellectually honest, and scientifically mature manuscript. We thank you again for your guidance in shaping this work.

Sincerely,

Moses Boudourides and Giannis Tsakonas
