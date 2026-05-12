# Additional Academic References for JLegal-ChatBot

These 8 references were not included in the original project report but would
strengthen the literature review and methodology sections. The team should integrate
them manually into the relevant report chapters. Do NOT modify the report DOCX
automatically — all additions are manual editorial decisions.

---

## [R1] Dense Passage Retrieval (DPR Architecture)

**IEEE Citation:**
V. Karpukhin, B. Oguz, S. Min, P. Lewis, L. Wu, S. Edunov, D. Chen, and W. Yih,
"Dense Passage Retrieval for Open-Domain Question Answering,"
in *Proc. 2020 Conf. Empirical Methods Natural Lang. Process. (EMNLP)*,
Online, Nov. 2020, pp. 6769–6781,
doi: 10.18653/v1/2020.emnlp-main.550.

**Why it strengthens the report:**
JLegal's retriever uses the same core principle as DPR — embed both documents and
queries into a shared dense vector space and retrieve by inner product / cosine
similarity. Citing Karpukhin et al. provides the foundational justification for
choosing AraBERTv02 + mean pooling over sparse BM25 retrieval.

**Relevant chapter:** Methodology — Retrieval Module

---

## [R2] BM25 and Probabilistic Relevance Ranking

**IEEE Citation:**
S. Robertson and H. Zaragoza,
"The Probabilistic Relevance Framework: BM25 and Beyond,"
*Found. Trends Inf. Retr.*, vol. 3, no. 4, pp. 333–389, Jan. 2009,
doi: 10.1561/1500000019.

**Why it strengthens the report:**
The report's model-comparison section (camelbert-mix → AraBERTv02 → MARBERT)
would benefit from a baseline comparison with BM25. Citing Robertson & Zaragoza
gives context for why dense retrieval was chosen over classical TF-IDF/BM25 for
legal Arabic text despite BM25's continued competitiveness in English benchmarks.

**Relevant chapter:** Related Work — Information Retrieval Methods

---

## [R3] Retrieval-Augmented Generation (RAG) Foundation

**IEEE Citation:**
P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler,
M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela,
"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks,"
in *Proc. 34th Conf. Neural Inf. Process. Syst. (NeurIPS)*, Online, Dec. 2020,
vol. 33, pp. 9459–9474.

**Why it strengthens the report:**
Lewis et al. is the canonical RAG paper. Every RAG-based system should cite it to
establish that the architecture is academically grounded, not ad-hoc. JLegal
implements the same retrieve-then-generate pattern described here.

**Relevant chapter:** System Architecture — RAG Pipeline Overview

---

## [R4] RAGAS: Automated Evaluation of RAG Systems

**IEEE Citation:**
S. Es, J. James, L. Espinosa-Anke, and S. Schockaert,
"RAGAS: Automated Evaluation of Retrieval Augmented Generation,"
in *Proc. 18th Conf. Eur. Chapter Assoc. Comput. Linguist. (EACL)*,
Malta, Mar. 2024, pp. 150–163.

**Why it strengthens the report:**
The evaluation chapter (Chapter 5) uses Hit Rate @K and manual assessment. Citing
RAGAS provides an academic framework for discussing faithfulness and answer relevance
evaluation — and gives the committee a reference to show the team is aware of
state-of-the-art evaluation methodology even if the full RAGAS pipeline was not
implemented within the project scope.

**Relevant chapter:** Evaluation — Metrics and Methodology

---

## [R5] BGE Cross-Encoder Re-Ranking

**IEEE Citation:**
J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu,
"BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text
Embeddings Through Self-Knowledge Distillation,"
arXiv preprint arXiv:2309.07597, Sep. 2023. [Online].
Available: https://arxiv.org/abs/2309.07597

**Why it strengthens the report:**
The retriever returns top-K by cosine similarity. A cross-encoder re-ranker (such as
BGE-Reranker) would re-score the top-K chunks with full cross-attention, improving
precision. Citing this paper allows the Future Work section to propose re-ranking as
a concrete next step, grounded in published work showing accuracy improvements.

**Relevant chapter:** Future Work — Retrieval Quality Improvements

---

## [R6] Arabic NLP and AraBERT

**IEEE Citation:**
W. Antoun, F. Baly, and H. Hajj,
"AraBERT: Transformer-based Model for Arabic Language Understanding,"
in *Proc. 4th Workshop Open-Source Arabic Corpora Arabic Lang. Process. Tools*,
Marseille, France, May 2020, pp. 9–15.
[Online]. Available: https://aclanthology.org/2020.osact-1.2

**Why it strengthens the report:**
JLegal uses `aubmindlab/bert-base-arabertv02`, which is the v02 release of AraBERT.
Citing the original paper is essential to justify the model selection and provides
the accuracy figures (MSA downstream tasks) that explain why AraBERT outperformed
camelbert-mix in the domain of formal Arabic legal text.

**Relevant chapter:** Methodology — Embedding Model Selection

---

## [R7] Hallucination in Large Language Models

**IEEE Citation:**
Y. Zhang, Y. Li, L. Cui, D. Cai, L. Liu, T. Fu, X. Huang, E. Zhao, Y. Zhang,
Y. Chen, L. Wang, A. T. Luu, W. Bi, F. Shi, and S. Shi,
"Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models,"
arXiv preprint arXiv:2309.01219, Sep. 2023. [Online].
Available: https://arxiv.org/abs/2309.01219

**Why it strengthens the report:**
Anti-hallucination is a core selling point of JLegal ("the system refuses to answer
when no relevant articles are found"). This survey provides the academic vocabulary
and taxonomy of hallucination types (intrinsic vs extrinsic factuality errors), giving
the committee a reference framework for evaluating the claim.

**Relevant chapter:** Introduction — Motivation; Evaluation — Anti-Hallucination

---

## [R8] ReAct: Synergizing Reasoning and Acting in Language Models

**IEEE Citation:**
S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao,
"ReAct: Synergizing Reasoning and Acting in Language Models,"
in *Proc. 11th Int. Conf. Learn. Representations (ICLR)*,
Kigali, Rwanda, May 2023.
[Online]. Available: https://openreview.net/forum?id=WE_vluYUL-X

**Why it strengthens the report:**
JLegal uses a static retrieve-then-generate pipeline. ReAct represents the next
evolutionary step: an agent that interleaves reasoning steps with retrieval calls,
potentially querying multiple legal databases iteratively. Citing Yao et al. allows
the Future Work section to propose agentic retrieval as a concrete research direction
beyond the current project scope.

**Relevant chapter:** Future Work — Agentic Retrieval and Multi-Hop Reasoning

---

## Integration Notes

- All 8 references are peer-reviewed or widely cited arXiv preprints (1000+ citations each).
- [R1], [R3], [R6] are the highest priority — cite in the methodology chapter.
- [R4] (RAGAS) strengthens the evaluation chapter most directly.
- [R7] (hallucination survey) strengthens the motivation and anti-hallucination sections.
- [R2], [R5], [R8] are best placed in Related Work and Future Work respectively.
