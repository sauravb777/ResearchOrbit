# Research Report: What are recent approaches to retrieval augmented generation for domain-specific question answering?

## Evidence Base
**Strength: MODERATE** (3 sources retrieved, but most evidence comes from abstracts rather than full text.) Read the sections below with this in mind — a report grounded in few sources should be treated as a starting point, not a settled answer.

## Executive Answer

- We propose a retrieval-augmented generation framework combining a dense retriever with a fine-tuned generator. [E1]
- Our method improves exact match accuracy by 8.4 points over the baseline on a domain-specific benchmark dataset. [E2]
- However, the approach incurs additional latency due to the retrieval step, which limits deployment in real-time settings. [E3]
- This survey reviews adaptive retrieval methods that adjust retrieval depth based on query difficulty. [E4]
- We introduce a hybrid retrieval method combining dense embeddings with learned sparse representations. [E5]

## Findings

### Methods and Approaches

- We propose a retrieval-augmented generation framework combining a dense retriever with a fine-tuned generator. [E1][S1]
- This survey reviews adaptive retrieval methods that adjust retrieval depth based on query difficulty. [E4][S2]
- We introduce a hybrid retrieval method combining dense embeddings with learned sparse representations. [E5][S3]

### Key Results

- Our method improves exact match accuracy by 8.4 points over the baseline on a domain-specific benchmark dataset. [E2][S1]
- Results show a 5 point improvement in recall at low latency cost compared to pure dense retrieval baselines on standard QA benchmarks. [E6][S3]

### Limitations and Open Challenges

- However, the approach incurs additional latency due to the retrieval step, which limits deployment in real-time settings. [E3][S1]
- A key limitation across methods is poor generalization to unseen domains. [E8][S2]

### Datasets and Benchmarks

- Several benchmark datasets are compared, including one collected specifically for multi-hop reasoning. [E7][S2]

## Agreements, Contradictions, and Trade-offs

No explicit contradictions were detected across the retrieved sources. This may reflect genuine consensus, or simply that too few directly comparable results were retrieved — treat convergence with the same caution as disagreement.

## Research Gaps and Open Questions

- Only 3 source(s) were retained; broader coverage may surface additional gaps or contradict early findings.
- All retained sources came from a single provider; consider whether other databases (e.g. IEEE, ACM, Google Scholar) would surface additional relevant work.

## Practical Takeaways


**Watch out for:**
- A key limitation across methods is poor generalization to unseen domains. [E8][S2]

## Limitations of This Review

- Fewer than 5 sources were retained; treat conclusions as provisional.
- No open full text could be fetched; findings rely on abstracts only.
- Evidence snippets are extracted from abstracts (and full text where available); full-paper nuance may be lost.
- No LLM was available/used for this run; the report was assembled deterministically from extracted evidence.

## References

- **[S1]** Retrieval-Augmented Generation for Domain-Specific QA. A. Author, B. Author · 2025 · arXiv
- **[S2]** A Survey of Adaptive Retrieval Methods for QA. C. Author · 2024 · ACL
- **[S3]** Efficient Dense Retrieval with Learned Sparse Representations. D. Author, E. Author · 2023 · EMNLP
