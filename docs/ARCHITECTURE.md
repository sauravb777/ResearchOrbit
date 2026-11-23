# ResearchPilot Architecture (v80.3.0)

ResearchPilot separates **source acquisition** from **source reading**. A complete PDF may be retained locally for provenance and exact quote verification, but the LLM is no longer required to consume the entire paper.

## Adaptive paper-reading pipeline

1. **Structural cartography**
   - Parse headings, page boundaries and section regions deterministically.
   - Classify regions as abstract, introduction, method, data, experiment, results, ablation, analysis, discussion, limitations, conclusion, appendix/supplement or references.
   - References are excluded from LLM reading by construction.

2. **Report-aware routing**
   - The requested report type changes the reading emphasis.
   - Comparison emphasizes datasets, protocols, baselines, metrics, results and comparability.
   - Gap analysis emphasizes limitations, failures, unsupported claims and untested assumptions.
   - Evidence review emphasizes study design, outcomes, uncertainty and methodological limitations.
   - Proposal-oriented reports emphasize the evidence-backed gap, baselines and evaluation lessons.
   - Long or structurally complex papers receive an LLM cartographer. Short, clean papers use deterministic routing to avoid an unnecessary planning call.

3. **Targeted deep reading**
   - Selected evidence-bearing sections are read deeply.
   - Adjacent sections are bundled into bounded prompts so the model can reason across method -> experiment -> result without rereading the paper framing repeatedly.
   - Large selected sections may be chunked with overlap.
   - Low-value sections are omitted rather than summarized just to claim coverage.

4. **Conditional appendix reading**
   - Appendices and supplementary material are not normal reading targets.
   - They are read only when the main paper or the cartographer identifies a concrete dependency, such as a referenced ablation, implementation detail or experiment needed to interpret a result.

5. **Feynman-style integration**
   - The integration stage reconstructs the mechanism as a dependency chain: design choice -> mechanism -> test -> result -> interpretation -> boundary condition.
   - It preserves exact conditions, negative findings, ablations, failures, trade-offs, reproducibility details and surprising observations.
   - It distinguishes author-stated claims, demonstrated evidence and analytical inference.

6. **Grounding**
   - Model-proposed evidence quotes are verified deterministically against the complete locally stored source text.
   - The complete source is therefore used for provenance verification, not as a reason to flood every LLM prompt with irrelevant material.

## Runtime design

The reader is bounded by section, bundle and integration budgets. Multiple selected papers can still be processed concurrently through the existing LLM scheduler. The goal is not to minimize the amount of paper understood. The goal is to minimize **irrelevant tokens** while retaining the evidence required by the question and report type.

## Compatibility

The production reader is adaptive by default. The legacy whole-paper reader is retained only as internal compatibility code and is not the normal production path. Existing report synthesis, claim verification, chat and export layers continue to consume the same `PaperAnalysis` and `Evidence` contracts.

## v81 BEAST reading path

The production paper-reading path is now `pipeline/beast_reader.py`:

`extracted paper -> section map -> question-aware skim -> mandatory core-section selection -> parallel evidence bundles -> conditional appendix read -> evidence/claim recovery -> dossier integration -> deterministic quote verification`

The full extracted text is retained locally for provenance and quote verification. It is not repeatedly sent to the LLM. The LLM receives compact section maps, targeted evidence-bearing regions, and structured reading packets. The research-level LangGraph pipeline remains responsible for post-reading synthesis, council/debate, evidence normalization, parallel drafting and publication audit.

An explicit UI paper selection is a locked research input. The semantic corpus validator can record an alignment warning, but it cannot silently replace that paper. A selected paper can leave the final evidence corpus only if its text cannot be read or every recovery path fails; that outcome is persisted per paper with a reason.
