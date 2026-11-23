from __future__ import annotations
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..llm import LLMClient
from ..schemas import DraftSection, Evidence, PaperAnalysis, SynthesisPlan, CrossPaperFinding
from .prompts import PROMPT_VERSIONS
PROMPT_VERSION = PROMPT_VERSIONS["drafting"]
_SYSTEM = """You are a senior research report author working inside an evidence-grounded research agent.
Write one substantive section of a publishable research report. The section must help the reader understand the research question and what the literature actually establishes.
Knowledge-retention contract: this report is intended to compress the selected papers into a high-density scholarly learning artifact. Preserve nearly all question-relevant knowledge a researcher would need to understand, evaluate, reproduce, compare, apply, or extend the selected work. Do not confuse compression with omission. Preserve unique high-value observations including the paper's direct answer, mechanism, decisive design choices, experimental conditions, quantitative results, controls and ablations, negative results, failure modes, robustness and transfer boundaries, important trade-offs, implementation or cost implications, reproducibility constraints, contradictions, surprising findings, and research implications when they materially matter to the question. Remove repetition, boilerplate, peripheral background, retrieval metadata, and low-value detail. When multiple papers repeat a finding, synthesize the common point while retaining distinct conditions or evidence that change its interpretation. Specialized report types should change the allocation of attention, not lower this retention standard.
Do not produce an outline, generic filler, paper-by-paper abstracts, or repeated executive-summary language.
Use the supplied paper dossiers as research memory, not as a checklist. For every report type, write at research-survey quality: organize evidence around the question, meaningful approach families or analytical dimensions, explain mechanisms and experimental conditions, preserve important quantitative findings and negative results, compare results only under defensible conditions, and derive cross-paper insights. The report-type specification controls the emphasis, but never lowers the required depth. Individual papers are evidence inside the argument, not standalone summaries. The evidence_inventory and research_relevance_map are especially important because they preserve relevant information that did not fit predefined fields. Follow the research question and evidence, not the presence or absence of a field. Explain mechanisms, experimental context, results, limitations and unexpected relevant details when they materially help answer the question.
Each dossier contains field-level mappings to verified evidence passages. Use the closest supporting [E#]
mapping whenever making a paper-specific factual claim. Cite only supplied [E#] or [S#] identifiers.
A paraphrased analysis field is not self-authenticating.
Never invent numbers, datasets, venues, methods, causal explanations or citations. Do not turn unavailable fields, extraction gaps, absent schema values or screening gaps into report topics. Treat the dossiers as research memory, not a completeness checklist. Silently omit dimensions that simply are not relevant or not discussed. Only expose an absence when the paper explicitly tests or claims something that makes the absence scientifically consequential, such as an omitted control, an untested condition that limits the stated conclusion, or a result that fails to establish a claim. In those cases, describe the scientific implication naturally, not as a workflow complaint such as "not reported" or "missing information". Never compare results unless the experimental conditions make the comparison defensible. When evidence is insufficient for a claim, state the evidentiary boundary in natural scholarly language.
Single-source rule: when only one paper supplies the material for a point, write it, cite that paper and say nothing about the other papers, since the citation already shows who supports it. Never write sentences such as "the other studies do not report this", "no paper addressed", or "this was not found". Coverage discipline: a finding supported by only one or a few papers must be attributed to those papers and must not be written as a field-wide conclusion. If no retained paper contains evidence for a topic, omit that topic rather than announcing that the corpus failed to find it. Do not mention missing information simply because another paper lacks it. Unexpected evidence discovered in a paper is valid and should be integrated when it materially answers or changes the research question.
Report-type behavior:
literature_review: synthesize themes and disagreements across studies;
state_of_art: explain approach families, evolution, evaluation practice and bottlenecks;
gap_analysis: separate established findings from explicit gaps, inferred gaps and insufficient evidence;
comparison: align methods and conditions, then explain trade-offs and non-comparability;
evidence_review: explain screening and evidence characteristics, consistency and uncertainty;
research_proposal: separate literature evidence from proposed research and make the gap operational;
paper_analysis: preserve paper-specific claims before technical, experimental and cross-paper synthesis.
Use clear paragraphs with occasional compact bullets when they improve readability. Do not use tables unless the section genuinely requires a compact comparison. Do not use em dashes or en dashes. Avoid Oxford-comma constructions.
Citation discipline: every externally verifiable factual statement, numerical result, method detail, dataset claim, limitation or cross-paper conclusion must be traceable to supplied evidence. Cite at the end of the sentence or tightly connected claim cluster, using one or more specific evidence IDs. Do not mechanically cite every sentence when several consecutive sentences are clearly supported by the same evidence, but never leave a substantive analytical paragraph entirely uncited. Do not attach one citation to a paragraph if it does not support all of the paragraph's claims. If a point is an interpretation across papers, label it as an interpretation and cite the supporting papers/evidence. Citations are provenance, not decoration: maximize information density and readability first, then place the minimum sufficient citations needed to make the argument auditable. Never cite a paper merely because it was retrieved.
Writing style: direct, academically professional and concise. Explain concepts in plain language when useful, but do not oversimplify technical distinctions. Do not pad sections with generic background.
Survey-paper standard: write like a section of a peer-reviewed survey article. Open with a topic sentence that states the section's claim, develop it through mechanisms, quantitative evidence and conditions, compare studies explicitly (name the papers via their citation tags), and close with the implication for the research question. Optimise for knowledge density, not length: use as many words as the evidence justifies and no more, never pad or repeat. For every major finding answer the chain: what was observed, why (mechanism, only if the evidence supports it), under what conditions (dataset, setting, baseline, scale), how strong the evidence is, how it compares with other studies, and why it matters. Keep the epistemic level visible with natural phrasing: direct evidence ('Study X observed'), cross-paper synthesis ('Across the reviewed studies'), interpretation ('This suggests') and hypothesis ('A possible implication is'). Never present your own synthesis as if a paper said it. Keep demonstrated results separate from proposed transfer to other domains, populations or settings, and say which is which. Preserve negative results, failed methods, weak improvements and contradictions. Use short subsection labels in bold only when the section covers several distinct approaches.
OUTPUT: markdown section body only, no heading, no preamble."""
_REVISION = """You are a report revision editor. Correct only the supplied section using the supplied verification failures and valid evidence. Preserve supported content. Remove or qualify unsupported statements. Use only supplied citation IDs. Do not add external facts. Return markdown only."""
def _allowed_tags(text: str) -> set[str]:
    return set(re.findall(r"\[(?:E|S)\d+\]", text or ""))
def _has_substantive_uncited_paragraph(text: str) -> bool:
    for raw in re.split(r"\n\s*\n", text or ""):
        paragraph = re.sub(r"\s+", " ", raw).strip()
        if len(paragraph) < 55:
            continue
        if re.search(r"\[(?:E|S)\d+", paragraph):
            continue
        if re.match(r"^(not established|insufficient evidence|the retained (?:corpus|evidence)|this report|within the stated scope)", paragraph, flags=re.I):
            continue
        return True
    return False
_SECTION_CRITIQUE = """You are the quality-control reader for one section of a scholarly research report.
Audit the section against the supplied evidence and paper dossier. Identify only concrete defects:
unsupported factual claims, citations that do not support the sentence, missing conditions that materially
change interpretation, unjustified cross-paper comparison, author-stated versus inferred limitation confusion,
or important requested evidence omitted from the section. Do not rewrite the section. Return JSON only:
{"issues":[{"problem":"...","correction":"...","evidence_ids":["E1"]}]}
An empty issue list is a valid pass."""
def _quality_review_and_revise(section_markdown: str, title: str, evidence: list[Evidence], dossier: list[dict], llm: LLMClient) -> tuple[str, bool]:
    reviewer = getattr(llm, "complete_json", None)
    if not llm.available or not callable(reviewer):
        return section_markdown, False
    evidence_text = "\n".join(f"[{e.id}] ({e.kind}; {e.locator}) {e.text}" for e in evidence)
    try:
        audit = reviewer(
            _SECTION_CRITIQUE,
            f"SECTION: {title}\n\nDRAFT:\n{section_markdown}\n\nVERIFIED EVIDENCE:\n{evidence_text}\n\nPAPER DOSSIER:\n{json.dumps(dossier, ensure_ascii=False)}",
            max_tokens=1200,
        )
    except Exception:
        return section_markdown, False
    issues = audit.get("issues") if isinstance(audit, dict) else None
    if not isinstance(issues, list) or not issues:
        return section_markdown, False
    valid = [i for i in issues if isinstance(i, dict) and i.get("problem") and isinstance(i.get("evidence_ids", []), list)]
    if not valid:
        return section_markdown, False
    allowed = {e.id for e in evidence}
    if any(not set(i.get("evidence_ids") or []).issubset(allowed) for i in valid):
        return section_markdown, False
    corrections = "\n".join(
        f"- Problem: {i['problem']}\n  Correction: {i.get('correction','')}\n  Valid evidence: {', '.join(i.get('evidence_ids') or [])}"
        for i in valid[:8]
    )
    try:
        revised = llm.complete(
            _REVISION,
            f"SECTION TITLE: {title}\nORIGINAL SECTION:\n{section_markdown}\n\nQUALITY-CONTROL FINDINGS:\n{corrections}\n\nVALID EVIDENCE:\n{evidence_text}",
            max_tokens=1900,
        )
    except Exception:
        return section_markdown, False
    if not revised or not revised.strip():
        return section_markdown, False
    tags = _allowed_tags(revised)
    if not tags.issubset({f"[{e.id}]" for e in evidence}):
        return section_markdown, False
    if _has_substantive_uncited_paragraph(revised):
        return section_markdown, False
def draft_sections(question: str, report_type: str, depth: str, plan: SynthesisPlan, evidence: list[Evidence], analyses: list[PaperAnalysis], findings: list[CrossPaperFinding], llm: LLMClient, council: dict | None = None, knowledge_model: dict | None = None) -> tuple[list[DraftSection], bool]:
    """Draft independent report sections with bounded concurrency.
    Sections do not depend on one another. Running them sequentially made a
    long report wait on every network round trip. The LLM client has its own
    global concurrency gate, so this fan-out is deliberately small and cannot
    create an uncontrolled API burst.
    """
    if not llm.available or not plan.sections:
        return [], False
    council = council or {}
    analysis_payload = []
    for a in analyses:
        analysis_payload.append({
            "source_id": a.source_id,
            "source_tag": f"S{len(analysis_payload)+1}", "depth": a.depth, "method_category": a.method_category,
            "problem": a.problem, "objective": a.research_objective, "method": a.method_summary,
            "mechanism": a.mechanism, "setup": a.experimental_setup, "datasets": a.dataset_details,
            "baselines": a.baselines, "metrics": a.metrics, "training": a.training_protocol,
            "uncertainty": a.statistical_uncertainty, "error_analysis": a.error_analysis,
            "results": a.key_results, "quantitative_results": [q.model_dump() for q in a.quantitative_results],
            "ablations": a.ablations, "robustness": a.robustness, "limitations": a.limitations,
            "limitations_author_stated": a.limitations_author_stated,
            "limitations_inferred": a.limitations_inferred, "established": a.established,
            "not_established": a.not_established, "assumptions": a.assumptions,
            "failure_modes": a.failure_modes, "generalization": a.generalization,
            "implementation": a.implementation_details, "dependencies": a.dependency_risks,
            "interpretation": a.result_interpretation, "open_questions": a.open_questions,
            "field_evidence": getattr(a, "field_evidence", {}) or {},
            "section_map": getattr(a, "section_map", []) or [],
            "claim_ledger": getattr(a, "claim_ledger", []) or [],
            "experiment_records": getattr(a, "experiment_records", []) or [],
            "negative_results": getattr(a, "negative_results", ""),
            "causal_claims": getattr(a, "causal_claims", ""),
            "transfer_conditions": getattr(a, "transfer_conditions", ""),
            "evidence_conflicts": getattr(a, "evidence_conflicts", ""),
            "citation_context": getattr(a, "citation_context", ""),
            "evidence_quality_notes": getattr(a, "evidence_quality_notes", ""),
            "question_answer": getattr(a, "question_answer", ""),
            "key_evidence_chain": getattr(a, "key_evidence_chain", [])[:20],
            "decision_relevant_insights": getattr(a, "decision_relevant_insights", [])[:30],
            "method_tradeoffs": getattr(a, "method_tradeoffs", [])[:20],
            "replication_requirements": getattr(a, "replication_requirements", [])[:20],
            "research_implications": getattr(a, "research_implications", [])[:20],
            "first_principles": getattr(a, "first_principles", ""),
            "design_rationale": getattr(a, "design_rationale", [])[:12],
            "equation_explanations": getattr(a, "equation_explanations", [])[:10],
            "experiment_chain": getattr(a, "experiment_chain", [])[:14],
            "ablation_analysis": getattr(a, "ablation_analysis", [])[:12],
            "baseline_fairness": getattr(a, "baseline_fairness", ""),
            "claim_evidence_audit": getattr(a, "claim_evidence_audit", [])[:14],
            "hidden_assumptions": getattr(a, "hidden_assumptions", [])[:12],
            "section_consistency": getattr(a, "section_consistency", ""),
            "novelty_breakdown": getattr(a, "novelty_breakdown", ""),
            "significance": getattr(a, "significance", ""),
            "research_memory": getattr(a, "research_memory", [])[:120],
            "evidence_inventory": getattr(a, "evidence_inventory", [])[:120],
            "research_relevance_map": getattr(a, "research_relevance_map", {}) or {},
            "unmapped_observations": getattr(a, "unmapped_observations", [])[:70],
            "paper_evidence": [
                {"id": e.id, "kind": e.kind, "locator": e.locator, "text": e.text}
                for e in evidence if e.source_id == a.source_id
            ][:60],
        })
    finding_payload = [
        {"id": f.id, "type": f.finding_type, "text": f.text,
         "evidence_ids": f.evidence_ids, "source_ids": f.source_ids,
         "caveat": f.caveat}
        for f in findings
    ]
    def _one(idx: int, raw: object) -> DraftSection | None:
        title = str(raw.get("title") or raw.get("name") or f"Section {idx}") if isinstance(raw, dict) else f"Section {idx}"
        purpose = str(raw.get("purpose", "")) if isinstance(raw, dict) else ""
        requested_ids = list(raw.get("evidence_ids") or []) if isinstance(raw, dict) else []
        requested_sources = list(raw.get("source_ids") or []) if isinstance(raw, dict) else []
        analytical_tasks = list(raw.get("analytical_tasks") or []) if isinstance(raw, dict) else []
        must_answer = list(raw.get("must_answer") or []) if isinstance(raw, dict) else []
        relevant = [
            e for e in evidence
            if (not requested_ids or e.id in requested_ids)
            and (not requested_sources or e.source_id in requested_sources)
        ]
        if not relevant:
            relevant = sorted(
                [e for e in evidence if not requested_sources or e.source_id in requested_sources],
                key=lambda e: e.relevance,
                reverse=True,
            )
        relevant = relevant[:60]
        evidence_block = "\n".join(
            f"[{e.id}] ({e.kind}; {e.locator}) {e.text}" for e in relevant
        )
        section_analyses = [
            a for a in analysis_payload
            if not requested_sources or a["source_id"] in requested_sources
        ]
        if not section_analyses:
            section_analyses = analysis_payload
        section_findings = [
            f for f in finding_payload
            if not requested_sources or any(s in requested_sources for s in f["source_ids"])
        ]
        deep_reading_section = title.strip().lower() == "deep technical readings"
        target_length = int(raw.get("target_length") or (6500 if deep_reading_section else 3200)) if isinstance(raw, dict) else 3200
        user = f"""QUESTION: {question}
REPORT TYPE: {report_type}
DEPTH: deep
REPORT STYLE: research-grade synthesis for every report type. Build an evidence-backed analytical structure appropriate to the requested type, preserve mechanisms, experimental conditions, quantitative findings, limitations and boundary conditions, and synthesize across papers rather than producing one mini-summary per paper. Standard Report additionally emphasizes a broad field map and approach taxonomy.
SECTION: {title}
PURPOSE: {purpose}
ANALYTICAL TASKS: {analytical_tasks}
MUST ANSWER: {must_answer}
LITERAL EVIDENCE:
{evidence_block}
TARGETED PAPER DOSSIERS:
{json.dumps(section_analyses, ensure_ascii=False)}
TARGETED CROSS-PAPER FINDINGS:
{json.dumps(section_findings, ensure_ascii=False)}
RESEARCH-WRITER BRIEF:
{json.dumps(getattr(plan, "writer_brief", {}) or {}, ensure_ascii=False)}
PROFESSOR COUNCIL ARGUMENT:
{json.dumps(council, ensure_ascii=False)}
SHARED RESEARCH KNOWLEDGE MODEL (reasoning aid, never an independent source):
{json.dumps(knowledge_model or {}, ensure_ascii=False)[:18000]}
Write a coherent research-paper-quality analytical section. Prefer synthesis over enumeration, but do not compress away important experimental conditions. Use the paper-specific context to explain why a result matters, exactly under what conditions it was obtained, what controls/ablations support the interpretation, and what the evidence does not establish. For the Deep Technical Readings section, do not compress the papers into short summaries. Give each selected paper its own clearly labeled subsection and teach it step by step: problem and motivation, core idea, prerequisite concepts, mechanism from input to output, design rationale, implementation/training details that affect behavior, experimental setup, results with conditions, ablations and what they isolate, failures/robustness, limitations and reproducibility, then the paper's exact contribution to the research question. Preserve distinctive observations from the evidence inventory and research memory. The goal is high-retention scholarly understanding, not a short literature-review paragraph. Use experiment records for result claims and the claim ledger to distinguish demonstrated findings from author interpretation, inference, causal claims and unsupported generalization. Use the question_answer and evidence chains to keep the section oriented toward the user's actual question. Use decision-relevant insights and trade-offs to explain why findings matter rather than merely restating them. Preserve negative results, boundary conditions and non-comparable studies when they change the interpretation. Never manufacture missing details. IMPORTANT EDITORIAL RULE: the report is a scholarly learning artifact, not an extraction log. Include only information that materially helps the reader understand, evaluate or use the research. Do not narrate where the system found information, retrieval steps, relevance scores, internal evidence IDs, extraction failures, empty fields, screening mechanics, or requests for proof. Do not write 'this was not reported', 'the paper did not mention', 'not found', or similar absence language merely because an extraction field is empty. Mention an absence only when the absence itself is scientifically meaningful to the argument, such as a missing control that materially limits an experimental conclusion. Source provenance, exact proof, page-level evidence, 'which paper supports this?', and deeper verification questions belong to Research Chat/Verify. The report should read as if an expert researcher synthesized the literature after reading it, not as if an AI exposed its research workflow. Cite every substantive factual paragraph with the closest valid evidence IDs."""
        try:
            draft = llm.complete(_SYSTEM, user, max_tokens=min(8500, max(3800, int(target_length * 1.35))), tier="M", priority="report")
        except Exception:
            draft = None
        if not draft or not draft.strip():
            return None
        allowed = {f"[{e.id}]" for e in relevant}
        allowed |= {f"[{item.get('source_tag')}]" for item in section_analyses if item.get('source_tag')}
        tags = _allowed_tags(draft)
        if not tags.issubset(allowed) or _has_substantive_uncited_paragraph(draft):
            return None
        final_draft, revised = _quality_review_and_revise(
            draft.strip(), title, relevant, section_analyses, llm
        )
        minimum_words = 900 if deep_reading_section else 220
        if len(re.findall(r"\b\w+\b", final_draft)) < minimum_words and len(relevant) >= 3:
            try:
                expanded = llm.complete(
                    _SYSTEM,
                    user + "\n\nEXPANSION REQUIRED: the previous draft was too thin for a survey article. "
                    "Rewrite it with more depth. Add mechanisms, experimental conditions, quantitative results, negative results "
                    "and explicit cross-paper comparison drawn ONLY from the material above. Do not pad; add only substance the evidence supports.\n\nPREVIOUS DRAFT:\n" + final_draft,
                    max_tokens=min(8500, max(4500, int(target_length * 1.35))),
                )
            except Exception:
                expanded = None
            if expanded and expanded.strip() and _allowed_tags(expanded).issubset(allowed) \
                    and not _has_substantive_uncited_paragraph(expanded) \
                    and len(expanded.split()) > len(final_draft.split()):
                final_draft, revised = expanded.strip(), True
        return DraftSection(
            id=f"DS{idx}", title=title, purpose=purpose, markdown=final_draft,
            evidence_ids=[e.id for e in relevant],
            source_ids=list(dict.fromkeys(e.source_id for e in relevant)),
            status="revised" if revised else "drafted",
            revision_count=1 if revised else 0,
        )
    workers = min(3, max(1, len(plan.sections)))
    results: dict[int, DraftSection | None] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="researchpilot-draft") as pool:
        futures = {
            pool.submit(_one, idx, raw): idx
            for idx, raw in enumerate(plan.sections, 1)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = None
    sections = [results[i] for i in sorted(results) if results.get(i) is not None]
    return sections, bool(sections)
def revise_sections(sections: list[DraftSection], audit_items: dict[str, list[str]], evidence: list[Evidence], llm: LLMClient, max_cycles: int = 2) -> tuple[list[DraftSection], int, bool]:
    if not llm.available or not sections or not audit_items:
        return sections, 0, False
    evidence_map = {e.id: e for e in evidence}
    cycles = 0
    used = False
    for _ in range(max_cycles):
        changed = False
        for section in sections:
            failures = audit_items.get(section.id, [])
            if not failures or section.revision_count >= max_cycles:
                continue
            valid_evidence = "\n".join(f"[{e.id}] ({e.kind}) {e.text}" for e in evidence if e.id in section.evidence_ids)
            prompt = f"SECTION TITLE: {section.title}\nORIGINAL SECTION:\n{section.markdown}\n\nVERIFICATION FAILURES:\n" + "\n".join(failures) + f"\n\nVALID EVIDENCE:\n{valid_evidence}"
            try:
                revised = llm.complete(_REVISION, prompt, max_tokens=2600, tier="M", priority="report")
            except Exception:
                revised = None
            if not revised or not revised.strip():
                continue
            section_evidence = [e for e in evidence if e.id in section.evidence_ids]
            allowed = {f"[{e.id}]" for e in section_evidence}
            if not _allowed_tags(revised).issubset(allowed):
                continue
            section.markdown = revised.strip()
            section.revision_count += 1
            section.status = "revised"
            changed = True
            used = True
        cycles += 1
        if not changed:
            break
    return sections, cycles, used
