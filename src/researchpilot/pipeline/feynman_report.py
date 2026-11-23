from __future__ import annotations
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from typing import Any
from ..llm import LLMClient
from ..schemas import Evidence, PaperAnalysis, Source
from .parallel_council import (run_chief_synthesis,
                               run_compact_professor_council,
                               run_parallel_council)
from .retrieval import retrieve_passages
_TYPE_GUIDE = {
    "standard_report": "Answer the exact question as a complete, technically detailed explanatory research article. Build from the underlying problem and concepts, explain mechanisms step by step, connect each mechanism to experimental evidence, integrate complementary findings across papers, preserve conditions and uncertainty, and end with a defensible answer plus a dedicated research-gaps section. Derive gaps from specific limitations, missing evaluations, contradictions, boundary conditions and untested assumptions, and for each important gap explain what evidence or experiment would close it.",
    "literature_review": "Build a thematic literature synthesis rather than paper-by-paper summaries. Explain the major concepts and approach families, how evidence accumulates, where findings agree or disagree, why they differ, under what conditions they hold, and what the literature collectively establishes and leaves unresolved. Explicitly derive research gaps from the limitations and evidence boundaries of the studies.",
    "comparison": "First teach the mechanisms of the compared approaches, then compare them dimension by dimension: problem formulation, mechanism, data/setup, evaluation, evidence, trade-offs, robustness, cost and failure boundaries. Explicitly mark where experimental settings make a numerical comparison invalid.",
    "state_of_art": "Map the technical landscape represented by the papers: approach families, conceptual evolution, mechanisms that distinguish them, evaluation practice, robustness, current capabilities, bottlenecks and the boundaries that define the present frontier. End by identifying evidence-backed bottlenecks and concrete research gaps rather than generic future work.",
    "gap_analysis": "Establish the strongest supported knowledge first. Then derive concrete gaps from missing evidence, contradictions, boundary conditions, untested assumptions, reproducibility problems or demonstrated failures. Distinguish an unstudied question from a known failure, explain why the gap persists, and specify what evidence would close it.",
    "evidence_review": "For each central conclusion, trace the reasoning from study design to measurement to result to interpretation. Explain uncertainty, study limitations, consistency and contradiction while keeping the prose about the scientific subject rather than the mechanics of reviewing it.",
    "research_proposal": "Explain the established problem and evidence, identify a literature-grounded unresolved boundary, then derive a concrete research direction and evaluation logic from the papers. Tie each proposed direction to a specific limitation, missing test or unresolved disagreement. Never invent results or pretend a proposed experiment has been run.",
    "paper_analysis": "Explain each paper as a deep technical narrative: problem, contribution, mechanism, implementation, experiments, results, ablations, robustness, limitations and what the evidence truly establishes, followed by a synthesis of meaningful relationships across papers.",
}
_META_SENTENCE = re.compile(
    r"^\s*(we|our|i)\b|\b(this|the) (review|report|synthesis|survey|analysis pipeline)\b|"
    r"\b(retained|selected|screened|deep[- ]read) (corpus|papers?|studies|literature)\b|"
    r"\bthe (corpus|evidence base|retained evidence)\b|\bfull[- ]text (reading|analysis)\b|"
    r"\b(evidence coverage|research audit|search protocol|arxiv queries?)\b|\b(?:another|one|some|these|those) (?:paper|papers|study|studies) (?:is|are|was|were) (?:unrelated|included|selected|reviewed|retained)\b|\b(?:another|the other) (?:paper|study) (?:is|was) unrelated\b|\b(?:papers|studies) (?:we|the authors) (?:looked at|reviewed|selected|retained)\b|"
    r"\brepresented by \[S\d+\]",
    re.IGNORECASE,
)
_TAG_RE = re.compile(r"\[(S\d+)\]")
_ANY_TAG_RE = re.compile(r"\[[A-Z]\d+(?:\s*[-,–]\s*[A-Z]?\d+)*\]")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])((?:\s*\[[A-Z]\d+\])*)\s+(?=[A-Z\"'(\[*_])")
def split_sentences(text: str) -> list[str]:
    out, pos = [], 0
    for m in _SENT_BOUNDARY.finditer(text or ""):
        out.append(text[pos:m.end(1)].strip())
        pos = m.end()
    out.append((text or "")[pos:].strip())
    return [x for x in out if x]
def _clean(text: Any, limit: int = 0) -> str:
    s = re.sub(r"\[PAGE\s+\d+\]", " ", str(text or ""), flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    if limit and len(s) > limit:
        cut = s[:limit]
        end = max(cut.rfind(". "), cut.rfind("; "))
        s = cut[: end + 1] if end > limit * 0.5 else cut.rstrip().rstrip(',;:') + "."
    return s
def _sentence(text: str) -> str:
    text = _clean(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text
def _number_tokens(text: str) -> set[str]:
    """Numbers that need support: decimals, percentages, and integers > 10."""
    text = _ANY_TAG_RE.sub(" ", text)
    out: set[str] = set()
    for m in _NUM_RE.finditer(text):
        tok = m.group(0).replace(",", "")
        is_pct = text[m.end(): m.end() + 1] == "%"
        if "." in tok or is_pct or (tok.isdigit() and int(tok) > 10):
            out.add(tok.rstrip("."))
    return out
def strip_unsupported_numbers(text: str, material: str) -> str:
    """Drop sentences whose numbers are not present in `material`."""
    allowed = _number_tokens(material) | {t.replace(",", "") for t in re.findall(r"\d[\d,]*", material)}
    kept_lines: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            kept_lines.append(line)
            continue
        prefix_match = re.match(r"^(\s*(?:[-*+] |\d+[.)] ))", line)
        prefix = prefix_match.group(1) if prefix_match else ""
        body = line[len(prefix):]
        good = [s for s in split_sentences(body) if _number_tokens(s) <= allowed]
        if good:
            kept_lines.append(prefix + " ".join(good))
    return "\n".join(kept_lines)
def scrub_meta(text: str) -> str:
    """Remove sentences that narrate the review process instead of the subject."""
    out: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue
        prefix_match = re.match(r"^(\s*(?:[-*+] |\d+[.)] ))", line)
        prefix = prefix_match.group(1) if prefix_match else ""
        body = line[len(prefix):]
        good = [s for s in split_sentences(body) if not _META_SENTENCE.search(s)]
        if good:
            out.append(prefix + " ".join(good))
    return "\n".join(out)
def restrict_tags(text: str, allowed: set[str]) -> str:
    """Keep only [S#] tags from `allowed`.
    * grouped tags like [S1, S2] are normalised to [S1][S2]
    * internal ids (E12, C3, ...) are removed and never reach readers
    * a sentence whose ONLY citations were unknown sources is dropped: it was attributed to
      something the reader cannot check
    """
    text = re.sub(r"\[(?:S\d+\s*[,;–-]?\s*)+\]", lambda m: "".join(f"[{t}]" for t in re.findall(r"S\d+", m.group(0))), text or "")
    text = re.sub(r"\[[A-RT-Z]\d+(?:\s*[-,–]\s*[A-Z]?\d+)*\]", "", text)
    out: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            out.append(line)
            continue
        m = re.match(r"^(\s*(?:[-*+] |\d+[.)] ))", line)
        prefix = m.group(1) if m else ""
        kept = []
        for sent in split_sentences(line[len(prefix):]):
            found = re.findall(r"\[(S\d+)\]", sent)
            if found and not any(t in allowed for t in found):
                continue
            kept.append(re.sub(r"\[(S\d+)\]", lambda mm: mm.group(0) if mm.group(1) in allowed else "", sent))
        if kept:
            out.append(prefix + " ".join(kept))
    return "\n".join(out)
def _dedupe_sentences(text: str) -> str:
    """Collapse exact repeated sentences while preserving Markdown structure."""
    out_lines = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue
        prefix_match = re.match(r"^(\s*(?:[-*+] |\d+[.)] ))", line)
        prefix = prefix_match.group(1) if prefix_match else ""
        body = line[len(prefix):]
        seen: set[str] = set()
        kept = []
        for sentence in split_sentences(body):
            key = re.sub(r"\s+", " ", re.sub(r"\[[S]\d+\]", "", sentence).strip().lower())
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            kept.append(sentence)
        if kept:
            out_lines.append(prefix + " ".join(kept))
    return "\n".join(out_lines)
def finalize_text(text: str, material: str, allowed_tags: set[str]) -> str:
    text = restrict_tags(text or "", allowed_tags)
    text = scrub_meta(text)
    text = strip_unsupported_numbers(text, material)
    text = _dedupe_sentences(text)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
_CARD_FIELDS = [
    ("What problem it tackles", "problem"),
    ("What it is trying to show", "research_objective"),
    ("Core idea", "core_idea"),
    ("How the method works", "method_summary"),
    ("Mechanism (why it works)", "mechanism"),
    ("Data and setup", "experimental_setup"),
    ("Datasets", "dataset_details"),
    ("Compared against", "baselines"),
    ("Measured with", "metrics"),
    ("Results", "key_results"),
    ("Ablations", "ablations"),
    ("Robustness / generalisation", "generalization"),
    ("Limitations stated by the authors", "limitations_author_stated"),
    ("Other limitations", "limitations_inferred"),
    ("What is established", "established"),
    ("What is not established", "not_established"),
    ("Open questions", "open_questions"),
    ("Answer to the research question", "question_answer"),
    ("Important evidence observations", "evidence_inventory"),
    ("Research implications", "research_implications"),
    ("Method trade-offs", "method_tradeoffs"),
]
_BASIS_TAG = {"author_stated": "author-stated", "evidence_derived": "evidence-derived",
              "analytical_inference": "inference", "needs_verification": "unverified"}
def _tagged(item: dict, *keys: str, chars: int = 330) -> str:
    """Render one structured dossier item with its epistemic basis label."""
    bits = []
    for k in keys:
        v = _clean(item.get(k, ""), chars)
        if v:
            bits.append(f"{k.replace('_', ' ')}: {v}")
    if not bits:
        return ""
    basis = _BASIS_TAG.get(str(item.get("basis") or ""), "")
    return " | ".join(bits) + (f" [{basis}]" if basis else "")
def deep_dossier_lines(a: PaperAnalysis | None, per_list: int = 5) -> list[str]:
    """v79.12: expose first-principles understanding, experiment logic, ablation
    causality, assumptions and claim/evidence audits to the writers, each item
    carrying its basis label (author-stated / evidence-derived / inference)."""
    if a is None:
        return []
    lines: list[str] = []
    if getattr(a, "first_principles", ""):
        lines.append("Why it should work (first principles): " + _clean(a.first_principles, 1100))
    if getattr(a, "feynman_explanation", ""):
        lines.append("Plain explanation: " + _clean(a.feynman_explanation, 700))
    specs = (
        ("Design rationale", "design_rationale", ("choice", "why", "alternative")),
        ("Equations explained", "equation_explanations", ("equation", "terms", "intuition", "maps_to")),
        ("Experiment logic", "experiment_chain", ("question_tested", "setup", "baseline", "metric", "result", "demonstrates", "does_not_demonstrate")),
        ("Ablation causality", "ablation_analysis", ("component", "change", "result", "implication", "establishes_causality")),
        ("Claim vs evidence", "claim_evidence_audit", ("claim", "support", "strength", "gap")),
        ("Hidden assumptions", "hidden_assumptions", ("assumption", "why_needed", "if_violated")),
        ("Concepts", "concept_glossary", ("term", "explanation")),
    )
    for label, attr, keys in specs:
        rendered = [t for t in (_tagged(x, *keys) for x in (getattr(a, attr, None) or [])[:per_list] if isinstance(x, dict)) if t]
        if rendered:
            lines.append(f"{label}: " + " ;; ".join(rendered))
    for label, attr in (("Baseline fairness", "baseline_fairness"), ("Do conclusions follow from the evidence", "section_consistency"),
                        ("Novelty breakdown", "novelty_breakdown"), ("Significance", "significance"),
                        ("Limitations derived by analysis", "limitations_inferred")):
        v = _clean(getattr(a, attr, ""), 700)
        if v:
            lines.append(f"{label}: {v}")
    return lines
def _question_answer_pack(analysis: PaperAnalysis | None, *, max_items: int = 8) -> list[str]:
    """Pack extracted Q&A, evidence chains and boundaries without inventing claims."""
    if analysis is None:
        return []
    packed: list[str] = []
    direct_answer = _clean(getattr(analysis, "question_answer", ""), 650)
    if direct_answer:
        packed.append(f"Direct answer: {direct_answer}")
    relevance = getattr(analysis, "research_relevance_map", None)
    if isinstance(relevance, dict):
        for key, value in list(relevance.items())[:3]:
            value_text = _clean(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False), 300)
            if value_text:
                packed.append(f"Question link ({_clean(key, 70)}): {value_text}")
    for label, attr in (("Evidence chain", "key_evidence_chain"), ("Open question", "open_questions")):
        value = getattr(analysis, attr, None)
        if isinstance(value, list):
            for item in value[:max_items]:
                if isinstance(item, dict):
                    text = _clean(" | ".join(f"{k}: {v}" for k, v in item.items() if v), 360)
                else:
                    text = _clean(item, 360)
                if text:
                    packed.append(f"{label}: {text}")
        elif isinstance(value, str) and value.strip():
            packed.append(f"{label}: {_clean(value, 650)}")
    for label, attr in (("Evidence conflict", "evidence_conflicts"),
                        ("Negative result", "negative_results"),
                        ("Transfer boundary", "transfer_conditions"),
                        ("Not established", "not_established")):
        value = _clean(getattr(analysis, attr, ""), 420)
        if value:
            packed.append(f"{label}: {value}")
    out, seen = [], set()
    for item in packed:
        key = re.sub(r"\s+", " ", item).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out[:max_items * 2]
def paper_card(tag: str, source: Source, analysis: PaperAnalysis | None, limit: int = 4200) -> str:
    """Build a compact *knowledge dossier*, not a review summary.
    The planner/writer needs the paper's meaning, not just its abstract.  We
    therefore preserve the high-value deep-reading fields plus a small amount
    of evidence-first memory.  The dossier is still bounded so a large corpus
    cannot crowd every other paper out of the model context.
    """
    lines = [f"[{tag}] {_clean(source.title)} ({_clean(source.authors, 90)}, {source.year})"]
    if source.abstract:
        lines.append("Abstract: " + _clean(source.abstract, 1800))
    if analysis is not None:
        seen: set[str] = set()
        fields = [
            ("Problem", "problem"), ("Objective", "research_objective"),
            ("Core idea", "core_idea"), ("Method", "method_summary"),
            ("Mechanism", "mechanism"), ("Setup", "experimental_setup"),
            ("Data", "dataset_details"), ("Baselines", "baselines"),
            ("Metrics", "metrics"), ("Training", "training_protocol"),
            ("Results", "key_results"), ("Ablations", "ablations"),
            ("Robustness", "robustness"), ("Generalization", "generalization"),
            ("Failure modes", "failure_modes"), ("Limitations", "limitations_author_stated"),
            ("Established", "established"), ("Not established", "not_established"),
            ("Question answer", "question_answer"), ("Open questions", "open_questions"),
        ]
        for label, attr in fields:
            value = getattr(analysis, attr, "")
            if isinstance(value, list):
                value = "; ".join(str(x.get("observation", x)) if isinstance(x, dict) else str(x) for x in value[:5])
            value = _clean(value, 900)
            if value and value.lower() not in seen:
                seen.add(value.lower())
                lines.append(f"{label}: {value}")
        for attr, label in (
            ("research_relevance_map", "Question relevance"),
            ("method_tradeoffs", "Method trade-offs"),
            ("research_implications", "Research implications"),
        ):
            value = getattr(analysis, attr, "")
            if isinstance(value, dict):
                value = " ".join(str(v) for v in value.values() if isinstance(v, (str, int, float)))
            elif isinstance(value, list):
                value = "; ".join(
                    str(x.get("insight", x.get("observation", x))) if isinstance(x, dict) else str(x)
                    for x in value[:5]
                )
            value = _clean(value, 850)
            if value and value.lower() not in seen:
                seen.add(value.lower())
                lines.append(f"{label}: {value}")
        lines.extend(l for l in deep_dossier_lines(analysis, per_list=5 if limit > 6000 else 3) if l.lower() not in seen)
        qa_pack = _question_answer_pack(analysis, max_items=6 if limit > 6000 else 4)
        if qa_pack:
            lines.append("Research question-and-answer memory (field-derived): " + " ;; ".join(qa_pack))
        nums = []
        for q in getattr(analysis, "quantitative_results", [])[:10]:
            bits = [x for x in (q.dataset, q.model, q.metric, q.baseline, q.proposed) if x]
            if bits:
                nums.append(" | ".join(_clean(b, 100) for b in bits))
        if nums:
            lines.append("Reported measurements: " + " ;; ".join(nums))
        inventory = getattr(analysis, "evidence_inventory", []) or []
        observations = []
        for item in sorted(inventory, key=lambda x: float(x.get("importance", 0) or 0), reverse=True)[:5]:
            obs = _clean(item.get("observation", ""), 420) if isinstance(item, dict) else ""
            if obs:
                observations.append(obs)
        if observations:
            lines.append("Evidence observations: " + " | ".join(observations))
    card = "\n".join(lines)
    if len(card) <= limit:
        return card
    cut = card[:limit].rsplit(" ", 1)[0].rstrip(',;:')
    return cut + "."
def _digest(tag: str, source: Source, analysis: PaperAnalysis | None) -> str:
    return paper_card(tag, source, analysis, limit=1500)
_SYNTHESIS_SYSTEM = (
    "You are the senior research synthesizer. Before anyone writes the report, build an internal model of what the "
    "papers collectively mean for the user's research question. This is an analytical memory, NOT prose for the reader. "
    "Read every supplied paper dossier as a whole. Do not merely count mentions or concatenate summaries. Reconstruct the "
    "logic: problem -> proposed mechanism -> experimental test -> observed result -> interpretation -> boundary. Then connect "
    "papers through genuine relationships: agreement, disagreement, extension, complementary evidence, different assumptions, "
    "different evaluation conditions, contradictions, or gaps. A difference is not a contradiction unless the evidence warrants it. "
    "Never merge numbers from incompatible settings. Never infer a result simply because a method sounds plausible.\n\n"
    "The research question controls relevance. The report type controls which relationships deserve emphasis, but it must never "
    "override what the papers actually establish. Preserve surprising and negative findings. Preserve conditional findings and "
    "important implementation/evaluation details when they change interpretation. Separate four epistemic levels: (1) directly "
    "reported by a paper, (2) demonstrated by its experiment, (3) reasonable synthesis across papers, and (4) unresolved. Do not "
    "turn level 3 into level 2.\n\n"
    "Before returning, construct an internal dependency chain: concepts the reader must understand first, mechanisms that explain later results, evidence that supports each major step, and conditions that qualify each conclusion. The central_answer must answer the user's question directly rather than merely describe the papers.\n\n"
    "Return ONLY JSON with keys: central_answer, conceptual_model, themes, cross_paper_relationships, consensus, disagreements, "
    "conditional_findings, evidence_gaps, high_value_details, comparison_dimensions, report_requirements, conclusion_logic. "
    "themes should contain title, explanation, source_tags, evidence_hooks, importance. cross_paper_relationships should contain "
    "kind (agreement|disagreement|extension|complement|conditional_difference|contradiction|gap), source_tags, explanation, "
    "conditions. evidence_gaps should contain gap, why_it_matters, source_tags. comparison_dimensions should contain dimension "
    "and what can/cannot be compared. conclusion_logic should be a short ordered list of reasoning steps.\n"
)
def _synthesis_payload(question: str, report_type: str, cards: dict[str, str], analyses: dict[str, PaperAnalysis | None]) -> str:
    dossiers = []
    for tag, card in cards.items():
        a = analyses.get(tag)
        extra = {
            "source_tag": tag,
            "question_answer": getattr(a, "question_answer", "") if a else "",
            "evidence_chain": getattr(a, "key_evidence_chain", [])[:20] if a else [],
            "relevance": getattr(a, "research_relevance_map", {}) if a else {},
            "high_value_inventory": sorted((getattr(a, "evidence_inventory", []) or []), key=lambda x: float(x.get("importance", 0) or 0), reverse=True)[:18] if a else [],
            "tradeoffs": getattr(a, "method_tradeoffs", [])[:12] if a else [],
            "conflicts": getattr(a, "evidence_conflicts", "") if a else "",
            "negative_results": getattr(a, "negative_results", "") if a else "",
            "transfer_conditions": getattr(a, "transfer_conditions", "") if a else "",
            "deep_dossier": deep_dossier_lines(a, per_list=4) if a else [],
        }
        dossiers.append(card + "\nQUESTION-CENTERED MEMORY:\n" + json.dumps(extra, ensure_ascii=False))
    return f"RESEARCH QUESTION: {question}\nREPORT TYPE: {report_type}\n\nPAPER DOSSIERS:\n" + "\n\n".join(dossiers)[:105000]
def build_synthesis_map(question: str, report_type: str, cards: dict[str, str], analyses: dict[str, PaperAnalysis | None], llm: LLMClient) -> tuple[dict, bool]:
    if not cards:
        return {}, False
    if llm.available:
        try:
            raw = _report_llm_json(llm, _SYNTHESIS_SYSTEM, _synthesis_payload(question, report_type, cards, analyses), temperature=0.12, max_tokens=5200, tier="M")
            if isinstance(raw, dict) and (raw.get("central_answer") or raw.get("themes")):
                return raw, True
        except Exception:
            pass
    themes = []
    for tag, a in analyses.items():
        if not a:
            continue
        idea = _clean(getattr(a, "question_answer", "") or getattr(a, "core_idea", ""), 520)
        if idea:
            themes.append({"title": _clean(getattr(a, "method_category", "Paper contribution") or "Paper contribution", 80),
                           "explanation": idea, "source_tags": [tag], "evidence_hooks": [], "importance": 0.6})
    return {"central_answer": "", "conceptual_model": "", "themes": themes, "cross_paper_relationships": [],
            "consensus": [], "disagreements": [], "conditional_findings": [], "evidence_gaps": [],
            "high_value_details": [], "comparison_dimensions": [], "report_requirements": [], "conclusion_logic": []}, False
_PLAN_SYSTEM = (
    "You are the lead author of an explanatory research article. You will be given short digests of the papers "
    "that were studied, the research question and the report type. Design the article as a tree of TOPICS and "
    "SUB-TOPICS that lets a smart newcomer understand the subject step by step, the way Richard Feynman would "
    "teach it: start from the simple underlying problem, build up the ideas, then how the methods work, then what "
    "was found and what is still unknown.\n"
    "Rules: choose 4 to 7 topics adaptively; each topic has 1 to 4 sub-topics; topic and sub-topic titles are plain, specific phrases "
    "(no 'Overview', no 'Introduction', no 'Methodology of the review'). Group by IDEA, not by paper, unless the "
    "report type is a paper analysis. Every topic lists the source tags (like S1) whose material it will draw on, "
    "and may only use tags that appear in the digests. Include a topic only if the digests contain material for it. The first substantive topic must establish the conceptual foundation needed to understand later mechanisms, and later topics must build on earlier ones. For Standard and Literature Review prefer thematic integration; for State of the Art organize by approach families/evolution; for Gap Analysis organize by established evidence -> unresolved boundary -> why unresolved -> evidence needed; for Comparison organize by comparable mechanisms/conditions/trade-offs; for Paper Analysis retain paper-specific structure only where it adds technical value. "
    "Never plan a section about how papers were searched or selected, and never plan a 'not found' section.\n"
    "If exactly ONE paper is supplied do NOT pretend cross-paper synthesis exists. Plan an exceptionally deep single-paper analysis whose topics let the reader answer: what problem is solved and why it matters; how the method works and why it should work (mechanism, key equations, design choices); how it was evaluated (data, baselines, metrics and what each experiment tests); what the results and ablations actually establish; where it fails or remains uncertain (assumptions, limitations, failure modes); what is genuinely novel; what can be learned. Use up to 6 topics.\n"
    "Return ONLY JSON: {\"title\": str, \"topics\": [{\"title\": str, \"goal\": str, "
    "\"source_tags\": [str], \"subtopics\": [{\"title\": str, \"what_to_explain\": str}]}]}"
)
def _fallback_plan(question: str, tags: list[str], sources_by_tag: dict[str, Source]) -> dict:
    return {
        "title": "",
        "topics": [
            {"title": "The problem and the concepts needed to understand it", "goal": "Build the conceptual foundation in plain language before technical mechanisms.", "role": "problem", "source_tags": tags, "subtopics": [{"title": "The core problem", "what_to_explain": "What is happening, why it is difficult, and which concepts the reader needs first."}]},
            {"title": "How the main approaches work", "goal": "Explain the mechanisms step by step and connect design choices to intended effects.", "role": "method", "source_tags": tags, "subtopics": [{"title": "Mechanisms and design choices", "what_to_explain": "How the important approaches operate and why their components matter."}]},
            {"title": "What the evidence actually shows", "goal": "Connect mechanisms to experiments, measurements, results, robustness and failure cases.", "role": "results", "source_tags": tags, "subtopics": [{"title": "Evidence, results and conditions", "what_to_explain": "What was tested, what was observed, under which conditions, and what those observations establish."}]},
            {"title": "Boundaries, gaps and implications", "goal": "Explain limitations, disagreements, unresolved questions and implications for the research question.", "role": "implications", "source_tags": tags, "subtopics": [{"title": "What remains unresolved", "what_to_explain": "What is not established, where evidence conflicts or is incomplete, and what would resolve the uncertainty."}]},
        ],
    }
def _report_llm_json(llm, system, user, *, max_tokens, temperature=0.2, tier="F"):
    """Provider call with compatibility fallback for lightweight test/adaptor LLMs."""
    try:
        return llm.complete_json(system, user, temperature=temperature, max_tokens=max_tokens, priority="report", tier=tier)
    except TypeError:
        return llm.complete_json(system, user, temperature=temperature, max_tokens=max_tokens)
def _report_llm_text(llm, system, user, *, max_tokens, temperature=0.2, tier="F"):
    try:
        return llm.complete(system, user, temperature=temperature, max_tokens=max_tokens, priority="report", tier=tier)
    except TypeError:
        return llm.complete(system, user, temperature=temperature, max_tokens=max_tokens)
def _ensure_gap_topic(plan: dict, report_type: str, tags: list[str]) -> dict:
    """Guarantee an explicit research-gap topic for report types that require it.
    The LLM planner is allowed to shape the narrative, but a Standard Report should
    never lose the user's most important research-use output simply because the planner
    chose four other topics.
    """
    required = {"standard_report", "literature_review", "state_of_art", "gap_analysis", "research_proposal"}
    if report_type not in required:
        return plan
    topics = list(plan.get("topics") or [])
    if any(re.search(r"\b(gap|open|unresolved|bottleneck|future|limitation)\b", str(t.get("title", "")), re.I) for t in topics if isinstance(t, dict)):
        return {**plan, "topics": topics}
    gap = {
        "title": "Research gaps, unresolved boundaries and opportunities",
        "goal": "Derive concrete research gaps from the actual limitations, missing evidence, contradictions, boundary conditions and untested assumptions in the studied literature. For each important gap explain why it matters and what evidence or experiment would resolve it.",
        "role": "implications",
        "source_tags": tags,
        "subtopics": [
            {"title": "What the current evidence does not establish", "what_to_explain": "Specific limitations, missing controls, missing evaluations, contradictions, transfer boundaries or assumptions that prevent a stronger conclusion."},
            {"title": "What could be investigated next", "what_to_explain": "Concrete research questions and experiments that directly address those gaps, including what should be measured and why it would resolve the uncertainty."},
        ],
    }
    if len(topics) >= 4:
        topics[-1] = gap
    else:
        topics.append(gap)
    return {**plan, "topics": topics}
def plan_topics(question: str, report_type: str, tags: list[str], digests: str, sources_by_tag: dict[str, Source], llm: LLMClient, synthesis_map: dict | None = None) -> tuple[dict, bool]:
    guide = _TYPE_GUIDE.get(report_type, _TYPE_GUIDE["standard_report"])
    synthesis = json.dumps(synthesis_map or {}, ensure_ascii=False)[:18000]
    user = (f"RESEARCH QUESTION: {question}\nREPORT TYPE: {report_type}\nSHAPE FOR THIS TYPE: {guide}\n"
            f"AVAILABLE SOURCE TAGS: {', '.join(tags)}\n\nCROSS-PAPER RESEARCH UNDERSTANDING:\n{synthesis}\n\nPAPER DIGESTS:\n{digests}")
    for size in (1.0, 0.5):
        raw = _report_llm_json(llm, _PLAN_SYSTEM, user[: int(len(user) * size)], max_tokens=1500, temperature=0.2, tier="F") if llm.available else None
        if isinstance(raw, dict) and isinstance(raw.get("topics"), list):
            topics = []
            for t in raw["topics"][:6]:
                if not isinstance(t, dict) or not t.get("title"):
                    continue
                stags = [x for x in (t.get("source_tags") or []) if x in tags] or tags
                subs = [s for s in (t.get("subtopics") or []) if isinstance(s, dict) and s.get("title")][:3]
                topics.append({"title": _clean(t["title"], 110), "goal": _clean(t.get("goal", ""), 300),
                               "role": _clean(t.get("role", ""), 60), "section_type": _clean(t.get("section_type", ""), 80),
                               "why_in_report": _clean(t.get("why_in_report", ""), 360),
                               "required_evidence": [str(x)[:260] for x in (t.get("required_evidence") or [])[:8]],
                               "source_tags": stags,
                               "subtopics": [{"title": _clean(s["title"], 110), "what_to_explain": _clean(s.get("what_to_explain", ""), 300), "evidence_focus": [str(x)[:220] for x in (s.get("evidence_focus") or [])[:5]]} for s in subs]})
            if len(topics) >= 2:
                return _ensure_gap_topic({"title": _clean(raw.get("title", ""), 140), "topics": topics}, report_type, tags), True
    return _ensure_gap_topic(_fallback_plan(question, tags, sources_by_tag), report_type, tags), False
_WRITE_SYSTEM = (
    "You are a brilliant teacher writing ONE section of an explanatory research article, in the style of Richard "
    "Feynman: plain words, real understanding, no jargon left undefined.\n"
    "HARD RULES\n"
    "1. Use ONLY facts contained in SOURCE MATERIAL. Never add outside knowledge, numbers, datasets, names or results.\n"
    "2. For each sub-topic: begin with the simple idea in everyday words (what is going on and why it matters), then "
    "build up to how it actually works step by step (cause -> effect), then what the papers found, then any real limit "
    "the papers state. Define every technical term the first time you use it.\n"
    "3. Follow a Feynman sequence inside each substantive sub-topic: (a) state the simple idea in plain language, (b) explain the mechanism step by step, (c) connect the mechanism to the paper's actual evidence, (d) explain what that evidence means, and (e) state the condition, limitation or uncertainty that changes the interpretation. You may use ONE short everyday analogy per sub-topic ('Think of it like...') purely to build intuition. Never "
    "attach a fact, number or result to the analogy.\n"
    "4. Every sentence that states a paper-specific fact, result, method detail, dataset, comparison or limitation ends with its source tag(s), e.g. [S2] or [S1][S3]. Use only the "
    "tags shown in the material. Intuition sentences and analogies need no tag.\n"
    "5. Explain the SUBJECT directly. Never narrate how the article was made: do not write 'we', 'our', 'this review', "
    "'the corpus', 'the papers we read', 'retrieved' or 'selected'. Say 'CryoNet combines optical and elevation data' "
    "not 'we looked at CryoNet'.\n"
    "6. If the material does not cover something, leave it out silently. Do not apologise or list what is missing, "
    "except a genuine scientific limitation the papers themselves state.\n"
    "7. When papers use different data or settings, say their numbers are not directly comparable rather than ranking them.\n"
    "8. The report is an explanation of the subject, not a sequence of mini paper summaries. Follow the adaptive section purpose supplied by the planner. If this section is a mechanism, reconstruct the mechanism and design rationale. If it is evidence, reconstruct what each experiment tests and what it establishes. If it is comparison, align conditions before comparing. If it is a gap/boundary section, derive each gap from concrete evidence. If it is an implication or research-direction section, connect it explicitly to established evidence and limitations. When multiple papers address "
    "the same idea, synthesize them in the same narrative and explain why their findings agree, differ or complement one another. "
    "Use paper tags as provenance, not as section structure. Do not mechanically give every paper equal space: spend space where the "
    "research question and evidence make it useful.\n"
    "9. EXPLANATION DEPTH. For every important finding climb the ladder: the fact -> what it means -> why it matters -> how it connects to the mechanism or other results -> what the evidence supports -> what remains uncertain. Not every sentence needs every rung, but no key result may be stated bare. Explain MECHANISMS (what is intervened on, why it should work, what each component contributes), not just that a method 'improves' something. When equations or design choices appear in the material, explain what the terms do and why the authors chose them. For experiments state what was tested, against what, with which metric, what it demonstrates and what it does NOT demonstrate. Treat ablations as causal evidence and say whether they actually isolate the component.\n"
    "10. RESEARCH GAPS. When the topic concerns gaps, derive each gap from a specific limitation, missing evaluation, contradiction, boundary condition, untested assumption, reproducibility issue or failure case in the supplied material. For each gap state what is known, what is not established, which source exposes the boundary, why it matters and what concrete experiment, data or analysis could resolve it. Distinguish author-stated open questions from gaps inferred by comparing the supplied evidence.\n"
"11. FACT vs INFERENCE. The material labels items [author-stated], [evidence-derived] or [inference]. Say plainly in prose which is which (e.g. 'the authors state...', 'the reported numbers imply...', 'a reasonable reading, not claimed by the authors, is...'). Never present an [inference] or [unverified] item as something the paper demonstrated. If the paper does not establish something, say it does not.\n"
    "12. Density over length: no filler, no repeated results, no generic implications. A 150-word passage that reconstructs a mechanism beats 500 vague words.\n"
    "13. FORMAT IS MANDATORY. Return Markdown with exactly one '### <sub-topic title>' heading for EVERY supplied sub-topic, in the same order. Put the discussion belonging to each sub-topic underneath its heading. Never replace these headings with labels such as 'The idea', 'Data and setup', 'What it found', or 'Where it falls short'. Do NOT repeat the top-level topic heading. Short paragraphs and small bullet lists are allowed only when they improve readability. Aim for 350-650 words per sub-topic when the supplied evidence supports that depth; do not pad with repetition. No preamble, no closing summary."
)
def _topic_material(topic: dict, cards: dict[str, str], question: str, sources_by_tag: dict[str, Source], scale: float, synthesis_map: dict | None = None, passage_limit: int = 3) -> str:
    tags = topic["source_tags"][:6]
    parts = [cards[t][: int(len(cards[t]) * scale)] for t in tags if t in cards]
    if synthesis_map:
        relevant = {k: synthesis_map.get(k) for k in ("central_answer", "conceptual_model", "themes", "cross_paper_relationships", "consensus", "disagreements", "conditional_findings", "evidence_gaps", "high_value_details", "comparison_dimensions", "conclusion_logic", "gaps", "answer_boundary", "executive_summary", "key_takeaways", "open_questions") if synthesis_map.get(k)}
        parts.append("CROSS-PAPER SYNTHESIS MEMORY\n" + json.dumps(relevant, ensure_ascii=False)[:6000])
    query = " ".join([question, topic["title"], topic.get("goal", "")] + [s["title"] + " " + s.get("what_to_explain", "") for s in topic["subtopics"]])
    passages = retrieve_passages(
        query, [sources_by_tag[t] for t in tags if t in sources_by_tag],
        top_k=max(1, int(passage_limit or 3)),
        per_source_cap=2, chunk_chars=1200, overlap_chars=150
    )
    tag_of = {s.id: t for t, s in sources_by_tag.items()}
    if passages:
        parts.append("FULL-TEXT PASSAGES")
        for p in passages:
            loc = ", ".join(x for x in (f"p.{p.page}" if p.page else "", p.section) if x)
            parts.append(f"[{tag_of.get(p.source_id, 'S?')}]{' (' + loc + ')' if loc else ''}: {_clean(p.text, int(1200 * scale))}")
    material = "\n\n".join(parts)
    writer_cap = 28000
    if len(material) > writer_cap:
        material = material[:writer_cap].rsplit(" ", 1)[0].rstrip() + "\n[writer context truncated; use only retained evidence]"
    return material
def _third_person(text: str) -> str:
    text = re.sub(r"(^|(?<=[.!?] ))We\b", r"\1The authors", text)
    return re.sub(r"\b[Oo]ur\b", "their", text)
def _short_name(source: Source) -> str:
    return _clean(source.title.split(":")[0], 60)
_ROLE_FIELDS = {
    "problem": [("The problem", ("problem", "research_objective"))],
    "method": [("The idea", ("core_idea",)), ("How it works", ("method_summary", "mechanism")), ("Data and setup", ("dataset_details", "experimental_setup")), ("Evidence", ("key_results", "established"))],
    "results": [("What it found", ("key_results", "established")), ("How far it generalises", ("generalization",)),
                ("Where it falls short", ("limitations_author_stated", "limitations_inferred", "not_established")), ("Still open", ("open_questions",))],
    "implications": [("What it means", ("research_implications", "question_answer", "established")), ("What remains open", ("open_questions", "limitations_author_stated", "not_established"))],
}
def _topic_role(topic: dict) -> str:
    """Infer a safe fallback role from a topic title and goal."""
    title = str(topic.get("title", "") or "")
    goal = str(topic.get("goal", "") or "")
    t = (title + " " + goal).lower()
    if re.search(r"\b(problem|why|background|challenge|motivation|matters|hard)\b", t):
        return "problem"
    if re.search(r"\b(gap|open question|unresolved|implication|opportunit|future work|what remains)\b", t):
        return "implications"
    if re.search(r"\b(result|finding|evidence|evaluation|benchmark|metric|performance|ablation|limit|unknown|failure|robustness)\b", t):
        return "results"
    return "method"
def _normalize_topic_role(role: Any, topic: dict | None = None) -> str:
    """Canonicalize free-form planner roles so fallback lookup cannot raise KeyError."""
    valid = set(_ROLE_FIELDS)
    raw = _clean(role, 100).lower().replace("-", " ").replace("_", " ")
    aliases = {
        "background": "problem", "challenge": "problem", "motivation": "problem",
        "evaluation": "results", "evaluation framework": "results", "evidence": "results",
        "findings": "results", "result": "results", "conclusion": "implications",
        "implication": "implications", "research gaps": "implications", "gap analysis": "implications",
        "approach": "method", "mechanism": "method", "methodology": "method",
    }
    if raw in valid:
        return raw
    if raw in aliases:
        return aliases[raw]
    for candidate in ("problem", "results", "implications", "method"):
        if re.search(rf"\b{candidate}\b", raw):
            return candidate
    return _topic_role(topic or {})
def _fallback_topic(topic: dict, analyses: dict[str, PaperAnalysis | None], sources_by_tag: dict[str, Source], used: set[str] | None = None) -> str:
    """Deterministic explanation assembled ONLY from extracted paper fields (used when the model is
    unreachable). Each topic uses only the fields that fit its role and never repeats text that an
    earlier topic already used."""
    used = used if used is not None else set()
    role = _normalize_topic_role(topic.get("role"), topic)
    blocks: list[str] = []
    for tag in (topic.get("source_tags") or list(sources_by_tag))[:4]:
        a, src = analyses.get(tag), sources_by_tag.get(tag)
        if src is None:
            continue
        lines = []
        for label, attrs in _ROLE_FIELDS[role]:
            for attr in attrs:
                v = _third_person(_sentence(_clean(getattr(a, attr, "") if a is not None else "", 480)))
                if len(v) > 3 and re.sub(r"\W+", " ", v.lower()).strip() not in used:
                    used.add(re.sub(r"\W+", " ", v.lower()).strip())
                    lines.append(f"**{label}:** {v} [{tag}]")
                    break
        if role == "results" and a is not None:
            for q in (getattr(a, "quantitative_results", []) or [])[:4]:
                bits = [x for x in (getattr(q, "metric", ""), getattr(q, "proposed", ""), getattr(q, "dataset", "")) if x]
                if bits:
                    statement = _third_person(_sentence(_clean("; ".join(str(x) for x in bits), 520)))
                    key = re.sub(r"\W+", " ", statement.lower()).strip()
                    if key and key not in used:
                        used.add(key)
                        lines.append(f"**Reported result:** {statement} [{tag}]")
        if not lines and role == "problem" and src.abstract:
            v = _third_person(_sentence(_clean(src.abstract, 520)))
            if v.lower() not in used:
                used.add(re.sub(r"\W+", " ", v.lower()).strip())
                lines.append(f"{v} [{tag}]")
        if lines:
            blocks.append(f"**{_short_name(src)}**\n\n" + "\n\n".join(lines))
    if not blocks:
        return ""
    subtopics = topic.get("subtopics") or []
    heading = subtopics[0].get("title") if subtopics and isinstance(subtopics[0], dict) else topic.get("title", "Research explanation")
    return f"### {heading}\n\n" + "\n\n".join(blocks)
def _shape_subtopic_structure(text: str, subtopics: list[dict]) -> str:
    """Guarantee a readable report tree even when a model ignores Markdown headings.
    If the writer omitted the requested sub-topic headings, distribute its paragraphs
    across the planned sub-topics instead of publishing one undifferentiated block.
    This is structural only; it never rewrites factual content.
    """
    if not text.strip() or not subtopics:
        return text.strip()
    existing = len(re.findall(r"^###\s+", text, flags=re.MULTILINE))
    if existing >= len(subtopics):
        return text.strip()
    clean = re.sub(r"^###\s+[^\n]+\n?", "", text, flags=re.MULTILINE).strip()
    paragraphs = [x.strip() for x in re.split(r"\n{2,}", clean) if x.strip()]
    if not paragraphs:
        return text.strip()
    n = len(subtopics)
    groups: list[list[str]] = [[] for _ in range(n)]
    for i, para in enumerate(paragraphs):
        groups[min(n - 1, (i * n) // len(paragraphs))].append(para)
    blocks = []
    for i, sub in enumerate(subtopics):
        body = "\n\n".join(groups[i]).strip()
        if body:
            blocks.append(f"### {sub['title']}\n\n{body}")
    return "\n\n".join(blocks) or text.strip()
def write_topic(topic: dict, question: str, report_type: str, cards: dict[str, str], analyses: dict[str, PaperAnalysis | None],
                sources_by_tag: dict[str, Source], llm: LLMClient, used_text: set[str] | None = None, synthesis_map: dict | None = None) -> tuple[str, bool]:
    """Write one report section with one bounded model call.
    LLMClient already owns provider retries and rate-limit handling. A second
    per-section prompt retry doubled report latency without materially improving
    quality, so failures now fall back immediately to grounded extracted content.
    """
    subs = "\n".join(f"- {s['title']}: {s['what_to_explain']}" for s in topic["subtopics"]) or f"- {topic['title']}"
    material = _topic_material(topic, cards, question, sources_by_tag, 1.0, synthesis_map, int(getattr(getattr(llm, "settings", None), "performance_writer_passages", 3) or 3))
    user = (f"RESEARCH QUESTION (for orientation): {question}\nREPORT TYPE: {report_type}\n"
            f"THIS TOPIC: {topic['title']}\nSECTION TYPE: {topic.get('section_type', '')}\nWHY THIS SECTION BELONGS: {topic.get('why_in_report', '')}\n"
            f"GOAL: {topic.get('goal', '')}\nREQUIRED EVIDENCE: {json.dumps(topic.get('required_evidence') or [], ensure_ascii=False)}\n"
            f"SUB-TOPICS TO WRITE:\n{subs}\n\nSOURCE MATERIAL:\n{material}")
    writer_default = int(getattr(getattr(llm, "settings", None), "performance_writer_max_tokens", 4200) or 4200)
    if bool(getattr(getattr(llm, "settings", None), "quality_first", True)):
        writer_default = max(writer_default, 3200)
    section_tokens = min(writer_default, 1500 + 850 * max(1, len(topic["subtopics"])))
    raw = _report_llm_text(llm, _WRITE_SYSTEM, user, max_tokens=section_tokens, temperature=0.2, tier="F") if llm.available else None
    if raw:
        text = finalize_text(raw, material, set(topic["source_tags"]))
        text = _shape_subtopic_structure(text, topic["subtopics"])
        if len(text.split()) >= 60:
            return text, True
    return _fallback_topic(topic, analyses, sources_by_tag, used_text), False
_FINALE_SYSTEM = (
    "You are finishing an explanatory research article written in Feynman's plain style. You get the research "
    "question and the finished topic sections. Write:\n"
    "- short_answer: 2-4 sentences that directly answer the research question in everyday words, using only what the "
    "sections say.\n"
    "- takeaways: 3-6 one-sentence key ideas a reader should remember.\n"
    "- open_questions: 2-5 things that are genuinely still unsolved or unstudied according to the sections (not "
    "'the corpus lacks...' remarks).\n"
    "Facts must carry the [S#] tags already used in the sections; add no new facts or numbers; never write 'we', "
    "'this review' or anything about how the article was made. Return ONLY JSON: "
    "{\"short_answer\": str, \"takeaways\": [str], \"open_questions\": [str]}"
)
def write_finale(question: str, topic_texts: list[str], allowed: set[str], llm: LLMClient) -> tuple[dict, bool]:
    material = "\n\n".join(t[:5200] for t in topic_texts)
    raw = _report_llm_json(llm, _FINALE_SYSTEM, f"RESEARCH QUESTION: {question}\n\nSECTIONS:\n{material}", max_tokens=900, temperature=0.2, tier="F") if llm.available else None
    if not isinstance(raw, dict):
        return {}, False
    def fix(x: Any) -> str:
        return finalize_text(str(x or ""), material, allowed)
    out = {
        "short_answer": fix(raw.get("short_answer")),
        "takeaways": [y for y in (fix(x) for x in (raw.get("takeaways") or [])[:6]) if y],
        "open_questions": [y for y in (fix(x) for x in (raw.get("open_questions") or [])[:5]) if y],
    }
    return out, bool(out["short_answer"] or out["takeaways"])
def _fallback_finale(analyses: dict[str, PaperAnalysis | None]) -> dict:
    answer_bits, gaps = [], []
    for tag, a in analyses.items():
        if a is None:
            continue
        qa = _clean(getattr(a, "question_answer", ""), 380)
        if qa:
            answer_bits.append(_sentence(qa) + f" [{tag}]")
        oq = _clean(getattr(a, "open_questions", "") or getattr(a, "limitations_author_stated", ""), 300)
        if oq:
            gaps.append(_sentence(oq) + f" [{tag}]")
    return {"short_answer": " ".join(answer_bits[:3]), "takeaways": [], "open_questions": gaps[:4]}
def _references(tags_used: list[str], sources_by_tag: dict[str, Source]) -> str:
    lines = []
    for tag in tags_used:
        s = sources_by_tag.get(tag)
        if not s:
            continue
        link = s.url or (f"https://doi.org/{s.doi}" if s.doi else "")
        lines.append(f"- **[{tag}]** {_clean(s.authors, 160)} ({s.year}). {_clean(s.title)}." + (f" {link}" if link else ""))
    return "\n".join(lines)
_EDITOR_SYSTEM = (
    "You are the final scientific editor and publication checker of an explanatory research report. "
    "First check the supplied draft for unsupported claims, invalid numerical comparisons, lost caveats, paper-by-paper repetition, and failure to answer the research question. Then rewrite only where needed to make it read like a coherent, "
    "well-written research article that directly answers the research question. "
    "Do not add knowledge. Every factual statement must be supported by the supplied "
    "draft/source material and retain an appropriate [S#] citation. Remove review-process "
    "narration, audit language, generic filler, repeated explanations and unsupported claims. "
    "Keep useful technical detail. Preserve the report's topic structure. Do not introduce "
    "new sections called methodology, evidence base, audit, quality, corpus or screening. "
    "Return ONLY the revised Markdown body, with no commentary."
)
def _dedupe_report_sections(markdown: str) -> str:
    """Remove accidental repeated paragraphs/sections without doing semantic rewriting."""
    blocks = re.split(r"\n{2,}", markdown or "")
    seen: set[str] = set()
    out: list[str] = []
    for block in blocks:
        norm = re.sub(r"\s+", " ", block).strip().lower()
        if norm and not norm.startswith("#"):
            key = re.sub(r"\[[sS]\d+\]", "", norm)
            key = re.sub(r"[^a-z0-9 ]+", "", key).strip()
            if len(key) > 80 and key in seen:
                continue
            if len(key) > 80:
                seen.add(key)
        out.append(block)
    return "\n\n".join(out)
def _editor_material(question: str, report_type: str, topics: list[str], cards: dict[str, str], synthesis_map: dict | None = None) -> str:
    return (
        f"RESEARCH QUESTION: {question}\nREPORT TYPE: {report_type}\n\n"
        "SOURCE DOSSIERS:\n" + "\n\n".join(cards.values())[:26000] +
        "\n\nDRAFT:\n" + "\n\n".join(topics)[:30000]
    )
def _dedupe_topic_blocks(topic_blocks: list[str]) -> list[str]:
    """Drop near-identical topic bodies produced by an over-eager model."""
    kept: list[str] = []
    signatures: list[str] = []
    for block in topic_blocks:
        body = re.sub(r"^## .*$", "", block, flags=re.MULTILINE)
        body = re.sub(r"\s+", " ", body).strip().lower()
        if len(body) < 80:
            continue
        if any(SequenceMatcher(None, body, prior).ratio() >= 0.88 for prior in signatures):
            continue
        signatures.append(body)
        kept.append(block)
    return kept
def _ensure_source_coverage(
    topic_blocks: list[str],
    analyses: dict[str, PaperAnalysis | None],
    sources_by_tag: dict[str, Source],
) -> list[str]:
    """Make sure every deep-read paper can actually contribute to the narrative.
    This does not dump a paper summary into the report. It adds at most one
    compact, source-cited explanatory paragraph when the model accidentally
    ignores an otherwise selected paper.
    """
    if not topic_blocks:
        return topic_blocks
    text = "\n".join(topic_blocks)
    missing = [tag for tag in sources_by_tag if f"[{tag}]" not in text]
    if not missing:
        return topic_blocks
    out = list(topic_blocks)
    target_index = len(out) - 1
    additions = []
    for tag in missing:
        a = analyses.get(tag)
        s = sources_by_tag.get(tag)
        if not a or not s:
            continue
        idea = _clean(a.core_idea or a.method_summary or a.problem or s.abstract, 500)
        result = _clean(a.key_results or a.established, 420)
        if not idea:
            continue
        paragraph = f"{idea.rstrip('.')} [{tag}]."
        if result and result.lower() not in idea.lower():
            paragraph += f" {result.rstrip('.')} [{tag}]."
        additions.append(paragraph)
    if additions:
        out[target_index] = out[target_index].rstrip() + "\n\n" + "\n\n".join(additions)
    return out
def final_edit_report(
    question: str,
    report_type: str,
    topic_blocks: list[str],
    cards: dict[str, str],
    llm: LLMClient,
    synthesis_map: dict | None = None,
) -> tuple[list[str], bool]:
    """One optional coherence pass after section writing.
    This is deliberately the last LLM operation before rendering. It is not a
    new research pass: the model sees only the already-built sections and the
    bounded paper dossiers. If it fails, the validated original sections remain.
    """
    if not llm.available or not topic_blocks:
        return topic_blocks, False
    material = _editor_material(question, report_type, topic_blocks, cards, synthesis_map)
    raw = _report_llm_text(llm, _EDITOR_SYSTEM, material, max_tokens=min(7000, max(2200, sum(len(x) for x in topic_blocks) // 3)), temperature=0.15, tier="F")
    if not raw:
        return topic_blocks, False
    edited = finalize_text(raw, "\n\n".join(cards.values()), set(re.findall(r"\[(S\d+)\]", "\n".join(cards.values()))))
    edited = _dedupe_report_sections(edited)
    if len(edited.split()) < max(120, int(sum(len(x.split()) for x in topic_blocks) * 0.35)):
        return topic_blocks, False
    headings = re.findall(r"^## ", edited, flags=re.MULTILINE)
    if len(headings) < max(2, len(topic_blocks) - 1):
        return topic_blocks, False
    chunks = re.split(r"(?=^## )", edited, flags=re.MULTILINE)
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks, True
_REPORT_VERIFY_SYSTEM = (
    "You are the final publication gate for an explanatory research report. Inspect the draft against the supplied "
    "research question, report type, cross-paper synthesis memory and paper dossiers. Do not rewrite prose yet. Find "
    "substantive defects that a style editor can miss: the draft does not answer the question, a section is merely a paper "
    "summary, a cross-paper relationship is asserted without support, author interpretation is presented as demonstrated fact, "
    "numbers are compared under incompatible conditions, an important qualifying result is lost, a conclusion is stronger than "
    "the evidence, a genuine limitation is omitted, a key mechanism is named but never explained, a headline result is given without what it demonstrates or does not demonstrate, an analytical inference is presented as something the paper stated, an ablation is used as causal proof it does not provide, or the report type's purpose is not actually satisfied. Ignore harmless "
    "style preferences. Do not invent a correction.\n\n"
    "Return ONLY JSON: {\"approved\": true|false, \"issues\": [{\"severity\": \"critical|major|minor\", "
    "\"section\": str, \"problem\": str, \"why\": str, \"source_tags\": [str], \"required_change\": str}]} . "
    "Approve when the report is scientifically coherent and sufficiently complete for the supplied corpus, even if it is not perfect.\n"
)
_REPAIR_SYSTEM = (
    "You are a senior scientific editor repairing an explanatory research report after a publication-gate review. "
    "Apply ONLY the listed issues. Preserve all correct content, citations, topic structure and useful technical detail. "
    "Use only the supplied paper dossiers and synthesis memory. Do not add facts, numbers, interpretations or sources. "
    "The result must read like a research article that explains the subject, not like an audit of the research process. "
    "Do not introduce methodology, corpus, screening, audit or quality sections. Return ONLY the complete revised Markdown report body.\n"
)
def _publication_gate(question: str, report_type: str, report: str, cards: dict[str, str], synthesis_map: dict, llm: LLMClient) -> tuple[str, bool, bool]:
    if not llm.available or not report.strip():
        return report, False, False
    payload = (
        f"RESEARCH QUESTION: {question}\nREPORT TYPE: {report_type}\n\n"
        "CROSS-PAPER SYNTHESIS MEMORY:\n" + json.dumps(synthesis_map or {}, ensure_ascii=False)[:14000] + "\n\n"
        "PAPER DOSSIERS:\n" + "\n\n".join(cards.values())[:30000] + "\n\nDRAFT:\n" + report[:36000]
    )
    try:
        verdict = _report_llm_json(llm, _REPORT_VERIFY_SYSTEM, payload, max_tokens=2600, temperature=0.0, tier="F")
    except Exception:
        return report, False, False
    if not isinstance(verdict, dict):
        return report, False, False
    approved = bool(verdict.get("approved"))
    issues = [x for x in (verdict.get("issues") or []) if isinstance(x, dict) and x.get("problem")]
    if approved or not issues:
        return report, True, approved
    repair_payload = payload + "\n\nPUBLICATION-GATE ISSUES:\n" + json.dumps(issues[:8], ensure_ascii=False)
    try:
        repaired = _report_llm_text(llm, _REPAIR_SYSTEM, repair_payload, max_tokens=min(8500, max(3000, len(report) // 2)), temperature=0.1, tier="F")
    except Exception:
        repaired = None
    if not repaired:
        return report, True, False
    cleaned = finalize_text(repaired, "\n\n".join(cards.values()), set(re.findall(r"\[(S\d+)\]", "\n".join(cards.values()))))
    if len(cleaned.split()) < max(150, int(len(report.split()) * 0.55)):
        return report, True, False
    return cleaned, True, False
def build_feynman_report(
    question: str,
    report_type: str,
    title: str,
    sources: list[Source],
    analyses: list[PaperAnalysis],
    evidence: list[Evidence] | None,
    llm: LLMClient,
    knowledge_model: dict | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a high-quality report with a parallel council and one decisive join.
    Hot path:
        immutable paper dossiers
          ├─ parallel evidence / methods / mechanisms / inquiry council
          └─ independent cross-paper synthesis
                    ↓
             one chief-researcher join
                    ↓
             parallel thematic writers
                    ↓
          one bounded publication gate
    The key performance property is that the council and independent synthesis are
    one parallel barrier, and there is no serial director -> specialists -> chair ->
    planner chain.  The final writers never reopen PDFs; they consume the shared
    research memory produced by the reading stage.
    """
    report_started = time.perf_counter()
    tags = [f"S{i}" for i in range(1, len(sources) + 1)]
    sources_by_tag = dict(zip(tags, sources))
    analysis_by_source = {a.source_id: a for a in analyses}
    analysis_by_tag = {t: analysis_by_source.get(s.id) for t, s in sources_by_tag.items()}
    n_papers = len(sources_by_tag)
    fast_pipeline = bool(getattr(getattr(llm, "settings", None), "fast_report_pipeline", False))
    ultra_fast = bool(getattr(llm, "ultra_fast_override", False) or getattr(getattr(llm, "settings", None), "ultra_fast_report", False))
    if ultra_fast and n_papers:
        card_limit = int(getattr(getattr(llm, "settings", None), "ultra_fast_card_chars", 13000))
        cards = {t: paper_card(t, s, analysis_by_tag[t], limit=card_limit) for t, s in sources_by_tag.items()}
        t_reason = time.perf_counter()
        council, council_used = run_compact_professor_council(question, report_type, cards, llm)
        reasoning_seconds = round(time.perf_counter() - t_reason, 3)
        topics = council.get("topics") if isinstance(council, dict) else None
        if not isinstance(topics, list) or len(topics) < 2:
            topics = _fallback_plan(question, tags, sources_by_tag).get("topics", [])
        topics = _ensure_gap_topic({"title": council.get("title", "") if isinstance(council, dict) else "", "topics": topics}, report_type, tags).get("topics") or topics
        normalized_fast_topics = []
        for index, item in enumerate(topics, 1):
            if not isinstance(item, dict):
                continue
            topic = dict(item)
            topic["title"] = _clean(topic.get("title") or f"Topic {index}", 110)
            topic["goal"] = _clean(topic.get("goal", ""), 500)
            raw_tags = topic.get("source_tags")
            if not isinstance(raw_tags, list):
                raw_tags = tags
            topic["source_tags"] = [tag for tag in raw_tags if isinstance(tag, str) and tag in tags] or list(tags)
            subs = topic.get("subtopics")
            if not isinstance(subs, list):
                subs = []
            topic["subtopics"] = [
                {"title": _clean(sub.get("title") or f"Key idea {j}", 110),
                 "what_to_explain": _clean(sub.get("what_to_explain", ""), 360)}
                for j, sub in enumerate(subs, 1) if isinstance(sub, dict)
            ][:4]
            if not topic["subtopics"]:
                topic["subtopics"] = [{"title": topic["title"], "what_to_explain": topic["goal"] or topic["title"]}]
            topic["role"] = _normalize_topic_role(topic.get("role"), topic)
            normalized_fast_topics.append(topic)
        topics = normalized_fast_topics or _fallback_plan(question, tags, sources_by_tag)["topics"]
        max_topics = max(2, int(getattr(getattr(llm, "settings", None), "ultra_fast_max_topics", 3)))
        topics = topics[:max_topics]
        writer_synthesis = {**(council if isinstance(council, dict) else {})}
        topic_results: dict[int, tuple[str, bool]] = {}
        workers = min(3, max(1, len(topics)))
        t_writers = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="researchpilot-v84-writer") as pool:
            futures = {pool.submit(write_topic, topic, question, report_type, cards, analysis_by_tag, sources_by_tag, llm, set(), writer_synthesis): i for i, topic in enumerate(topics, 1)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    topic_results[i] = fut.result()
                except Exception:
                    topic_results[i] = (_fallback_topic(topics[i - 1], analysis_by_tag, sources_by_tag, set()), False)
        writer_seconds = round(time.perf_counter() - t_writers, 3)
        blocks = []
        topic_llm = 0
        for i, topic in enumerate(topics, 1):
            body, used = topic_results.get(i, ("", False))
            if body.strip():
                topic_llm += int(used)
                blocks.append(f"## {topic.get('title') or f'Topic {i}'}\n\n{body}")
        blocks = _dedupe_topic_blocks(blocks)
        blocks = _ensure_source_coverage(blocks, analysis_by_tag, sources_by_tag)
        if not any(re.search(r"\b(gap|open|unresolved|bottleneck|limitation)\b", b.split("\n", 1)[0], re.I) for b in blocks):
            gap_topic = next((t for t in topics if isinstance(t, dict) and re.search(r"\b(gap|open|unresolved|bottleneck|limitation)\b", str(t.get("title", "")), re.I)), None)
            if gap_topic is None:
                gap_topic = {"title": "Research gaps, unresolved boundaries and opportunities", "goal": "Derive concrete gaps from evidence boundaries.", "source_tags": tags, "subtopics": []}
            gap_body = _fallback_topic(gap_topic, analysis_by_tag, sources_by_tag, set())
            gap_items = (council.get("gaps") or []) if isinstance(council, dict) else []
            if gap_items:
                gap_body = "\n\n".join([gap_body] + [f"- **{str(g.get('gap') or '').strip()}**: {str(g.get('evidence_basis') or '').strip()} {str(g.get('resolving_test') or '').strip()}" for g in gap_items[:4] if isinstance(g, dict) and str(g.get('gap') or '').strip()])
            blocks.append(f"## {gap_topic.get('title')}\n\n" + gap_body)
        short = _clean(council.get("executive_summary") or council.get("central_answer"), 2200) if isinstance(council, dict) else ""
        takeaways = [_clean(x, 500) for x in (council.get("key_takeaways") or [])[:5] if _clean(x, 500)] if isinstance(council, dict) else []
        open_questions = [_clean(x, 600) for x in (council.get("open_questions") or [])[:5] if _clean(x, 600)] if isinstance(council, dict) else []
        if not short:
            short = _fallback_finale(analysis_by_tag).get("short_answer", "")
        heading = _clean(title, 160) or _clean(council.get("title", ""), 160) or _clean(question, 160)
        parts = [f"# {heading}", "", "## Executive summary", "", short, ""]
        if blocks: parts += ["\n\n".join(blocks), ""]
        if takeaways or open_questions:
            parts += ["## Conclusion", ""]
            if takeaways: parts += ["### Key takeaways", ""] + [f"- {x}" for x in takeaways] + [""]
            if open_questions: parts += ["### What remains open", ""] + [f"- {x}" for x in open_questions] + [""]
        used_tags = []
        for t in re.findall(r"\[(S\d+)\]", "\n".join(parts)):
            if t not in used_tags: used_tags.append(t)
        refs = _references(sorted(set(used_tags), key=lambda x: int(x[1:])) or tags, sources_by_tag)
        if refs: parts += ["## References", "", refs, ""]
        markdown = re.sub(r"(\[S\d+(?:, p\. \d+)?\])(?:\s*\1)+", r"\1", "\n".join(parts))
        markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"
        return markdown, {
            "council_pass_llm": council_used, "chief_synthesis_llm": False, "synthesis_llm": council_used,
            "plan_llm": False, "topics": len(blocks), "topics_llm": topic_llm, "finale_llm": False,
            "editor_llm": False, "publication_gate_llm": False, "publication_gate_approved": None,
            "llm_used": bool(council_used or topic_llm), "parallel_council_roles": 1,
            "architecture": "v84_bounded_compact_council", "bounded_report_calls": 1 + len(topics),
            "timing": {"reasoning_seconds": reasoning_seconds, "writer_seconds": writer_seconds, "report_seconds": round(time.perf_counter() - report_started, 3)},
        }
    perf_card = int(getattr(getattr(llm, "settings", None), "performance_council_card_chars", 3200) or 3200)
    card_limit = perf_card if n_papers > 1 else max(perf_card, 6500)
    cards = {t: paper_card(t, s, analysis_by_tag[t], limit=card_limit) for t, s in sources_by_tag.items()}
    _st = getattr(llm, "settings", None)
    quality_first = bool(getattr(_st, "quality_first", True)) if _st is not None else True
    compact_mode = bool(getattr(_st, "compact_report_reasoning", False)) if _st is not None else False
    if quality_first:
        compact_mode = False
    if compact_mode:
        t_reason = time.perf_counter()
        chief, chief_used = run_compact_professor_council(question, report_type, cards, llm)
        reasoning_seconds = round(time.perf_counter() - t_reason, 3)
        council_views = {"compact_professor_council": chief} if chief else {}
        council_used = chief_used
    else:
        council_views, council_used = run_parallel_council(question, report_type, cards, analysis_by_tag, llm)
        chief, chief_used = run_chief_synthesis(question, report_type, cards, council_views, llm)
    if not isinstance(chief, dict):
        chief = {}
    if not isinstance(council_views, dict):
        council_views = {}
    if knowledge_model:
        chief["shared_research_knowledge"] = knowledge_model
    planned_topics = chief.get("topics") if isinstance(chief, dict) else None
    if not isinstance(planned_topics, list) or len(planned_topics) < 2:
        if compact_mode:
            deterministic_plan = _fallback_plan(question, tags, sources_by_tag)
            chief = {**deterministic_plan, **(chief if isinstance(chief, dict) else {})}
            planned_topics = deterministic_plan.get("topics") or []
            chief["topics"] = planned_topics
            plan_llm = False
        else:
            fallback_synthesis, fallback_used = build_synthesis_map(question, report_type, cards, analysis_by_tag, llm)
            chief = {**fallback_synthesis, **chief}
            plan, plan_llm = plan_topics(question, report_type, tags, "\n".join(_digest(t, s, analysis_by_tag[t]) for t, s in sources_by_tag.items()), sources_by_tag, llm, synthesis_map=chief)
            planned_topics = plan.get("topics") or _fallback_plan(question, tags, sources_by_tag)["topics"]
            chief["topics"] = planned_topics
            chief["title"] = chief.get("title") or plan.get("title")
            chief_used = chief_used or fallback_used
    else:
        plan_llm = chief_used
    plan = _ensure_gap_topic({"title": chief.get("title", ""), "topics": planned_topics}, report_type, tags)
    planned_topics = plan.get("topics") or _fallback_plan(question, tags, sources_by_tag)["topics"]
    _st = getattr(llm, "settings", None)
    if quality_first:
        min_topics = max(4, int(getattr(_st, "quality_report_min_topics", 4) or 4))
        max_topics = max(min_topics, int(getattr(_st, "quality_report_max_topics", 7) or 7))
        max_topics = min(9, max_topics)
    else:
        max_topics = max(2, min(4, int(getattr(_st, "performance_max_report_topics", 3)))) if compact_mode else int(getattr(_st, "v81_max_report_topics", 4))
        max_topics = max(2, max_topics)
    if len(planned_topics) > max_topics:
        gap_idx = next((i for i, t in enumerate(planned_topics) if re.search(r"(gap|open|unresolved|bottleneck|limitation)", str(t.get("title", "")), re.I)), None)
        if gap_idx is not None and gap_idx >= max_topics:
            planned_topics = planned_topics[:max_topics - 1] + [planned_topics[gap_idx]]
        else:
            planned_topics = planned_topics[:max_topics]
    normalized_topics = []
    for index, item in enumerate(planned_topics, 1):
        if not isinstance(item, dict):
            continue
        topic = dict(item)
        topic["title"] = _clean(topic.get("title") or f"Topic {index}", 110)
        topic["goal"] = _clean(topic.get("goal", ""), 500)
        raw_tags = topic.get("source_tags")
        if not isinstance(raw_tags, list):
            raw_tags = tags
        topic["source_tags"] = [tag for tag in raw_tags if isinstance(tag, str) and tag in tags] or list(tags)
        subs = topic.get("subtopics")
        if not isinstance(subs, list):
            subs = []
        topic["subtopics"] = [
            {"title": _clean(sub.get("title") or f"Key idea {j}", 110),
             "what_to_explain": _clean(sub.get("what_to_explain", ""), 360),
             "evidence_focus": sub.get("evidence_focus", [])}
            for j, sub in enumerate(subs, 1) if isinstance(sub, dict)
        ][:4]
        if not topic["subtopics"]:
            topic["subtopics"] = [{"title": topic["title"], "what_to_explain": topic["goal"] or topic["title"]}]
        topic["role"] = _normalize_topic_role(topic.get("role"), topic)
        normalized_topics.append(topic)
    planned_topics = normalized_topics or _fallback_plan(question, tags, sources_by_tag)["topics"]
    writer_synthesis = {
        **(chief if isinstance(chief, dict) else {}),
        "parallel_council": council_views,
    }
    topic_results: dict[int, tuple[str, bool]] = {}
    workers = min(max(1, int(getattr(_st, "llm_max_concurrency", 5))), max(1, len(planned_topics)))
    t_writers = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="researchpilot-report") as pool:
        futures = {
            pool.submit(write_topic, topic, question, report_type, cards, analysis_by_tag, sources_by_tag, llm, set(), writer_synthesis): i
            for i, topic in enumerate(planned_topics, 1)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                topic_results[i] = fut.result()
            except Exception:
                topic_results[i] = (_fallback_topic(planned_topics[i - 1], analysis_by_tag, sources_by_tag, set()), False)
    writer_seconds = round(time.perf_counter() - t_writers, 3)
    topic_blocks: list[str] = []
    topic_llm = 0
    for i, topic in enumerate(planned_topics, 1):
        body, used = topic_results.get(i, ("", False))
        if not body.strip():
            continue
        topic_llm += int(used)
        topic_blocks.append(f"## {topic['title']}\n\n{body}")
    topic_blocks = _dedupe_topic_blocks(topic_blocks)
    topic_blocks = _ensure_source_coverage(topic_blocks, analysis_by_tag, sources_by_tag)
    topic_blocks = [re.sub(r"^## \d+\. ", "## ", block, count=1) for block in topic_blocks]
    finale = {
        "short_answer": _clean(chief.get("executive_summary") or chief.get("central_answer"), 1800),
        "takeaways": [_clean(x, 500) for x in (chief.get("key_takeaways") or [])[:6] if _clean(x, 500)],
        "open_questions": [_clean(x, 600) for x in (chief.get("open_questions") or [])[:6] if _clean(x, 600)],
    }
    if not finale["short_answer"]:
        finale = _fallback_finale(analysis_by_tag)
    heading = _clean(title, 160) or _clean(chief.get("title", ""), 160) or _clean(question, 160)
    parts = [f"# {heading}", ""]
    parts += ["<!-- ## Answer to the research question -->"]
    if finale.get("short_answer"):
        parts += ["## Executive summary", "", finale["short_answer"], ""]
    if topic_blocks:
        parts += ["<!-- ## 1. -->", "\n\n".join(topic_blocks), ""]
    if finale.get("takeaways") or finale.get("open_questions"):
        parts += ["## Conclusion", ""]
        if finale.get("takeaways"):
            parts += ["### Key takeaways", ""] + [f"- {x}" for x in finale["takeaways"]] + [""]
        if finale.get("open_questions"):
            parts += ["### What remains open", ""] + [f"- {x}" for x in finale["open_questions"]] + [""]
    used_tags = []
    for t in re.findall(r"\[(S\d+)\]", "\n".join(parts)):
        if t not in used_tags:
            used_tags.append(t)
    refs = _references(sorted(set(used_tags), key=lambda x: int(x[1:])) or tags, sources_by_tag)
    if refs:
        parts += ["## References", "", refs, ""]
    markdown = re.sub(r"(\[S\d+(?:, p\. \d+)?\])(?:\s*\1)+", r"\1", "\n".join(parts))
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"
    if compact_mode and (type(llm) is LLMClient or bool(getattr(llm, "force_fast_path", False))):
        publication_gate_ran, publication_gate_approved = False, None
    else:
        markdown, publication_gate_ran, publication_gate_approved = _publication_gate(
            question, report_type, markdown, cards, writer_synthesis, llm
        )
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip() + "\n"
    return markdown, {
        "council_pass_llm": council_used,
        "chief_synthesis_llm": chief_used,
        "synthesis_llm": bool(council_views.get("independent_synthesis")) if isinstance(council_views, dict) else False,
        "plan_llm": plan_llm,
        "topics": len(topic_blocks),
        "topics_llm": topic_llm,
        "finale_llm": False,
        "editor_llm": False,
        "publication_gate_llm": publication_gate_ran,
        "publication_gate_approved": publication_gate_approved if publication_gate_ran else None,
        "llm_used": bool(council_used or chief_used or topic_llm or publication_gate_ran),
        "parallel_council_roles": len(council_views),
        "architecture": "compact_professor_fast_v1" if compact_mode else "parallel_professor_council_v2",
        "bounded_report_calls": (1 + len(planned_topics)) if compact_mode else (7 + len(planned_topics)),
        "timing": {"reasoning_seconds": round(locals().get("reasoning_seconds", 0.0), 3), "writer_seconds": round(locals().get("writer_seconds", 0.0), 3), "report_seconds": round(time.perf_counter() - report_started, 3)},
    }
