from __future__ import annotations
from dataclasses import dataclass
import json
import re
from typing import Iterable
from ..llm import LLMClient
from ..schemas import Source, PaperAnalysis
_STOP = {
    "what", "which", "where", "when", "does", "do", "how", "can", "could", "would",
    "should", "the", "and", "for", "from", "with", "into", "that", "this", "these",
    "those", "their", "about", "using", "use", "used", "research", "study", "studies",
    "paper", "papers", "better", "understand", "understanding", "based", "toward", "towards",
    "improve", "improved", "improving", "monitor", "monitored", "monitoring",
    "predict", "predicted", "predicting", "prediction", "predictions",
}
@dataclass
class CorpusAlignment:
    source_id: str
    score: float
    aligned: bool
    reason: str = ""
    method: str = "deterministic"
    relationship: str = ""
def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9][a-z0-9-]{2,}", (text or "").lower())
    return {w for w in words if w not in _STOP}
def _normal_forms(tokens: set[str]) -> set[str]:
    """Conservative dependency-free morphology normalization for topic safety."""
    out = set(tokens)
    for token in list(tokens):
        for suffix in ("ing", "ed", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 5:
                out.add(token[:-len(suffix)])
                break
        if token.endswith("tion") and len(token) > 7:
            out.add(token[:-4])
        if token.endswith("tive") and len(token) > 7:
            out.add(token[:-4])
    return out
def _topic_signals(question: str, source: Source, research_blueprint=None) -> tuple[float, float, float]:
    """Return question, title and blueprint overlap used by the safety gate.
    This is deliberately separate from ranking: semantic retrieval can surface
    paraphrases, but a paper with zero topic signal must not enter an automatic
    corpus solely because recency, venue metadata or an optimistic model score
    pushed it upward.
    """
    q = _normal_forms(_tokens(question))
    if not q:
        return 0.0, 0.0, 0.0
    title = _normal_forms(_tokens(source.title))
    abstract = _normal_forms(_tokens(source.abstract))
    body = title | abstract
    direct = len(q & body) / max(1, len(q))
    title_overlap = len(q & title) / max(1, len(q))
    blueprint_terms: set[str] = set()
    if research_blueprint is not None:
        for field in ("retrieval_targets",):
            for value in (getattr(research_blueprint, field, []) or []):
                blueprint_terms |= _normal_forms(_tokens(str(value)))
    blueprint_overlap = len(blueprint_terms & body) / max(1, len(blueprint_terms)) if blueprint_terms else 0.0
    return direct, title_overlap, blueprint_overlap
def _lexical_alignment(question: str, source: Source) -> float:
    direct, title_overlap, _ = _topic_signals(question, source)
    return min(1.0, 0.65 * direct + 0.35 * title_overlap)
def _batch(items: list, size: int = 8) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
def rescue_source_corpus(
    question: str,
    sources: list[Source],
    alignments: list[CorpusAlignment],
    *,
    minimum_anchor_score: float = 0.39,
    minimum_overlap: float = 0.055,
) -> tuple[list[Source], list[str]]:
    """Recover a useful pre-read corpus when semantic validation false-negatives.
    Retrieval score + verified full text are independent signals already established
    before this validator runs.  The validator therefore must not be allowed to
    erase the entire evidence set because one LLM classification pass was noisy.
    Catastrophic lexical mismatches remain excluded.
    """
    amap = {a.source_id: a for a in alignments}
    q = _tokens(question)
    kept: list[Source] = []
    rescued: list[str] = []
    for source in sources:
        a = amap.get(source.id)
        if a is None or a.aligned:
            kept.append(source)
            continue
        direct, title_overlap, blueprint_overlap = _topic_signals(question, source)
        overlap = direct
        ranking = max(
            float(getattr(source, "score", 0.0) or 0.0),
            float(getattr(source, "relevance_score", 0.0) or 0.0) / 100.0,
        )
        zero_topic_signal = direct < 0.025 and title_overlap < 0.025 and blueprint_overlap < 0.025
        catastrophic = zero_topic_signal or (overlap < 0.025 and a.score < 0.30 and ranking < minimum_anchor_score)
        readable = bool(source.full_text.strip()) and bool(source.full_text_parsed) and bool(getattr(source, "full_text_complete", True))
        if readable and not catastrophic and (ranking >= minimum_anchor_score or overlap >= minimum_overlap):
            kept.append(source)
            rescued.append(source.id)
    return kept, rescued
def validate_sources(
    question: str,
    sources: list[Source],
    llm: LLMClient | None = None,
    *,
    threshold: float = 0.52,
    research_blueprint=None,
) -> list[CorpusAlignment]:
    """Run the corpus safety gate.
    This is intentionally NOT a second ranking system. Ranking answers
    "which eligible papers are most useful?" The integrity gate answers only
    "is this paper genuinely related enough to enter the research corpus?"
    Supporting evidence is valid. A paper does not need to study the exact
    geographic population or reproduce the exact phenomenon if its method,
    mechanism, measurements, or evidence materially answers a sub-question in
    the research blueprint. Truly adjacent or unrelated papers are rejected.
    """
    if not sources:
        return []
    results: dict[str, CorpusAlignment] = {}
    llm_ok = bool(llm and llm.available and callable(getattr(llm, "complete_json", None)))
    if llm_ok:
        system = """You are the corpus safety judge for a scholarly research agent.
Your job is NOT to rank papers and NOT to demand that every paper exactly match the
full wording of the research question.
Classify each paper's relationship to the research question:
- core: directly studies the requested phenomenon/problem or a central part of it.
- supporting: materially contributes a method, measurement, mechanism, dataset, evidence,
  or sub-question needed to answer the research question, even if the paper studies a
  different but scientifically transferable setting.
- adjacent: related subject matter but its contribution would mostly require speculation
  or a large conceptual jump to use in this research.
- unrelated: does not materially help answer the question.
Both CORE and SUPPORTING are aligned and may enter a scholarly evidence corpus.
ADJACENT and UNRELATED are not aligned.
For example, a paper on glacier icequake monitoring can be SUPPORTING for a question
about glacier-collapse hazard prediction if its monitoring/prediction evidence is
materially useful. A paper on open-pit mining slope instability is not automatically
supporting merely because it concerns slope failure; the connection must be substantive.
Do not use prestige, recency, citation count, venue, or geography alone.
Do not reject a useful methodological paper merely because it is not in the exact study
region. Do not infer relevance from a title alone when the abstract contradicts it.
Return JSON only:
{"items":[
  {"source_id":"...", "alignment":0.0, "relationship":"core|supporting|adjacent|unrelated",
   "aligned":true|false, "reason":"concise evidence-based reason"}
]}
Use alignment as a 0..1 semantic relatedness score. Around 0.8-1.0 is direct/core,
around 0.55-0.8 is substantial supporting relevance, around 0.3-0.55 is weak/adjacent,
below 0.3 is generally unrelated."""
        for batch in _batch(sources):
            payload = [{
                "source_id": s.id,
                "title": s.title,
                "abstract": (s.abstract or "")[:6500],
            } for s in batch]
            blueprint_payload = (
                research_blueprint.model_dump()
                if hasattr(research_blueprint, "model_dump")
                else (research_blueprint or {})
            )
            try:
                data = llm.complete_json(
                    system,
                    json.dumps({
                        "question": question,
                        "research_blueprint": blueprint_payload,
                        "papers": payload,
                    }, ensure_ascii=False),
                    max_tokens=2800,
                )
            except Exception:
                data = None
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                allowed = {s.id for s in batch}
                for item in data["items"]:
                    if not isinstance(item, dict) or item.get("source_id") not in allowed:
                        continue
                    try:
                        score = max(0.0, min(1.0, float(item.get("alignment", 0.0))))
                    except Exception:
                        continue
                    relationship = str(item.get("relationship", "")).strip().lower()
                    if relationship not in {"core", "supporting", "adjacent", "unrelated"}:
                        relationship = "core" if score >= 0.72 else ("supporting" if score >= 0.45 else "adjacent")
                    aligned = relationship in {"core", "supporting"} and score >= 0.45
                    if relationship == "core":
                        aligned = score >= 0.52
                    results[item["source_id"]] = CorpusAlignment(
                        source_id=item["source_id"],
                        score=score,
                        aligned=aligned,
                        reason=str(item.get("reason", ""))[:500],
                        method="llm",
                        relationship=relationship,
                    )
    final: list[CorpusAlignment] = []
    for source in sources:
        lexical = _lexical_alignment(question, source)
        direct, title_overlap, blueprint_overlap = _topic_signals(question, source, research_blueprint)
        zero_topic_signal = direct < 0.025 and title_overlap < 0.025 and blueprint_overlap < 0.025
        candidate = results.get(source.id)
        if candidate:
            semantic_escape = (candidate.relationship == "supporting" and candidate.score >= 0.55)
            catastrophic = zero_topic_signal and not semantic_escape
            aligned = candidate.aligned and not catastrophic
            reason = candidate.reason
            if catastrophic:
                reason = (reason + " Deterministic topic-consistency guard rejected this paper: "
                          "its title/abstract has no meaningful overlap with the question or research blueprint.").strip()
            final.append(CorpusAlignment(
                source.id, candidate.score, aligned, reason,
                "llm+topic_guard" if candidate.method == "llm" else candidate.method,
                candidate.relationship,
            ))
        else:
            aligned = not zero_topic_signal
            reason = "Conservative deterministic coherence check used because semantic validation was unavailable."
            if zero_topic_signal:
                reason += " Deterministic topic-consistency guard rejected this paper."
            final.append(CorpusAlignment(source.id, lexical, aligned, reason, "deterministic"))
    return final
def rescue_analysis_corpus(
    question: str,
    sources: list[Source],
    analyses: list[PaperAnalysis],
    rejected_ids: set[str],
    *,
    minimum_anchor_score: float = 0.39,
    minimum_overlap: float = 0.055,
) -> tuple[list[PaperAnalysis], list[Source], list[str]]:
    """Recover from false-negative post-reading alignment judgments.
    The post-reading semantic judge is a *safety signal*, not a second corpus
    selector.  A model can call a useful methodological paper ``adjacent`` even
    though the retrieval stage already established strong relevance and the
    deep reader produced question-relevant evidence.  We therefore only remove
    a rejected dossier when there is no independent signal that it contributes.
    This preserves the hard safety boundary against obviously unrelated papers
    while preventing a single LLM adjudication failure from destroying an
    otherwise valid research run.
    """
    source_map = {s.id: s for s in sources}
    kept: list[PaperAnalysis] = []
    kept_sources: list[Source] = []
    rescued: list[str] = []
    q = _tokens(question)
    for analysis in analyses:
        source = source_map.get(analysis.source_id)
        if source is None:
            continue
        if analysis.source_id not in rejected_ids:
            kept.append(analysis)
            kept_sources.append(source)
            continue
        dossier = " ".join(str(x) for x in [
            source.title, source.abstract, analysis.problem,
            analysis.research_objective, analysis.core_idea,
            analysis.method_summary, analysis.question_answer,
            analysis.established, analysis.key_results, analysis.research_implications,
        ])
        d = _tokens(dossier)
        overlap = len(q & d) / max(1, len(q))
        ranking = max(float(getattr(source, "score", 0.0) or 0.0),
                      float(getattr(source, "relevance_score", 0.0) or 0.0) / 100.0)
        evidence_signal = bool(analysis.evidence_inventory or analysis.key_evidence_chain or analysis.field_evidence)
        admit = (ranking >= minimum_anchor_score and evidence_signal) or (overlap >= minimum_overlap and evidence_signal)
        if admit:
            kept.append(analysis)
            kept_sources.append(source)
            rescued.append(analysis.source_id)
    return kept, kept_sources, rescued
def validate_analyses(
    question: str,
    sources: list[Source],
    analyses: list[PaperAnalysis],
    llm: LLMClient | None = None,
    *,
    threshold: float = 0.52,
) -> list[CorpusAlignment]:
    """Validate deep-read dossiers as a safety check, not a second ranking pass."""
    source_map = {s.id: s for s in sources}
    usable = [a for a in analyses if a.source_id in source_map]
    if not usable:
        return []
    results: dict[str, CorpusAlignment] = {}
    if llm and llm.available and callable(getattr(llm, "complete_json", None)):
        system = """You are auditing whether a deep-read paper belongs in a scholarly synthesis.
Classify its relationship to the user's research question as core, supporting, adjacent, or unrelated.
Core and supporting are acceptable. Supporting includes a useful method, measurement, mechanism,
dataset, or sub-question that materially contributes even when the paper studies another setting.
Adjacent and unrelated should be rejected. Do not turn this into a ranking exercise.
Return JSON only:
{"items":[{"source_id":"...","alignment":0.0,
"relationship":"core|supporting|adjacent|unrelated",
"aligned":true|false,"reason":"..."}]}"""
        for batch in _batch(usable, 6):
            payload = [{
                "source_id": a.source_id,
                "problem": a.problem,
                "objective": a.research_objective,
                "core_idea": a.core_idea,
                "question_answer": a.question_answer,
                "established": a.established,
                "results": a.key_results,
            } for a in batch]
            try:
                data = llm.complete_json(
                    system,
                    json.dumps({"question": question, "papers": payload}, ensure_ascii=False),
                    max_tokens=2400,
                )
            except Exception:
                data = None
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                allowed = {a.source_id for a in batch}
                for item in data["items"]:
                    if not isinstance(item, dict) or item.get("source_id") not in allowed:
                        continue
                    try:
                        score = max(0.0, min(1.0, float(item.get("alignment", 0.0))))
                    except Exception:
                        continue
                    relationship = str(item.get("relationship", "")).strip().lower()
                    if relationship not in {"core", "supporting", "adjacent", "unrelated"}:
                        relationship = "core" if score >= 0.72 else ("supporting" if score >= 0.45 else "adjacent")
                    aligned = relationship in {"core", "supporting"} and score >= 0.45
                    if relationship == "core":
                        aligned = score >= threshold
                    results[item["source_id"]] = CorpusAlignment(
                        item["source_id"], score, aligned,
                        str(item.get("reason", ""))[:500], "llm"
                    )
    out = []
    for a in usable:
        c = results.get(a.source_id)
        if c:
            out.append(c)
        else:
            source = source_map[a.source_id]
            q = _tokens(question)
            d = _tokens(" ".join([
                a.problem, a.research_objective, a.core_idea,
                a.question_answer, a.established, source.title, source.abstract
            ]))
            overlap = len(q & d) / max(1, len(q))
            out.append(CorpusAlignment(
                a.source_id, overlap, overlap >= 0.075,
                "Dossier topic overlap used because semantic validation was unavailable.",
                "deterministic"
            ))
    return out
