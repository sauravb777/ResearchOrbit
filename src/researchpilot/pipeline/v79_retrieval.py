from __future__ import annotations
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from ..llm import LLMClient
from ..schemas import Source
@dataclass
class Facet:
    name: str
    why_needed: str
    synonyms: list[str] = field(default_factory=list)
    importance: str = "supporting"
@dataclass
class CandidateDecision:
    source_id: str
    directness: float = 0.0
    answers_question: float = 0.0
    evidence_type: str = "context"
    domain_match: str = "far"
    reasons: list[str] = field(default_factory=list)
    facets: list[str] = field(default_factory=list)
    verified: bool = False
    usable: str = "no"
@dataclass
class RetrievalV79Result:
    selected: list[Source]
    candidates: list[Source]
    facets: list[Facet]
    coverage: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    warnings: list[str]
    llm_used: bool
SYSTEM_FACETS = """You are the retrieval architect for an academic research agent.
Given a research question and report type, decompose the information need into 4-10 answer facets.
A facet is a distinct piece of evidence the final report needs, not a generic topic label.
For each facet give why_needed, 3-6 synonyms in authors' vocabulary, and importance core/supporting/context.
Also return entities, methods, sub_questions, exclusions, likely_arxiv_categories, and report_type_needs.
Do not answer the question. JSON only."""
SYSTEM_TRIAGE = """You are an extremely strict academic paper triage judge.
Use ONLY the question, facet definitions, title and abstract supplied. Do not reward shared keywords alone.
For every paper score directness 0-3, answers_question 0-3, domain_match same/adjacent/far, identify evidence_type,
and list the facets actually served. A paper that merely mentions a topic should be rejected.
Return JSON only."""
SYSTEM_RERANK = """You are a listwise scholarly retrieval judge. Rank the supplied papers for how much each would improve
an expert answer to the exact question for the requested report type, considering what the other papers already cover.
Do not rank by prestige. Prefer papers that add missing answer facets, direct evidence, decisive mechanisms,
contrasting findings, benchmarks or boundary conditions. Return an ordered list of source ids with a short reason."""
SYSTEM_VERIFY = """You are verifying whether a paper actually does what its abstract suggests.
Read ONLY the supplied introduction and conclusion. State what it actually did, what it did NOT do,
which requested facets it truly serves, and whether it is usable as evidence for the report type.
Never infer beyond the supplied text. JSON only."""
def _json(llm: LLMClient, system: str, payload: Any, max_tokens: int = 3000, tier: str = "M", priority: str = "retrieval"):
    if not llm.available:
        return None
    try:
        return llm.complete_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=max_tokens, tier=tier, priority=priority)
    except Exception:
        return None
def _workers(llm: LLMClient, n: int) -> int:
    st = getattr(llm, "settings", None)
    cap = int(getattr(st, "v80_retrieval_llm_workers", 4) or 4)
    cap = min(cap, int(getattr(st, "llm_max_concurrency", cap) or cap))
    return max(1, min(cap, n))
def _pmap(fn, items: list, workers: int) -> list:
    """Order-preserving bounded parallel map (independent LLM shards)."""
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="researchpilot-retrieval") as pool:
        return list(pool.map(fn, items))
def build_facets(question: str, report_type: str, llm: LLMClient) -> tuple[dict[str, Any], bool]:
    data = _json(llm, SYSTEM_FACETS, {"question": question, "report_type": report_type}, 2400, tier="S", priority="retrieval")
    if isinstance(data, dict) and isinstance(data.get("facets"), list):
        return data, True
    return {
        "facets": [
            {"name": "direct answer", "why_needed": "Evidence that directly answers the research question", "synonyms": [], "importance": "core"},
            {"name": "methods and mechanisms", "why_needed": "How relevant approaches work", "synonyms": [], "importance": "core"},
            {"name": "empirical evidence", "why_needed": "Results, datasets, metrics and conditions", "synonyms": [], "importance": "core"},
            {"name": "limitations and boundaries", "why_needed": "Conditions where conclusions do not transfer", "synonyms": [], "importance": "supporting"},
            {"name": "contrasts and gaps", "why_needed": "Disagreement, missing evidence and open questions", "synonyms": [], "importance": "supporting"},
        ],
        "sub_questions": [], "entities": [], "methods": [], "exclusions": [], "likely_arxiv_categories": [], "report_type_needs": report_type,
    }, False
def build_query_families(question: str, plan: dict[str, Any]) -> list[str]:
    queries: list[str] = [question.strip()]
    facets = plan.get("facets") or []
    for f in facets:
        if not isinstance(f, dict):
            continue
        name = str(f.get("name") or "").strip()
        synonyms = [str(x).strip() for x in f.get("synonyms", []) if str(x).strip()][:5]
        cats = [str(x).strip() for x in plan.get("likely_arxiv_categories", []) if str(x).strip()][:2]
        if name:
            terms = " OR ".join([name] + synonyms)
            if cats:
                queries.append(f"cat:{cats[0]} AND ({terms})")
            queries.append(f"all:({terms}) AND all:{question.strip()[:180]}")
            queries.append(" ".join([name] + synonyms[:2] + [question.strip()[:180]]))
            queries.append(f"ti:survey OR ti:review OR ti:\"state of the art\" AND ({name})")
    for sub in plan.get("sub_questions", []) or []:
        q = str(sub).strip()
        if q:
            queries.append(q)
    out, seen = [], set()
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()[:420]
        if not q or q.lower() in seen:
            continue
        seen.add(q.lower()); out.append(q)
    return out[:24]
def triage(question: str, facets: list[dict], sources: list[Source], llm: LLMClient) -> tuple[dict[str, CandidateDecision], bool]:
    out: dict[str, CandidateDecision] = {}
    batches = [sources[i:i+10] for i in range(0, len(sources), 10)]
    def run_batch(batch: list[Source]):
        data = _json(llm, SYSTEM_TRIAGE, {
            "question": question, "facets": facets,
            "papers": [{"source_id": s.id, "title": s.title, "abstract": (s.abstract or "")[:6000]} for s in batch]
        }, 3200, tier="F", priority="retrieval")
        return batch, data
    used = False
    for batch, data in _pmap(run_batch, batches, _workers(llm, len(batches))):
        if not isinstance(data, dict):
            continue
        items = data.get("papers") or data.get("assessments") or []
        if not isinstance(items, list):
            continue
        ids = {s.id for s in batch}
        for x in items:
            if not isinstance(x, dict) or x.get("source_id") not in ids:
                continue
            sid = x["source_id"]
            try:
                out[sid] = CandidateDecision(
                    source_id=sid,
                    directness=max(0,min(3,float(x.get("directness",0)))),
                    answers_question=max(0,min(3,float(x.get("answers_question",0)))),
                    evidence_type=str(x.get("evidence_type") or "context"),
                    domain_match=str(x.get("domain_match") or "far"),
                    reasons=[str(r) for r in x.get("reasons",[])][:4],
                    facets=[str(r) for r in x.get("facets",[])][:10],
                )
            except Exception:
                continue
        used = used or bool(items)
    return out, used
def listwise_rerank(question: str, report_type: str, sources: list[Source], triage_map: dict[str, CandidateDecision], llm: LLMClient) -> tuple[list[str], bool]:
    if not sources:
        return [], False
    batches = [sources[i:i+10] for i in range(0, len(sources), 10)]
    def run_batch(batch: list[Source]):
        data = _json(llm, SYSTEM_RERANK, {
            "question": question, "report_type": report_type,
            "papers": [{"source_id": s.id, "title": s.title, "abstract": (s.abstract or "")[:4500],
                         "triage": triage_map.get(s.id).__dict__ if s.id in triage_map else {}} for s in batch]
        }, 2600, tier="S", priority="retrieval")
        return batch, data
    ordered: list[str] = []
    used = False
    for batch, data in _pmap(run_batch, batches, _workers(llm, len(batches))):
        if isinstance(data, dict):
            ranks = data.get("ranking") or data.get("ranked") or []
            if isinstance(ranks, list):
                ids = {s.id for s in batch}
                for x in ranks:
                    sid = x.get("source_id") if isinstance(x, dict) else x
                    if sid in ids and sid not in ordered:
                        ordered.append(sid)
                used = used or bool(ranks)
    return ordered, used
def verify_finalists(question: str, report_type: str, facets: list[dict], sources: list[Source], llm: LLMClient) -> dict[str, dict[str, Any]]:
    todo = [s for s in sources if s.full_text]
    def run_one(s: Source):
        text = s.full_text or ""
        parts = re.split(r"(?im)^\[PAGE\s+\d+\]\s*$", text)
        intro = "\n".join(parts[:5])[:26000]
        conclusion = "\n".join(parts[-5:])[-26000:]
        data = _json(llm, SYSTEM_VERIFY, {
            "question": question, "report_type": report_type, "facets": facets,
            "source_id": s.id, "title": s.title, "introduction": intro, "conclusion": conclusion
        }, 2200, tier="M", priority="retrieval")
        return s, data
    out: dict[str, dict[str, Any]] = {}
    for s, data in _pmap(run_one, todo, _workers(llm, len(todo))):
        if isinstance(data, dict):
            data["verified"] = True
            out[s.id] = data
    return out
def select_with_coverage(question: str, report_type: str, sources: list[Source], facets: list[dict], triage_map: dict[str, CandidateDecision], ranks: list[str], verification: dict[str, dict[str, Any]], max_papers: int = 12) -> tuple[list[Source], list[dict[str, Any]]]:
    by_id = {s.id:s for s in sources}
    rank_pos = {sid:i for i,sid in enumerate(ranks)}
    def score(s: Source) -> float:
        t = triage_map.get(s.id, CandidateDecision(s.id))
        v = verification.get(s.id, {})
        usable = str(v.get("usable_for_report", v.get("usable", "yes"))).lower()
        verify_bonus = 0.35 if usable in {"yes","true","mostly"} else 0.0
        return 0.45*(t.answers_question/3) + 0.35*(t.directness/3) + verify_bonus + 0.05*(1/(1+rank_pos.get(s.id,999)))
    remaining = [s for s in sources if s.id in triage_map and triage_map[s.id].answers_question >= 1]
    remaining.sort(key=lambda s:(-score(s), rank_pos.get(s.id,999), s.title.lower()))
    selected: list[Source] = []
    covered: set[str] = set()
    coverage: list[dict[str,Any]] = []
    role_counts = {"core":0,"supporting":0,"context":0}
    target = min(max_papers, max(1, len(remaining)))
    while remaining and len(selected) < target:
        best = None; best_gain = -1e9
        for s in remaining:
            d = triage_map.get(s.id, CandidateDecision(s.id))
            facets_hit = set(d.facets)
            gain = score(s) + 0.30*len(facets_hit-covered)
            if d.domain_match == "far": gain -= 0.5
            if gain > best_gain:
                best_gain, best = gain, s
        if best is None: break
        remaining.remove(best); selected.append(best)
        d = triage_map.get(best.id, CandidateDecision(best.id))
        new = [f for f in d.facets if f not in covered]
        covered.update(new)
        core = any((str(f.get("name")) in new and str(f.get("importance"))=="core") for f in facets if isinstance(f,dict))
        role = "core" if core or d.answers_question >= 2.5 else "supporting"
        if role_counts[role] == 0 and role != "core" and any(str(f.get("importance"))=="core" for f in facets if isinstance(f,dict)):
            role = "core"
        role_counts[role] += 1
        best.role = "deep_read"
        best.score = round(score(best),4)
        best.relevance_score = round(best.score*100,2)
        best.relevance_reasons = list(dict.fromkeys(d.reasons + [f"Facets added: {', '.join(new[:4])}" if new else "Adds corroborating evidence"]))[:6]
    for f in facets:
        name = str(f.get("name") if isinstance(f,dict) else f)
        hit = [s.id for s in selected if name in (triage_map.get(s.id, CandidateDecision(s.id)).facets)]
        coverage.append({"facet":name,"importance":f.get("importance","supporting") if isinstance(f,dict) else "supporting","covered":bool(hit),"source_ids":hit})
    return selected, coverage
def extract_arxiv_reference_ids(text: str, limit: int = 40) -> list[str]:
    """Mine explicit arXiv identifiers from already-read reference lists."""
    ids = re.findall(r"(?:arXiv\s*:\s*)?((?:\d{4}\.\d{4,5})(?:v\d+)?)", text or "", flags=re.I)
    return list(dict.fromkeys(ids))[:limit]
def run_v79_retrieval(question: str, report_type: str, candidates: list[Source], llm: LLMClient, max_papers: int = 12, search_fn=None) -> RetrievalV79Result:
    plan, intent_used = build_facets(question, report_type, llm)
    facets = [Facet(str(f.get("name","")), str(f.get("why_needed","")), [str(x) for x in f.get("synonyms",[])], str(f.get("importance","supporting"))) for f in plan.get("facets",[]) if isinstance(f,dict)]
    trace = [{"stage":"R0","facets":len(facets),"llm_used":intent_used}]
    _ = build_query_families(question, plan)
    triage_map, triage_used = triage(question, [f.__dict__ for f in facets], candidates, llm)
    trace.append({"stage":"R2","triaged":len(triage_map),"llm_used":triage_used})
    triage_sources = [s for s in candidates if s.id in triage_map and triage_map[s.id].answers_question >= 1]
    triage_shortlist = sorted(
        triage_sources,
        key=lambda s: (
            -(triage_map.get(s.id, CandidateDecision(s.id)).answers_question),
            -(triage_map.get(s.id, CandidateDecision(s.id)).directness),
            s.title.lower(),
        ),
    )[:min(24, len(triage_sources))]
    from concurrent.futures import ThreadPoolExecutor, as_completed
    ranks = []
    rank_used = False
    verification = {}
    verify_thread = None
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="researchpilot-retrieval-join") as pool:
        rank_future = pool.submit(listwise_rerank, question, report_type, triage_sources, triage_map, llm)
        verify_future = pool.submit(verify_finalists, question, report_type, [f.__dict__ for f in facets], triage_shortlist, llm)
        for future in as_completed((rank_future, verify_future)):
            if future is rank_future:
                ranks, rank_used = future.result()
            else:
                verification = future.result()
    trace.append({"stage":"R3-R4_parallel","ranked":len(ranks),"verified":len(verification),"llm_used":bool(rank_used or verification)})
    finalists = sorted(triage_sources, key=lambda s: ranks.index(s.id) if s.id in ranks else 999)[:min(20,len(triage_sources))]
    late_finalists = [s for s in finalists if s.id not in verification]
    if late_finalists:
        verification.update(verify_finalists(question, report_type, [f.__dict__ for f in facets], late_finalists, llm))
    trace.append({"stage":"R4_finalize","verified":len(verification),"late_verified":len(late_finalists)})
    selected, coverage = select_with_coverage(question, report_type, finalists, [f.__dict__ for f in facets], triage_map, ranks, verification, max_papers=max_papers)
    uncovered_core = [c["facet"] for c in coverage if c["importance"]=="core" and not c["covered"]]
    warnings = []
    for round_no in range(1, 3):
        if not uncovered_core or search_fn is None:
            break
        queries=[]
        for f in facets:
            if f.name in uncovered_core:
                queries.append(" ".join([f.name] + f.synonyms[:3] + [question[:180]]))
        ref_ids=[]
        for src in selected[:3]:
            ref_ids.extend(extract_arxiv_reference_ids(src.full_text))
        queries.extend(["id:"+rid for rid in list(dict.fromkeys(ref_ids))[:12]])
        queries=list(dict.fromkeys(q[:420] for q in queries if q))[:8]
        if not queries: break
        try:
            extra=search_fn(queries) or []
        except Exception:
            extra=[]
        known={s.dedupe_key() for s in candidates}
        added=[x for x in extra if x.dedupe_key() not in known]
        candidates.extend(added)
        trace.append({"stage":f"R6_round_{round_no}","queries":len(queries),"added_candidates":len(added)})
        if added:
            triage_map2, triage_used2 = triage(question, [f.__dict__ for f in facets], candidates, llm)
            triage_map.update(triage_map2); triage_used = triage_used or triage_used2
            eligible=[x for x in candidates if x.id in triage_map and triage_map[x.id].answers_question >= 1 and x.full_text_complete]
            ranks2, rank_used2=listwise_rerank(question,report_type,eligible,triage_map,llm)
            ranks=ranks2 or ranks; rank_used=rank_used or rank_used2
            finalists=sorted(eligible,key=lambda x:ranks.index(x.id) if x.id in ranks else 999)[:20]
            verification=verify_finalists(question,report_type,[f.__dict__ for f in facets],finalists,llm)
            selected,coverage=select_with_coverage(question,report_type,finalists,[f.__dict__ for f in facets],triage_map,ranks,verification,max_papers=max_papers)
            uncovered_core=[c["facet"] for c in coverage if c["importance"]=="core" and not c["covered"]]
        else:
            break
    if uncovered_core:
        warnings.append("Not covered by arXiv corpus: " + ", ".join(uncovered_core))
    trace.append({"stage":"R5-R6","selected":len(selected),"uncovered_core":uncovered_core})
    return RetrievalV79Result(selected=selected,candidates=candidates,facets=facets,coverage=coverage,trace=trace,warnings=warnings,llm_used=any([intent_used,triage_used,rank_used,bool(verification)]))
