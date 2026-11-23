from __future__ import annotations
import re
from dataclasses import dataclass, field
from ..schemas import FollowupQA, PaperAnalysis, Source
@dataclass(frozen=True)
class ChatDecision:
    action: str = "answer"
    effective_question: str = ""
    intent: str = "research_question"
    context_mode: str = "general"
    confidence: float = 1.0
    clarification_question: str = ""
    clarification_options: list[str] = field(default_factory=list)
    reason: str = ""
def _recent_source_ids(prior: list[FollowupQA]) -> list[str]:
    for turn in reversed(prior or []):
        if turn.source_ids:
            return list(dict.fromkeys(turn.source_ids))
    return []
def _last_clarification(prior: list[FollowupQA]) -> FollowupQA | None:
    if prior and getattr(prior[-1], "needs_clarification", False):
        return prior[-1]
    return None
def _target_count(question: str, sources: list[Source]) -> int:
    q = question.lower()
    count = 0
    if re.search(r"\b(?:paper\s+)?(?:first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)\b", q):
        count += 1
    if re.search(r"\b\[s\d+\]\b|\bs\d+\b", q, re.I):
        count += 1
    if re.search(r"\b(all|these|both|each)\s+papers?\b", q):
        count += 2
    return count
def _looks_like_ambiguous_reference(q: str) -> bool:
    return bool(re.search(
        r"\b(?:this|that|it|they|them|those|these|the\s+method|the\s+approach|the\s+result|the\s+limitation|the\s+one|that\s+one|this\s+one)\b",
        q, re.I,
    ))
def _looks_like_vague_task(q: str) -> bool:
    return bool(re.search(
        r"^(?:explain|describe|compare|analyze|analyse|discuss|summarize|summarise|expand on|tell me about|what about|why|how)\s+(?:this|that|it|them|these|those|one|the same|more)\b",
        q.strip(), re.I,
    ))
def _intent(question: str) -> str:
    q = question.lower()
    if re.search(r"\b(compare|comparison|difference|different|versus|vs\.?|contrast)\b", q):
        return "comparison"
    if re.search(r"\b(methodology|method|architecture|algorithm|implementation|training|pipeline|procedure)\b", q):
        return "methodology"
    if re.search(r"\b(why|how does|how do|mechanism|work|cause|leads to|results in)\b", q):
        return "mechanism"
    if re.search(r"\b(limitation|limit|weakness|failure|caveat|problem with|flaw|risk|bias)\b", q):
        return "critique"
    if re.search(r"\b(gap|missing|future work|not studied|open question|unexplored)\b", q):
        return "gap"
    if re.search(r"\b(agree|disagree|contradict|conflict|inconsistent)\b", q):
        return "contradiction"
    if re.search(r"\b(result|results|performance|score|accuracy|iou|f1|precision|recall|latency|loss|improved|decreased|increase|decrease|metric|benchmark|bleu|rouge|perplexity)\b", q):
        return "result"
    if re.search(r"\b(where did|where does|which paper|which source|source of|proof|evidence for|show me|quote|citation|cite|page|section)\b", q):
        return "provenance"
    if re.search(r"\b(report|section|conclusion|summary|thesis|main point)\b", q):
        return "report"
    if re.search(r"\b(what is|what are|what does|explain|meaning|mean)\b", q):
        return "explanation"
    return "research_question"
def _topic_score(question: str, sources: list[Source], analyses: list[PaperAnalysis], case_question: str, report_title: str = "") -> float:
    """Cheap topic affinity, deliberately broader than exact paper retrieval.
    This is used only for routing. Retrieval itself remains evidence-first.
    """
    q_tokens = {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", question)}
    if not q_tokens:
        return 0.0
    corpus = " ".join([case_question, report_title] + [s.title + " " + s.abstract + " " + (s.full_text or "")[:6000] for s in sources[:15]] + [
        " ".join(str(x) for x in [a.core_idea, a.method_summary, a.research_objective, a.key_results]) for a in analyses[:15]
    ])
    c_tokens = {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,}", corpus)}
    stop = {"what","does","how","why","can","could","would","should","the","and","for","from","with","this","that","paper","research","method","about","into","your","our","their","they","them"}
    q = {x.rstrip("s") for x in q_tokens if len(x) >= 3 and x not in stop}
    c = {x.rstrip("s") for x in c_tokens if len(x) >= 3}
    return len(q & c) / max(1, len(q))
def decide_chat_turn(
    question: str,
    *,
    sources: list[Source],
    analyses: list[PaperAnalysis],
    prior_followups: list[FollowupQA],
    case_question: str = "",
    report_title: str = "",
) -> ChatDecision:
    q = (question or "").strip()
    prior = prior_followups or []
    pending = _last_clarification(prior)
    if pending:
        effective = f"{pending.question}\nUser clarification: {q}"
        return ChatDecision(
            action="answer", effective_question=effective, intent=_intent(pending.question),
            context_mode="direct", confidence=0.98, reason="resumed pending clarification",
        )
    intent = _intent(q)
    recent_ids = _recent_source_ids(prior)
    has_prior = bool(prior)
    has_target = _target_count(q, sources) > 0
    selected_context = bool(re.search(r"\bselected\s+(?:passage|text|report\s+passage)\b", q, re.I))
    single_paper_job = (
        len(sources) == 1
        and intent in {"methodology", "result", "critique", "gap", "report", "provenance"}
        and not re.search(r"\b(?:how does|how do)\s+(?:a|an|the)?\s*(?:transformer|llm|language model|neural network|cnn|rnn|attention)\b", q, re.I)
    )
    if single_paper_job:
        return ChatDecision(
            action="answer", effective_question=q, intent=intent, context_mode="direct",
            confidence=0.98, reason="single active paper makes the research-specific task unambiguous",
        )
    affinity = _topic_score(q, sources, analyses, case_question, report_title)
    generic_concept = bool(re.search(r"\b(?:a|an|the)?\s*(?:transformer|llm|language model|neural network|cnn|rnn|attention)\b", q, re.I))
    if affinity >= 0.45 and intent in {"result", "critique", "gap", "report", "provenance"} and not generic_concept:
        return ChatDecision(
            action="answer", effective_question=q, intent=intent, context_mode="direct",
            confidence=0.94, reason="question strongly matches the research corpus",
        )
    if len(sources) == 1 and intent in {"research_question", "explanation", "mechanism"} and affinity >= 0.45 and not generic_concept:
        return ChatDecision(
            action="answer", effective_question=q, intent=intent, context_mode="direct",
            confidence=0.94, reason="question strongly matches the only selected paper",
        )
    if selected_context:
        return ChatDecision(
            action="answer", effective_question=q, intent="provenance", context_mode="direct",
            confidence=0.99, reason="UI supplied an explicit selected passage",
        )
    if len(sources) > 1 and re.search(r"\b(?:methodology|method|architecture|results?|dataset|experiment|limitations?|approach)\b", q, re.I) and not has_target and not has_prior:
        return ChatDecision(
            action="clarify", effective_question=q, intent=intent, context_mode="direct", confidence=0.25,
            clarification_question="Which paper do you mean? You can say Paper 1, Paper 2, or the paper title.",
            reason="paper-specific task has multiple possible targets",
        )
    if re.search(r"\bcompare\s+(?:these|them|those|the two|the three|the papers)\b", q, re.I) and len(sources) > 2 and not has_target and not has_prior:
        return ChatDecision(
            action="clarify", effective_question=q, intent="comparison", context_mode="direct", confidence=0.3,
            clarification_question="Which papers should I compare: all selected papers or a specific pair?",
            clarification_options=["All selected papers", "A specific pair"],
            reason="comparison target is materially ambiguous",
        )
    if _looks_like_vague_task(q) and not has_prior and not has_target:
        return ChatDecision(
            action="clarify", effective_question=q, intent=intent, context_mode="direct", confidence=0.2,
            clarification_question="What would you like me to focus on? You can name the paper, method, result, or concept.",
            reason="deictic request has no resolvable conversational referent",
        )
    if _looks_like_ambiguous_reference(q) and not has_prior and not has_target and (len(sources) != 1) and not case_question:
        return ChatDecision(
            action="clarify", effective_question=q, intent=intent, context_mode="direct", confidence=0.25,
            clarification_question="What does “this” refer to here? Give me the paper, method, result, or concept you mean.",
            reason="reference cannot be resolved from conversation",
        )
    if re.search(r"\b(?:this|that|it|they|them|the method|the result)\b", q, re.I) and has_prior:
        mode = "direct" if recent_ids else "adjacent"
    else:
        score = _topic_score(q, sources, analyses, case_question, report_title)
        if has_target or re.search(r"\b(?:paper|report|authors?|study|evidence|according to|citation|section)\b", q, re.I):
            mode = "direct"
        elif score >= 0.18:
            mode = "adjacent"
        else:
            mode = "general"
    return ChatDecision(
        action="answer", effective_question=q, intent=intent, context_mode=mode,
        confidence=0.9 if mode != "general" else 0.75, reason="resolved without material ambiguity",
    )
