from __future__ import annotations
import json
from ..llm import LLMClient
SYSTEM="""You are the evidence editor for a research report. Build a compact Creator Pack strictly from the supplied
paper evidence and report claims. Never invent facts. Each claim card needs a strength label, evidence ids,
conditions, safe wording and what would overturn it. Also produce say/say-carefully/don't-say, misconceptions,
60-second concepts, analogy bank and a script outline. JSON only."""
def build_creator_pack(question, report_markdown, evidence, analyses, llm: LLMClient):
    if not llm.available: return {}, False
    payload={"question":question,"report":report_markdown[:60000],"evidence":[{"id":e.id,"source_id":e.source_id,"text":e.text,"locator":e.locator} for e in evidence[:180]],"papers":[{"id":a.source_id,"question_answer":a.question_answer,"established":a.established,"not_established":a.not_established,"limitations":a.limitations} for a in analyses]}
    try:
        out=llm.complete_json(SYSTEM,json.dumps(payload,ensure_ascii=False),max_tokens=5200,priority="report")
        return (out if isinstance(out,dict) else {}), isinstance(out,dict)
    except Exception:
        return {},False
