from __future__ import annotations
import argparse
import json
import os
import re
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
def score(result, expected_terms: list[str]) -> dict:
    srcs = result.sources
    hit = sum(1 for s in srcs if any(t in f"{s.title} {s.abstract}".lower() for t in expected_terms))
    ev_ids = {e.id for e in result.evidence}
    claims = result.claims
    supported = sum(1 for c in claims if any(i in ev_ids for i in (getattr(c, "evidence_ids", None) or [])))
    m = result.metrics
    return {
        "sources": len(srcs),
        "retrieval_topicality": round(hit / len(srcs), 3) if srcs else 0.0,
        "claim_support_rate": round(supported / len(claims), 3) if claims else 0.0,
        "report_words": len(result.report_markdown.split()),
        "llm_calls": m.get("llm_calls", 0),
        "estimated_cost_usd": m.get("estimated_cost_usd", 0),
        "runtime_seconds": round(m.get("total_runtime_seconds", 0), 1),
    }
def run_once(question: str, council: bool):
    from researchpilot import config
    os.environ["ENABLE_COUNCIL"] = "true" if council else "false"
    config.reset_settings_cache()
    from researchpilot.pipeline.orchestrator import ResearchAgent
    return ResearchAgent().run(question)
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--ablate-council", action="store_true")
    args = ap.parse_args()
    from researchpilot.config import get_settings
    if not get_settings().llm_available:
        print("No LLM key configured; live evaluation needs GROQ_API_KEY or OPENAI_API_KEY.", file=sys.stderr)
        return 2
    rows = [json.loads(l) for l in (Path(__file__).parent / "questions.jsonl").read_text().splitlines() if l.strip()][: args.limit]
    out = []
    for row in rows:
        q, terms = row["question"], [t.lower() for t in row["expected_terms"]]
        entry = {"question": q, "with_council": score(run_once(q, True), terms)}
        if args.ablate_council:
            entry["without_council"] = score(run_once(q, False), terms)
        out.append(entry)
        print(f"done: {q}", file=sys.stderr)
    (Path(__file__).parent / "live_results.json").write_text(json.dumps(out, indent=2))
    lines = ["# Live evaluation\n", "| Question | Mode | Topicality | Claim support | Words | Calls | Cost $ | Sec |", "|---|---|---|---|---|---|---|---|"]
    for e in out:
        for mode in ("with_council", "without_council"):
            if mode in e:
                m = e[mode]
                lines.append(f"| {e['question'][:60]} | {mode} | {m['retrieval_topicality']} | {m['claim_support_rate']} | {m['report_words']} | {m['llm_calls']} | {m['estimated_cost_usd']} | {m['runtime_seconds']} |")
    (Path(__file__).parent / "live_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
