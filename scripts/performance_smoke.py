from __future__ import annotations
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from researchpilot.config import Settings
from researchpilot.llm import LLMClient
from researchpilot.pipeline.beast_reader import read_paper_beast
from researchpilot.pipeline.feynman_report import build_feynman_report
from researchpilot.schemas import PaperAnalysis, Source
def paper(i: int) -> Source:
    text = "\n".join([
        "1 Introduction\nThis study investigates a concrete research problem and motivates a measurable intervention.",
        "2 Method\nThe proposed method changes a defined mechanism, with explicit assumptions and implementation details.",
        "3 Data and Experiments\nThe method is evaluated on a named dataset against stated baselines using defined metrics.",
        "4 Results and Ablations\nThe experiments report quantitative results and isolate important components through ablations.",
        "5 Discussion and Limitations\nThe authors discuss failure modes, boundary conditions and unresolved limitations.",
        "6 Conclusion\nThe paper states what the evidence establishes and what remains uncertain.",
    ])
    return Source(id=f"smoke-{i}", title=f"ResearchOrbit performance smoke paper {i}", authors="Smoke Test", year="2026", abstract="Synthetic benchmark paper.", full_text=text, full_text_complete=True, full_text_parsed=True, pdf_acquired=True, role="deep_read", score=0.9)
class SimulatedLLM(LLMClient):
    """Deterministic provider emulator that preserves the production call graph."""
    def __init__(self, settings: Settings, latency: float = 0.08):
        super().__init__(settings=settings)
        self.latency = latency
        self.force_fast_path = True
        self.lock = threading.Lock()
        self.calls = []
        self.inflight = 0
        self.max_inflight = 0
    @property
    def available(self):
        return True
    def complete(self, system, user, temperature=None, max_tokens=1500, priority="normal", tier=None, reasoning_effort=None, response_format=None, **kwargs):
        with self.lock:
            self.inflight += 1
            self.max_inflight = max(self.max_inflight, self.inflight)
            self.calls.append({"kind": "json" if response_format else "text", "prompt_chars": len(system) + len(user), "max_tokens": max_tokens, "tier": tier, "priority": priority})
        try:
            time.sleep(self.latency)
            if response_format:
                if "chief professor-level research council" in system or "chief research scientist" in system:
                    return json.dumps({
                        "title": "Performance benchmark report",
                        "central_answer": "The selected studies provide complementary evidence. [S1] [S2] [S3] [S4]",
                        "executive_summary": "The studies provide complementary evidence. [S1] [S2] [S3] [S4]",
                        "key_takeaways": ["The mechanisms are condition-dependent. [S1] [S2]"],
                        "open_questions": ["The evidence leaves a boundary unresolved. [S3] [S4]"],
                        "topics": [
                            {"title":"Problem and evidence", "goal":"Explain the problem and evidence", "role":"problem", "source_tags":["S1","S2","S3","S4"], "subtopics":[{"title":"Problem","what_to_explain":"Explain the problem"}]},
                            {"title":"Mechanisms", "goal":"Explain mechanisms", "role":"method", "source_tags":["S1","S2","S3","S4"], "subtopics":[{"title":"Mechanisms","what_to_explain":"Explain mechanisms"}]},
                            {"title":"Results and research gaps", "goal":"Explain results and gaps", "role":"results", "source_tags":["S1","S2","S3","S4"], "subtopics":[{"title":"Evidence and gaps","what_to_explain":"Explain evidence and gaps"}]},
                        ],
                    })
                return json.dumps({"problem":"problem", "core_idea":"mechanism", "method_summary":"method", "experimental_setup":"evaluation", "key_results":"result", "limitations_author_stated":"limitation", "question_answer":"answer", "evidence_inventory":[{"observation":"Result", "quote":"This result is directly demonstrated by the paper."}], "research_memory":[{"observation":"Mechanism", "quote":"The mechanism is described in this paper."}], "experiment_records":[{"result":"Result", "evidence_quote":"This result is directly demonstrated by the paper."}], "claim_ledger":[{"claim":"Claim", "evidence_quote":"The mechanism is described in this paper."}]})
            return "### Explanation\n\nThe evidence supports the mechanism and measured result [S1] [S2] [S3] [S4]."
        finally:
            with self.lock:
                self.inflight -= 1
def _run_readers(llm, n):
    started = time.perf_counter()
    analyses = [None] * n
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(read_paper_beast, "How does the proposed mechanism work and what does the evidence establish?", paper(i), llm, "standard_report"): i - 1 for i in range(1, n + 1)}
        for fut in as_completed(futures):
            i = futures[fut]
            a, _, meta = fut.result()
            if a is None:
                raise RuntimeError(f"reader {i+1} failed: {meta}")
            analyses[i] = a
    return analyses, time.perf_counter() - started
def run_simulated(n: int) -> int:
    settings = Settings(groq_api_key="simulated", quality_first=True, llm_max_concurrency=5, paper_read_concurrency=n)
    llm = SimulatedLLM(settings)
    analyses, read_s = _run_readers(llm, n)
    sources = [paper(i) for i in range(1, n + 1)]
    started = time.perf_counter()
    md, info = build_feynman_report("How do the proposed mechanisms work and what does the evidence establish?", "standard_report", "Performance smoke", sources, analyses, [], llm)
    report_s = time.perf_counter() - started
    total = read_s + report_s
    assert md.strip()
    assert info.get("bounded_report_calls", 99) <= 20
    assert info.get("topics", 0) >= 4
    assert llm.max_inflight >= 2, f"parallelism not observed: max_inflight={llm.max_inflight}"
    max_prompt = max(c["prompt_chars"] for c in llm.calls)
    print(f"SIMULATED {n}-paper benchmark: read={read_s:.3f}s report={report_s:.3f}s total={total:.3f}s")
    print(f"calls={len(llm.calls)} max_inflight={llm.max_inflight} max_prompt_chars={max_prompt} report_calls={info.get('bounded_report_calls')}")
    print("PASS: quality-first reader/report barriers are parallel and bounded")
    return 0
def run_live(n: int) -> int:
    settings = Settings()
    if not settings.groq_api_key:
        print("FAIL: GROQ_API_KEY is not configured.")
        return 2
    llm = LLMClient(settings=settings)
    sources = [paper(i) for i in range(1, n + 1)]
    analyses, read_s = _run_readers(llm, n)
    started = time.perf_counter()
    markdown, info = build_feynman_report("How do the proposed mechanisms work and what does the evidence establish?", "standard_report", "ResearchOrbit live performance smoke", sources, analyses, [], llm)
    report_s = time.perf_counter() - started
    total = read_s + report_s
    usage = llm.usage.summary()
    print(f"LIVE {n}-paper benchmark: read={read_s:.2f}s report={report_s:.2f}s total={total:.2f}s")
    print(f"calls={usage['llm_calls']} prompt_tokens={usage['prompt_tokens']} completion_tokens={usage['completion_tokens']} total_tokens={usage['total_tokens']} llm_latency={usage['llm_latency_seconds']:.2f}s")
    print(f"architecture={info.get('architecture')} report_calls={info.get('bounded_report_calls')} timing={info.get('timing')}")
    print("Provider events:")
    for event in llm.events[-12:]:
        print(f"  {event.ok} {event.model}: {event.detail}")
    if not markdown.strip():
        print("FAIL: no report was produced")
        return 1
    if total > 600:
        print("FAIL: exceeded 10-minute performance contract")
        return 1
    print("PASS: live smoke completed within 10 minutes")
    return 0
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("simulated", "live"), default="simulated")
    ap.add_argument("--papers", type=int, default=4, choices=range(1, 6))
    args = ap.parse_args()
    return run_simulated(args.papers) if args.mode == "simulated" else run_live(args.papers)
if __name__ == "__main__":
    raise SystemExit(main())
