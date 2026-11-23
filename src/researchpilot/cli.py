from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .config import get_settings
from .export import sources_to_bibtex
from .pipeline.orchestrator import ResearchAgent
from .storage import ResultStore
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="researchpilot", description="Evidence-grounded research agent")
    parser.add_argument("question", help="The research question to investigate")
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-evidence", type=int, default=None)
    parser.add_argument("--no-save", action="store_true", help="Don't persist the run to the local database")
    parser.add_argument(
        "--interactive", action="store_true",
        help="After the report, start an interactive Q&A loop grounded in the retrieved evidence",
    )
    parser.add_argument("--estimate-only", action="store_true",
                        help="Print an estimated cost/latency range and exit without running")
    parser.add_argument("--max-cost", type=float, default=None, help="Per-run LLM spend cap in USD (0 = no cap)")
    parser.add_argument("--no-council", action="store_true", help="Skip the multi-role council (leaner, cheaper run)")
    args = parser.parse_args(argv)
    import os
    from .config import reset_settings_cache
    if args.max_cost is not None:
        os.environ["MAX_RUN_COST_USD"] = str(max(0.0, args.max_cost))
    if args.no_council:
        os.environ["ENABLE_COUNCIL"] = "false"
    if args.max_cost is not None or args.no_council:
        reset_settings_cache()
    settings = get_settings()
    if args.estimate_only:
        from .usage import estimate_run_cost
        est = estimate_run_cost(settings)
        for k, v in est.items():
            print(f"{k}: {v}")
        return 0
    agent = ResearchAgent()
    print(f"Researching: {args.question}", file=sys.stderr)
    if not agent.llm.available:
        print(
            "No GROQ_API_KEY configured. v79 requires Groq for retrieval, "
            "deep reading and report publication; the run will stop rather "
            "than present an offline heuristic report as a research deliverable.",
            file=sys.stderr,
        )
    result = agent.run(
        args.question,
        max_sources=settings.max_sources if args.max_sources is None else max(0, args.max_sources),
        max_evidence=settings.max_evidence if args.max_evidence is None else max(0, args.max_evidence),
    )
    out_dir = Path(settings.reports_dir) / result.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(result.report_markdown, encoding="utf-8")
    (out_dir / "audit.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "references.bib").write_text(sources_to_bibtex(result.sources), encoding="utf-8")
    if not args.no_save:
        ResultStore(settings.db_path).save(result)
    print(f"\nDone. {result.metrics.get('sources_selected', len(result.sources))} sources, "
          f"{result.metrics.get('evidence_count', len(result.evidence))} evidence snippets, "
          f"{result.metrics.get('claim_count', len(result.claims))} claims.", file=sys.stderr)
    print(
        f"Runtime: {result.metrics.get('total_runtime_seconds', 0):.2f}s   "
        f"LLM calls: {result.metrics.get('llm_calls', 0)}   "
        f"Est. cost: ${result.metrics.get('estimated_cost_usd', 0):.4f}",
        file=sys.stderr,
    )
    print(f"Report:      {out_dir / 'report.md'}", file=sys.stderr)
    print(f"Audit:       {out_dir / 'audit.json'}", file=sys.stderr)
    print(f"Bibliography: {out_dir / 'references.bib'}", file=sys.stderr)
    if result.warnings:
        print("\nWarnings:", file=sys.stderr)
        for w in result.warnings:
            print(f"  - {w}", file=sys.stderr)
    print(result.report_markdown)
    if args.interactive:
        print("\n--- Interactive Q&A (grounded in the evidence above). Type 'exit' to quit. ---", file=sys.stderr)
        while True:
            try:
                followup = input("\nAsk a follow-up question: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not followup or followup.lower() in {"exit", "quit"}:
                break
            qa = agent.ask_followup(result, followup)
            print(f"\n{qa.answer}\n")
        if not args.no_save:
            ResultStore(settings.db_path).save(result)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
