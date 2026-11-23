from __future__ import annotations
import os
os.environ.setdefault("REQUIRE_VERIFIED_DEEP_REPORT", "false")
os.environ.setdefault("REQUIRE_RESEARCH_GRADE_REPORT", "false")
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from researchpilot.pipeline.orchestrator import ResearchAgent
from researchpilot.search.aggregator import SearchAggregator
from eval.checks import ALL_CHECKS
from eval.fixtures import SCENARIOS, FixedProvider
def run_scenario(scenario) -> dict:
    agent = ResearchAgent(aggregator=SearchAggregator(providers=[FixedProvider(scenario.sources)]))
    result = agent.run(scenario.question, max_sources=scenario.max_sources, max_iterations=scenario.max_iterations)
    checks_run = []
    for check_fn in ALL_CHECKS:
        passed, detail = check_fn(result)
        checks_run.append({"check": check_fn.__name__, "passed": passed, "detail": detail})
    n_selected = len(result.sources)
    scenario_checks = [
        (
            "expect_min_sources_selected",
            n_selected >= scenario.expect_min_sources_selected,
            f"selected={n_selected}, expected>={scenario.expect_min_sources_selected}",
        )
    ]
    if scenario.expect_max_sources_selected is not None:
        scenario_checks.append((
            "expect_max_sources_selected",
            n_selected <= scenario.expect_max_sources_selected,
            f"selected={n_selected}, expected<={scenario.expect_max_sources_selected}",
        ))
    for expected_warning in scenario.expect_warning_contains:
        found = any(expected_warning in w for w in result.warnings)
        scenario_checks.append(("expect_warning", found, f"looking for '{expected_warning}' in warnings"))
    for expected_snippet in scenario.expect_report_contains:
        found = expected_snippet in result.report_markdown
        scenario_checks.append(("expect_report_contains", found, f"looking for '{expected_snippet}' in report"))
    for name, passed, detail in scenario_checks:
        checks_run.append({"check": name, "passed": passed, "detail": detail})
    all_passed = all(c["passed"] for c in checks_run)
    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "question": scenario.question,
        "passed": all_passed,
        "checks": checks_run,
        "metrics": result.metrics,
    }
def render_markdown(results: list[dict]) -> str:
    lines = ["# ResearchPilot Evaluation Results\n"]
    n_passed = sum(1 for r in results if r["passed"])
    lines.append(f"**{n_passed}/{len(results)} scenarios passed.**\n")
    lines.append("| Scenario | Status | Sources | Evidence | Claims |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        m = r["metrics"]
        lines.append(
            f"| {r['scenario']} | {status} | {m.get('sources_selected')} | "
            f"{m.get('evidence_count')} | {m.get('claim_count')} |"
        )
    lines.append("")
    for r in results:
        lines.append(f"## {r['scenario']}")
        lines.append(f"_{r['description']}_\n")
        lines.append(f"Question: \"{r['question']}\"\n")
        for c in r["checks"]:
            mark = "✅" if c["passed"] else "❌"
            lines.append(f"- {mark} `{c['check']}` — {c['detail']}")
        lines.append("")
    return "\n".join(lines)
def main() -> int:
    results = [run_scenario(s) for s in SCENARIOS]
    out_dir = Path(__file__).parent
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out_dir / "results.md").write_text(render_markdown(results), encoding="utf-8")
    n_passed = sum(1 for r in results if r["passed"])
    print(f"{n_passed}/{len(results)} scenarios passed.")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['scenario']}")
        if not r["passed"]:
            for c in r["checks"]:
                if not c["passed"]:
                    print(f"           - {c['check']}: {c['detail']}")
    return 0 if n_passed == len(results) else 1
if __name__ == "__main__":
    raise SystemExit(main())
