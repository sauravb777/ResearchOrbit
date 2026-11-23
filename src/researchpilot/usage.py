from __future__ import annotations
from dataclasses import dataclass
from threading import Lock
_PRICING_PER_MILLION: dict[str, tuple[float, float]] = {
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
}
_DEFAULT_PRICING = (0.50, 1.50)
def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_price, out_price = _PRICING_PER_MILLION.get(model, _DEFAULT_PRICING)
    return round((prompt_tokens / 1_000_000) * in_price + (completion_tokens / 1_000_000) * out_price, 6)
@dataclass
class UsageRecord:
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    estimated_cost_usd: float
class UsageTracker:
    """Accumulates UsageRecords for one run and reports totals."""
    def __init__(self) -> None:
        self.records: list[UsageRecord] = []
        self._lock = Lock()
    def record(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int, latency_seconds: float) -> None:
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
        with self._lock:
            self.records.append(UsageRecord(provider, model, prompt_tokens, completion_tokens, latency_seconds, cost))
    def summary(self) -> dict[str, float | int]:
        with self._lock:
            records = list(self.records)
        return {
            "llm_calls": len(records),
            "prompt_tokens": sum(r.prompt_tokens for r in records),
            "completion_tokens": sum(r.completion_tokens for r in records),
            "total_tokens": sum(r.prompt_tokens + r.completion_tokens for r in records),
            "llm_latency_seconds": round(sum(r.latency_seconds for r in records), 3),
            "estimated_cost_usd": round(sum(r.estimated_cost_usd for r in records), 6),
        }
def estimate_run_cost(settings, model: str | None = None, papers: int | None = None) -> dict:
    """Rough *pre-run* cost/latency estimate so users can see what a run will
    cost before starting it. Heuristic: a deep read costs about 3 calls per
    paper, the council costs a director + up to 5 specialists + a chair, and
    planning/drafting/review/audit add a fixed overhead. Returned as a range
    (low = fewer papers/cheaper calls, high = full budget)."""
    if model is None:
        if settings.groq_api_key and settings.groq_models:
            model = settings.groq_models[0]
        elif not getattr(settings, "enable_v79_architecture", False) and settings.openai_api_key and settings.openai_models:
            model = settings.openai_models[0]
    if model is None:
        return {"llm_configured": False, "estimated_cost_usd": (0.0, 0.0), "estimated_llm_calls": (0, 0),
                "note": "No LLM key configured: offline deterministic mode, no LLM cost."}
    n = papers if papers is not None else settings.auto_selection_max
    n = max(1, n)
    council_calls = (1 + 5 + 1) if settings.enable_council else 0
    deep = (n * 3) if settings.enable_deep_read else 0
    overhead = 14
    high_calls = deep + council_calls + overhead
    low_calls = max(1, int(high_calls * 0.6))
    per_call_in, per_call_out = 6000, 1200
    def cost(calls: int) -> float:
        return round(calls * estimate_cost_usd(model, per_call_in, per_call_out), 4)
    high_cost, low_cost = cost(high_calls), cost(low_calls)
    cap = settings.max_run_cost_usd
    if cap:
        high_cost = min(high_cost, cap)
        low_cost = min(low_cost, cap)
    secs = lambda calls: int(calls * 8 / max(1, settings.llm_max_concurrency))
    return {
        "llm_configured": True, "model": model, "papers": n,
        "estimated_llm_calls": (low_calls, high_calls),
        "estimated_cost_usd": (low_cost, high_cost),
        "estimated_seconds": (secs(low_calls), secs(high_calls)),
        "budget_cap_usd": cap or None,
        "note": "Heuristic estimate; actual usage is reported in the run metrics.",
    }
