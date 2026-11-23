from __future__ import annotations
import json
import re
import time
import threading
from dataclasses import dataclass, field
from typing import Any
import requests
from .config import Settings, get_settings
from .usage import UsageTracker
from .llm_scheduler import GroqTokenScheduler
def _safe_text(value: str) -> str:
    """Replace isolated UTF-16 surrogates before sending JSON over HTTP."""
    return "".join("\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in value)
@dataclass
class LLMEvent:
    provider: str
    model: str
    ok: bool
    detail: str = ""
_LLM_REQUEST_SEMAPHORE: threading.BoundedSemaphore | None = None
_LLM_REQUEST_SEMAPHORE_SIZE: int = 0
_LLM_REQUEST_SEMAPHORE_LOCK = threading.Lock()
def _llm_request_semaphore(limit: int) -> threading.BoundedSemaphore:
    global _LLM_REQUEST_SEMAPHORE, _LLM_REQUEST_SEMAPHORE_SIZE
    limit = max(1, int(limit))
    with _LLM_REQUEST_SEMAPHORE_LOCK:
        if _LLM_REQUEST_SEMAPHORE is None or _LLM_REQUEST_SEMAPHORE_SIZE != limit:
            _LLM_REQUEST_SEMAPHORE = threading.BoundedSemaphore(limit)
            _LLM_REQUEST_SEMAPHORE_SIZE = limit
        return _LLM_REQUEST_SEMAPHORE
class LLMHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"HTTP {status_code}: {detail}")
        self.status_code = status_code
@dataclass
class LLMClient:
    settings: Settings = field(default_factory=get_settings)
    events: list[LLMEvent] = field(default_factory=list)
    usage: UsageTracker = field(default_factory=UsageTracker)
    _tls: threading.local = field(default_factory=threading.local, repr=False, compare=False)
    _run_deadline: float | None = field(default=None, repr=False, compare=False)
    _scheduler: GroqTokenScheduler = field(init=False, repr=False, compare=False)
    def __post_init__(self) -> None:
        self._scheduler = GroqTokenScheduler(getattr(self.settings, "groq_tpm_hint", 8000))
    def set_run_deadline(self, seconds: float | None) -> None:
        """Set a hard wall-clock deadline for the current research run."""
        self._run_deadline = (time.monotonic() + float(seconds)) if seconds is not None else None
    def clear_run_deadline(self) -> None:
        self._run_deadline = None
    def remaining_run_seconds(self) -> float | None:
        if self._run_deadline is None:
            return None
        return max(0.0, self._run_deadline - time.monotonic())
    def last_failure(self) -> str:
        """Reason the calling thread's most recent LLM call returned None ('' if it succeeded)."""
        return str(getattr(self._tls, "last_error", "") or "")
    def _set_failure(self, reason: str) -> None:
        self._tls.last_error = reason
    @property
    def available(self) -> bool:
        return self.settings.llm_available
    def list_groq_models(self) -> list[str]:
        """Fetch active Groq model ids without exposing credentials."""
        if not self.settings.groq_api_key:
            return []
        try:
            resp = requests.get(
                f"{self.settings.groq_base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {self.settings.groq_api_key}"},
                timeout=min(15, self.settings.llm_timeout),
            )
            resp.raise_for_status()
            data = resp.json()
            return [str(item.get("id")) for item in (data.get("data") or []) if isinstance(item, dict) and item.get("id")]
        except Exception as exc:
            self.events.append(LLMEvent("groq", "-", False, f"model-list: {exc}"))
            return []
    def health(self) -> dict[str, Any]:
        """Return non-secret provider diagnostics suitable for the API/UI."""
        providers = []
        for name, base_url, api_key, models in self._providers():
            providers.append({
                "provider": name,
                "configured": bool(api_key),
                "base_url": base_url,
                "models": list(models),
            })
        budget = self.budget_exhausted()
        return {
            "available": self.available,
            "providers": providers,
            "budget_exhausted": bool(budget),
            "budget_reason": budget or "",
            "usage": self.usage.summary(),
            "recent_events": [
                {"provider": e.provider, "model": e.model, "ok": e.ok, "detail": e.detail[:500]}
                for e in self.events[-20:]
            ],
        }
    def budget_exhausted(self) -> str | None:
        """Return a reason string if this run's cost/call budget is spent."""
        summary = self.usage.summary()
        max_cost = self.settings.max_run_cost_usd
        if max_cost and summary["estimated_cost_usd"] >= max_cost:
            return f"cost budget ${max_cost:.2f} reached (${summary['estimated_cost_usd']:.4f} spent)"
        max_calls = self.settings.max_llm_calls
        if max_calls and summary["llm_calls"] >= max_calls:
            return f"call budget {max_calls} reached"
        return None
    def _providers(self) -> list[tuple[str, str, str, list[str]]]:
        """Return Groq only. OpenAI fields remain readable for legacy settings."""
        if self.settings.groq_api_key:
            return [("groq", self.settings.groq_base_url, self.settings.groq_api_key, self.settings.groq_models)]
        return []
    def _call_one(
        self, base_url: str, api_key: str, model: str, system: str, user: str, temperature: float, max_tokens: int,
        priority: str = "normal", reasoning_effort: str | None = None, response_format: dict | None = None,
        timeout_override: float | None = None,
    ) -> tuple[str, int, int]:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        system = _safe_text(system)
        user = _safe_text(user)
        payload = {
            "model": _safe_text(model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        payload["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if response_format:
            payload["response_format"] = response_format
        estimated_tokens = max(1, int((len(system) + len(user)) / 4) + int(max_tokens))
        remaining = self.remaining_run_seconds()
        if remaining is not None and remaining <= 0:
            raise TimeoutError("research run deadline reached before LLM admission")
        scheduler_timeout = max(1.0, min(float(self.settings.llm_timeout), 10.0, remaining if remaining is not None else 10.0))
        self._scheduler.acquire(
            model, estimated_tokens, priority=priority, timeout=scheduler_timeout
        )
        try:
            gate = _llm_request_semaphore(self.settings.llm_max_concurrency)
            request_timeout = float(timeout_override if timeout_override is not None else self.settings.llm_timeout)
            remaining = self.remaining_run_seconds()
            if remaining is not None:
                if remaining <= 0:
                    raise TimeoutError("research run deadline reached during LLM request")
                request_timeout = min(request_timeout, max(1.0, remaining))
            acquired = gate.acquire(timeout=max(0.01, request_timeout))
            if not acquired:
                raise TimeoutError("LLM concurrency gate timed out before request admission")
            try:
                remaining = self.remaining_run_seconds()
                if remaining is not None:
                    if remaining <= 0:
                        raise TimeoutError("research run deadline reached during LLM request")
                    request_timeout = min(request_timeout, max(1.0, remaining))
                resp = requests.post(
                    f"{base_url}/chat/completions", headers=headers, json=payload, timeout=request_timeout
                )
            finally:
                gate.release()
            self._scheduler.update_headers(model, getattr(resp, "headers", {}) or {})
        except Exception:
            self._scheduler.refund_estimate(model, estimated_tokens, 0)
            raise
        if resp.status_code >= 400:
            if resp.status_code == 429:
                retry_after = resp.headers.get("retry-after")
                try:
                    if retry_after:
                        remaining = self.remaining_run_seconds()
                        wait = min(5.0, max(0.1, float(retry_after)))
                        if remaining is not None: wait = min(wait, max(0.0, remaining))
                        if wait > 0: time.sleep(wait)
                except ValueError: pass
            raise LLMHTTPError(resp.status_code, resp.text[:300])
        data = resp.json()
        text = data["choices"][0]["message"].get("content", "")
        if not text or not text.strip():
            raise RuntimeError("empty completion")
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        return text, prompt_tokens, completion_tokens
    @staticmethod
    def _tier_models(models: list[str], tier: str | None) -> list[str]:
        """Order Groq models by the v79 S/M/F tier contract.
        GPT-OSS 120B is the S/M model and GPT-OSS 20B is the F model. If a
        user configured only one model, it remains usable rather than making
        the tiering feature a hidden availability requirement.
        """
        if not tier:
            return list(models)
        tier = str(tier).upper()
        if tier == "F":
            preferred = [m for m in models if "gpt-oss-20b" in m]
        else:
            preferred = [m for m in models if "gpt-oss-120b" in m]
        rest = [m for m in models if m not in preferred]
        return preferred + rest
    @staticmethod
    def _default_reasoning_effort(tier: str | None) -> str | None:
        return {"S": "high", "M": "medium", "F": "low"}.get(str(tier).upper()) if tier else None
    def complete(
        self, system: str, user: str, temperature: float | None = None, max_tokens: int = 1500,
        priority: str = "normal", tier: str | None = None, reasoning_effort: str | None = None,
        response_format: dict | None = None, timeout_override: float | None = None
    ) -> str | None:
        """Run a Groq completion with optional v79 model/reasoning tiering."""
        temperature = self.settings.llm_temperature if temperature is None else temperature
        reasoning_effort = reasoning_effort or self._default_reasoning_effort(tier)
        self._set_failure("")
        reason = self.budget_exhausted()
        if reason:
            if not any(e.provider == "budget" for e in self.events):
                self.events.append(LLMEvent("budget", "-", False, reason + "; LLM stages stopped"))
            self._set_failure("budget exhausted: " + reason)
            return None
        last_failure = "no provider configured"
        for name, base_url, api_key, models in self._providers():
            for model in self._tier_models(models, tier):
                last_err = None
                for attempt in range(self.settings.llm_retries + 1):
                    start = time.perf_counter()
                    try:
                        text, prompt_tok, completion_tok = self._call_one(
                            base_url, api_key, model, system, user, temperature, max_tokens, priority,
                            reasoning_effort=reasoning_effort, response_format=response_format, timeout_override=timeout_override
                        )
                        elapsed = time.perf_counter() - start
                        self.usage.record(name, model, prompt_tok, completion_tok, elapsed)
                        self.events.append(LLMEvent(
                            name, model, True,
                            f"attempt {attempt + 1}, {elapsed:.2f}s, prompt={prompt_tok}tok/{len(system)+len(user)}chars, "
                            f"output={completion_tok}tok, tier={tier or '-'}, reasoning={reasoning_effort or '-'}"
                        ))
                        return text
                    except Exception as exc:
                        last_err = exc
                        retryable = not isinstance(exc, LLMHTTPError) or exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500
                        if retryable and attempt < self.settings.llm_retries:
                            time.sleep(0.6 * (attempt + 1))
                        elif not retryable:
                            break
                self.events.append(LLMEvent(name, model, False, str(last_err)))
                last_failure = str(last_err)
        self._set_failure(last_failure)
        return None
    @staticmethod
    def extract_json(raw: str) -> Any:
        """Best-effort extraction of a JSON value from LLM prose."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        for open_ch, close_ch in (("{", "}"), ("[", "]")):
            start, end = text.find(open_ch), text.rfind(close_ch)
            if start != -1 and end > start:
                candidate = text[start : end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        continue
        raise ValueError("could not extract valid JSON from LLM response")
    def complete_json(
        self, system: str, user: str, temperature: float | None = None, max_tokens: int = 1500,
        priority: str = "normal", tier: str | None = None, reasoning_effort: str | None = None,
        salvage: bool = False, timeout_override: float | None = None,
    ) -> dict | list | None:
        """JSON completion.  With ``salvage=True`` a reply cut off by the token
        limit is repaired to its longest valid prefix instead of being discarded
        (used by paper reading, where verified partial output beats no output)."""
        complete_kwargs = dict(
            temperature=temperature, max_tokens=max_tokens, priority=priority,
            tier=tier, reasoning_effort=reasoning_effort, response_format={"type": "json_object"},
        )
        if timeout_override is not None:
            complete_kwargs["timeout_override"] = timeout_override
        text = self.complete(system, user, **complete_kwargs)
        if text is None:
            return None
        try:
            return self.extract_json(text)
        except ValueError as exc:
            if salvage:
                try:
                    from .pipeline.reading_resilience import salvage_truncated_json
                    value = salvage_truncated_json(text)
                    if isinstance(value, (dict, list)) and value:
                        self.events.append(LLMEvent("json", "-", True, "salvaged truncated JSON reply"))
                        return value
                except Exception:
                    pass
            self.events.append(LLMEvent("json", "-", False, str(exc)))
            self._set_failure("could not extract valid JSON (reply truncated or malformed)")
            return None
