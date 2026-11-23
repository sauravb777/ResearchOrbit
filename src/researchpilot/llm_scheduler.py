from __future__ import annotations
import heapq
import threading
import re
import time
from dataclasses import dataclass
@dataclass
class _Bucket:
    limit: float = 0.0
    remaining: float = 0.0
    reset_at: float = 0.0
class GroqTokenScheduler:
    PRIORITY = {"chat": 0, "report": 1, "reading": 2, "retrieval": 3, "normal": 4}
    def __init__(self, initial_tpm: int = 0) -> None:
        self._lock = threading.Condition()
        self._initial_tpm = max(0, int(initial_tpm or 0))
        self._buckets: dict[str, _Bucket] = {}
        self._queue: list[tuple[int, int, str, int]] = []
        self._seq = 0
        self._last_admit = 0.0
    def _bucket(self, model: str) -> _Bucket:
        if model not in self._buckets:
            initial = float(self._initial_tpm)
            self._buckets[model] = _Bucket(limit=initial, remaining=initial, reset_at=time.monotonic() + 60.0 if initial else 0.0)
        return self._buckets[model]
    def update_headers(self, model: str, headers) -> None:
        def num(name, default=0.0):
            try: return float(headers.get(name, default))
            except (TypeError, ValueError): return default
        limit = num("x-ratelimit-limit-tokens")
        remaining = num("x-ratelimit-remaining-tokens")
        raw_reset = headers.get("x-ratelimit-reset-tokens")
        reset = num("x-ratelimit-reset-tokens")
        if raw_reset is not None and not reset:
            text = str(raw_reset).strip().lower()
            total = 0.0
            for amount, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
                factor = {"h": 3600.0, "m": 60.0, "s": 1.0}[unit]
                total += float(amount) * factor
            reset = total
        if reset > 0 and reset < 1_000_000:
            reset_at = time.monotonic() + reset
        else:
            reset_at = time.monotonic() + 1.0
        with self._lock:
            b = self._bucket(model)
            if limit: b.limit = limit
            if headers.get("x-ratelimit-remaining-tokens") is not None: b.remaining = remaining
            if reset: b.reset_at = reset_at
            self._lock.notify_all()
    def acquire(self, model: str, estimated_tokens: int, priority: str = "normal", timeout: float | None = None) -> None:
        """Admit one request against the model's token bucket.
        ``timeout`` is a *stall* timeout, not a total-wait cap.  Under fan-out
        (papers x regions) many threads legitimately queue behind each other for
        a token bucket that refills once a minute; a fixed wall-clock cap made
        the tail of the queue fail even though the provider was healthy and the
        queue was draining.  The deadline therefore restarts whenever *any*
        request is admitted, so only a queue that makes no progress at all for
        ``timeout`` seconds raises TimeoutError. The timeout is measured from the
        request's own admission attempt and is intentionally NOT extended by other
        requests making progress.
        A request whose estimate exceeds the whole bucket can never be satisfied
        by ``remaining >= estimate`` (remaining is capped at the limit), so the
        requirement is clamped to the bucket limit instead of starving forever.
        """
        estimated_tokens = max(1, int(estimated_tokens))
        rank = self.PRIORITY.get(priority, self.PRIORITY["normal"])
        with self._lock:
            started = time.monotonic()
            deadline = started + float(timeout) if timeout is not None else None
            self._seq += 1
            ticket = (rank, self._seq, model, estimated_tokens)
            heapq.heappush(self._queue, ticket)
            try:
                while True:
                    now = time.monotonic()
                    for bucket in self._buckets.values():
                        if bucket.limit > 0 and bucket.reset_at and now >= bucket.reset_at:
                            bucket.remaining = bucket.limit
                            bucket.reset_at = 0.0
                    chosen_index = None
                    earliest_reset = None
                    for idx, queued in enumerate(sorted(self._queue)):
                        q_rank, _q_seq, q_model, q_est = queued
                        qb = self._bucket(q_model)
                        need = min(q_est, qb.limit) if qb.limit > 0 else q_est
                        if qb.limit <= 0 or qb.remaining >= need:
                            chosen_index = idx
                            break
                        if qb.reset_at:
                            earliest_reset = qb.reset_at if earliest_reset is None else min(earliest_reset, qb.reset_at)
                    ordered = sorted(self._queue)
                    if chosen_index is not None and ordered[chosen_index] == ticket:
                        b = self._bucket(model)
                        if b.limit > 0:
                            b.remaining = max(0.0, b.remaining - min(estimated_tokens, b.limit))
                        self._queue.remove(ticket)
                        heapq.heapify(self._queue)
                        self._last_admit = time.monotonic()
                        self._lock.notify_all()
                        return
                    wait = 0.05
                    if chosen_index is None and earliest_reset is not None:
                        wait = max(0.05, earliest_reset - now)
                    if deadline is not None:
                        remaining = deadline - now
                        if remaining <= 0:
                            raise TimeoutError(f"Groq token scheduler stalled for {float(timeout):.1f}s with no admissions (model {model})")
                        wait = min(wait, remaining)
                    self._lock.wait(timeout=wait)
            finally:
                try:
                    self._queue.remove(ticket)
                    heapq.heapify(self._queue)
                except ValueError:
                    pass
    def refund_estimate(self, model: str, estimated_tokens: int, actual_tokens: int) -> None:
        delta = max(0, int(estimated_tokens) - int(actual_tokens))
        if not delta: return
        with self._lock:
            b = self._bucket(model)
            if b.limit > 0: b.remaining = min(b.limit, b.remaining + delta)
            self._lock.notify_all()
GLOBAL_GROQ_SCHEDULER = GroqTokenScheduler()
