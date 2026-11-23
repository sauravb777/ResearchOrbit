from __future__ import annotations
import re
from dataclasses import asdict, dataclass, field
from typing import Any
LEVEL_FULL = 1
LEVEL_TARGETED = 2
LEVEL_DETERMINISTIC_INTEGRATION = 3
LEVEL_PARTIAL = 4
LEVEL_EXCLUDED = 5
LEVEL_LABELS = {
    LEVEL_FULL: "full adaptive deep reading",
    LEVEL_TARGETED: "targeted regional deep reading",
    LEVEL_DETERMINISTIC_INTEGRATION: "regional deep reading with deterministic integration",
    LEVEL_PARTIAL: "verified partial dossier",
    LEVEL_EXCLUDED: "excluded (no verified dossier)",
}
THROTTLED = "provider_throttled"
TRUNCATED = "truncated_or_malformed_json"
EMPTY = "empty_provider_response"
SCHEMA = "schema_mismatch"
NO_TEXT = "no_readable_text"
NO_STRUCTURE = "structure_unmappable"
NO_PACKETS = "no_usable_reading_packets"
THIN = "insufficient_substantive_fields"
CANCELLED = "cancelled"
UNKNOWN = "unknown_error"
STRATEGY_CHANGE_CLASSES = {TRUNCATED, SCHEMA, THIN, NO_PACKETS, EMPTY}
def classify_failure(reason: Any) -> str:
    """Map an exception / reason string onto the failure taxonomy."""
    text = str(reason or "").lower()
    if not text:
        return UNKNOWN
    if "cancel" in text:
        return CANCELLED
    if any(k in text for k in ("429", "rate limit", "rate-limit", "throttl", "scheduler wait", "scheduler stalled", "stalled", "timed out", "timeout", "503", "502", "overloaded")):
        return THROTTLED
    if any(k in text for k in ("could not extract valid json", "truncat", "json", "unterminated", "expecting")):
        return TRUNCATED
    if "empty completion" in text or "empty_provider" in text or "empty provider" in text:
        return EMPTY
    if any(k in text for k in ("validation", "schema", "input should be")):
        return SCHEMA
    if "missing complete paper text" in text or "no readable" in text or "no full text" in text:
        return NO_TEXT
    if "could not be mapped" in text or "structure" in text:
        return NO_STRUCTURE
    if "no usable reading packets" in text:
        return NO_PACKETS
    if "insufficient substantive" in text:
        return THIN
    return UNKNOWN
@dataclass
class PaperReadStatus:
    """What actually happened to one selected paper.  Serialised into the run result."""
    source_id: str
    title: str = ""
    level: int = LEVEL_EXCLUDED
    status: str = "excluded"
    verified: bool = False
    reason: str = ""
    failure_class: str = ""
    attempts: int = 0
    regions_planned: int = 0
    regions_read: int = 0
    regions_failed: int = 0
    integration: str = ""
    cache_hits: int = 0
    notes: list[str] = field(default_factory=list)
    def to_dict(self) -> dict:
        d = asdict(self)
        d["level_label"] = LEVEL_LABELS.get(self.level, "")
        return d
def status_from_meta(source_id: str, title: str, meta: dict | None, *, attempts: int = 1) -> PaperReadStatus:
    """Translate a reader's metadata into a ledger entry for a *successful* read."""
    meta = meta or {}
    planned = int(meta.get("selected_jobs") or meta.get("reading_bundles") or 0)
    failed = int(meta.get("failed_jobs") or 0)
    read = max(0, planned - failed)
    degraded = bool(meta.get("degraded_mode"))
    det = meta.get("integration") == "deterministic"
    if failed:
        level, status = LEVEL_PARTIAL, "partial"
    elif degraded:
        level, status = LEVEL_TARGETED, "targeted"
    elif det:
        level, status = LEVEL_DETERMINISTIC_INTEGRATION, "full"
    else:
        level, status = LEVEL_FULL, "full"
    notes = []
    if failed:
        notes.append(f"{failed} of {planned} planned reading regions could not be read; the dossier contains only evidence verified from the regions that were read")
    if det:
        notes.append("integration call failed; dossier assembled deterministically from verified regional reading packets")
    if degraded:
        notes.append("first attempt failed; recovered with a reduced targeted reading plan")
    return PaperReadStatus(
        source_id=source_id, title=title, level=level, status=status, verified=True,
        attempts=attempts, regions_planned=planned, regions_read=read, regions_failed=failed,
        integration=str(meta.get("integration") or "llm"), cache_hits=int(meta.get("cache_hits") or 0), notes=notes,
    )
def excluded_status(source_id: str, title: str, reason: str, *, attempts: int = 0, diagnosis: str | None = None) -> PaperReadStatus:
    fc = diagnosis or classify_failure(reason)
    return PaperReadStatus(
        source_id=source_id, title=title, level=LEVEL_EXCLUDED, status="excluded", verified=False,
        reason=str(reason or "no verified dossier could be produced")[:400], failure_class=fc, attempts=attempts,
        integration="none",
    )
_NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven"}
def _word(n: int) -> str:
    return _NUMBER_WORDS.get(n, str(n))
def coverage_statement(statuses: list[PaperReadStatus | dict], *, include_titles: bool = True) -> str:
    """Honest, human-readable statement of which selected papers were analysed.
    Returns "" when every selected paper was analysed to level 1-3 (nothing to
    disclose).  Never claims a failed paper was analysed."""
    rows = [s.to_dict() if isinstance(s, PaperReadStatus) else dict(s) for s in statuses]
    if not rows:
        return ""
    total = len(rows)
    verified = [r for r in rows if r.get("verified")]
    excluded = [r for r in rows if not r.get("verified")]
    partial = [r for r in verified if r.get("status") == "partial"]
    if not excluded and not partial:
        return ""
    parts: list[str] = []
    if excluded:
        n_ok = len(verified)
        head = f"{_word(n_ok)} of the {_word(total).lower()} selected papers were analysed in depth" if n_ok else f"None of the {_word(total).lower()} selected papers could be analysed in depth"
        if n_ok and len(excluded) == 1:
            head += "; the remaining paper could not be verified and was therefore not used for any claim in this report."
        elif n_ok:
            head += f"; the remaining {_word(len(excluded)).lower()} could not be verified and were therefore not used for any claim in this report."
        else:
            head += "."
        parts.append(head)
        if include_titles:
            for r in excluded:
                title = (r.get("title") or r.get("source_id") or "").strip()
                reason = (r.get("reason") or "no verified dossier").strip().rstrip(".")
                parts.append(f"- Not analysed: \"{title[:140]}\" ({reason}).")
    if partial:
        parts.append(
            ("Partially analysed: " if len(partial) > 1 else "Partially analysed: ")
            + "; ".join(f"\"{(r.get('title') or r.get('source_id') or '')[:100]}\" ({r.get('regions_read', 0)} of {r.get('regions_planned', 0)} planned reading regions)" for r in partial)
            + ". Findings from these papers are limited to the regions that were read and verified."
        )
    return "\n".join(parts)
def insert_coverage_note(report_md: str, note: str) -> str:
    """Insert an 'Evidence coverage' block right after the report's title line."""
    lines = report_md.split("\n")
    block = ["", "> **Evidence coverage.** " + note.replace("\n", "\n> "), ""]
    for i, ln in enumerate(lines[:12]):
        if ln.startswith("# "):
            return "\n".join(lines[: i + 1] + block + lines[i + 1:])
    return "\n".join(block[1:] + [""] + lines)
_LIST_KEYS = ("facts", "mechanisms", "experiments", "results", "ablations", "limitations", "failures", "assumptions", "surprises")
def _obs_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for k in ("observation", "finding", "detail", "summary", "claim", "text", "result", "description"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        vals = [str(v) for v in item.values() if isinstance(v, (str, int, float)) and str(v).strip()]
        return "; ".join(vals)[:600]
    return ""
def _join(items: list[Any], limit: int = 12, width: int = 500) -> str:
    seen, out = set(), []
    for it in items:
        t = re.sub(r"\s+", " ", _obs_text(it)).strip()
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        out.append(t[:width])
        if len(out) >= limit:
            break
    return " | ".join(out)
def build_partial_dossier(packets: list[dict], section_map: list[dict] | None = None) -> dict:
    """Assemble a dossier dict *without any LLM call* from reading packets.
    Only content the regional readers actually returned is used (no
    paraphrasing, no invention).  The caller verifies quotes against the source
    text afterwards, exactly as for the LLM-integrated dossier."""
    buckets: dict[str, list[Any]] = {k: [] for k in _LIST_KEYS}
    quotes: list[Any] = []
    notes: list[str] = []
    for p in packets or []:
        pk = p.get("packet", p) if isinstance(p, dict) else None
        if not isinstance(pk, dict):
            continue
        for k in _LIST_KEYS:
            v = pk.get(k)
            if isinstance(v, list):
                buckets[k].extend(v)
            elif isinstance(v, (str, dict)) and v:
                buckets[k].append(v)
        q = pk.get("evidence_quotes")
        if isinstance(q, list):
            quotes.extend(q)
        fn = pk.get("feynman_notes")
        if isinstance(fn, str) and fn.strip():
            notes.append(fn.strip())
        elif isinstance(fn, (list, dict)):
            t = _join(fn if isinstance(fn, list) else [fn], limit=4)
            if t:
                notes.append(t)
    facts = buckets["facts"]
    by_type: dict[str, list[Any]] = {}
    for f in facts:
        if isinstance(f, dict):
            by_type.setdefault(str(f.get("type") or "other").lower(), []).append(f)
    def grp(*types: str) -> list[Any]:
        out: list[Any] = []
        for t in types:
            out.extend(by_type.get(t, []))
        return out
    dossier: dict[str, Any] = {
        "problem": _join(grp("claim", "positioning", "open_question")[:6]),
        "core_idea": _join(grp("claim", "method")[:4], limit=4),
        "method_summary": _join(grp("method", "implementation") + buckets["mechanisms"][:4], limit=8),
        "mechanism": _join(buckets["mechanisms"] + grp("mechanism"), limit=8),
        "experimental_setup": _join(buckets["experiments"] + grp("dataset", "metric", "baseline"), limit=10),
        "key_results": _join(buckets["results"] + grp("result"), limit=12),
        "ablations": _join(buckets["ablations"] + grp("ablation"), limit=8),
        "limitations_author_stated": _join(buckets["limitations"] + grp("limitation"), limit=8),
        "failure_modes": _join(buckets["failures"] + grp("failure", "negative_result"), limit=8),
        "assumptions": _join(buckets["assumptions"] + grp("assumption"), limit=8),
        "negative_results": _join(grp("negative_result"), limit=6),
        "feynman_explanation": _join(notes, limit=4, width=900),
    }
    dossier = {k: v for k, v in dossier.items() if v}
    inventory: list[dict] = []
    for f in facts:
        if isinstance(f, dict) and _obs_text(f):
            inventory.append({k: f.get(k) for k in ("type", "observation", "conditions", "quote", "location", "section_id", "importance") if f.get(k) not in (None, "")})
    for k in ("mechanisms", "experiments", "results", "ablations", "limitations", "failures", "assumptions", "surprises"):
        for it in buckets[k]:
            if _obs_text(it):
                rec = dict(it) if isinstance(it, dict) else {"observation": _obs_text(it)}
                rec.setdefault("type", k.rstrip("s"))
                inventory.append(rec)
    dossier["evidence_inventory"] = inventory[:200]
    dossier["research_memory"] = inventory[:200]
    dossier["evidence_quotes"] = quotes[:200]
    if section_map:
        dossier["section_map"] = section_map
    return dossier
def salvage_truncated_json(text: str) -> Any:
    """Recover the longest valid prefix of a JSON document cut off mid-stream.
    Reasoning models spend completion tokens on hidden reasoning, so a long
    structured answer is frequently truncated.  Rather than discard the whole
    reply we close any open strings/brackets and drop an incomplete trailing
    member.  Returns the parsed value, or raises ValueError."""
    import json
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    start = min([i for i in (s.find("{"), s.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("no JSON object found")
    s = s[start:]
    stack: list[str] = []
    in_str = False
    esc = False
    last_good = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
            last_good = i + 1
        elif ch == ",":
            last_good = i
    candidates = []
    if last_good > 0:
        candidates.append(s[:last_good])
    candidates.append(s)
    for cand in candidates:
        st: list[str] = []
        ins = False
        es = False
        for ch in cand:
            if ins:
                if es:
                    es = False
                elif ch == "\\":
                    es = True
                elif ch == '"':
                    ins = False
                continue
            if ch == '"':
                ins = True
            elif ch in "{[":
                st.append("}" if ch == "{" else "]")
            elif ch in "}]" and st:
                st.pop()
        body = cand + ('"' if ins else "")
        body = re.sub(r",\s*$", "", body)
        for _ in range(3):
            try:
                return json.loads(body + "".join(reversed(st)))
            except json.JSONDecodeError:
                body2 = re.sub(r',?\s*"[^"\\]*"\s*:?\s*$', "", body)
                body2 = re.sub(r",\s*$", "", body2)
                if body2 == body:
                    break
                body = body2
    raise ValueError("could not salvage truncated JSON")
