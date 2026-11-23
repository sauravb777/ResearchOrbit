from __future__ import annotations
import importlib
import sys
import time
OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
def _line(tag: str, msg: str) -> None:
    print(f"{tag} {msg}")
def main() -> int:
    problems = 0
    print("ResearchPilot doctor\n" + "=" * 40)
    _line(OK if sys.version_info >= (3, 10) else BAD, f"Python {sys.version.split()[0]} (needs 3.10+)")
    for mod, why in [("fastapi", "API server"), ("uvicorn", "API server"), ("pydantic", "schemas"), ("requests", "HTTP"),
                     ("langgraph", "agent graph"), ("fitz", "PDF reading (PyMuPDF)"), ("docx", "DOCX export"), ("reportlab", "PDF export")]:
        try:
            importlib.import_module(mod)
            _line(OK, f"import {mod}")
        except Exception as exc:
            if mod == "fitz":
                try:
                    importlib.import_module("pypdf")
                    _line(WARN, "PyMuPDF missing; falling back to pypdf (works, slightly weaker on tables). pip install PyMuPDF")
                    continue
                except Exception:
                    pass
            problems += 1
            _line(BAD, f"import {mod} ({why}) failed: {exc}. Fix: pip install -r requirements.txt")
    from .config import get_settings
    from .llm import LLMClient
    settings = get_settings()
    llm = LLMClient(settings=settings)
    if not llm.available:
        problems += 1
        _line(BAD, "No GROQ_API_KEY found. v79 requires Groq for retrieval, deep reading and report generation.")
    else:
        for p in llm.health()["providers"]:
            if not p["configured"]:
                continue
            _line(OK, f"key configured for {p['provider']}; configured models: {', '.join(p['models'])}")
            if p["provider"] == "groq":
                active = llm.list_groq_models()
                if active:
                    missing = [m for m in p["models"] if m not in active]
                    if missing:
                        problems += 1
                        _line(BAD, f"configured Groq model(s) are not active: {', '.join(missing)}")
                    else:
                        _line(OK, f"Groq model list verified ({len(active)} active models)")
                else:
                    _line(WARN, "Could not verify Groq model list; the first completion will validate the configured ids.")
        t0 = time.time()
        reply = llm.complete("You answer with one word.", "Reply with the word READY.", max_tokens=20)
        if reply and reply.strip():
            _line(OK, f"LLM call works ({time.time() - t0:.1f}s): {reply.strip()[:40]!r}")
        else:
            problems += 1
            _line(BAD, "LLM call failed. Recent provider errors:")
            for e in llm.health()["recent_events"][-4:]:
                print(f"        {e['provider']}/{e['model']}: {e['detail'][:200]}")
            print("        Typical causes: wrong/expired key, model name no longer offered (edit GROQ_MODELS), "
                  "rate limit or no credit.")
        if settings.llm_timeout < 60:
            _line(WARN, f"LLM_TIMEOUT={settings.llm_timeout}s is low for long paper-reading prompts; use 90 or more.")
    try:
        from .search.arxiv import ArxivProvider
        t0 = time.time()
        found = ArxivProvider().search("retrieval augmented generation", limit=3)
        if found:
            _line(OK, f"arXiv search works ({len(found)} results in {time.time() - t0:.1f}s)")
            import requests
            url = found[0].pdf_url or found[0].url
            if url:
                r = requests.get(url, stream=True, timeout=25, headers={"User-Agent": "ResearchPilot-doctor"})
                head = next(r.iter_content(1024), b"")
                r.close()
                if r.status_code == 200 and (head.startswith(b"%PDF") or b"pdf" in r.headers.get("content-type", "").lower()):
                    _line(OK, "arXiv PDF download works (full-text deep reading is possible)")
                else:
                    problems += 1
                    _line(BAD, f"arXiv PDF download returned HTTP {r.status_code}. Check firewall/proxy; deep reading needs PDFs.")
        else:
            problems += 1
            _line(BAD, "arXiv returned no results. Check internet/proxy; arXiv also rate-limits (wait a minute and retry).")
    except Exception as exc:
        problems += 1
        _line(BAD, f"arXiv check failed: {exc}")
    print("=" * 40)
    if problems:
        print(f"{problems} problem(s) found. Fix the [FAIL] lines above, then restart the API and regenerate.")
        return 1
    print("Everything needed for a full report is working.")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
