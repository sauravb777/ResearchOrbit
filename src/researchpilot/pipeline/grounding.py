from __future__ import annotations
import re
from ..llm import LLMClient
from ..textutils import split_sentences
_CITATION_RE = re.compile(r"\[([A-Z]\d+)\]")
def extract_citation_tags(text: str) -> set[str]:
    return set(_CITATION_RE.findall(text))
def safe_grounded_text(
    system_prompt: str,
    bullets: list[str],
    allowed_tags: set[str],
    llm: LLMClient | None,
    max_tokens: int = 500,
    preamble: str = "",
) -> tuple[str | None, bool]:
    """Ask the LLM to turn `bullets` into prose grounded ONLY in the
    citation tags present in `allowed_tags`. Returns (text, llm_used).
    `text` is None if the LLM is unavailable, fails, or hallucinates a
    citation outside `allowed_tags` — callers must have a deterministic
    fallback ready.
    `preamble`, when given, is prepended to the user message *before* the
    evidence bullets (e.g. "FOLLOW-UP QUESTION: ..." or "CLAIM TO CHECK:
    ..."). This matters more than it looks: a caller that needs the model
    to answer something specific about the bullets, rather than just turn
    them into prose, must actually put that "something specific" in the
    message the model sees — evidence bullets alone don't communicate a
    question. (This fixes a real bug: `qa.py` used to call this helper
    with only the evidence bullets and no question text at all, so the
    model had nothing to answer and correctly said so.)
    """
    llm = llm or LLMClient()
    if not getattr(llm, "available", False) or not bullets:
        return None, False
    user_msg = "\n".join(f"- {b}" for b in bullets)
    if preamble:
        user_msg = f"{preamble}\n\nEVIDENCE:\n{user_msg}"
    text = llm.complete(system_prompt, user_msg, max_tokens=max_tokens)
    if not text:
        return None, False
    used_tags = extract_citation_tags(text)
    if not used_tags.issubset(allowed_tags):
        return None, False
    marked = re.sub(
        r"(?<=[.!?])\s*((?:\[[A-Z]\d+\]\s*)+)",
        lambda m: f" {m.group(1).strip()}\u241f",
        text,
    )
    sentence_chunks = [chunk.strip() for chunk in marked.split("\u241f") if chunk.strip()]
    if not sentence_chunks or any(not extract_citation_tags(chunk) for chunk in sentence_chunks):
        return None, False
    list_item_re = re.compile(r"(?m)^\s*(?:[-*+] |\d+[.)] )(.+?)\s*$")
    for match in list_item_re.finditer(text):
        item = match.group(1).strip()
        if item and not extract_citation_tags(item):
            return None, False
    return text.strip(), True
