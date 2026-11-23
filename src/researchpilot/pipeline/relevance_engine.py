from __future__ import annotations
import math
import re
from datetime import datetime, timezone
from typing import Iterable
from ..schemas import RelevanceAssessment, Source, canonical_report_type
RELEVANCE_WEIGHTS = {
    "semantic_relevance": 0.70,
    "recency": 0.15,
    "publication_signal": 0.15,
}
AUTO_RELEVANCE_THRESHOLD = 0.39
AUTO_PRIORITY_THRESHOLD = 0.44
AUTO_MIN_PAPERS = 3
AUTO_FALLBACK_PAPERS = 3
AUTO_MAX_PAPERS = 4
def automatic_selection_target(ranked_sources: Iterable[Source], score_getter=None) -> int:
    """Return the automatic deep-read target for a ranked eligible corpus.
    Contract:
      * >44% papers are preferred when they form a meaningful evidence set.
      * Keep at least the configured minimum when enough eligible candidates exist.
      * If the priority set is empty, use a small fallback evidence set.
      * Never exceed the configured product ceiling.
    This function only decides the target count. Callers still require the raw
    >=39% relevance gate and verified readable full text before a paper can be
    admitted.
    """
    items = list(ranked_sources)
    if not items:
        return 0
    get_score = score_getter or (lambda s: float(getattr(s, "score", 0.0) or 0.0))
    eligible = [s for s in items if get_score(s) >= AUTO_RELEVANCE_THRESHOLD]
    if not eligible:
        return 0
    priority = [s for s in eligible if get_score(s) > AUTO_PRIORITY_THRESHOLD]
    if not priority:
        return min(AUTO_FALLBACK_PAPERS, len(eligible))
    return min(AUTO_MAX_PAPERS, max(AUTO_MIN_PAPERS, len(priority)), len(eligible))
def weights(report_type: str) -> dict[str, float]:
    """Compatibility helper. The score is intentionally report-type invariant."""
    return dict(RELEVANCE_WEIGHTS)
def _year_value(year: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(year or ""))
    return int(m.group(0)) if m else None
def recency_score(year: str, now: datetime | None = None) -> float | None:
    """Return a 0..1 exponential recency score.
    The newest publication year receives 1.0. The exponential decay means the
    absolute point loss is larger near the present and progressively smaller
    for older years, matching the intended "front-loaded" recency penalty.
    Half-life is three years.
    """
    y = _year_value(year)
    if y is None:
        return None
    current = (now or datetime.now(timezone.utc)).year
    age = max(0, current - y)
    return max(0.0, min(1.0, 2.0 ** (-age / 3.0)))
_JOURNAL_QUARTILE_POINTS = {
    "1": 15.0, "2": 12.0, "3": 9.0, "4": 7.0, "5": 5.0,
}
_CONFERENCE_TIERS: dict[int, tuple[str, ...]] = {
    1: (
        "neurips", "neural information processing systems",
        "icml", "international conference on machine learning",
        "iclr", "international conference on learning representations",
        "cvpr", "computer vision and pattern recognition",
        "iccv", "international conference on computer vision",
        "eccv", "european conference on computer vision",
        "acl", "annual meeting of the association for computational linguistics",
        "emnlp", "empirical methods in natural language processing",
        "naacl", "north american chapter of the association for computational linguistics",
        "aaai", "association for the advancement of artificial intelligence",
        "ijcai", "international joint conference on artificial intelligence",
        "kdd", "knowledge discovery and data mining",
        "sigir", "special interest group on information retrieval",
        "chi", "human factors in computing systems",
        "the web conference", "www conference",
        "miccai", "medical image computing and computer assisted intervention",
    ),
    2: (
        "aistats", "artificial intelligence and statistics",
        "uai", "uncertainty in artificial intelligence",
        "coling", "international conference on computational linguistics",
        "eacl", "european chapter of the association for computational linguistics",
        "cikm", "conference on information and knowledge management",
        "wsdm", "web search and data mining",
        "recsys", "recommender systems",
        "ecir", "european conference on information retrieval",
        "bmvc", "british machine vision conference",
        "accv", "asian conference on computer vision",
        "icassp", "acoustics speech and signal processing",
        "icme", "international conference on multimedia and expo",
        "interspeech", "international conference on spoken language processing",
        "ubicomp", "pervasive and ubiquitous computing",
        "sigmod", "management of data",
        "vldb", "very large databases",
        "pods", "principles of database systems",
        "focs", "foundations of computer science",
        "soda", "discrete algorithms",
        "stoc", "theory of computing",
    ),
    3: (
        "icdm", "international conference on data mining",
        "sdm", "siam international conference on data mining",
        "pkdd", "principles and practice of knowledge discovery in databases",
        "ecml-pkdd", "european conference on machine learning and principles",
        "paclic", "pacific asia conference on language information and computation",
        "asonam", "advances in social networks analysis and mining",
        "wacv", "winter conference on applications of computer vision",
        "3dv", "international conference on 3d vision",
        "icpr", "international conference on pattern recognition",
        "acmmm", "multimedia conference",
        "mm", "multimedia conference",
        "naacl findings", "acl findings", "emnlp findings",
    ),
    4: (
        "workshop on", "workshops", "workshop",
        "lrec", "language resources and evaluation conference",
        "icdar", "document analysis and recognition",
        "apsipa", "asia-pacific signal and information processing association",
    ),
    5: (
        "proceedings of the international conference",
        "proceedings of the annual conference",
    ),
}
def _venue_text(source: Source) -> str:
    return " ".join(
        x.strip() for x in (source.journal_ref, source.venue) if x and x.strip()
    ).lower()
def _venue_name_matches(text: str, name: str) -> bool:
    """Match a venue name without allowing short acronyms to hit substrings.
    For example, ``CHI`` must not match ``machine`` and ``MM`` must not match
    arbitrary words. Longer venue names can safely use substring matching.
    """
    name = name.strip().lower()
    if len(name) <= 4 and re.fullmatch(r"[a-z0-9]+", name):
        return bool(re.search(rf"\b{re.escape(name)}\b", text))
    return name in text
def publication_signal_details(source: Source) -> tuple[float, str]:
    """Return normalized publication signal and its auditable basis.
    The return value is normalized to 0..1 because the production weight is
    15%. The basis is metadata only and is never another scoring dimension.
    """
    text = _venue_text(source)
    if not text or text == "arxiv":
        return 0.0, "unverified venue / arXiv preprint"
    q = re.search(r"\bq([1-5])\b", text, flags=re.I)
    if q:
        points = _JOURNAL_QUARTILE_POINTS[q.group(1)]
        return points / 15.0, f"journal Q{q.group(1)} ({points:g}/15)"
    tier = re.search(r"\b(?:conference|venue)[ -]?tier[ :]?([1-5])\b", text, flags=re.I)
    if tier:
        t = int(tier.group(1))
        points = {1:15.0, 2:12.0, 3:9.0, 4:7.0, 5:5.0}[t]
        return points / 15.0, f"conference/venue tier {t} ({points:g}/15)"
    tier_word = re.search(r"\btop[- ]tier\b|\btier[- ]([1-5])\b", text, flags=re.I)
    if tier_word:
        if tier_word.group(1):
            t = int(tier_word.group(1))
        else:
            t = 1
        points = {1:15.0, 2:12.0, 3:9.0, 4:7.0, 5:5.0}[t]
        return points / 15.0, f"explicit venue tier {t} ({points:g}/15)"
    findings_names = ("acl findings", "findings of acl", "emnlp findings", "findings of emnlp", "naacl findings", "findings of naacl")
    for name in findings_names:
        if _venue_name_matches(text, name):
            return 9.0 / 15.0, f"recognized conference tier 3 (9/15): {name}"
    for tier_num in sorted(_CONFERENCE_TIERS):
        for name in sorted(_CONFERENCE_TIERS[tier_num], key=len, reverse=True):
            if name in findings_names:
                continue
            if _venue_name_matches(text, name):
                points = {1:15.0, 2:12.0, 3:9.0, 4:7.0, 5:5.0}[tier_num]
                return points / 15.0, f"recognized conference tier {tier_num} ({points:g}/15): {name}"
    return 0.0, "venue identified but no recognized tier verified"
def publication_signal(source: Source) -> float:
    """Return the single 0..1 publication signal used by the 15% budget."""
    return publication_signal_details(source)[0]
def publication_score(source: Source) -> float:
    """Return publication points on the exact 0..15 scale."""
    return round(publication_signal(source) * 15.0, 2)
def venue_score(source: Source) -> float:
    """Compatibility alias returning the old 0..1-shaped venue signal."""
    return publication_signal(source)
def citation_score(count: int | None) -> float:
    """Legacy compatibility helper. Citations are NOT used in ranking."""
    if count is None or count <= 0:
        return 0.0
    return min(1.0, math.log1p(count) / math.log1p(5000))
def foundational_score(source: Source) -> float:
    """Legacy compatibility helper. Foundational importance is NOT ranked."""
    return 0.0
def normalize_available(values: dict[str, float | None], weight_map: dict[str, float]) -> tuple[float, float, list[str]]:
    """Legacy utility retained for external callers/tests.
    The production relevance score does NOT use missing-signal renormalization:
    unavailable recency/publication metadata contributes zero rather than
    silently increasing the semantic component.
    """
    numerator = 0.0
    denominator = 0.0
    missing = []
    for name, weight in weight_map.items():
        value = values.get(name)
        if value is None:
            missing.append(name)
            continue
        numerator += max(0.0, min(1.0, float(value))) * weight
        denominator += weight
    return ((numerator / denominator) if denominator else 0.0), denominator, missing
def deterministic_metadata(source: Source) -> dict[str, float]:
    return {
        "recency": recency_score(source.year),
        "publication_signal": publication_signal(source),
    }
def score_assessment(a: RelevanceAssessment, source: Source, report_type: str) -> RelevanceAssessment:
    """Apply exactly the three production signals."""
    semantic = getattr(a, "semantic_relevance", 0.0)
    semantic = max(0.0, min(1.0, float(semantic or 0.0)))
    meta = deterministic_metadata(source)
    recency = float(meta["recency"] or 0.0)
    pub = float(meta["publication_signal"] or 0.0)
    overall = (
        RELEVANCE_WEIGHTS["semantic_relevance"] * semantic
        + RELEVANCE_WEIGHTS["recency"] * recency
        + RELEVANCE_WEIGHTS["publication_signal"] * pub
    )
    a.semantic_relevance = round(semantic, 4)
    a.recency = round(recency, 4) if meta["recency"] is not None else None
    a.publication_signal = round(pub, 4)
    a.venue_quality = a.publication_signal
    a.citation_influence = None
    a.foundational_importance = None
    a.overall = round(overall, 4)
    a.importance = round(
        RELEVANCE_WEIGHTS["recency"] * recency
        + RELEVANCE_WEIGHTS["publication_signal"] * pub, 4
    )
    a.available_weight = 1.0
    a.corpus_value = 0.0
    missing = []
    if _year_value(source.year) is None:
        missing.append("recency")
    if publication_signal(source) <= 0:
        missing.append("publication_signal")
    a.missing_signals = missing
    a.ranking_confidence = round(semantic, 4)
    _, publication_basis = publication_signal_details(source)
    a.reasons = list(dict.fromkeys([
        f"Semantic research relevance: {semantic:.0%}",
        f"Recency: {recency:.0%}",
        f"Publication signal: {meta['publication_signal']:.0%} ({publication_basis})",
    ] + list(a.reasons)))[:8]
    return a
def apply_ranking(sources: list[Source], assessments: dict[str, RelevanceAssessment], report_type: str) -> None:
    for s in sources:
        a = assessments.get(s.id)
        if a is None:
            continue
        s.relevance_score = round(a.overall * 100, 2)
        s.importance_score = round(a.importance * 100, 2)
        s.corpus_value_score = 0.0
        s.ranking_confidence = round(a.ranking_confidence * 100, 2)
        s.ranking_components = {
            "semantic_relevance": a.semantic_relevance,
            "recency": a.recency,
            "publication_signal": a.publication_signal,
        }
        s.missing_ranking_signals = list(a.missing_signals)
        s.score = round(a.overall, 4)
def batch_candidates(sources: Iterable[Source], batch_size: int = 10) -> list[list[Source]]:
    items = list(sources)
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
