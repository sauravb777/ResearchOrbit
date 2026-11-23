from __future__ import annotations
import re
def _clean(value: str) -> str:
    return " ".join(str(value or "").split()).strip()[:420]
def is_cross_view_geolocalization_query(question: str) -> bool:
    """Identify cross-view visual localization/geolocation research intent."""
    low = _clean(question).casefold().replace("geolocalisation", "geolocalization")
    domain = ("geolocal", "geo-local", "geolocation", "cross-view", "cross view",
              "crossview", "satellite", "ground-to-aerial", "ground-to-satellite",
              "aerial-ground", "video geolocation")
    visual_task = ("image", "video", "local", "camera", "satellite", "cross-view",
                   "cross view", "aerial", "annotation", "dataset", "trajectory",
                   "street view", "ground view")
    return any(term in low for term in domain) and any(term in low for term in visual_task)
def build_discovery_queries(question: str, planned_queries: list[str] | None = None, limit: int = 10) -> list[str]:
    """Return compact, prioritized, non-duplicated arXiv queries.
    Technical domain queries run before long natural-language prose. For domain
    searches, a few dataset/annotation and video/camera-pose facets are included.
    The original question is always reserved as the final query when non-empty.
    """
    q = _clean(question)
    planned = [_clean(x) for x in (planned_queries or []) if _clean(x)]
    targeted: list[str] = []
    if is_cross_view_geolocalization_query(q):
        targeted = [
            'all:"cross-view geo-localization" OR all:"cross-view geolocalization" OR all:"cross-view geolocation"',
            'all:"ground-to-aerial" OR all:"ground-to-satellite" OR all:"cross-view image matching"',
            'all:"video geolocation" OR all:"video localization" OR all:"camera trajectory" OR all:"camera localization"',
            'all:"visual geo-localization" OR all:"visual geolocation" OR all:"image-based geolocation" OR all:"visual place recognition"',
            'all:"satellite image" AND all:localization',
            'all:geolocalization AND all:dataset OR all:"cross-view" AND all:annotation',
            'all:"ground-to-satellite" AND all:benchmark OR all:"camera trajectory" AND all:localization',
        ]
    max_items = max(1, min(16, int(limit)))
    reserve_question = bool(q)
    budget = max(0, max_items - int(reserve_question))
    ordered = targeted + planned
    result: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        item = _clean(item)
        key = re.sub(r"\s+", " ", item).casefold()
        if item and key not in seen:
            result.append(item)
            seen.add(key)
        if len(result) >= budget:
            break
    if q:
        key = re.sub(r"\s+", " ", q).casefold()
        if key not in seen:
            result.append(q)
        elif q in result:
            result.remove(q)
            result.append(q)
    return result[:max_items]
def geo_localization_relevance(question: str, title: str, abstract: str) -> tuple[float, list[str]]:
    """Small transparent topical prior; the LLM relevance judge remains primary."""
    if not is_cross_view_geolocalization_query(question):
        return 0.0, []
    text = f"{title or ''} {abstract or ''}".casefold()
    groups = {
        "geolocation/localization": ("geolocal", "geo-local", "geolocation", "localization", "localisation", "camera pose", "place recognition", "location estimation"),
        "cross-view imagery": ("cross-view", "cross view", "crossview", "ground-to-aerial", "ground-to-satellite", "aerial-ground", "ground view", "street view", "satellite image", "aerial image"),
        "video/trajectory": ("video geolocation", "video localization", "video localisation", "camera trajectory", "camera localization", "camera localisation", "visual odometry", "trajectory estimation"),
        "visual matching/retrieval": ("image matching", "image retrieval", "visual localization", "visual localisation", "visual place", "feature matching", "image-based", "image based geolocation"),
    }
    hits = [label for label, terms in groups.items() if any(term in text for term in terms)]
    if len(hits) >= 2:
        return 0.16, ["Domain match: " + ", ".join(hits)]
    if len(hits) == 1:
        return 0.06, ["Partial domain match: " + hits[0]]
    return -0.10, ["Weak topical match; verify relevance before selection"]
