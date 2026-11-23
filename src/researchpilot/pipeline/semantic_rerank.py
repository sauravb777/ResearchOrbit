from __future__ import annotations
from ..config import get_settings
from ..schemas import Source
from .ranker import score_source
_model_cache: dict[str, object] = {}
_EMBED_CACHE: dict[tuple[str, str], object] = {}
def _load_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    _model_cache[model_name] = model
    return model
def semantic_rerank(query: str, sources: list[Source], blend: float = 0.4) -> bool:
    """Mutates `source.score` in place, blending heuristic and semantic
    similarity. Returns True if semantic scoring was actually applied,
    False if it was skipped for any reason (not enabled, not installed,
    model load/embedding failure) — callers can use this to decide whether
    to mention it in the run's audit trail, but never need to branch on it
    for correctness, since the heuristic score is always computed first.
    """
    settings = get_settings()
    if not settings.enable_semantic_rerank or not sources:
        return False
    try:
        import numpy as np
        model = _load_model(settings.semantic_rerank_model)
        keys = [(settings.semantic_rerank_model, s.id) for s in sources]
        missing = [i for i, k in enumerate(keys) if k not in _EMBED_CACHE]
        if missing:
            texts = [f"{sources[i].title}. {sources[i].abstract}"[:2000] for i in missing]
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            for i, v in zip(missing, vecs):
                _EMBED_CACHE[keys[i]] = v
            if len(_EMBED_CACHE) > 20000:
                for k in list(_EMBED_CACHE)[:2000]:
                    _EMBED_CACHE.pop(k, None)
        query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        doc_vecs = np.stack([_EMBED_CACHE[k] for k in keys])
        similarities = np.dot(doc_vecs, query_vec)
        for source, sim in zip(sources, similarities):
            heuristic = score_source(query, source)
            semantic = max(0.0, min(1.0, (float(sim) + 1.0) / 2.0))
            source.score = round(min(1.0, blend * semantic + (1 - blend) * heuristic), 4)
        return True
    except Exception:
        return False
