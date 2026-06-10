"""Query-relevance scoring with a graceful backend fallback chain.

Backends, in preference order:
  1. embeddings — semantic cosine similarity via sentence-transformers (best
     quality; needs the optional dependency + a cached model).
  2. tfidf — scikit-learn TF-IDF cosine (no model download, term-rarity aware).
  3. heuristic — rarity-weighted token overlap (always available, no deps).

``relevance_scores`` always returns a list of floats in [0, 1], one per text,
and never raises: if the requested backend is unavailable it silently degrades
to the next one and logs which backend actually ran.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

import structlog

log = structlog.get_logger()

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Small, high-frequency stopword set — kept inline to avoid an nltk dependency.
_STOPWORDS = frozenset("""
a an the of in on at by to for from with without and or but if then else
will would shall should can could may might must do does did is are was were
be been being has have had this that these those it its as into over under
than then so such not no yes more most less least up down out about after
before between during through how what when where which who whom why
""".split())


# --------------------------------------------------------------------------
# Tokenization helpers
# --------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1]


def _normalize(scores: list[float]) -> list[float]:
    """Scale scores to [0, 1] by their max (0 if all non-positive)."""
    hi = max(scores, default=0.0)
    if hi <= 0:
        return [0.0 for _ in scores]
    return [max(0.0, s) / hi for s in scores]


# --------------------------------------------------------------------------
# Backend 3: heuristic (rarity-weighted overlap) — always available
# --------------------------------------------------------------------------

def _heuristic_scores(query: str, texts: list[str]) -> list[float]:
    q_tokens = set(_tokens(query))
    if not q_tokens:
        return [0.0 for _ in texts]

    doc_tokens = [set(_tokens(t)) for t in texts]
    n_docs = len(texts) or 1
    # Inverse document frequency over THIS candidate set: a query word that
    # appears in every candidate carries no discriminating signal.
    df: dict[str, int] = {}
    for toks in doc_tokens:
        for w in toks & q_tokens:
            df[w] = df.get(w, 0) + 1

    scores: list[float] = []
    for toks in doc_tokens:
        overlap = toks & q_tokens
        s = 0.0
        for w in overlap:
            idf = math.log((n_docs + 1) / (df.get(w, 0) + 1)) + 1.0
            s += idf
        # Length-normalize so long docs don't win by volume.
        s = s / (1.0 + math.log(len(toks) + 1)) if toks else 0.0
        scores.append(s)
    return _normalize(scores)


# --------------------------------------------------------------------------
# Backend 2: TF-IDF cosine (scikit-learn)
# --------------------------------------------------------------------------

def _tfidf_scores(query: str, texts: list[str]) -> list[float] | None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=4096)
        matrix = vec.fit_transform([query] + texts)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).ravel().tolist()
        return _normalize(sims)
    except Exception as e:  # empty vocab, etc.
        log.debug("relevance.tfidf_failed", error=str(e))
        return None


# --------------------------------------------------------------------------
# Backend 1: embeddings (sentence-transformers)
# --------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _get_embedder(model_name: str):
    """Lazily load and cache a SentenceTransformer; return None if unavailable."""
    # Quiet the HF Hub chatter that was spamming the trading terminal
    # ("unauthenticated requests" warnings + "Loading weights" progress
    # bars). Loading still works without a token; these are cosmetic.
    import logging as _logging
    import os as _os
    _os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    _os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _logging.getLogger("huggingface_hub").setLevel(_logging.ERROR)
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        log.info("relevance.embeddings_unavailable", reason="sentence-transformers not installed")
        return None
    try:
        model = SentenceTransformer(model_name)
        log.info("relevance.embeddings_loaded", model=model_name)
        return model
    except Exception as e:
        log.warning("relevance.embeddings_load_failed", model=model_name, error=str(e))
        return None


def _embedding_scores(query: str, texts: list[str], model_name: str) -> list[float] | None:
    model = _get_embedder(model_name)
    if model is None:
        return None
    try:
        import numpy as np

        vecs = model.encode([query] + texts, normalize_embeddings=True, show_progress_bar=False)
        q = np.asarray(vecs[0])
        docs = np.asarray(vecs[1:])
        sims = (docs @ q).tolist()
        # cosine of normalized vectors is in [-1, 1]; clamp negatives to 0.
        return _normalize([max(0.0, s) for s in sims])
    except Exception as e:
        log.warning("relevance.embeddings_encode_failed", error=str(e))
        return None


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def relevance_scores(
    query: str,
    texts: list[str],
    *,
    backend: str = "embeddings",
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float]:
    """Score each text's relevance to ``query`` in [0, 1].

    Falls through embeddings -> tfidf -> heuristic depending on ``backend`` and
    what's actually available. Never raises.
    """
    if not texts:
        return []

    order = {
        "embeddings": ("embeddings", "tfidf", "heuristic"),
        "tfidf": ("tfidf", "heuristic"),
        "heuristic": ("heuristic",),
    }.get(backend, ("embeddings", "tfidf", "heuristic"))

    for b in order:
        if b == "embeddings":
            scores = _embedding_scores(query, texts, model_name)
        elif b == "tfidf":
            scores = _tfidf_scores(query, texts)
        else:
            scores = _heuristic_scores(query, texts)
        if scores is not None:
            if b != backend:
                log.debug("relevance.backend_fallback", requested=backend, used=b)
            return scores

    # Should be unreachable (heuristic never returns None), but be safe.
    return [0.0 for _ in texts]
