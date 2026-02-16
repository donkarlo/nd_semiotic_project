# -----------------------------
# Policy + instructions
# -----------------------------
from typing import List

from nd_semiotic.language.natural.large_model.rag.retrieval_hit import \
    RetrievalHit


class AnswerPolicy:
    def __init__(
            self,
            vector_min_score: float = 0.22,
            keyword_min_score: float = 0.6,
            allow_general_fallback: bool = True,
            show_sources: bool = True,
            max_sources: int = 6,
    ):
        self._vector_min_score = float(vector_min_score)
        self._keyword_min_score = float(keyword_min_score)
        self._allow_general_fallback = bool(allow_general_fallback)
        self._show_sources = bool(show_sources)
        self._max_sources = int(max_sources)

    def should_answer_with_context(self, vec_hits: List[RetrievalHit], kw_hits: List[RetrievalHit]) -> bool:
        if not vec_hits and not kw_hits:
            return False
        best_vec = vec_hits[0].score if vec_hits else None
        best_kw = kw_hits[0].score if kw_hits else None
        vec_ok = (best_vec is not None) and (best_vec >= self._vector_min_score)
        kw_ok = (best_kw is not None) and (best_kw >= self._keyword_min_score)
        return bool(vec_ok or kw_ok)

    def allow_general_fallback(self) -> bool:
        return self._allow_general_fallback

    def show_sources(self) -> bool:
        return self._show_sources

    def max_sources(self) -> int:
        return self._max_sources