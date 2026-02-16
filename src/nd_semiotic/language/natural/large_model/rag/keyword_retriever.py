import re
from typing import List
from pathlib import Path
from nd_semiotic.language.natural.large_model.rag.chunk import Chunk
from nd_semiotic.language.natural.large_model.rag.retrieval_hit import \
    RetrievalHit


# -----------------------------
# Keyword retrieval
# -----------------------------
class KeywordRetriever:
    def __init__(self, chunks: List[Chunk]):
        self._chunks = chunks

    def search(self, query: str, k: int = 8) -> List[RetrievalHit]:
        q = (query or "").strip().lower()
        if not q:
            return []
        terms = [t for t in re.split(r"\s+", q) if t]

        scored: List[RetrievalHit] = []
        for ch in self._chunks:
            tl = ch.text.lower()
            path_lower = ch.doc_path.lower()
            stem_lower = Path(ch.doc_path).stem.lower()
            haystack = tl + "\n" + path_lower + "\n" + stem_lower

            if q in haystack:
                count = haystack.count(q)
                score = 2.0 + min(1.0, count * 0.1)
                scored.append(RetrievalHit(score=float(score), chunk=ch))
                continue

            if len(terms) == 1:
                t0 = terms[0]
                if t0 in haystack:
                    count = haystack.count(t0)
                    score = 1.0 + min(1.0, count * 0.05)
                    scored.append(RetrievalHit(score=float(score), chunk=ch))
                continue

            hit_count = sum(1 for t in terms if t in haystack)
            if hit_count > 0:
                score = hit_count / max(1, len(terms))
                scored.append(RetrievalHit(score=float(score), chunk=ch))

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[: int(k)]