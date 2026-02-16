from dataclasses import dataclass

from nd_semiotic.language.natural.large_model.rag.chunk import \
    Chunk


@dataclass(frozen=True)
class RetrievalHit:
    score: float
    chunk: Chunk