from pathlib import Path
from typing import List, Optional
import faiss
import numpy as np

from nd_semiotic.language.natural.large_model.rag.chunk import Chunk
from nd_semiotic.language.natural.large_model.rag.retrieval_hit import \
    RetrievalHit


# -----------------------------
# Index
# -----------------------------
class FaissIndex:
    def __init__(self):
        self._index: Optional[faiss.Index] = None
        self._chunks: List[Chunk] = []
        self._dim: Optional[int] = None

    def build(self, embeddings: np.ndarray, chunks: List[Chunk]) -> None:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D.")
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks/embeddings mismatch.")
        self._dim = int(embeddings.shape[1])
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(embeddings.astype(np.float32))
        self._chunks = chunks

    def search(self, query_emb: np.ndarray, k: int) -> List[RetrievalHit]:
        if self._index is None:
            raise RuntimeError("Index not built.")
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        faiss.normalize_L2(query_emb)
        scores, ids = self._index.search(query_emb.astype(np.float32), int(k))
        out: List[RetrievalHit] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            out.append(RetrievalHit(score=float(score), chunk=self._chunks[int(idx)]))
        return out

    def save(self, index_path: Path, meta_path: Path) -> None:
        if self._index is None:
            raise RuntimeError("Index not built.")
        faiss.write_index(self._index, str(index_path))
        meta = {"dim": self._dim, "chunks": [c.__dict__ for c in self._chunks]}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def load(self, index_path: Path, meta_path: Path) -> None:
        self._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._dim = int(meta["dim"])
        self._chunks = [Chunk(**c) for c in meta["chunks"]]

    def chunks(self) -> List[Chunk]:
        return self._chunks