import re

from nd_semiotic.language.natural.large_model.rag.chunk import Chunk
from typing import List, Tuple


# -----------------------------
# Chunking
# -----------------------------


class Chunker:
    def __init__(self, chunk_size_chars: int = 1500, overlap_chars: int = 150, max_chunks_total: int = 5000):
        self._chunk_size = int(chunk_size_chars)
        self._overlap = int(overlap_chars)
        self._max_chunks_total = int(max_chunks_total)

    def chunk_all(self, docs: List[Tuple[str, str]]) -> List[Chunk]:
        chunks: List[Chunk] = []
        for doc_path, text in docs:
            for ch in self.chunk(doc_path, text):
                chunks.append(ch)
                if len(chunks) >= self._max_chunks_total:
                    return chunks
        return chunks

    def chunk(self, doc_path: str, text: str) -> List[Chunk]:
        s = self._normalize(text)
        if not s.strip():
            return []
        out: List[Chunk] = []
        start = 0
        cid = 0
        n = len(s)
        while start < n:
            end = min(n, start + self._chunk_size)
            part = s[start:end].strip()
            if part:
                out.append(Chunk(doc_path=doc_path, chunk_id=cid, text=part))
                cid += 1
            if end == n:
                break
            start = max(0, end - self._overlap)
        return out

    def _normalize(self, s: str) -> str:
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s