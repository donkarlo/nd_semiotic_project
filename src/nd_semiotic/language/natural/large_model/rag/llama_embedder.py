from typing import List

import faiss
import numpy as np
from llama_cpp import Llama

# -----------------------------
# Embedding
# -----------------------------
class LlamaEmbedder:
    def __init__(self, embed_llm: Llama, max_input_tokens: int, batch_size: int = 8):
        self._llm = embed_llm
        self._max_input_tokens = int(max_input_tokens)
        self._batch_size = max(1, int(batch_size))

    def embed(self, texts: List[str]) -> np.ndarray:
        safe = [self._truncate_by_tokens(t) for t in texts]
        try:
            mat = self._embed_batched(safe)
        except RuntimeError:
            mat = self._embed_one_by_one(safe)
        if mat.shape[0] != len(safe):
            raise RuntimeError(f"Embedding count mismatch: {mat.shape[0]} vs {len(safe)}")
        faiss.normalize_L2(mat)
        return mat

    def _embed_batched(self, texts: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            out = self._llm.create_embedding(input=batch)
            data = out.get("data", [])
            if not isinstance(data, list) or len(data) != len(batch):
                raise RuntimeError("Embedding batch mismatch.")
            try:
                data = sorted(data, key=lambda x: int(x.get("index", 0)))
            except Exception:
                pass
            for item in data:
                vecs.append(np.asarray(item["embedding"], dtype=np.float32))
        return np.vstack(vecs).astype(np.float32)

    def _embed_one_by_one(self, texts: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for t in texts:
            out = self._llm.create_embedding(input=t)
            data = out.get("data", [])
            if not isinstance(data, list) or not data:
                raise RuntimeError("Embedding response is empty.")
            vecs.append(np.asarray(data[0]["embedding"], dtype=np.float32))
        return np.vstack(vecs).astype(np.float32)

    def _truncate_by_tokens(self, text: str) -> str:
        if not text:
            return text
        toks = self._llm.tokenize(text.encode("utf-8", errors="ignore"))
        if len(toks) <= self._max_input_tokens:
            return text
        toks = toks[: self._max_input_tokens]
        detok = self._llm.detokenize(toks)
        try:
            return detok.decode("utf-8", errors="ignore")
        except Exception:
            return text