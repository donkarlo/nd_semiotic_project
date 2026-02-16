# -----------------------------
# RAG engine
# -----------------------------

from nd_semiotic.language.natural.large_model.rag.answer_policy import \
    AnswerPolicy
from nd_semiotic.language.natural.large_model.rag.chunker import Chunker
from nd_semiotic.language.natural.large_model.rag.document_loader import \
    DocumentLoader
from nd_semiotic.language.natural.large_model.rag.document_name_header_builder import \
    DocumentNameHeaderBuilder
from nd_semiotic.language.natural.large_model.rag.faisse_index import \
    FaissIndex
from nd_semiotic.language.natural.large_model.rag.file_scanner import \
    FileScanner
from nd_semiotic.language.natural.large_model.rag.instructions import \
    Instructions
from nd_semiotic.language.natural.large_model.rag.keyword_retriever import \
    KeywordRetriever
from nd_semiotic.language.natural.large_model.rag.llama_embedder import \
    LlamaEmbedder
from nd_semiotic.language.natural.large_model.rag.repo_finger_print import \
    RepoFingerprint
from nd_semiotic.language.natural.large_model.rag.retrieval_hit import \
    RetrievalHit
from nd_semiotic.language.natural.large_model.rag.runtime_config import \
    RuntimeConfig
from llama_cpp import Llama
from typing import List, Tuple, Optional, Set


class RagEngine:
    def __init__(self, cfg: RuntimeConfig, policy: AnswerPolicy):
        self._cfg = cfg
        self._policy = policy

        self._exclude_dirs = {
            ".git", ".rag_cache", "__pycache__", ".venv", "venv",
            "node_modules", "build", "dist", "out", ".idea", ".pytest_cache",
        }

        self._instructions = Instructions(self._cfg.instructions_yaml)

        if not self._cfg.root.exists():
            raise ValueError(f"Root does not exist: {self._cfg.root}")
        if not self._cfg.instructions_yaml.exists():
            raise ValueError(f"Instructions file does not exist: {self._cfg.instructions_yaml}")
        if not self._cfg.embed_model_gguf.exists():
            raise ValueError(f"Embed model does not exist: {self._cfg.embed_model_gguf}")
        if not self._cfg.chat_model_gguf.exists():
            raise ValueError(f"Chat model does not exist: {self._cfg.chat_model_gguf}")

        self._cache_dir = self._cfg.cache_dir or (self._cfg.root / ".rag_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._fp_path = self._cache_dir / "fingerprint.txt"
        self._faiss_path = self._cache_dir / "index.faiss"
        self._meta_path = self._cache_dir / "index.meta.json"

        self._embed_llm = Llama(
            model_path=str(self._cfg.embed_model_gguf),
            n_ctx=int(self._cfg.embed_n_ctx),
            n_threads=int(self._cfg.n_threads),
            n_batch=int(self._cfg.embed_n_batch),
            embedding=True,
            verbose=False,
        )
        self._chat_llm = Llama(
            model_path=str(self._cfg.chat_model_gguf),
            n_ctx=int(self._cfg.chat_n_ctx),
            n_threads=int(self._cfg.n_threads),
            n_batch=int(self._cfg.chat_n_batch),
            embedding=False,
            verbose=False,
        )

        max_input_tokens = max(128, int(self._cfg.embed_n_ctx) - int(self._cfg.embed_token_margin))

        self._scanner = FileScanner(
            root=self._cfg.root,
            extensions=self._cfg.extensions,
            exclude_dirs=self._exclude_dirs,
            include_roots=self._cfg.include_roots,
            max_files=int(self._cfg.max_files),
        )
        self._loader = DocumentLoader()
        self._name_header = DocumentNameHeaderBuilder(root=self._cfg.root)
        self._chunker = Chunker(
            chunk_size_chars=int(self._cfg.chunk_size_chars),
            overlap_chars=int(self._cfg.chunk_overlap_chars),
            max_chunks_total=int(self._cfg.max_chunks_total),
        )
        self._embedder = LlamaEmbedder(
            embed_llm=self._embed_llm,
            max_input_tokens=max_input_tokens,
            batch_size=int(self._cfg.embed_batch_size),
        )
        self._index = FaissIndex()
        self._fingerprint = RepoFingerprint(
            root=self._cfg.root,
            extensions=self._cfg.extensions,
            exclude_dirs=self._exclude_dirs,
            include_roots=self._cfg.include_roots,
            max_files=int(self._cfg.max_files),
        )
        self._kw: Optional[KeywordRetriever] = None

    def print_settings(self) -> None:
        print("RAG terminal is ready. Type a question and press Enter.", flush=True)
        print("Commands: :reload, :exit", flush=True)
        print(f"RAG root = {self._cfg.root}", flush=True)
        print(f"Instructions = {self._cfg.instructions_yaml}", flush=True)
        print(f"Embed model = {self._cfg.embed_model_gguf}", flush=True)
        print(f"Chat model = {self._cfg.chat_model_gguf}", flush=True)
        print(f"Cache dir = {self._cache_dir}", flush=True)
        print(f"Extensions = {self._cfg.extensions}", flush=True)
        print(f"Max files = {self._cfg.max_files}", flush=True)
        if self._cfg.include_roots:
            print(f"Include roots = {[str(p) for p in self._cfg.include_roots]}", flush=True)
        else:
            print("Include roots = (all)", flush=True)

    def _merge_hits(self, vec_hits: List[RetrievalHit], kw_hits: List[RetrievalHit], top_k: int) -> List[RetrievalHit]:
        seen: Set[Tuple[str, int]] = set()
        merged: List[RetrievalHit] = []

        for hit in vec_hits:
            key = (hit.chunk.doc_path, hit.chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)

        kw_boost = 0.05
        for hit in kw_hits:
            key = (hit.chunk.doc_path, hit.chunk.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(RetrievalHit(score=float(hit.score) + kw_boost, chunk=hit.chunk))

        merged.sort(key=lambda h: h.score, reverse=True)
        return merged[: int(top_k)]

    def ensure_index(self) -> None:
        current_fp = self._fingerprint.compute()
        cached_fp = self._fp_path.read_text(encoding="utf-8").strip() if self._fp_path.exists() else ""

        if self._faiss_path.exists() and self._meta_path.exists() and cached_fp == current_fp:
            self._index.load(self._faiss_path, self._meta_path)
            self._kw = KeywordRetriever(self._index.chunks())
            print("Index loaded from cache.", flush=True)
            return

        self.rebuild_index()
        self._fp_path.write_text(current_fp, encoding="utf-8")

    def rebuild_index(self) -> None:
        print("Building index...", flush=True)

        docs: List[Tuple[str, str]] = []
        paths = list(self._scanner.iter_files())
        print(f"Found {len(paths)} files to index.", flush=True)

        for i, path in enumerate(paths, 1):
            try:
                txt = self._loader.load_text(path)
            except Exception:
                continue

            header = self._name_header.build_header(path)
            docs.append((str(path), header + txt))

            if i % 50 == 0:
                print(f"Loaded {i}/{len(paths)} files...", flush=True)

        chunks = self._chunker.chunk_all(docs)
        print(f"Produced {len(chunks)} chunks.", flush=True)

        if not chunks:
            raise RuntimeError("No chunks produced. Check root/extensions/include_roots/max_files.")

        texts = [c.text for c in chunks]
        print("Embedding chunks...", flush=True)
        emb = self._embedder.embed(texts)
        print("Embedding done.", flush=True)

        self._index.build(embeddings=emb, chunks=chunks)
        self._index.save(self._faiss_path, self._meta_path)
        self._kw = KeywordRetriever(self._index.chunks())
        print("Index built and saved.", flush=True)

    def answer(self, question: str, top_k: int = 10, max_tokens: int = 200) -> str:
        q = (question or "").strip()
        if not q:
            return ""

        if self._kw is None:
            self._kw = KeywordRetriever(self._index.chunks())

        kw_hits = self._kw.search(q, k=max(8, int(top_k)))
        q_emb = self._embedder.embed([q])
        vec_hits = self._index.search(q_emb, k=max(8, int(top_k)))
        merged = self._merge_hits(vec_hits=vec_hits, kw_hits=kw_hits, top_k=int(top_k))

        use_context = self._policy.should_answer_with_context(vec_hits=vec_hits, kw_hits=kw_hits)
        if not use_context:
            if self._policy.allow_general_fallback():
                return self._general_answer(q=q, max_tokens=int(max_tokens))
            return "In indexed files, I cannot find it."

        if len(q.split()) == 1:
            hits = (kw_hits or merged)[: self._policy.max_sources()]
            return self._snippet_answer_from_hits(hits)

        return self._contextual_answer(q=q, hits=merged, max_tokens=int(max_tokens))

    def _snippet_answer_from_hits(self, hits: List[RetrievalHit]) -> str:
        lines: List[str] = []
        for hit in hits:
            ch = hit.chunk
            snippet = ch.text[:800]
            if self._policy.show_sources():
                lines.append(f"[source={ch.doc_path} chunk={ch.chunk_id} score={hit.score:.4f}]\n{snippet}")
            else:
                lines.append(snippet)
        return "\n\n".join(lines)

    def _general_answer(self, q: str, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": self._instructions.general_system_prompt()},
            {"role": "user", "content": q},
        ]
        out = self._chat_llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=0.2,
        )
        return out["choices"][0]["message"]["content"]

    def _contextual_answer(self, q: str, hits: List[RetrievalHit], max_tokens: int) -> str:
        kept = self._fit_context_to_n_ctx(q=q, hits=hits, max_tokens=int(max_tokens), per_chunk_char_limit=600)
        context_lines: List[str] = []
        for hit in kept[: self._policy.max_sources()]:
            ch = hit.chunk
            txt = ch.text[:600]
            if self._policy.show_sources():
                context_lines.append(f"[source={ch.doc_path} chunk={ch.chunk_id} score={hit.score:.4f}]\n{txt}")
            else:
                context_lines.append(txt)
        context = "\n\n".join(context_lines)

        messages = [
            {"role": "system", "content": self._instructions.system_prompt()},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{q}"},
        ]

        # hard guard
        n_ctx = int(self._chat_llm.n_ctx())
        if self._count_chat_prompt_tokens(messages) + int(max_tokens) > n_ctx:
            if self._policy.allow_general_fallback():
                return self._general_answer(q=q, max_tokens=int(max_tokens))
            return "In indexed files, I cannot fit context into the model window."

        out = self._chat_llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=0.2,
        )
        return out["choices"][0]["message"]["content"]

    def _count_chat_prompt_tokens(self, messages: List[dict]) -> int:
        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts.append(f"{role.upper()}:\n{content}\n")
        prompt = "\n".join(parts)
        toks = self._chat_llm.tokenize(prompt.encode("utf-8", errors="ignore"))
        return len(toks)

    def _fit_context_to_n_ctx(
            self,
            q: str,
            hits: List[RetrievalHit],
            max_tokens: int,
            per_chunk_char_limit: int = 600,
            min_hits: int = 1,
    ) -> List[RetrievalHit]:
        if max_tokens < 1:
            max_tokens = 1
        n_ctx = int(self._chat_llm.n_ctx())

        def build_messages(candidate_hits: List[RetrievalHit], limit: int) -> List[dict]:
            lines: List[str] = []
            for hit in candidate_hits[: self._policy.max_sources()]:
                ch = hit.chunk
                lines.append(f"[source={ch.doc_path} chunk={ch.chunk_id} score={hit.score:.4f}]\n{ch.text[:limit]}")
            ctx = "\n\n".join(lines)
            return [
                {"role": "system", "content": self._instructions.system_prompt()},
                {"role": "user", "content": f"CONTEXT:\n{ctx}\n\nQUESTION:\n{q}"},
            ]

        candidate = hits[:]
        limit = int(per_chunk_char_limit)

        while True:
            msgs = build_messages(candidate, limit)
            prompt_tokens = self._count_chat_prompt_tokens(msgs)

            if prompt_tokens + max_tokens <= n_ctx:
                return candidate[:]

            if limit > 200:
                limit = max(200, int(limit * 0.8))
                continue

            if len(candidate) > min_hits:
                candidate = candidate[:-1]
                continue

            if limit > 80:
                limit = max(80, int(limit * 0.7))
                continue

            return candidate[:]