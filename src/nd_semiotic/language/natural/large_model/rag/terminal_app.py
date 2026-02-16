import time

from nd_semiotic.language.natural.large_model.rag.rag_engine import RagEngine


# -----------------------------
# Terminal app
# -----------------------------
class TerminalApp:
    def __init__(self, rag: RagEngine):
        self._rag = rag

    def run(self) -> None:
        self._rag.print_settings()
        self._rag.ensure_index()
        print("Ready for questions.", flush=True)

        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                return

            if not q:
                continue
            if q == ":exit":
                return
            if q == ":reload":
                self._rag._instructions.reload()
                start = time.time()
                self._rag.rebuild_index()
                elapsed = time.time() - start
                print(f"Ready for questions. (reindex: {elapsed:.1f}s)", flush=True)
                continue

            ans = self._rag.answer(q, top_k=12, max_tokens=220)
            print("\n" + ans, flush=True)