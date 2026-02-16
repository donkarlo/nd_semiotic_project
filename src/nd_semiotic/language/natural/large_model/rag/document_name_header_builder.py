import re
from pathlib import Path
from typing import List


# -----------------------------
# LinearAlgebra header injection
# -----------------------------
class DocumentNameHeaderBuilder:
    def __init__(self, root: Path):
        self._root = root

    def build_header(self, path: Path) -> str:
        rel = self._safe_relative(path).as_posix()
        stem = path.stem
        tokens = " ".join(self._tokenize_path(rel))
        lines = [f"__DOC_PATH__ {rel}", f"__DOC_STEM__ {stem}"]
        if tokens:
            lines.append(f"__DOC_TOKENS__ {tokens}")
        return "\n".join(lines) + "\n\n"

    def _safe_relative(self, path: Path) -> Path:
        try:
            return path.relative_to(self._root)
        except Exception:
            return path

    def _tokenize_path(self, rel_posix: str) -> List[str]:
        parts = re.split(r"[\/\._\-]+", rel_posix)
        return [p.lower() for p in parts if p]