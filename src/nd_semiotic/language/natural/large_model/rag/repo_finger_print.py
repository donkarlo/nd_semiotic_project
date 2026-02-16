from pathlib import Path
from typing import List, Tuple, Optional, Set
import hashlib

from nd_semiotic.language.natural.large_model.rag.file_scanner import \
    FileScanner


# -----------------------------
# Fingerprint
# -----------------------------
class RepoFingerprint:
    def __init__(
            self,
            root: Path,
            extensions: Tuple[str, ...],
            exclude_dirs: Set[str],
            include_roots: Optional[List[Path]],
            max_files: int,
    ):
        self._root = root
        self._extensions = tuple(e.lower() for e in extensions)
        self._exclude_dirs = set(exclude_dirs)
        self._include_roots = include_roots
        self._max_files = int(max_files)

    def compute(self) -> str:
        h = hashlib.sha256()
        scanned = 0
        scanner = FileScanner(
            root=self._root,
            extensions=self._extensions,
            exclude_dirs=self._exclude_dirs,
            include_roots=self._include_roots,
            max_files=self._max_files,
        )
        for p in scanner.iter_files():
            if scanned >= self._max_files:
                break
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            h.update(str(p).encode("utf-8", errors="ignore"))
            h.update(str(int(st.st_mtime)).encode("utf-8"))
            h.update(str(int(st.st_size)).encode("utf-8"))
            scanned += 1
        return h.hexdigest()