import os
from typing import Iterable, List, Tuple, Optional, Set, Dict
from pathlib import Path

class FileScanner:
    def __init__(
            self,
            root: Path,
            extensions: Tuple[str, ...],
            exclude_dirs: Optional[Set[str]] = None,
            include_roots: Optional[List[Path]] = None,
            max_files: int = 500,
    ):
        self._root = root
        self._extensions = tuple(e.lower() for e in extensions)
        self._exclude_dirs = set(exclude_dirs) if exclude_dirs else {
            ".git", ".rag_cache", "__pycache__", ".venv", "venv",
            "node_modules", "build", "dist", "out", ".idea", ".pytest_cache",
        }
        self._include_roots = include_roots
        self._max_files = int(max_files)

    def iter_files(self) -> Iterable[Path]:
        roots: List[Path] = []
        if not self._include_roots:
            roots.append(self._root)
        else:
            for r in self._include_roots:
                roots.append(r if r.is_absolute() else (self._root / r))

        seen = 0
        for base in roots:
            if not base.exists():
                continue

            for dirpath, dirnames, filenames in os.walk(str(base), followlinks=False):
                dirnames[:] = [d for d in dirnames if d not in self._exclude_dirs]
                dirnames.sort()
                filenames.sort()

                for fn in filenames:
                    if seen >= self._max_files:
                        return
                    p = Path(dirpath) / fn
                    if p.suffix.lower() not in self._extensions:
                        continue
                    yield p
                    seen += 1