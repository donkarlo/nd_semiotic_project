from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Set, Dict
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeConfig:
    root: Path
    instructions_yaml: Path
    embed_model_gguf: Path
    chat_model_gguf: Path

    extensions: Tuple[str, ...] = (".schema.yaml", ".yml", ".tex", ".md", ".txt", ".pdf")
    max_files: int = 5000
    include_roots: Optional[List[Path]] = None

    cache_dir: Optional[Path] = None

    max_chunks_total: int = 12000
    chunk_size_chars: int = 1500
    chunk_overlap_chars: int = 150

    embed_n_ctx: int = 2048
    chat_n_ctx: int = 2048
    n_threads: int = 4
    embed_n_batch: int = 64
    chat_n_batch: int = 32

    embed_batch_size: int = 8
    embed_token_margin: int = 64