from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    doc_path: str
    chunk_id: int
    text: str