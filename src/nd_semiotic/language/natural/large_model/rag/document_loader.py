from pathlib import Path

# -----------------------------
# Document loading
# -----------------------------
class DocumentLoader:
    def __init__(self, max_chars_per_file: int = 250_000):
        self._max_chars_per_file = int(max_chars_per_file)

    def load_text(self, path: Path) -> str:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if len(raw) > self._max_chars_per_file:
            raw = raw[: self._max_chars_per_file]
        if path.suffix.lower() in (".schema.yaml", ".yml"):
            return self._load_yaml_as_pretty_text(raw)
        return raw

    def _load_yaml_as_pretty_text(self, raw: str) -> str:
        try:
            data = yaml.safe_load(raw)
        except Exception:
            return raw
        try:
            return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        except Exception:
            return raw