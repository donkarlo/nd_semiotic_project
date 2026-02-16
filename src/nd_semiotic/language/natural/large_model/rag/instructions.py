from pathlib import Path
from typing import Dict
import yaml

class Instructions:
    def __init__(self, path: Path):
        self._path = path
        self._data: Dict[str, object] = {}
        self.reload()

    def reload(self) -> None:
        try:
            raw = self._path.read_text(encoding="utf-8", errors="ignore")
            data = yaml.safe_load(raw)
            self._data = data if isinstance(data, dict) else {}
        except Exception:
            self._data = {}

    def system_prompt(self) -> str:
        sp = self._data.get("system_prompt")
        if isinstance(sp, str) and sp.strip():
            return sp.strip()
        return (
            "Use CONTEXT if it helps. If CONTEXT is irrelevant or weak, answer directly. "
            "Be concise and actionable."
        )

    def general_system_prompt(self) -> str:
        gsp = self._data.get("general_system_prompt")
        if isinstance(gsp, str) and gsp.strip():
            return gsp.strip()
        return "Answer concisely and actionably."