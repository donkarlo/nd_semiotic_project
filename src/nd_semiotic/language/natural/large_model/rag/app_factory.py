# -----------------------------
# AppFactory (OOP)
# -----------------------------
from nd_semiotic.language.natural.large_model.rag.answer_policy import \
    AnswerPolicy
from nd_semiotic.language.natural.large_model.rag.rag_engine import \
    RagEngine
from nd_semiotic.language.natural.large_model.rag.runtime_config import \
    RuntimeConfig
from nd_semiotic.language.natural.large_model.rag.terminal_app import \
    TerminalApp


class AppFactory:
    def __init__(self, cfg: RuntimeConfig, policy: AnswerPolicy):
        self._cfg = cfg
        self._policy = policy

    def build(self) -> TerminalApp:
        rag = RagEngine(cfg=self._cfg, policy=self._policy)
        return TerminalApp(rag=rag)